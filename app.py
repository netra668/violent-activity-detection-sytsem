from flask import Flask, render_template, request, send_from_directory, Response
import os
import cv2
import logging
import csv
from datetime import datetime, timedelta
from playsound import playsound
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image
import torch
import numpy as np
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load classification model (MobileNetV2)
classification_model = models.mobilenet_v2(pretrained=False)
classification_model.classifier[1] = torch.nn.Linear(classification_model.classifier[1].in_features, 2)
classification_model.load_state_dict(torch.load('best_model_optimized.pth'))
classification_model.eval()

# Load detection model (YOLO)
detection_model = YOLO("yolov8n.pt", verbose=False)
weapon_detection_model = YOLO("Weapon-Detection-YOLOv8/runs/detect/train6/weights/best.pt", verbose=False)

# Define transforms for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

interested_classes = [0, 1, 2, 3, 5, 7, 9, 10, 13, 25, 30, 31, 32, 34, 36, 37, 38, 42, 43, 56, 67, 68, 69, 70, 75, 76]
COCO_CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

weapon_interested_classes = [0, 3, 4, 5]
weapon_classes = ['pistol', 'smartphone', 'knife', 'monedero', 'bill', 'card']

# Set up logging
log_file = 'live_feed_log.csv'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Create CSV file
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Label', 'Detected Objects'])

violence_start_time = None
violence_duration_threshold = timedelta(seconds=30)  
siren_sound_file = 'warning-alarm.mp3'  

def process_frame(frame):
    global violence_start_time
    # Classification (MobileNetV2)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_frame).unsqueeze(0)
    with torch.no_grad():
        classification_output = classification_model(input_tensor)
        is_violent = classification_output.argmax(1).item() == 1

    # Detection (YOLO - general)
    results = detection_model(frame)
    annotated_frame = frame.copy()

    for result in results[0].boxes:
        class_id = int(result.cls)
        if class_id in interested_classes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            class_name = COCO_CLASSES[class_id]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
            cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Detection (YOLO - Weapon)
    weapon_results = weapon_detection_model(frame)

    for result in weapon_results[0].boxes:
        class_id = int(result.cls)
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        class_name = weapon_classes[class_id]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    label = "Violent" if is_violent else "Nonviolent"
    color = (0, 0, 255) if is_violent else (0, 255, 0)
    cv2.putText(annotated_frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    detected_objects = [COCO_CLASSES[int(result.cls)] for result in results[0].boxes if int(result.cls) in interested_classes]
    detected_weapons = [weapon_classes[int(result.cls)] for result in weapon_results[0].boxes]

    log_entry = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Label': label,
        'Detected Objects': ', '.join(detected_objects + detected_weapons)
    }
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(log_entry.values())

     # Check for prolonged violence detection
    if is_violent:
        if violence_start_time is None:
            violence_start_time = datetime.now()
        elif datetime.now() - violence_start_time > violence_duration_threshold:
            # Trigger alert
            print("Alert: Prolonged violent activity detected!")
            playsound(siren_sound_file)
            # Reset the start time to avoid repeated alerts
            violence_start_time = datetime.now()
    else:
        violence_start_time = None
        
    return annotated_frame

@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html')

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = process_frame(frame)
        _, jpeg = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        file_ext = os.path.splitext(file.filename)[1]
        input_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}{file_ext}")
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{os.path.basename(input_path)}")
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 
                              int(cap.get(cv2.CAP_PROP_FPS)), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(process_frame(frame))
        cap.release()
        out.release()

        return render_template("index.html", message=f"Video processed: {os.path.basename(output_path)}")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()