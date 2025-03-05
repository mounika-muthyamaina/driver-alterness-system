import cv2
import dlib
import time
import numpy as np
from scipy.spatial import distance
from pygame import mixer
import torch
import pyttsx3
import mysql.connector

# Initialize Pygame mixer for alarm
mixer.init()
mixer.music.load('alarm.wav')  # Ensure you have an alarm.wav file in the working directory

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Load facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Constants
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 20
DISTRACTION_CLASSES = ['cell phone', 'bottle', 'cup', 'remote', 'mouse', 'food', 'bread', 'juice', 'glass', 'fruit']
EYE_CLOSED_DURATION_THRESHOLD = 2  # Changed to 2 seconds

# Confidence threshold for object detection
CONFIDENCE_THRESHOLD = 0.5

# Initialize counters and timers
COUNTER = 0
ALARM_ON = False
eyes_closed_start_time = None

# Initialize database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="mona_1379",
    database="driver_alertness"
)
cursor = db.cursor()

def log_incident(incident_type):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('INSERT INTO incidents (incident_type, timestamp) VALUES (%s, %s)', (incident_type, timestamp))
    db.commit()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness_and_distraction(frame):
    global COUNTER, ALARM_ON, eyes_closed_start_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        leftEye = shape[36:42]
        rightEye = shape[42:48]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    mixer.music.play()
                    engine.say("Warning: Drowsiness detected.")
                    engine.runAndWait()
                    eyes_closed_start_time = time.time()  # Start the timer when eyes are closed
                    log_incident('Drowsiness detected')
        else:
            COUNTER = 0
            ALARM_ON = False
            mixer.music.stop()
            if eyes_closed_start_time:
                eyes_closed_start_time = None  # Reset the timer if eyes open

    # Check for drowsiness based on closed eyes duration
    if eyes_closed_start_time:
        eyes_closed_duration = time.time() - eyes_closed_start_time
        if eyes_closed_duration >= EYE_CLOSED_DURATION_THRESHOLD:
            if not ALARM_ON:
                ALARM_ON = True
                mixer.music.play()
                engine.say("Warning: Drowsiness detected.")
                engine.runAndWait()
                log_incident('Drowsiness detected')

    # Object detection for distractions
    results = model(frame)

    # Filter detections based on confidence threshold
    filtered_results = results.xyxy[0][results.xyxy[0][:, 4] > CONFIDENCE_THRESHOLD]

    # Move filtered results to CPU for NMS
    filtered_results = filtered_results.cpu()

    for i, (xmin, ymin, xmax, ymax, confidence, cls) in enumerate(filtered_results):
        if results.names[int(cls)] in DISTRACTION_CLASSES:
            # Define a region representing the driver's hand (adjust as needed)
            hand_region = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
            if hand_region.size > 0:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"WARNING: Distraction Detected.", (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                engine.say(f"Warning:Distraction detected.")
                engine.runAndWait()
                log_incident(f'{results.names[int(cls)]} detected')
    
    return frame

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_drowsiness_and_distraction(frame)
        
        cv2.imshow("Driver Alertness Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):  # Press 'a' to turn off alarm
            mixer.music.stop()
        elif key == ord('h'):  # Press 'h' to show history
            cursor.execute("SELECT * FROM incidents")
            data = cursor.fetchall()
            for row in data:
                print(row)
    
    cap.release()
    cv2.destroyAllWindows()
    db.close()

if __name__ == "__main__":
    main()
