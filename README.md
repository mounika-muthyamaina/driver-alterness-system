# Driver Alertness System

## Abstract
The **Real-Time Driver Alertness System with Alarm** is designed to enhance driver safety by monitoring facial landmarks and detecting signs of drowsiness and distractions. It utilizes **dlib** for facial landmark detection, **OpenCV (cv2)** for real-time video processing, and **YOLOv5** for object detection. The system can identify drowsy behavior through **eye aspect ratio (EAR) analysis** and detect distractions such as **mobile phone usage and eating**. When a potential risk is detected, the system triggers an alarm and logs the incident into a **MySQL database**, helping to prevent accidents and promote safe driving habits.

---

## Features
- **Real-time facial landmark detection** using **dlib**.
- **Eye Aspect Ratio (EAR) monitoring** for drowsiness detection.
- **Object detection with YOLOv5** to identify distractions like mobile phone use, food consumption, and more.
- **Audio alarm alerts** using Pygame's mixer.
- **Text-to-Speech warnings** via pyttsx3.
- **Incident logging** in a MySQL database.
- **User-friendly controls**: Press 'q' to quit, 'a' to stop the alarm, and 'h' to view logged incidents.

---

## Technologies Used
- **Python**
- **OpenCV (cv2)**
- **dlib (Facial Landmark Detection)**
- **YOLOv5 (Object Detection)**
- **Pygame (Alarm Sound)**
- **pyttsx3 (Text-to-Speech Alerts)**
- **MySQL (Incident Logging)**

---

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/driver-alertness-system.git
cd driver-alertness-system
```

### Step 2: Install Dependencies
```bash
pip install opencv-python dlib numpy scipy pygame torch pyttsx3 mysql-connector-python
```

> **Note:** You must also download the YOLOv5 model and **shape_predictor_68_face_landmarks.dat** for facial landmark detection.

### Step 3: Set Up MySQL Database
1. Open MySQL and create a database:
    ```sql
    CREATE DATABASE driver_alertness;
    ```
2. Create a table for logging incidents:
    ```sql
    USE driver_alertness;
    CREATE TABLE incidents (
        id INT AUTO_INCREMENT PRIMARY KEY,
        incident_type VARCHAR(255),
        timestamp DATETIME
    );
    ```

---

## Usage
Run the system using:
```bash
python main.py
```

### Controls:
- **Press 'q'** → Quit the application.
- **Press 'a'** → Stop the alarm.
- **Press 'h'** → View the incident history (logged in MySQL database).

---

## File Structure
```
driver-alertness-system/
│── main.py  # Main script to run the detection system
│── alarm.wav  # Alarm sound file
│── shape_predictor_68_face_landmarks.dat  # Facial landmark model
│── README.md  # Documentation
```

---

## Future Enhancements
- **Integration with IoT-based vehicle systems** for automatic braking.
- **Improved deep learning models** for more accurate detection.
- **Mobile app integration** to send alerts to emergency contacts.

---

## License
This project is licensed under the **MIT License**.

---

## Acknowledgments
Special thanks to the developers of **OpenCV, dlib, and YOLOv5** for providing powerful tools to build this system.

