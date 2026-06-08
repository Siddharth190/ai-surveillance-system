# 🎥 AI Surveillance System

> AI-powered real-time surveillance system for detecting people and vehicles, monitoring restricted areas, and generating intelligent security alerts.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v11-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Dashboard-orange)

---

# 📌 Overview

AI Surveillance System is an intelligent video analytics platform built using YOLO, OpenCV, and Streamlit.

The system performs:

* Real-time object detection
* Multi-object tracking
* Intrusion detection
* Loitering detection
* Crowd detection
* Wrong-direction detection
* CCTV monitoring through RTSP streams
* Vehicle and person analytics

Suitable for:

* Smart Cities
* Airports
* Railway Stations
* Campuses
* Industrial Areas
* Shopping Malls
* Parking Lots
* Restricted Security Zones

---

# ✨ Features

## Object Detection

Detects:

| Class      | Supported |
| ---------- | --------- |
| Person     | ✅         |
| Car        | ✅         |
| Motorcycle | ✅         |
| Bus        | ✅         |
| Truck      | ✅         |

Powered by YOLO11 models:

* YOLO11n (Fastest)
* YOLO11s
* YOLO11m
* YOLO11l
* YOLO11x (Highest Accuracy)

---

## Event Detection

### 🚨 Intrusion Detection

Detects when a person or vehicle enters a restricted zone.

### ⏳ Loitering Detection

Triggers an alert when an object remains inside a zone longer than a configured threshold.

### 👥 Crowd Detection

Detects when the number of people inside a zone exceeds a configured limit.

### ↔ Wrong Direction Detection

Detects movement opposite to the allowed direction.

Supported directions:

* Left
* Right
* Up
* Down

---

## Video Sources

### Supported Inputs

* MP4 Videos
* AVI Videos
* MOV Videos
* MKV Videos
* USB Webcam
* RTSP Cameras
* RTMP Streams
* HTTP/MJPEG Streams

---

## Interactive Zone Management

Create monitoring zones directly on video frames.

Features:

* Polygon based zones
* Multiple zones
* Save zones
* Load zones
* Editable zone configuration

---

# 🖥️ System Requirements

## Minimum

* Windows 10/11
* Ubuntu 20.04+
* macOS 11+
* 8 GB RAM
* 10 GB Free Storage
* Python 3.8+

## Recommended

* 16 GB RAM
* NVIDIA GPU with CUDA support
* Python 3.10

---

# 📂 Project Structure

```text
AI-Surveillance-System/
│
├── ui_streamlit.py
├── complete_detector.py
├── detector.py
├── tracker.py
├── analyzer.py
├── config.py
│
├── draw_zone_interactive.py
├── diagnostic.py
│
├── requirements.txt
├── setup.py
├── setup.bat
├── setup.sh
│
├── zones.pkl
│
└── README.md
```

---

# 🚀 Installation Guide

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/AI-Surveillance-System.git

cd AI-Surveillance-System
```

---

## Step 2: Verify Python

Check Python version:

```bash
python --version
```

or

```bash
python3 --version
```

Required:

```text
Python 3.8 - 3.11
```

---

# 🪟 Windows Installation

## Create Virtual Environment

```cmd
python -m venv venv
```

---

## Activate Environment

### Command Prompt

```cmd
venv\Scripts\activate
```

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

---

## Upgrade Pip

```cmd
python -m pip install --upgrade pip
```

---

## Install Dependencies

```cmd
pip install -r requirements.txt
```

---

## Install Local Package

```cmd
pip install -e .
```

---

## Verify Installation

```cmd
python diagnostic.py
```

Expected output:

```text
✓ Ultralytics
✓ OpenCV
✓ PyTorch
✓ FastAPI
✓ AI Surveillance Model
```

---

# 🍎 macOS Installation

```bash
python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

pip install -e .
```

Or run:

```bash
chmod +x setup.sh

./setup.sh
```

---

# 🐧 Ubuntu/Linux Installation

## Install System Packages

```bash
sudo apt update

sudo apt install python3-pip python3-venv ffmpeg -y
```

---

## Create Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Requirements

```bash
pip install --upgrade pip

pip install -r requirements.txt

pip install -e .
```

---

# 📦 Required Python Packages

```text
ultralytics
opencv-python
numpy
torch
streamlit
fastapi
uvicorn
python-multipart
pillow
matplotlib
```

---

# 🎯 First-Time Setup

When launched for the first time:

YOLO automatically downloads the selected model.

Example:

```text
yolo11n.pt
yolo11s.pt
yolo11m.pt
yolo11l.pt
yolo11x.pt
```

Internet connection is required only once.

---

# ▶ Running the Dashboard

## Launch Streamlit UI

```bash
streamlit run ui_streamlit.py
```

Open browser:

```text
http://localhost:8501
```

---

# ▶ Running the Console Version

```bash
python complete_detector.py
```

You will be asked:

```text
1. Object Type
2. Direction
3. Loitering Time
4. Crowd Threshold
5. Confidence Threshold
6. YOLO Model
7. Video Source
```

---

# 🎥 Using a Video File

Example:

```text
C:\Videos\sample.mp4
```

or

```text
/home/user/sample.mp4
```

Supported formats:

```text
.mp4
.avi
.mov
.mkv
```

---

# 📷 Using Webcam

Select:

```text
Video Source → Webcam
```

The system automatically detects available cameras.

---

# 📡 Using RTSP CCTV Camera

Example:

```text
rtsp://username:password@192.168.1.100:554/Streaming/Channels/101
```

or

```text
rtsp://admin:admin123@192.168.1.10:554/live
```

---

# 🗺️ Creating Monitoring Zones

Option 1:

Use the built-in interactive zone editor.

```bash
python draw_zone_interactive.py
```

Instructions:

* Left Click → Add Point
* Right Click → Save Zone
* C → Clear
* R → Reset
* Q → Quit

---

Option 2:

Create zones from the Streamlit dashboard.

Example:

```text
100,100 500,100 500,400 100,400
```

---

# ⚙️ Performance Optimization

## Low-End PC

Use:

```text
YOLO11n
Confidence = 0.3
Frame Skip = 5
```

---

## Mid-Range PC

Use:

```text
YOLO11m
Confidence = 0.4
Frame Skip = 2
```

---

## High-End GPU

Use:

```text
YOLO11x
Confidence = 0.5
Frame Skip = 1
```

---

# 🧪 Troubleshooting

## OpenCV Error

```text
ModuleNotFoundError: cv2
```

Fix:

```bash
pip install opencv-python
```

---

## Streamlit Not Found

```bash
pip install streamlit
```

---

## CUDA Not Detected

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## RTSP Not Working

Check:

* Camera IP
* Username
* Password
* Firewall
* Port 554

Test URL using VLC first.

---

## Model Download Failed

Download manually:

```bash
yolo11n.pt
yolo11s.pt
yolo11m.pt
```

Place them in project root directory.

---

# 📈 Future Improvements

* Email Alerts
* SMS Alerts
* WhatsApp Alerts
* Face Recognition
* License Plate Recognition
* Heatmaps
* Cloud Storage
* Multi-Camera Dashboard
* Database Logging
* REST API Integration

---

# 👨‍💻 Author

Developed as an AI-powered surveillance and video analytics platform using:

* Python
* YOLO11
* OpenCV
* Streamlit
* PyTorch

---

