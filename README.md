# 🎥 AI Surveillance System

An AI-powered real-time surveillance platform that detects people and vehicles, monitors restricted zones, and generates intelligent alerts for security events such as intrusion, loitering, crowd formation, and wrong-direction movement.

---

# 🚀 Features

## Object Detection

Using YOLO11, the system can detect:

* Person
* Car
* Motorcycle
* Bus
* Truck

---

## Event Detection

### 🚨 Intrusion Detection

Detects when a person or vehicle enters a restricted area.

### ⏳ Loitering Detection

Triggers alerts when a person remains inside a zone longer than a configured threshold.

### 👥 Crowd Detection

Detects when the number of people inside a zone exceeds a configured limit.

### ↔ Wrong Direction Detection

Detects movement against an allowed direction.

Supported directions:

* Left
* Right
* Up
* Down

---

## Supported Video Sources

### Offline Sources

* MP4
* AVI
* MOV
* MKV

### Live Sources

* USB Webcam
* RTSP CCTV Cameras
* RTMP Streams
* HTTP/MJPEG Streams

---

# 📂 Repository Structure

```text
AI-Surveillance-System/
│
├── ai_surveillance_model/
│   ├── detector.py
│   ├── tracker.py
│   ├── analyzer.py
│   ├── config.py
│
├── complete_detector.py
├── ui_streamlit.py
├── draw_zone_interactive.py
├── diagnostic.py
│
├── requirements.txt
├── setup.py
├── setup.bat
├── setup.sh
│
├── zones.pkl
├── README.md
```

---

# 💻 System Requirements

## Minimum Requirements

| Component | Requirement                             |
| --------- | --------------------------------------- |
| OS        | Windows 10/11, Ubuntu 20.04+, macOS 11+ |
| RAM       | 8 GB                                    |
| Storage   | 10 GB Free                              |
| Python    | 3.8 - 3.11                              |
| Internet  | Required for first model download       |

---

## Recommended Requirements

| Component | Requirement          |
| --------- | -------------------- |
| RAM       | 16 GB+               |
| GPU       | NVIDIA GPU with CUDA |
| Storage   | SSD                  |
| Python    | 3.10                 |

---

# 📥 Clone the Repository

## Option 1: Using Git (Recommended)

### Step 1: Install Git

Download Git:

https://git-scm.com/downloads

Verify installation:

```bash
git --version
```

Expected output:

```text
git version 2.x.x
```

---

### Step 2: Clone Repository

Replace with your actual repository URL:

```bash
git clone https://github.com/USERNAME/AI-Surveillance-System.git
```

Example:

```bash
git clone https://github.com/siddharthshetty/AI-Surveillance-System.git
```

---

### Step 3: Move into Project Folder

```bash
cd AI-Surveillance-System
```

---

### Step 4: Verify Files

```bash
dir
```

Windows

or

```bash
ls
```

Linux/macOS

You should see:

```text
complete_detector.py
ui_streamlit.py
requirements.txt
setup.py
README.md
```

---

# 📦 Download ZIP Instead of Git

If you don't want to use Git:

1. Open repository
2. Click Code
3. Click Download ZIP
4. Extract ZIP
5. Open terminal inside extracted folder

---

# 🪟 Windows Installation

## Step 1: Create Virtual Environment

```cmd
python -m venv venv
```

---

## Step 2: Activate Environment

### Command Prompt

```cmd
venv\Scripts\activate
```

### PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell gives an error:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

---

## Step 3: Upgrade Pip

```cmd
python -m pip install --upgrade pip
```

---

## Step 4: Install Dependencies

```cmd
pip install -r requirements.txt
```

---

## Step 5: Install Project Package

```cmd
pip install -e .
```

---

## Step 6: Verify Installation

```cmd
python diagnostic.py
```

Expected Output:

```text
✓ Ultralytics
✓ OpenCV
✓ PyTorch
✓ FastAPI
✓ AI Surveillance Model
✓ All tests passed
```

---

# 🍎 macOS Installation

## Create Environment

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## Upgrade Pip

```bash
python -m pip install --upgrade pip
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Package

```bash
pip install -e .
```

---

## Verify

```bash
python diagnostic.py
```

---

# 🐧 Ubuntu/Linux Installation

## Install Python and FFmpeg

```bash
sudo apt update

sudo apt install python3 python3-pip python3-venv ffmpeg -y
```

---

## Create Environment

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt

pip install -e .
```

---

## Verify Installation

```bash
python diagnostic.py
```

---

# 🤖 First-Time Model Download

The first time you launch the application, YOLO downloads the selected model automatically.

Examples:

```text
yolo11n.pt
yolo11s.pt
yolo11m.pt
yolo11l.pt
yolo11x.pt
```

Internet is required only once.

Downloaded models are cached automatically.

---

# ▶ Running the Web Dashboard

Launch Streamlit:

```bash
streamlit run ui_streamlit.py
```

Open browser:

```text
http://localhost:8501
```

---

# ▶ Running the Command-Line Version

```bash
python complete_detector.py
```

---

# 🎥 Using a Video File

Select:

```text
Video File
```

Then provide:

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
Webcam
```

The system automatically detects connected cameras.

---

# 📡 Using RTSP CCTV Cameras

Example URLs:

## Hikvision

```text
rtsp://username:password@ip:554/Streaming/Channels/101
```

## Dahua

```text
rtsp://username:password@ip:554/cam/realmonitor?channel=1&subtype=0
```

## Generic

```text
rtsp://admin:admin123@192.168.1.10:554/live
```

---

# 🗺️ Creating Monitoring Zones

## Method 1: Interactive Zone Drawing Tool

Run:

```bash
python draw_zone_interactive.py
```

Controls:

| Key         | Action             |
| ----------- | ------------------ |
| Left Click  | Add Point          |
| Right Click | Save Zone          |
| C           | Clear Current Zone |
| R           | Reset All Zones    |
| S           | Save Zones         |
| L           | Load Zones         |
| Q           | Exit               |

---

## Method 2: Streamlit Dashboard

Example coordinates:

```text
100,100 500,100 500,400 100,400
```

Minimum 3 points required.

---

# ⚡ Performance Tuning

## Low-End Systems

```text
Model: YOLO11n
Confidence: 0.3
Frame Skip: 5
```

---

## Mid-Range Systems

```text
Model: YOLO11m
Confidence: 0.4
Frame Skip: 2
```

---

## High-End GPU Systems

```text
Model: YOLO11x
Confidence: 0.5
Frame Skip: 1
```

---

# 🧪 Troubleshooting

## Python Not Found

```text
'python' is not recognized
```

Fix:

Install Python:

https://python.org/downloads

Check:

```bash
python --version
```

---

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

## CUDA Not Working

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected:

```text
True
```

---

## RTSP Camera Not Connecting

Check:

* Camera IP
* Username
* Password
* Port 554
* Firewall Rules

Test URL first using VLC Media Player.

---

# 📈 Future Enhancements

* Face Recognition
* License Plate Recognition
* Email Alerts
* WhatsApp Alerts
* SMS Notifications
* Cloud Storage
* Database Logging
* REST APIs
* Multi-Camera Dashboard
* Heatmaps
* Object Re-Identification

---

# 👨‍💻 Author

Siddharth Shetty

Built with:

* Python
* YOLO11
* OpenCV
* PyTorch
* Streamlit

---
