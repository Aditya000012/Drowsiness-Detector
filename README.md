# 🛡️ DrowsGuard — Real-Time Drowsiness Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Face_Landmarker-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>

**DrowsGuard** is an AI-powered real-time drowsiness detection system that runs as a web application in your browser. It uses your webcam to continuously monitor facial landmarks, computes the **Eye Aspect Ratio (EAR)**, and triggers loud audio-visual alerts when signs of fatigue are detected — helping prevent accidents caused by drowsy driving, late-night studying, or shift work.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real-Time Face Tracking** | Detects and tracks 468 facial landmarks using Google's MediaPipe Face Landmarker |
| 👁️ **Eye Aspect Ratio (EAR)** | Calculates EAR from 6 eye landmarks per eye to measure eye openness |
| ⏱️ **Timed Drowsiness Detection** | Triggers alert only after eyes stay closed for **2+ continuous seconds** (reduces false positives) |
| 🔊 **Browser Audio Alarm** | Loud 800Hz alarm generated via the Web Audio API — plays directly in your browser |
| 🔴 **Visual Alert Overlay** | Flashing red border + pulsing "DROWSY! WAKE UP!" warning overlay on the video feed |
| 📊 **Live HUD Dashboard** | Shows EAR value, EAR bar graph, session uptime, current time, and AWAKE/DROWSY status |
| 🌐 **Web-Based Interface** | Runs as a Flask web app — access from any browser on `localhost:5000` |
| 🟢 **Eye Contour Visualization** | Green/red convex hull drawn around detected eyes with cyan landmark dots |

---

## 🧠 How It Works

```
Webcam Frame → MediaPipe Face Landmarker → Extract Eye Landmarks
    → Calculate EAR (Eye Aspect Ratio) → Compare Against Threshold (0.25)
        → If EAR < 0.25 for > 2 seconds → TRIGGER ALARM 🚨
        → If EAR ≥ 0.25 → Status: AWAKE ✅
```

### Eye Aspect Ratio (EAR) Formula

```
        |P2 - P6| + |P3 - P5|
EAR =  ────────────────────────
            2 × |P1 - P4|
```

- **P1–P6** are the six landmark points around each eye
- EAR ≈ **0.3** when eyes are fully open
- EAR ≈ **0.05** when eyes are closed
- Threshold: **0.25** (configurable in `app.py`)

---

## 📁 Project Structure

```
Drowsiness Detector/
├── app.py                    # Flask backend — video streaming, face detection, EAR logic
├── face_landmarker.task      # MediaPipe Face Landmarker AI model (auto-downloaded)
├── templates/
│   └── index.html            # Frontend — dark-themed UI with live video + audio alerts
├── requirements.txt          # Python dependencies
└── README.md                 # You are here
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed and added to PATH
- A working **webcam**
- A modern web browser (Chrome, Edge, Firefox)

### 1. Clone or Download

```bash
git clone https://github.com/Aditya000012/Drowsiness-Detector.git
cd Drowsiness-Detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the AI Model

The app requires the MediaPipe Face Landmarker model file. Download it with:

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'face_landmarker.task')"
```

### 4. Run the Application

```bash
python app.py
```

### 5. Open in Browser

Navigate to **[http://localhost:5000](http://localhost:5000)** and **click anywhere on the page** to enable browser audio alarms.

> ⚠️ **Important:** You must click on the page at least once after loading — browsers block auto-playing audio until the user interacts with the page.

---

## ⚙️ Configuration

You can tweak the detection sensitivity by editing these constants in `app.py`:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | EAR value below which eyes are considered "closed" |
| `DROWSY_TIME_THRESHOLD` | `2.0` | Seconds of continuous eye closure before alarm triggers |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3** | Core programming language |
| **Flask** | Lightweight web server for streaming and API |
| **OpenCV** | Video capture, image processing, and HUD rendering |
| **MediaPipe** | Google's Face Landmarker for real-time facial landmark detection |
| **NumPy / SciPy** | Numerical computation for EAR calculation |
| **Web Audio API** | Browser-native alarm sound generation (no server-side audio needed) |
| **HTML5 / CSS3 / JS** | Responsive dark-themed frontend UI |

---

## 🖥️ Architecture

```
┌─────────────┐        MJPEG Stream         ┌─────────────────┐
│   Webcam    │ ──→  /video_feed  ─────────→ │   Browser UI    │
└─────────────┘                              │                 │
       │                                     │  - Video Feed   │
       ▼                                     │  - EAR Display  │
┌─────────────┐        JSON API              │  - Status Badge │
│  MediaPipe  │ ──→  /status  ────────────→  │  - Audio Alarm  │
│  + OpenCV   │                              │  (Web Audio API)│
│  (Python)   │                              └─────────────────┘
└─────────────┘
    Flask Server (port 5000)                    Browser Client
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `python` not recognized | Install Python from [python.org](https://python.org) and check **"Add to PATH"** during installation |
| `pip` not recognized | Use `python -m pip install -r requirements.txt` instead |
| Broken video feed / no camera | Close other apps using the camera (Zoom, Teams, etc.) and restart the server |
| No alarm sound | Click anywhere on the page first to enable browser audio |
| Multiple stale processes | Run `taskkill /F /IM python.exe` (Windows) to kill zombie processes, then restart |
| `ModuleNotFoundError: mediapipe` | Run `pip install mediapipe` — make sure you're using the same Python environment |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) — Face Landmarker model
- [OpenCV](https://opencv.org/) — Computer vision library
- Eye Aspect Ratio algorithm based on the paper: *"Real-Time Eye Blink Detection using Facial Landmarks"* by Soukupová & Čech (2016)

---

<p align="center">
  Made with ❤️ by <strong>Aditya</strong>
</p>
