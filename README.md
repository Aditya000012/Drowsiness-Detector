# 🛡️ DrowsGuard — Real-Time Drowsiness Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-FaceMesh_JS-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Supabase-Database_%26_Auth-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white" />
  <img src="https://img.shields.io/badge/Chart.js-Data_Viz-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white" />
</p>

**DrowsGuard** is a full-stack AI-powered drowsiness detection web application. It uses your webcam to continuously monitor facial landmarks in real time, computes the **Eye Aspect Ratio (EAR)** entirely in the browser, and triggers audio-visual alerts the moment drowsiness is detected — with full user authentication, session management, and historical trend tracking powered by Supabase.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👁️ **Real-Time Detection** | MediaPipe FaceMesh tracks 468 facial landmarks every frame — entirely browser-side |
| 📐 **EAR Algorithm** | Eye Aspect Ratio calculated in JavaScript for zero-latency drowsiness detection |
| ⏱️ **Timed Sessions** | Set a duration (30min, 1hr, 2hr, 4hr, or custom) — app auto-stops with a countdown |
| ♾️ **Continuous Mode** | Run indefinitely until manually stopped |
| 🏷️ **Session Labels** | Tag sessions (Study, Work, Night Drive, Reading, Gaming, or custom) |
| 🔔 **Audio Alerts** | Web Audio API beep triggers the moment drowsiness is detected |
| 🔴 **Visual Alerts** | Pulsing red border around the entire screen when drowsy |
| 📊 **Live EAR Graph** | Scrolling real-time Chart.js graph of eye openness during session |
| 🎛️ **Live Sensitivity Control** | Adjust EAR threshold and alert delay mid-session |
| 📋 **Session Summary** | Full report on session end — duration, alerts, longest episode, avg EAR, alert timeline |
| 📈 **History & Trends** | Per-user session history with trend charts across all sessions |
| ⬇️ **CSV Export** | Download your full session history as a CSV file |
| 🔐 **User Authentication** | Register and login securely via Supabase Auth |
| ☁️ **Cloud Storage** | All session data persisted to Supabase (Postgres) per user |

---

## 🧠 How It Works

```
Webcam → MediaPipe FaceMesh (browser) → Extract Eye Landmarks
    → Calculate EAR (Eye Aspect Ratio) → Compare Against Threshold (default: 0.25)
        → If EAR < threshold for > 2 seconds → TRIGGER ALERT 🚨
        → If EAR ≥ threshold → Status: AWAKE ✅
            → Session ends → Save to Supabase → Show Summary
```

### Eye Aspect Ratio (EAR) Formula

```
        |P2 - P6| + |P3 - P5|
EAR =  ────────────────────────
            2 × |P1 - P4|
```

- **P1–P6** are the six landmark points around each eye
- EAR ≈ **0.30** when eyes are fully open
- EAR ≈ **0.05** when eyes are closed
- Default threshold: **0.25** (adjustable live via slider)

> Algorithm based on: *"Real-Time Eye Blink Detection using Facial Landmarks"* — Soukupová & Čech (2016)

---

## 📁 Project Structure

```
Drowsiness-Detector/
├── app.py                  # Flask backend — serves HTML, exposes /api/config
├── templates/
│   └── index.html          # Full single-page app (7 views, all pages)
├── static/
│   ├── app.js              # All detection logic, Supabase integration, session management
│   └── style.css           # Complete design system (dark navy dashboard aesthetic)
├── supabase_schema.sql     # SQL to set up the sessions table in Supabase
├── requirements.txt        # Python dependencies
├── .env.example            # Template for environment variables
└── README.md               # You are here
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- A working **webcam**
- A modern browser (Chrome recommended)
- A free [Supabase](https://supabase.com) account

### 1. Clone the repo
```bash
git clone https://github.com/Aditya000012/Drowsiness-Detector.git
cd Drowsiness-Detector
```

### 2. Install dependencies
```bash
pip install flask python-dotenv
```

### 3. Set up Supabase
- Create a free project at [supabase.com](https://supabase.com)
- Go to **SQL Editor** and run the contents of `supabase_schema.sql`
- Go to **Settings → API** and copy your Project URL and anon key

### 4. Create `.env` file
```
SUPABASE_URL=your-project-url
SUPABASE_ANON_KEY=your-anon-key
```

### 5. Run the app
```bash
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)**, register an account, and start your first session.

---

## ⚙️ Configuration

Adjust detection behaviour live during a session using the in-app sliders, or set defaults before starting:

| Parameter | Default | Description |
|---|---|---|
| EAR Threshold | `0.25` | EAR below which eyes are considered closed |
| Alert Delay | `2s` | Seconds of continuous eye closure before alert triggers |
| Session Mode | `Timed` | Timed (auto-stop) or Continuous (manual stop) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Detection | MediaPipe FaceMesh (browser-side, no server processing) |
| Charts | Chart.js |
| Audio Alerts | Web Audio API |
| Backend | Python Flask |
| Database | Supabase (Postgres) |
| Authentication | Supabase Auth |
| Deployment | Render |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Browser Client                     │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │   Webcam    │───▶│  MediaPipe   │                 │
│  │   (video)   │    │  FaceMesh JS │                 │
│  └─────────────┘    └──────┬───────┘                 │
│                            │ Landmarks               │
│                     ┌──────▼───────┐                 │
│                     │ EAR Engine   │                 │
│                     │ (app.js)     │                 │
│                     └──────┬───────┘                 │
│              ┌─────────────┼──────────────┐          │
│              ▼             ▼              ▼          │
│        Audio Alert    Visual HUD     Supabase JS     │
│        (Web Audio)   (Chart.js)    (Session Save)    │
└──────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Flask Backend    │
                    │  /api/config only  │
                    │  (serves HTML)     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │     Supabase       │
                    │  Auth + Postgres   │
                    │  (session history) │
                    └────────────────────┘
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| Buttons not working | Open browser console (F12) — check for JS errors |
| Camera not showing | Allow camera permissions in browser; close other apps using webcam |
| No audio alert | Click anywhere on the page first — browsers require user interaction before audio |
| Supabase not connecting | Check `/api/config` at `localhost:5000/api/config` — both keys should be non-empty |
| `python` not recognized | Install Python from [python.org](https://python.org) and check **"Add to PATH"** |
| Sessions not saving | Ensure `supabase_schema.sql` was run and RLS policy is enabled |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [Google MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_mesh) — FaceMesh model
- [Supabase](https://supabase.com) — Auth and database
- [Chart.js](https://chartjs.org) — Data visualization
- Eye Aspect Ratio algorithm based on: *"Real-Time Eye Blink Detection using Facial Landmarks"* — Soukupová & Čech (2016)

---

<p align="center">
  Made with ❤️ by <strong>Aditya Kumar Singh</strong>
</p>
