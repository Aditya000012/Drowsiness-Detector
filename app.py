from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import os
import math
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Compute absolute path to model file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')

# ==========================================
# CONFIGURATION
# ==========================================
EAR_THRESHOLD = 0.25
DROWSY_TIME_THRESHOLD = 2.0  # seconds

# MediaPipe Face Mesh Indices for Eyes
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Global State for Frontend Tracking
APP_STATE = {
    "is_drowsy": False,
    "ear": 0.0
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def euclidean_distance(p1, p2):
    return math.dist(p1, p2)

def calculate_ear(eye_points):
    if len(eye_points) < 6: return 0.0
    p1, p2, p3, p4, p5, p6 = eye_points
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h = euclidean_distance(p1, p4)
    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def draw_hud(frame, current_ear, is_drowsy, session_time):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "DrowsGuard", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                
    current_time_str = datetime.now().strftime("%H:%M:%S")
    time_size, _ = cv2.getTextSize(current_time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, current_time_str, (w - time_size[0] - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    ear_text = f"EAR: {current_ear:.2f}"
    cv2.putText(frame, ear_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    bar_x, bar_y, bar_w, bar_h = 160, h - 45, 150, 20
    ear_normalized = max(0.0, min(1.0, (current_ear - 0.15) / (0.35 - 0.15)))
    fill_w = int(bar_w * ear_normalized)
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
    fill_color = (0, 0, 255) if current_ear < EAR_THRESHOLD else (0, 255, 0)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

    status_text = "(!) DROWSY" if is_drowsy else "(*) AWAKE"
    status_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    
    if is_drowsy:
        pulse = int((math.sin(time.time() * 10) + 1) * 127) 
        status_color = (pulse, pulse, 255)

    status_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
    status_x = (w - status_size[0]) // 2
    cv2.putText(frame, status_text, (status_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3, cv2.LINE_AA)

    session_minutes = int(session_time // 60)
    session_seconds = int(session_time % 60)
    session_str = f"Uptime: {session_minutes:02d}:{session_seconds:02d}"
    sess_size, _ = cv2.getTextSize(session_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, session_str, (w - sess_size[0] - 20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

def trigger_drowsy_visuals(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
    
    alpha = (math.sin(time.time() * 10) + 1) * 0.15 
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
    
    text = " DROWSY! WAKE UP! "
    font_scale, thickness = 1.5, 4
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x, text_y = (w - text_size[0]) // 2, (h + text_size[1]) // 2
    
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ==========================================
# STREAM CAMERA FEED
# ==========================================
def generate_frames():
    global APP_STATE

    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        logging.info(f"Loading model from: {MODEL_PATH}")
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        logging.info("FaceLandmarker initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize FaceLandmarker: {e}")
        import traceback
        traceback.print_exc()
        return

    cap = cv2.VideoCapture(0)
    logging.info(f"Camera opened: {cap.isOpened()}")

    session_start_time = time.time()
    drowsy_start_time = None
    is_drowsy = False

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam Error/In Use!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            ret2, buffer = cv2.imencode('.jpg', frame)
            if ret2:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        current_ear = 0.0
        face_detected = False

        if detection_result.face_landmarks:
            face_detected = True
            for face_landmarks in detection_result.face_landmarks:
                left_eye_points = [(int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)) for idx in LEFT_EYE_INDICES]
                right_eye_points = [(int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)) for idx in RIGHT_EYE_INDICES]

                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)
                current_ear = (left_ear + right_ear) / 2.0

                left_eye_hull = cv2.convexHull(np.array(left_eye_points))
                right_eye_hull = cv2.convexHull(np.array(right_eye_points))

                is_currently_drowsy = current_ear < EAR_THRESHOLD
                hull_color = (0, 0, 255) if is_currently_drowsy else (0, 255, 0)

                cv2.drawContours(frame, [left_eye_hull], -1, hull_color, 1)
                cv2.drawContours(frame, [right_eye_hull], -1, hull_color, 1)

                cyan = (255, 255, 0)
                for point in left_eye_points + right_eye_points:
                    cv2.circle(frame, point, 2, cyan, -1)

                if is_currently_drowsy:
                    if drowsy_start_time is None:
                        drowsy_start_time = time.time()
                    elif time.time() - drowsy_start_time >= DROWSY_TIME_THRESHOLD:
                        is_drowsy = True
                else:
                    drowsy_start_time = None
                    is_drowsy = False
        else:
            is_drowsy = False
            drowsy_start_time = None
            text = " No Face Detected "
            t_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame, (w//2 - t_size[0]//2 - 10, h//2 - t_size[1] - 10), (w//2 + t_size[0]//2 + 10, h//2 + 10), (0, 200, 255), -1)
            cv2.putText(frame, text, (w//2 - t_size[0]//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        APP_STATE["is_drowsy"] = is_drowsy
        APP_STATE["ear"] = current_ear

        session_time = time.time() - session_start_time

        if face_detected:
            draw_hud(frame, current_ear, is_drowsy, session_time)
            if is_drowsy:
                trigger_drowsy_visuals(frame)
            else:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 4)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error(f"video_feed error: {e}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/debug')
def debug():
    import sys
    info = {
        "python": sys.version,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH,
        "cwd": os.getcwd(),
        "script_dir": SCRIPT_DIR
    }
    try:
        cap = cv2.VideoCapture(0)
        info["cam_opened"] = cap.isOpened()
        ret, frame = cap.read()
        info["cam_read"] = ret
        if ret:
            info["frame_shape"] = str(frame.shape)
        cap.release()
    except Exception as e:
        info["cam_error"] = str(e)
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)
        info["mediapipe_ok"] = True
    except Exception as e:
        info["mediapipe_error"] = str(e)
    return jsonify(info)

@app.route('/status')
def status():
    return jsonify(APP_STATE)

if __name__ == '__main__':
    # debug=False avoids double-execution which locks webcam on Windows
    app.run(debug=False, port=5000, threaded=True)
