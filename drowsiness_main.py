"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         AI-Powered Real-Time Driver Drowsiness Detection System              ║
║         Built with OpenCV · Dlib · Pygame · Python                           ║
║         Author  : Samiksha Salunkhe                                          ║                                                                              
╚══════════════════════════════════════════════════════════════════════════════╝

Tech Stack:
  - OpenCV       : Webcam capture, frame processing, UI rendering
  - Dlib         : Face detection + 68-point facial landmark prediction
  - SciPy        : EAR distance computations
  - Pygame       : Alarm sound management
  - imutils      : Frame resizing & landmark helpers
  - smtplib      : Email alert (optional)
  - logging      : Detection history logging
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import dlib
import pygame
import time
import os
import sys
import logging
import smtplib
import json
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ALARM_PATH = os.path.join(BASE_DIR, "alarm.wav")
from datetime import datetime
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque
from threading import Thread


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOGGING SETUP  –  writes every drowsiness event to drowsiness_log.json
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler("drowsiness_system.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("DrowsinessDetector")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GLOBAL CONSTANTS  –  tweak these without touching any other code
# ─────────────────────────────────────────────────────────────────────────────
# ── Eye Aspect Ratio ──────────────────────────────────────────────────────────
EAR_THRESHOLD        = 0.25   # below this → eye is "closed"
EAR_CONSEC_FRAMES    = 20     # frames eye must stay closed → alert
EAR_BLINK_FRAMES     = 4      # fewer frames → normal blink (not drowsy)

# ── Mouth / Yawn ─────────────────────────────────────────────────────────────
MAR_THRESHOLD        = 0.65   # Mouth Aspect Ratio above this → yawn
MAR_CONSEC_FRAMES    = 15

# ── Head-Pose ────────────────────────────────────────────────────────────────
HEAD_TILT_THRESHOLD  = 20     # degrees from vertical → attention warning

# ── AI Fatigue Score ─────────────────────────────────────────────────────────
FATIGUE_HISTORY_LEN  = 300    # last N frames used for rolling score

# ── UI / Display ─────────────────────────────────────────────────────────────
FRAME_WIDTH          = 900
DARK_MODE            = True   # toggle dark / light theme

# ── Landmark index ranges (dlib 68-point model) ───────────────────────────────
(L_START, L_END)     = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END)     = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(M_START, M_END)     = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(N_START, N_END)     = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH           = "shape_predictor_68_face_landmarks.dat"
ALARM_PATH           = "alarm.wav"
SCREENSHOT_DIR       = "screenshots"
LOG_PATH             = "drowsiness_log.json"

# ── Email Alert (optional – fill in or leave blank to disable) ───────────────
EMAIL_SENDER         = ""     # "your_email@gmail.com"
EMAIL_PASSWORD       = ""     # "app_password"
EMAIL_RECEIVER       = ""     # "receiver@example.com"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  COLOUR PALETTE  (dark-mode and light-mode)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "dark": {
        "bg":         (15,  15,  20),
        "panel":      (25,  28,  38),
        "accent":     (0,  200, 255),
        "green":      (0,  220,  80),
        "red":        (0,   40, 230),   # BGR
        "yellow":     (0,  200, 230),
        "text":       (220, 230, 255),
        "subtext":    (120, 130, 160),
        "border":     (45,  55,  80),
    },
    "light": {
        "bg":         (235, 235, 240),
        "panel":      (255, 255, 255),
        "accent":     (200, 100,   0),
        "green":      (30,  140,  30),
        "red":        (30,   30, 200),
        "yellow":     (20,  140, 200),
        "text":       (20,   20,  40),
        "subtext":    (90,   90, 120),
        "border":     (180, 180, 200),
    },
}
C = PALETTE["dark"] if DARK_MODE else PALETTE["light"]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Vertical distances A & B / (2 × horizontal distance C)
    """
    A = dist.euclidean(eye[1], eye[5])   # vertical – upper inner/outer
    B = dist.euclidean(eye[2], eye[4])   # vertical – lower inner/outer
    C = dist.euclidean(eye[0], eye[3])   # horizontal span
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """
    MAR = (||p2-p10|| + ||p4-p8||) / (2 * ||p1-p7||)
    Used for yawn detection.
    """
    A = dist.euclidean(mouth[2],  mouth[10])
    B = dist.euclidean(mouth[4],  mouth[8])
    C = dist.euclidean(mouth[0],  mouth[6])
    return (A + B) / (2.0 * C)


def draw_eye_contour(frame: np.ndarray, eye: np.ndarray, color: tuple) -> None:
    """Draw a closed polygon around the eye landmark points."""
    hull = cv2.convexHull(eye)
    cv2.drawContours(frame, [hull], -1, color, 1)


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


def put_text_shadow(frame, text, pos, font, scale, color, thickness=1):
    """Render text with a subtle drop-shadow for readability."""
    sx, sy = pos[0] + 1, pos[1] + 1
    cv2.putText(frame, text, (sx, sy), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos,      font, scale, color,      thickness,     cv2.LINE_AA)


def compute_head_pose(shape, frame_w, frame_h):
    """
    Estimate head yaw/pitch using solvePnP on 6 facial keypoints.
    Returns (yaw, pitch, roll) angles in degrees.
    """
    model_pts = np.array([
        (0.0,    0.0,    0.0),    # nose tip
        (0.0,  -330.0, -65.0),   # chin
        (-225.0, 170.0, -135.0), # left eye corner
        (225.0,  170.0, -135.0), # right eye corner
        (-150.0,-150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0), # right mouth corner
    ], dtype=np.float64)

    image_pts = np.array([
        shape[30],  # nose tip
        shape[8],   # chin
        shape[36],  # left eye corner
        shape[45],  # right eye corner
        shape[48],  # left mouth
        shape[54],  # right mouth
    ], dtype=np.float64)

    focal   = frame_w
    center  = (frame_w / 2, frame_h / 2)
    cam_mat = np.array([
        [focal, 0,      center[0]],
        [0,     focal,  center[1]],
        [0,     0,      1        ],
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(
        model_pts, image_pts, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles  # yaw, pitch, roll


def save_screenshot(frame: np.ndarray, reason: str) -> str:
    """Save current frame to /screenshots and return the filepath."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{reason}_{ts}.jpg")
    cv2.imwrite(path, frame)
    logger.info(f"Screenshot saved → {path}")
    return path


def log_event(event_type: str, details: dict) -> None:
    """Append a JSON-serialisable event to the detection history log."""
    entry = {"timestamp": datetime.now().isoformat(), "event": event_type, **details}
    try:
        history = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as f:
                history = json.load(f)
        history.append(entry)
        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Log write failed: {e}")


def send_email_alert(subject: str, body: str) -> None:
    """Fire-and-forget email alert (runs in background thread)."""
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        return

    def _send():
        try:
            msg = MIMEMultipart()
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = EMAIL_RECEIVER
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            logger.info("Email alert sent successfully.")
        except Exception as e:
            logger.warning(f"Email send failed: {e}")

    Thread(target=_send, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ALARM MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class AlarmManager:
    """Handles alarm sound playback without overlapping or blocking the main loop."""

    def __init__(self, alarm_path: str):
        pygame.mixer.init()
        self.alarm_path  = alarm_path
        self.is_playing  = False
        self._alarm_ok   = os.path.exists(alarm_path)
        if not self._alarm_ok:
            logger.warning(f"Alarm file not found: {alarm_path}. Using beep fallback.")

    def play(self):
        if self.is_playing:
            return
        if self._alarm_ok:
            pygame.mixer.music.load(self.alarm_path)
            pygame.mixer.music.play(-1)   # -1 = loop indefinitely
        else:
            # Synthetic beep as fallback (Windows only; silent on Linux/Mac)
            try:
                import winsound
                winsound.Beep(1000, 200)
            except Exception:
                pass
        self.is_playing = True

    def stop(self):
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False


# ─────────────────────────────────────────────────────────────────────────────
# 7.  AI FATIGUE SCORER
# ─────────────────────────────────────────────────────────────────────────────
class FatigueScorer:
    """
    Rolling-window fatigue score (0–100) combining:
      - EAR depression ratio
      - Yawn frequency
      - Head-pose deviations
      - Blink rate
    """

    def __init__(self, window: int = FATIGUE_HISTORY_LEN):
        self.ear_hist    = deque(maxlen=window)
        self.yawn_hist   = deque(maxlen=window)
        self.pose_hist   = deque(maxlen=window)
        self.blink_hist  = deque(maxlen=window)

    def update(self, ear, yawning, head_angle, blinked):
        self.ear_hist.append(ear)
        self.yawn_hist.append(1 if yawning  else 0)
        self.pose_hist.append(min(abs(head_angle), 90) / 90)
        self.blink_hist.append(1 if blinked  else 0)

    @property
    def score(self) -> int:
        if not self.ear_hist:
            return 0
        ear_score  = max(0, (EAR_THRESHOLD - np.mean(self.ear_hist)) / EAR_THRESHOLD) * 40
        yawn_score = np.mean(self.yawn_hist)  * 30
        pose_score = np.mean(self.pose_hist)  * 20
        blink_scr  = np.mean(self.blink_hist) * 10
        return int(min(100, ear_score + yawn_score + pose_score + blink_scr))

    def level(self) -> tuple:
        s = self.score
        if s < 25:  return "ALERT",    C["green"]
        if s < 55:  return "MODERATE", C["yellow"]
        if s < 75:  return "HIGH",     (0, 140, 255)
        return           "CRITICAL",   C["red"]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  PROFESSIONAL UI RENDERER
# ─────────────────────────────────────────────────────────────────────────────
class UIRenderer:
    """
    Composes the full dashboard overlay onto the video frame.
    Keeps the main loop clean by isolating all drawing logic here.
    """

    FONT       = cv2.FONT_HERSHEY_DUPLEX
    FONT_MONO  = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, frame_w, frame_h):
        self.fw = frame_w
        self.fh = frame_h

    # ── Side Panel ────────────────────────────────────────────────────────────
    def draw_panel(self, canvas, metrics: dict):
        """Right-side stats panel."""
        px = self.fw   # panel starts at original frame width
        pw = 260
        ph = self.fh

        # background
        panel = np.full((ph, pw, 3), C["panel"], dtype=np.uint8)

        y = 18
        # Title
        cv2.putText(panel, "DROWSINESS", (10, y), self.FONT, 0.55, C["accent"], 1, cv2.LINE_AA)
        y += 18
        cv2.putText(panel, "DETECTION SYSTEM", (10, y), self.FONT, 0.42, C["accent"], 1, cv2.LINE_AA)
        y += 8
        cv2.line(panel, (10, y), (pw - 10, y), C["border"], 1)
        y += 14

        def stat_row(label, value, color=None):
            nonlocal y
            color = color or C["text"]
            cv2.putText(panel, label, (12, y), self.FONT, 0.38, C["subtext"], 1, cv2.LINE_AA)
            y += 16
            cv2.putText(panel, str(value), (12, y), self.FONT, 0.50, color, 1, cv2.LINE_AA)
            y += 22

        stat_row("EAR",          f"{metrics['ear']:.3f}",
                 C["green"] if metrics['ear'] >= EAR_THRESHOLD else C["red"])
        stat_row("MAR",          f"{metrics['mar']:.3f}",
                 C["red"] if metrics['yawning'] else C["green"])
        stat_row("BLINKS",       metrics['blinks'])
        stat_row("YAWNS",        metrics['yawns'])
        stat_row("FPS",          f"{metrics['fps']:.1f}", C["accent"])
        stat_row("FACES",        metrics['faces'])

        y += 5
        cv2.line(panel, (10, y), (pw - 10, y), C["border"], 1)
        y += 14

        # Head pose
        yaw, pitch, roll = metrics.get('head_pose', (0, 0, 0))
        stat_row("HEAD YAW",   f"{yaw:.1f}°",
                 C["red"] if abs(yaw) > HEAD_TILT_THRESHOLD else C["green"])
        stat_row("HEAD PITCH", f"{pitch:.1f}°",
                 C["red"] if abs(pitch) > HEAD_TILT_THRESHOLD else C["green"])

        y += 5
        cv2.line(panel, (10, y), (pw - 10, y), C["border"], 1)
        y += 14

        # Fatigue bar
        fatigue_label, fatigue_color = metrics['fatigue_level']
        score = metrics['fatigue_score']
        cv2.putText(panel, "FATIGUE SCORE", (12, y), self.FONT, 0.38, C["subtext"], 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"{score}/100 – {fatigue_label}", (12, y), self.FONT, 0.42, fatigue_color, 1, cv2.LINE_AA)
        y += 14
        bar_w = pw - 24
        cv2.rectangle(panel, (12, y), (12 + bar_w, y + 10), C["border"], -1)
        cv2.rectangle(panel, (12, y), (12 + int(bar_w * score / 100), y + 10), fatigue_color, -1)
        y += 26

        y += 5
        cv2.line(panel, (10, y), (pw - 10, y), C["border"], 1)
        y += 14

        # Status
        status_txt   = "DROWSY!" if metrics['drowsy']  else ("YAWNING" if metrics['yawning'] else "AWAKE")
        status_color = C["red"]  if metrics['drowsy']  else (C["yellow"] if metrics['yawning'] else C["green"])
        cv2.putText(panel, "STATUS", (12, y), self.FONT, 0.38, C["subtext"], 1, cv2.LINE_AA)
        y += 18
        cv2.putText(panel, status_txt, (12, y), self.FONT, 0.65, status_color, 2, cv2.LINE_AA)
        y += 30

        # Timestamp
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(panel, ts, (12, ph - 18), self.FONT_MONO, 1.0, C["subtext"], 1, cv2.LINE_AA)

        # Merge panel onto canvas
        canvas[:, px:px + pw] = panel

    # ── Top Bar ───────────────────────────────────────────────────────────────
    def draw_top_bar(self, frame, drowsy: bool):
        bar_h = 36
        bar   = np.full((bar_h, self.fw, 3), C["bg"], dtype=np.uint8)
        title = "AI Driver Safety Monitor  v2.0"
        cv2.putText(bar, title, (10, 24), self.FONT, 0.55, C["accent"], 1, cv2.LINE_AA)
        date_str = datetime.now().strftime("%A, %d %b %Y")
        cv2.putText(bar, date_str, (self.fw - 200, 24), self.FONT, 0.40, C["subtext"], 1, cv2.LINE_AA)
        frame[0:bar_h, 0:self.fw] = bar

    # ── Drowsiness Alert Overlay ──────────────────────────────────────────────
    def draw_alert(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.fw, self.fh), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        text  = "DROWSINESS DETECTED!"
        scale = 1.1
        thick = 2
        tw, th = cv2.getTextSize(text, self.FONT, scale, thick)[0]
        cx = (self.fw - tw) // 2
        cy = self.fh // 2
        put_text_shadow(frame, text, (cx, cy), self.FONT, scale, (0, 60, 255), thick)
        put_text_shadow(frame, "Please take a break!", (cx + 30, cy + 40), self.FONT, 0.65, (0, 120, 255), 1)

    # ── EAR Graph (sparkline) ─────────────────────────────────────────────────
    def draw_ear_graph(self, frame, ear_history: deque):
        gx, gy, gw, gh = 10, self.fh - 80, 180, 60
        cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), C["border"], 1)
        cv2.putText(frame, "EAR", (gx + 2, gy - 4), self.FONT_MONO, 0.9, C["subtext"], 1, cv2.LINE_AA)

        if len(ear_history) < 2:
            return

        pts = list(ear_history)[-gw:]
        for i in range(1, len(pts)):
            y1 = int(gy + gh - pts[i-1] * gh * 1.5)
            y2 = int(gy + gh - pts[i]   * gh * 1.5)
            y1 = max(gy, min(gy + gh, y1))
            y2 = max(gy, min(gy + gh, y2))
            color = C["red"] if pts[i] < EAR_THRESHOLD else C["green"]
            cv2.line(frame, (gx + i - 1, y1), (gx + i, y2), color, 1)

        # threshold line
        thr_y = int(gy + gh - EAR_THRESHOLD * gh * 1.5)
        cv2.line(frame, (gx, thr_y), (gx + gw, thr_y), C["yellow"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────
class DrowsinessDetector:
    """
    Orchestrates:
      - Video stream capture
      - Face / landmark detection (dlib)
      - EAR / MAR computation
      - Head-pose estimation
      - Fatigue scoring
      - Alarm management
      - UI rendering
      - Logging & screenshots
    """

    def __init__(self, camera_index: int = 0):
        logger.info("Initialising Drowsiness Detection System …")

        # ── Validate model file ───────────────────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found: {MODEL_PATH}")
            logger.error("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            sys.exit(1)

        # ── Dlib detectors ────────────────────────────────────────────────────
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        logger.info("Dlib models loaded ✓")

        # ── Video stream ──────────────────────────────────────────────────────
        self.vs = VideoStream(src=camera_index).start()
        time.sleep(1.0)   # let camera warm up

        # ── State variables ───────────────────────────────────────────────────
        self.ear_counter   = 0     # consecutive low-EAR frames
        self.mar_counter   = 0
        self.blink_count   = 0
        self.yawn_count    = 0
        self.blink_total   = 0
        self.yawn_total    = 0
        self.drowsy        = False
        self.yawning       = False
        self.alert_sent    = False  # prevent duplicate email alerts

        # ── Rolling history ───────────────────────────────────────────────────
        self.ear_history   = deque(maxlen=200)
        self.fps_history   = deque(maxlen=30)

        # ── Sub-systems ───────────────────────────────────────────────────────
        self.alarm   = AlarmManager(ALARM_PATH)
        self.scorer  = FatigueScorer()

        # ── Determine canvas size ─────────────────────────────────────────────
        sample = self.vs.read()
        h, w   = sample.shape[:2]
        scale  = FRAME_WIDTH / w
        self.fh = int(h * scale)
        self.fw = FRAME_WIDTH
        self.ui = UIRenderer(self.fw, self.fh)

        logger.info(f"Canvas: {self.fw}×{self.fh}  |  Camera: {camera_index}")
        logger.info("System ready. Press 'q' to quit, 's' to screenshot, 'd' to toggle dark mode.")

    # ── Core per-frame processing ─────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run detection pipeline on a single frame; return annotated canvas."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)   # detect faces

        ear_val    = 0.0
        mar_val    = 0.0
        head_pose  = (0.0, 0.0, 0.0)
        blinked    = False

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # ── Eyes ─────────────────────────────────────────────────────────
            l_eye = shape[L_START:L_END]
            r_eye = shape[R_START:R_END]
            l_ear = eye_aspect_ratio(l_eye)
            r_ear = eye_aspect_ratio(r_eye)
            ear_val = (l_ear + r_ear) / 2.0

            draw_eye_contour(frame, l_eye, C["accent"])
            draw_eye_contour(frame, r_eye, C["accent"])

            # ── Mouth ─────────────────────────────────────────────────────────
            mouth   = shape[M_START:M_END]
            mar_val = mouth_aspect_ratio(mouth)
            hull_m  = cv2.convexHull(mouth)
            cv2.drawContours(frame, [hull_m], -1, C["yellow"], 1)

            # ── Head pose ─────────────────────────────────────────────────────
            try:
                head_pose = compute_head_pose(shape, frame.shape[1], frame.shape[0])
            except Exception:
                head_pose = (0.0, 0.0, 0.0)

            # ── Blink detection ───────────────────────────────────────────────
            if ear_val < EAR_THRESHOLD:
                self.ear_counter += 1
            else:
                if EAR_BLINK_FRAMES <= self.ear_counter < EAR_CONSEC_FRAMES:
                    self.blink_total += 1
                    blinked = True
                self.ear_counter = 0

            # ── Drowsiness ────────────────────────────────────────────────────
            if self.ear_counter >= EAR_CONSEC_FRAMES:
                if not self.drowsy:
                    self.drowsy = True
                    ts_now = datetime.now().isoformat()
                    log_event("DROWSINESS_DETECTED", {"ear": round(ear_val, 3)})
                    save_screenshot(frame, "drowsiness")
                    if not self.alert_sent:
                        send_email_alert(
                            "⚠️ Drowsiness Alert",
                            f"Driver drowsiness detected at {ts_now}.\nEAR={ear_val:.3f}"
                        )
                        self.alert_sent = True
                self.alarm.play()
            else:
                if self.drowsy:
                    self.drowsy     = False
                    self.alert_sent = False
                    log_event("DRIVER_AWAKE", {"ear": round(ear_val, 3)})
                self.alarm.stop()

            # ── Yawn detection ────────────────────────────────────────────────
            if mar_val > MAR_THRESHOLD:
                self.mar_counter += 1
                if self.mar_counter >= MAR_CONSEC_FRAMES and not self.yawning:
                    self.yawning   = True
                    self.yawn_total += 1
                    log_event("YAWN_DETECTED", {"mar": round(mar_val, 3)})
            else:
                self.mar_counter = 0
                self.yawning     = False

        # Only one face processed (first rect); for multi-face you'd loop.
        self.ear_history.append(ear_val)
        self.scorer.update(ear_val, self.yawning, head_pose[0], blinked)

        return ear_val, mar_val, head_pose, len(rects)

    # ── Main run loop ─────────────────────────────────────────────────────────
    def run(self):
        global DARK_MODE, C

        cv2.namedWindow("AI Drowsiness Detection System", cv2.WINDOW_NORMAL)

        t_prev = time.time()

        while True:
            raw = self.vs.read()
            if raw is None:
                logger.warning("Empty frame – skipping.")
                continue

            frame = cv2.resize(raw, (self.fw, self.fh))

            # ── FPS ───────────────────────────────────────────────────────────
            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history)

            # ── Detection pipeline ────────────────────────────────────────────
            ear_val, mar_val, head_pose, n_faces = self.process_frame(frame)

            # ── Compose full canvas (frame + side panel) ──────────────────────
            panel_w = 260
            canvas  = np.full((self.fh, self.fw + panel_w, 3), C["bg"], dtype=np.uint8)
            canvas[:, :self.fw] = frame

            metrics = {
                "ear":          ear_val,
                "mar":          mar_val,
                "blinks":       self.blink_total,
                "yawns":        self.yawn_total,
                "fps":          avg_fps,
                "faces":        n_faces,
                "head_pose":    head_pose,
                "drowsy":       self.drowsy,
                "yawning":      self.yawning,
                "fatigue_score": self.scorer.score,
                "fatigue_level": self.scorer.level(),
            }

            self.ui.draw_top_bar(canvas[:, :self.fw], self.drowsy)
            self.ui.draw_panel(canvas, metrics)
            self.ui.draw_ear_graph(canvas[:, :self.fw], self.ear_history)

            if self.drowsy:
                self.ui.draw_alert(canvas[:, :self.fw])

            # ── Show ──────────────────────────────────────────────────────────
            cv2.imshow("AI Drowsiness Detection System", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit signal received.")
                break
            elif key == ord("s"):
                save_screenshot(canvas, "manual")
            elif key == ord("d"):
                DARK_MODE = not DARK_MODE
                C = PALETTE["dark"] if DARK_MODE else PALETTE["light"]
                logger.info(f"Theme → {'Dark' if DARK_MODE else 'Light'}")

        self._cleanup()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def _cleanup(self):
        logger.info("Shutting down …")
        self.alarm.stop()
        self.vs.stop()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        logger.info("Session ended. Total blinks: %d  |  Yawns: %d",
                    self.blink_total, self.yawn_total)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="AI Driver Drowsiness Detection System")
    p.add_argument("--camera",  type=int, default=0,    help="Camera index (default: 0)")
    p.add_argument("--width",   type=int, default=900,  help="Frame width (default: 900)")
    p.add_argument("--light",   action="store_true",    help="Use light mode UI")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    FRAME_WIDTH = args.width
    if args.light:
        DARK_MODE = False
        C = PALETTE["light"]

    detector = DrowsinessDetector(camera_index=args.camera)
    detector.run()
