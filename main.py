# main.py
# MAIN ENTRY POINT — run this file only
# Merges YOLO tracking agent + VLM pipeline into one loop

import cv2
import numpy as np
import time
import threading
import tempfile
import os
import subprocess
import platform
import asyncio
from collections import deque

from ultralytics import YOLO
import edge_tts

from src.identity_manager import IdentityManager
from src.agent_core import AgentCore
from src.depth_model import DepthEstimator

from scripts.trigger_logic import is_uncertain, can_trigger
import scripts.vlm_agent as vlm_agent


# -------------------- TTS --------------------

VOICE = "en-US-AriaNeural"
last_spoken = None
last_speech_time = 0


def _play_audio(file_path, duration_secs=4):
    abs_path = os.path.abspath(file_path)
    if platform.system() == "Windows":
        ps_script = (
            f"Add-Type -AssemblyName presentationCore; "
            f"$mp = New-Object System.Windows.Media.MediaPlayer; "
            f"$mp.Open([System.Uri]'{abs_path}'); "
            f"$mp.Play(); "
            f"Start-Sleep -Seconds {duration_secs}"
        )
        subprocess.run(
            ['powershell', '-WindowStyle', 'Hidden', '-Command', ps_script],
            check=True
        )
    else:
        # Raspberry Pi — requires: sudo apt install mpg123
        subprocess.run(['mpg123', '-q', abs_path], check=True)


def _speak_async(text):
    async def run():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(path)
        # Dynamic duration: edge_tts speaks ~2.5 words/sec, +2s buffer
        word_count    = len(text.split())
        duration_secs = max(4, int(word_count / 2.5) + 2)
        _play_audio(path, duration_secs)
        os.remove(path)
    asyncio.run(run())


def speak(text):
    global last_spoken, last_speech_time
    now = time.time()
    if text == last_spoken and now - last_speech_time < 5:
        return
    last_spoken = text
    last_speech_time = now
    threading.Thread(target=_speak_async, args=(text,), daemon=True).start()


# -------------------- Controllers --------------------

class TemporalController:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def update(self, action):
        self.history.append(action)
        return max(set(self.history), key=self.history.count)


class SafetyPolicyEngine:
    def __init__(self):
        self.stop_threshold = 0.75
        self.slow_threshold = 0.55
        self.max_area_stop  = 0.88

    def decide(self, left_risk, center_risk, right_risk, max_area=None):

        if max_area is not None and max_area > self.max_area_stop:
            return "STOP", True

        if center_risk > self.stop_threshold:
            margin = abs(left_risk - right_risk)
            if margin < 0.05:
                return "TURN LEFT", True
            elif left_risk < right_risk:
                return "TURN LEFT", True
            else:
                return "TURN RIGHT", True

        if center_risk > self.slow_threshold:
            return "SLOW", False

        return "FORWARD CLEAR", False


# -------------------- Models --------------------

model       = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if platform.system() == "Windows" else cv2.VideoCapture(0)
identity    = IdentityManager()
agent       = AgentCore()
depth_model = DepthEstimator()
temporal    = TemporalController()
policy      = SafetyPolicyEngine()


# -------------------- Performance --------------------

FRAME_SKIP   = 3
DEPTH_SKIP   = 15

frame_count  = 0
cached_depth = None
cached_objects = {}
prev_time    = time.time()

print("SPACE = manual VLM assist | Q = quit")


# -------------------- Main Loop --------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 320))
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF

    # -------------------- VLM Pause --------------------
    # While VLM is running: freeze YOLO, show overlay, no print spam
    if vlm_agent.vlm_running:
        cv2.putText(frame, "VLM ACTIVE - ANALYZING SCENE...", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.imshow("Agent", frame)
        if key == ord("q"):
            break
        continue

    # -------------------- YOLO --------------------
    if frame_count % FRAME_SKIP == 0:

        results    = model(frame, verbose=False)
        detections = []
        boxes      = results[0].boxes

        if boxes is not None:
            for box in boxes:
                cls   = int(box.cls)
                label = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({"label": label, "bbox": [x1, y1, x2, y2]})

        cached_objects = identity.update(detections)

    objects = cached_objects
    agent.update(objects)

    # -------------------- Depth --------------------
    small_frame = cv2.resize(frame, (160, 120))

    if frame_count % DEPTH_SKIP == 0:
        cached_depth = depth_model.predict(small_frame)

    depth = cached_depth

    if depth is None:
        cv2.putText(frame, "INITIALIZING...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Agent", frame)
        if key == ord("q"):
            break
        continue

    # -------------------- Zone risk --------------------
    h, w = depth.shape

    # Center zone is 50% of frame width (25%→75%) to reduce false L/R triggers
    left   = depth[:, :w // 4]
    center = depth[:, w // 4:3 * w // 4]
    right  = depth[:, 3 * w // 4:]

    left_risk   = np.mean(left)
    center_risk = np.mean(center)
    right_risk  = np.mean(right)

    total = left_risk + center_risk + right_risk + 1e-6
    left_risk   /= total
    center_risk /= total
    right_risk  /= total

    # -------------------- YOLO risk boost --------------------
    for obj in objects.values():
        x1, y1, x2, y2  = obj["bbox"]
        area             = (x2 - x1) * (y2 - y1)
        frame_area       = frame.shape[0] * frame.shape[1]
        ratio            = area / frame_area
        obj_center_x     = (x1 + x2) / 2

        if ratio > 0.10 and frame.shape[1] * 0.25 < obj_center_x < frame.shape[1] * 0.75:
            center_risk += 0.25

    center_risk = min(center_risk, 1.0)

    # -------------------- Policy --------------------
    max_area = max(
        [0] + [
            ((obj["bbox"][2] - obj["bbox"][0]) * (obj["bbox"][3] - obj["bbox"][1]))
            / (frame.shape[0] * frame.shape[1])
            for obj in objects.values()
        ]
    )

    raw_action, collision = policy.decide(left_risk, center_risk, right_risk, max_area)
    action = temporal.update(raw_action)

    print(f"ACTION: {action} | L={left_risk:.2f} C={center_risk:.2f} R={right_risk:.2f}")

    # -------------------- Speech --------------------
    # YOLO directional cues are suppressed while VLM is active
    # VLM always takes the speaker; YOLO resumes after VLM finishes
    speak(action)

    # -------------------- VLM Trigger --------------------
    manual_trigger      = (key == 32)
    uncertainty_trigger = is_uncertain(frame)

    if (manual_trigger or uncertainty_trigger) and can_trigger():

        # Build zone context string from current YOLO + depth data
        detected_labels = list({obj["label"] for obj in objects.values()})
        
        def risk_level(r):
            if r > 0.6: return "high"
            if r > 0.35: return "moderate"
            return "low"

        zone_context = (
            f"Left side: {risk_level(left_risk)} risk. "
            f"Center: {risk_level(center_risk)} risk. "
            f"Right side: {risk_level(right_risk)} risk. "
            f"Detected objects: {', '.join(detected_labels) if detected_labels else 'none'}."
            f"Current navigation decision: {action}."
        )

        vlm_agent.trigger_vlm(frame, zone_context=zone_context, on_complete=speak)

    # -------------------- FPS --------------------
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # -------------------- Draw --------------------
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    for obj_id, obj in objects.items():
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{obj_id} {obj['label']}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f"ACTION: {action}", (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if collision else (0, 255, 0), 2)

    cv2.putText(frame, f"L:{left_risk:.2f} C:{center_risk:.2f} R:{right_risk:.2f}",
                (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Agent", frame)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 