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

if platform.system() != "Windows":
    from picamera2 import Picamera2
    from gpiozero import Button

from ultralytics import YOLO


from src.identity_manager import IdentityManager
from src.agent_core import AgentCore
from src.tof_sensor import ToFArray

from scripts.trigger_logic import is_uncertain, can_trigger
import scripts.vlm_agent as vlm_agent

from src.risk_weights import CLASS_WEIGHTS, DEFAULT_WEIGHT, BASE_BOOST



# -------------------- TTS (Piper, offline) --------------------

from piper import PiperVoice
import wave

PIPER_MODEL  = os.path.join(BASE_DIR, "voices", "en_GB-alba-medium.onnx") \
    if 'BASE_DIR' in dir() else os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "en_GB-alba-medium.onnx")
PIPER_VOLUME = 0.35              # tuned for MAX98357A to avoid clipping
AUDIO_DEVICE = "plughw:2,0"      # MAX98357A I2S amp (card 2)

# Load the voice model ONCE at startup, reuse for every phrase.
_piper_voice = PiperVoice.load(PIPER_MODEL)

last_spoken = None
last_speech_time = 0

_current_audio_proc = None
_vlm_speaking = False
_audio_lock = threading.Lock()


def _play_audio(file_path, duration_secs=4):
    global _current_audio_proc
    abs_path = os.path.abspath(file_path)
    if platform.system() == "Windows":
        ps_script = (
            f"Add-Type -AssemblyName presentationCore; "
            f"$mp = New-Object System.Windows.Media.MediaPlayer; "
            f"$mp.Open([System.Uri]'{abs_path}'); "
            f"$mp.Play(); "
            f"Start-Sleep -Seconds {duration_secs}"
        )
        proc = subprocess.Popen(
            ['powershell', '-WindowStyle', 'Hidden', '-Command', ps_script]
        )
    else:
        # Raspberry Pi — play WAV through the MAX98357A amp, interruptible
        proc = subprocess.Popen(['aplay', '-D', AUDIO_DEVICE, '-q', abs_path])

    with _audio_lock:
        _current_audio_proc = proc
    proc.wait()


def _stop_current_audio():
    global _current_audio_proc
    with _audio_lock:
        if _current_audio_proc is not None and _current_audio_proc.poll() is None:
            _current_audio_proc.terminate()
        _current_audio_proc = None


def _speak_async(text):
    import numpy as np
    # Generate raw audio from Piper, scale volume, write WAV, play it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        path = f.name

    # Collect Piper's audio chunks
    audio_chunks = []
    for chunk in _piper_voice.synthesize(text):
        audio_chunks.append(chunk.audio_int16_array)
    audio = np.concatenate(audio_chunks)

    # Apply volume scaling (0.35) to prevent MAX98357A clipping
    audio = (audio * PIPER_VOLUME).astype(np.int16)

    # Write WAV with correct headers
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_piper_voice.config.sample_rate)
        wav_file.writeframes(audio.tobytes())

    _play_audio(path)
    os.remove(path)


def speak(text, is_vlm=False):
    """
    is_vlm=True  -> high priority: interrupts any playing YOLO audio, blocks YOLO until done.
    is_vlm=False -> low priority (YOLO cue): stays silent while VLM owns the speaker.
    """
    global last_spoken, last_speech_time, _vlm_speaking

    # YOLO cues stay silent while the VLM owns the speaker.
    if not is_vlm and _vlm_speaking:
        return

    now = time.time()
    if text == last_spoken and now - last_speech_time < 1:
        return
    last_spoken = text
    last_speech_time = now

    if is_vlm:
        # VLM takes over: kill any YOLO audio, claim the speaker.
        _vlm_speaking = True
        _stop_current_audio()

    def _run():
        try:
            _speak_async(text)
        finally:
            if is_vlm:
                # release handled by caller (vlm_agent) after full sequence
                pass

    threading.Thread(target=_run, args=(), daemon=True).start()

def _release_vlm_audio():
    """Called by vlm_agent when the full VLM sequence ends — lets YOLO speak again."""
    global _vlm_speaking
    _vlm_speaking = False
    
    
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model    = YOLO(os.path.join(BASE_DIR, "models", "starvision_best.pt"))

if platform.system() == "Windows":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    picam = None
else:
    cap = None
    picam = Picamera2()
    config = picam.create_video_configuration(main={"size": (416, 320), "format": "RGB888"})
    picam.configure(config)
    picam.start()

identity    = IdentityManager()
agent       = AgentCore()
tof         = ToFArray()
temporal    = TemporalController()
policy      = SafetyPolicyEngine()

if platform.system() != "Windows":
    vlm_button = Button(6, pull_up=True)
else:
    vlm_button = None


# -------------------- Performance --------------------

FRAME_SKIP   = 3

frame_count    = 0
cached_objects = {}
prev_time      = time.time()

print("SPACE = manual VLM assist | Q = quit")


# -------------------- Main Loop --------------------

while True:

    if picam is not None:
        frame = picam.capture_array()
    else:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (416, 320))
    
    frame_count += 1

    if platform.system() == "Windows":
        key = cv2.waitKey(1) & 0xFF
    else:
        key = -1

    # -------------------- VLM Pause --------------------
    if vlm_agent.vlm_running:
        if platform.system() == "Windows":
            cv2.putText(frame, "VLM ACTIVE - ANALYZING SCENE...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            cv2.imshow("Agent", frame)
            if key == ord("q"):
                break
        else:
            print("VLM ACTIVE - ANALYZING SCENE...")
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
    agent.update(objects, frame.shape[0])

    # -------------------- ToF zone risk --------------------
    tof_risk    = tof.read()
    left_risk   = tof_risk["left"]
    center_risk = tof_risk["center"]
    right_risk  = tof_risk["right"]

    # -------------------- YOLO risk boost --------------------
    left_bound  = frame.shape[1] * 0.25
    right_bound = frame.shape[1] * 0.75
    frame_area  = frame.shape[0] * frame.shape[1] 
    
    for obj in objects.values():
        x1, y1, x2, y2 = obj["bbox"]
        area         = (x2 - x1) * (y2 - y1)
        ratio        = area / frame_area
        obj_center_x = (x1 + x2) / 2

        if ratio <= 0.10:
            continue

        weight = CLASS_WEIGHTS.get(obj["label"], DEFAULT_WEIGHT)
        boost  = BASE_BOOST * weight

        if obj_center_x < left_bound:
            left_risk += boost
        elif obj_center_x > right_bound:
            right_risk += boost
        else:
            center_risk += boost

    left_risk   = min(left_risk, 1.0)
    center_risk = min(center_risk, 1.0)
    right_risk  = min(right_risk, 1.0)

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
    speak(action)

    # -------------------- VLM Trigger --------------------
    if vlm_button is not None:
        manual_trigger = vlm_button.is_pressed
    else:
        manual_trigger = (key == 32)
    uncertainty_trigger = is_uncertain(frame)

    if (manual_trigger or uncertainty_trigger) and can_trigger():

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

        trigger_type = "manual" if manual_trigger else "auto"

        vlm_agent.trigger_vlm(
            frame,
            zone_context=zone_context,
            on_complete=lambda t: speak(t, is_vlm=True),
            trigger_type=trigger_type,
            announce=lambda t: speak(t, is_vlm=True),
            release=_release_vlm_audio,
        )

    # -------------------- FPS --------------------
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # -------------------- Draw --------------------
    if platform.system() == "Windows":
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
    else:
        print(f"Objects: {[(o['label']) for o in objects.values()]}")

if picam is not None:
    picam.stop()
else:
    cap.release()
    
if platform.system() == "Windows":
    cv2.destroyAllWindows()