# vlm_agent.py
# Local VLM agent using Ollama + LLaVA

import cv2
import threading
import requests
import base64
import time

# Prevent multiple VLM calls at once
vlm_running = False


def encode_image(frame):
    """
    Convert OpenCV frame to base64 JPG
    required by Ollama vision models
    """
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def build_prompt(zone_context=""):
    """
    Builds navigation-tuned prompt, optionally injecting fused risk
    (ToF distance + weighted YOLO detections) and detected object data.
    """
    sensor_block = ""
    if zone_context:
        sensor_block = f"""
Risk & detection context:
{zone_context}
"""

    return f"""You are a navigation assistant speaking directly to a blind person wearing smart glasses. Never reference "the image" or "the frame". Describe the real world as if you can see it.
{sensor_block}
Analyze the scene and respond in exactly this format:

Hazards: <what hazards are present and exactly where they are, e.g. "chair on the left side", "person directly ahead", "stairs to the right">

Action: <one clear spoken instruction the person should follow immediately, e.g. "Turn left to avoid the person ahead", "Stop, there are stairs directly in front of you", "Path is clear, move forward slowly">

Be specific about location. Never say "the image shows" or "I can see". Speak as if guiding someone in real time.

CRITICAL: Always direct the person AWAY from hazards. If a hazard is on the left, turn right. If a hazard is on the right, turn left. If ahead, stop or turn to the side with lower risk."""


def run_vlm_ollama(frame, zone_context=""):
    """
    Send image to local gemma3:4b model
    """
    img_b64 = encode_image(frame)
    if img_b64 is None:
        return "Image encoding failed."

    prompt = build_prompt(zone_context)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "moondream",
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def _worker(frame, zone_context="", on_complete=None):
    """
    Background inference thread.
    Calls on_complete(result) to pipe VLM output to speak().
    Holds vlm_running=True for estimated audio duration so
    YOLO directional cues stay suppressed until speech finishes.
    """
    global vlm_running

    try:
        result = run_vlm_ollama(frame, zone_context)

        print("\n========== VLM ==========")
        print(result)
        print("=========================\n")

        if on_complete:
            on_complete(result)

            # Approximate wait for audio to finish before releasing the lock.
            # edge_tts speaks ~2.5 words/sec — keeps YOLO cues silent
            # until VLM speech is done. Rough but sufficient for prototype.
            word_count   = len(result.split())
            est_duration = max(5, word_count / 2.5)
            time.sleep(est_duration)

    except Exception as e:
        print(f"\nVLM Error: {e}\n")

    finally:
        vlm_running = False


def trigger_vlm(frame, zone_context="", on_complete=None):
    """
    Start VLM if not already running.
    zone_context: string summary of fused risk zones + detected objects
    on_complete: callable that receives the VLM result string (e.g. speak)
    """
    global vlm_running

    if vlm_running:
        return

    vlm_running = True

    threading.Thread(
        target=_worker,
        args=(frame.copy(), zone_context, on_complete),
        daemon=True
    ).start()