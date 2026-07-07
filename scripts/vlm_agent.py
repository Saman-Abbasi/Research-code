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

    return f"""You are guiding a blind person using a camera on their chest. Describe only what is actually visible — never guess or invent objects.
{sensor_block}
In 2 short spoken sentences, tell them:
1. Any objects or hazards you see and where each is — left, center, or right.
2. Where it is safe to go — move left, move forward, move right, or stop.

Always guide them away from hazards. Speak plainly and directly, describing the real world around them, not the image."""


def run_vlm_ollama(frame, zone_context=""):
    """
    Send image to local model
    """
    img_b64 = encode_image(frame)
    if img_b64 is None:
        return "Image encoding failed."

    prompt = build_prompt(zone_context)

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava-phi3",
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def _worker(frame, zone_context="", on_complete=None, trigger_type="manual", announce=None, release=None):
    """
    Background inference thread.
      1. Announce (VLM-priority, interrupts YOLO).
      2. Wait 1s for the user to stop moving.
      3. Run analysis.
      4. Speak result.
      5. Release the VLM audio lock so YOLO resumes.
    """
    global vlm_running

    try:
        # 1. Announcement based on how the VLM was triggered
        if announce is not None:
            if trigger_type == "auto":
                announce("There has been an interruption in your path, hold still whilst I investigate your surrounding.")
            else:
                announce("Please hold still whilst I analyze the scene.")

        # 2. Give the user a moment to stop moving
        time.sleep(1)

        # 3. Analysis
        result = run_vlm_ollama(frame, zone_context)

        print("\n========== VLM ==========")
        print(result)
        print("=========================\n")

        # 4. Speak the result (also VLM-priority)
        if on_complete:
            on_complete(result)
            word_count   = len(result.split())
            est_duration = max(5, word_count / 2.5)
            time.sleep(est_duration)

    except Exception as e:
        print(f"\nVLM Error: {e}\n")

    finally:
        vlm_running = False
        # 5. Release the VLM audio priority so YOLO can speak again
        if release is not None:
            release()
        try:
            from scripts.trigger_logic import mark_trigger_complete
            mark_trigger_complete()
        except Exception:
            pass


def trigger_vlm(frame, zone_context="", on_complete=None, trigger_type="manual", announce=None, release=None):
    """
    Start VLM if not already running.
    trigger_type: "manual" (button) or "auto" (dark/bright/blur) — picks the announcement.
    announce: callable to speak the announcement at VLM priority.
    release:  callable to release VLM audio priority when the whole sequence ends.
    """
    global vlm_running

    if vlm_running:
        return

    vlm_running = True

    threading.Thread(
        target=_worker,
        args=(frame.copy(), zone_context, on_complete, trigger_type, announce, release),
        daemon=True
    ).start()