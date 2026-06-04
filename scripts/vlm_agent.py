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


def run_vlm_ollama(
    frame,
    prompt="""
You are a wearable navigation assistant.

Analyze the image and respond in exactly this format:

Scene: <short description>

Hazards: <obstacles, trip hazards, walls, furniture, cables, people, stairs, or none>

Guidance: <single navigation instruction>

Keep each field at most 3 sentences.
"""
):
    """
    Send image to local Ollama LLaVA model
    """
    img_b64 = encode_image(frame)
    if img_b64 is None:
        return "Image encoding failed."

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llava",
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


def _worker(frame, on_complete=None):
    """
    Background inference thread.
    Calls on_complete(result) to pipe VLM output to speak().
    Holds vlm_running=True for estimated audio duration so
    YOLO directional cues stay suppressed until speech finishes.
    """
    global vlm_running

    try:
        result = run_vlm_ollama(frame)

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


def trigger_vlm(frame, on_complete=None):
    """
    Start VLM if not already running.
    on_complete: callable that receives the VLM result string (e.g. speak)
    """
    global vlm_running

    if vlm_running:
        return

    vlm_running = True

    threading.Thread(
        target=_worker,
        args=(frame.copy(), on_complete),
        daemon=True
    ).start()