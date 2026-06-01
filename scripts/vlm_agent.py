# vlm_agent.py
# Local VLM agent using Ollama + LLaVA

import cv2
import threading
import requests
import base64

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


def _worker(frame):
    """
    Background inference thread
    """

    global vlm_running

    try:
        result = run_vlm_ollama(frame)

        print("\n========== VLM ==========")
        print(result)
        print("=========================\n")

    except Exception as e:
        print(f"\nVLM Error: {e}\n")

    finally:
        vlm_running = False


def trigger_vlm(frame):
    """
    Start VLM if not already running
    """

    global vlm_running

    if vlm_running:
        return

    vlm_running = True

    threading.Thread(
        target=_worker,
        args=(frame.copy(),),
        daemon=True
    ).start()