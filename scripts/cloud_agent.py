import os
import time
import threading
from dotenv import load_dotenv
from google import genai
from PIL import Image
import cv2

# ----------------------------
# Load API key
# ----------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=API_KEY)

model = "gemini-2.0-flash"

# ----------------------------
# Global inference lock + cooldown
# ----------------------------
inference_lock = threading.Lock()
last_call_time = 0
COOLDOWN_SEC = 10


def _convert_frame(frame):
    """
    Converts OpenCV frame (numpy) → JPEG → PIL Image (safe for Gemini)
    """

    # compress early to reduce latency + quota usage
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, buffer = cv2.imencode(".jpg", frame, encode_param)

    image = Image.fromarray(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
    return image


def run_gemini(frame):
    """
    SAFE entry point (called by controller thread)
    """

    global last_call_time

    try:
        image = _convert_frame(frame)

        response = client.models.generate_content(
            model=model,
            contents=[
                "Only return navigation-critical hazards, obstacles, and actionable movement instructions.",
                image
            ]
        )

        print("\n--- GEMINI OUTPUT ---")
        print(response.text)

    except Exception as e:
        print("Gemini Error:", e)

    finally:
        if inference_lock.locked():
            inference_lock.release()


def can_run():
    """
    Global gate: prevents quota burn + spam
    """
    global last_call_time

    now = time.time()

    if (now - last_call_time) < COOLDOWN_SEC:
        return False

    if not inference_lock.acquire(blocking=False):
        return False

    last_call_time = now
    return True