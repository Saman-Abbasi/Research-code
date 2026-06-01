import cv2
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os
import threading

# ----------------------------
# Load API key
# ----------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model = "gemini-2.5-flash"

# Prevent multiple overlapping Gemini calls
inference_running = False

# ----------------------------
# Gemini call (background thread)
# ----------------------------
def run_gemini(image):
    global inference_running

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                "Give only navigation-relevant hazards, obstacles, and actionable spatial info.",
                image
            ]
        )
        print("\n--- GEMINI OUTPUT ---")
        print(response.text)

    finally:
        inference_running = False


# ----------------------------
# Camera
# ----------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to capture, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # ----------------------------
    # SPACE (single trigger only)
    # ----------------------------
    if key == 32 and not inference_running:
        inference_running = True

        path = "data/frame.jpg"
        cv2.imwrite(path, frame)

        image = Image.open(path)
        image = image.resize((320, 240))
        image.save("data/temp.jpg", format="JPEG", quality=40)
        image = Image.open("data/temp.jpg")

        threading.Thread(target=run_gemini, args=(image,), daemon=True).start()

    # ----------------------------
    # EXIT
    # ----------------------------
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()