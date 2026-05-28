import cv2
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os
import threading

# ----------------------------
# Load API key from .env file
# ----------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini model selection
model = "gemini-2.5-flash"

# ----------------------------
# Send request to Gemini (runs in background thread)
# ----------------------------
def run_gemini(image):
    response = client.models.generate_content(
        model=model,
        contents=[
            "Give only navigation-relevant hazards, obstacles, and actionable spatial info.",
            image
        ]
    )
    print("\n--- GEMINI OUTPUT ---")
    print(response.text)

# ----------------------------
# Start webcam
# ----------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to capture frame, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # ----------------------------
    # SPACE BAR: capture + send to Gemini (non-blocking)
    # ----------------------------
    if key == 32:

        path = "data/frame.jpg"
        cv2.imwrite(path, frame)

        image = Image.open(path)
        image = image.resize((320, 240))
        image.save("data/temp.jpg", format="JPEG", quality=40)
        image = Image.open("data/temp.jpg")

        # run Gemini in background thread (IMPORTANT)
        threading.Thread(target=run_gemini, args=(image,)).start()

    # ----------------------------
    # Q key: exit loop
    # ----------------------------
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()