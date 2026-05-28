import cv2
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os

# ----------------------------
# Load API key from .env file
# ----------------------------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Gemini model selection
model = "gemini-2.5-flash"

# ----------------------------
# Start webcam (0 = default camera)
# ----------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to capture frame, Q to quit")

while True:
    # Read one frame from camera stream
    ret, frame = cap.read()
    if not ret:
        break

    # Show live camera feed in a window
    cv2.imshow("Live Camera", frame)

    # Wait for key press (1 ms loop delay)
    key = cv2.waitKey(1) & 0xFF

    # ----------------------------
    # SPACE BAR: capture + send to Gemini
    # ----------------------------
    if key == 32:

        # Save raw frame from OpenCV (BGR image)
        path = "data/frame.jpg"
        cv2.imwrite(path, frame)

        # Open image using PIL for Gemini compatibility
        image = Image.open(path)

        # Resize to reduce latency + bandwidth
        image = image.resize((320, 240))

        # Compress image (reduces upload size further)
        image.save("data/temp.jpg", format="JPEG", quality=40)

        # Reload compressed version for inference
        image = Image.open("data/temp.jpg")

        # ----------------------------
        # Send image + prompt to Gemini
        # ----------------------------
        response = client.models.generate_content(
            model=model,
            contents=[
                "Give only navigation-relevant hazards, obstacles, and actionable spatial info.",
                image
            ]
        )

        # Print model output
        print("\n--- GEMINI OUTPUT ---")
        print(response.text)

    # ----------------------------
    # Q key: exit program
    # ----------------------------
    if key == ord('q'):
        break

# Cleanup camera + windows
cap.release()
cv2.destroyAllWindows()