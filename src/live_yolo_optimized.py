from ultralytics import YOLO
import cv2
import time

# Load pretrained YOLOv8 model (nano version = fastest, lowest latency)
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera, CAP_DSHOW improves Windows performance)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Controls how often YOLO runs (performance vs accuracy tradeoff)
FRAME_SKIP = 2
frame_count = 0

# Used to compute FPS (frame rate monitoring)
prev_time = time.time()

while True:
    # Read one frame from the camera stream
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize frame to reduce compute cost and stabilize inference speed
    frame = cv2.resize(frame, (640, 480))

    # Default display frame (used when YOLO is not running this iteration)
    annotated = frame.copy()

    # Run YOLO only every N frames to improve FPS
    if frame_count % FRAME_SKIP == 0:
        results = model(frame, verbose=False)
        annotated = results[0].plot()

    # Compute FPS for performance monitoring
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay FPS onto output frame
    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display the final annotated frame
    cv2.imshow("Optimized YOLO", annotated)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resource and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()