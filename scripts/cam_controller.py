# cam_controller.py
# MAIN ENTRY POINT (RUN THIS FILE ONLY)

import cv2

from frame_buffer import update_frame, get_latest_frame
from trigger_logic import is_uncertain
from trigger_logic import can_trigger
from vlm_agent import trigger_vlm

# ----------------------------
# Camera setup
# ----------------------------
cap = cv2.VideoCapture(0)

print("SPACE = manual assist")
print("Q = quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Store latest frame
    update_frame(frame)

    # Display live feed
    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    latest = get_latest_frame()

    if latest is None:
        continue

    # ----------------------------
    # Manual trigger
    # ----------------------------
    manual_trigger = (key == 32)

    # ----------------------------
    # Uncertainty trigger
    # ----------------------------
    uncertainty_trigger = is_uncertain(latest)

    # ----------------------------
    # Combined trigger
    # ----------------------------
    should_trigger = manual_trigger or uncertainty_trigger

    # ----------------------------
    # Single gate into VLM
    # ----------------------------
    if should_trigger and can_trigger():
        trigger_vlm(latest)

    # ----------------------------
    # Exit
    # ----------------------------
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()