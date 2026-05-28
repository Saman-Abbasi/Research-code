from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

from identity_manager import IdentityManager
from agent_core import AgentCore
from depth_model import DepthEstimator


class TemporalController:
    def __init__(self, window_size=5):
        # Stores recent actions to reduce flickering
        self.history = deque(maxlen=window_size)

    def update(self, action):
        self.history.append(action)

        # Majority vote over recent frames
        return max(set(self.history), key=self.history.count)


class SafetyPolicyEngine:
    def __init__(self):
        # Risk thresholds
        self.stop_threshold = 0.58
        self.slow_threshold = 0.45

    def decide(self, left_risk, center_risk, right_risk):

        # Severe obstacle ahead
        if center_risk > self.stop_threshold:

            # Choose safer side
            if left_risk < right_risk:
                return "TURN LEFT", True
            else:
                return "TURN RIGHT", True

        # Moderate obstacle
        if center_risk > self.slow_threshold:
            return "SLOW", True

        return "FORWARD CLEAR", False


# -------------------- Models --------------------

# YOLO object detector
model = YOLO("yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Identity tracking layer
identity = IdentityManager()

# Agent memory / persistence layer
agent = AgentCore()

# Monocular depth estimation model
depth_model = DepthEstimator()

# Temporal anti-flicker layer
temporal = TemporalController()

# Navigation policy engine
policy = SafetyPolicyEngine()

# Run depth less frequently for performance
DEPTH_SKIP = 5
frame_count = 0

# Store last valid depth map
cached_depth = None

# -------------------- Main Loop --------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for consistent processing speed
    frame = cv2.resize(frame, (640, 480))

    # -------------------- YOLO Detection --------------------

    results = model(frame, verbose=False)

    detections = []

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            cls = int(box.cls)

            label = model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "bbox": [x1, y1, x2, y2]
            })

    # -------------------- Identity Layer --------------------

    objects = identity.update(detections)

    # -------------------- Agent Memory Layer --------------------

    agent.update(objects)

    # -------------------- Depth Estimation --------------------

    frame_count += 1

    # Smaller frame = faster inference
    small_frame = cv2.resize(frame, (256, 192))

    # Only run depth periodically
    if frame_count % DEPTH_SKIP == 0:

        cached_depth = depth_model.predict(small_frame)

    # Reuse previous depth map between updates
    depth = cached_depth

    # Skip loop until first depth map exists
    if depth is None:

        cv2.putText(
            frame,
            "INITIALIZING DEPTH...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Offline Navigation Agent", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue
    # -------------------- Navigation Zones --------------------

    h, w = depth.shape

    left_zone = depth[:, :w // 3]

    center_zone = depth[:, w // 3:2 * w // 3]

    right_zone = depth[:, 2 * w // 3:]

    # Average relative depth values
    left_score = np.mean(left_zone)

    center_score = np.mean(center_zone)

    right_score = np.mean(right_zone)

    # -------------------- Normalize Scores --------------------

    total = left_score + center_score + right_score + 1e-6

    left_norm = left_score / total

    center_norm = center_score / total

    right_norm = right_score / total

    # -------------------- Convert To Risk --------------------

    # Higher risk = obstacle more likely

    left_risk = left_norm

    center_risk = center_norm

    right_risk = right_norm

    # -------------------- YOLO Risk Boost --------------------

    # Increase danger if large object detected in center

    for obj_id, obj in objects.items():

        x1, y1, x2, y2 = obj["bbox"]

        box_area = (x2 - x1) * (y2 - y1)

        frame_area = frame.shape[0] * frame.shape[1]

        area_ratio = box_area / frame_area

        # Object center position
        obj_center_x = (x1 + x2) / 2

        # If object is large and near center
        if area_ratio > 0.10:

            if frame.shape[1] * 0.33 < obj_center_x < frame.shape[1] * 0.66:

                center_risk += 0.25

    # Clamp values
    left_risk = min(left_risk, 1.0)

    center_risk = min(center_risk, 1.0)

    right_risk = min(right_risk, 1.0)

    # -------------------- Policy Engine --------------------

    raw_action, collision_risk = policy.decide(
        left_risk,
        center_risk,
        right_risk
    )

    # -------------------- Temporal Stabilization --------------------

    action = temporal.update(raw_action)

    # -------------------- Visualization --------------------

    for obj_id, obj in objects.items():

        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{obj_id} {obj['label']}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    # -------------------- Risk Visualization --------------------

    cv2.putText(
        frame,
        f"ACTION: {action}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if collision_risk else (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"L:{left_risk:.2f} C:{center_risk:.2f} R:{right_risk:.2f}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # -------------------- Display --------------------

    cv2.imshow("Offline Navigation Agent", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -------------------- Cleanup --------------------

cap.release()

cv2.destroyAllWindows()