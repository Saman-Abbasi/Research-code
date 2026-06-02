from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

from identity_manager import IdentityManager
from agent_core import AgentCore
from depth_model import DepthEstimator


class TemporalController:
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)

    def update(self, action):
        self.history.append(action)
        return max(set(self.history), key=self.history.count)


class SafetyPolicyEngine:
    def __init__(self):
        self.stop_threshold = 0.58
        self.slow_threshold = 0.45

    def decide(self, left_risk, center_risk, right_risk):

        if center_risk > self.stop_threshold:
            if left_risk < right_risk:
                return "TURN LEFT", True
            else:
                return "TURN RIGHT", True

        if center_risk > self.slow_threshold:
            return "SLOW", True

        return "FORWARD CLEAR", False


# -------------------- Models --------------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

identity = IdentityManager()
agent = AgentCore()
depth_model = DepthEstimator()

temporal = TemporalController()
policy = SafetyPolicyEngine()

# -------------------- Performance controls --------------------
FRAME_SKIP = 2
DEPTH_SKIP = 10

frame_count = 0
cached_depth = None
cached_objects = {}

prev_time = time.time()

# -------------------- Main Loop --------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (416, 320))
    frame_count += 1

    # -------------------- YOLO (skipped frames) --------------------
    if frame_count % FRAME_SKIP == 0:

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

        cached_objects = identity.update(detections)

    objects = cached_objects

    # -------------------- Agent update --------------------
    agent.update(objects)

    # -------------------- Depth (skipped heavy inference) --------------------
    small_frame = cv2.resize(frame, (256, 192))

    if frame_count % DEPTH_SKIP == 0:
        cached_depth = depth_model.predict(small_frame)

    depth = cached_depth

    if depth is None:
        cv2.putText(frame, "INITIALIZING...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Agent", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # -------------------- Zone risk --------------------
    h, w = depth.shape

    left = depth[:, :w // 3]
    center = depth[:, w // 3:2 * w // 3]
    right = depth[:, 2 * w // 3:]

    left_risk = np.mean(left)
    center_risk = np.mean(center)
    right_risk = np.mean(right)

    total = left_risk + center_risk + right_risk + 1e-6

    left_risk /= total
    center_risk /= total
    right_risk /= total

    # -------------------- YOLO risk boost --------------------
    for obj in objects.values():

        x1, y1, x2, y2 = obj["bbox"]

        area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]

        ratio = area / frame_area
        obj_center_x = (x1 + x2) / 2

        if ratio > 0.10 and frame.shape[1] * 0.33 < obj_center_x < frame.shape[1] * 0.66:
            center_risk += 0.25

    center_risk = min(center_risk, 1.0)

    # -------------------- Policy --------------------
    raw_action, collision = policy.decide(
        left_risk, center_risk, right_risk
    )

    action = temporal.update(raw_action)

    # -------------------- FPS --------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # -------------------- Draw boxes --------------------
    for obj_id, obj in objects.items():

        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        cv2.putText(frame,
                    f"{obj_id} {obj['label']}",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # -------------------- UI --------------------
    cv2.putText(frame,
                f"ACTION: {action}",
                (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if collision else (0, 255, 0),
                2)

    cv2.putText(frame,
                f"L:{left_risk:.2f} C:{center_risk:.2f} R:{right_risk:.2f}",
                (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)

    cv2.imshow("Agent", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()