from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

from identity_manager import IdentityManager
from agent_core import AgentCore
from depth_model import DepthEstimator


# -------------------- Stabilization --------------------
class TemporalController:
    def __init__(self, window_size=7):
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
            return ("TURN LEFT" if left_risk < right_risk else "TURN RIGHT"), True

        if center_risk > self.slow_threshold:
            return "SLOW", True

        return "FORWARD CLEAR", False


# -------------------- Setup --------------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

identity = IdentityManager()
agent = AgentCore()
depth_model = DepthEstimator()

temporal = TemporalController()
policy = SafetyPolicyEngine()


# -------------------- Performance controls --------------------
FRAME_SKIP = 2
DEPTH_SKIP = 12
CAM_WIDTH, CAM_HEIGHT = 416, 320

frame_count = 0

cached_objects = {}
cached_depth = None

# FPS smoothing
fps_hist = deque(maxlen=30)
prev_time = time.time()

# rendering buffer (prevents tearing/jitter)
render_frame = None


# -------------------- Main Loop --------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
    frame_count += 1

    # -------------------- YOLO (throttled) --------------------
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

    # -------------------- Agent memory --------------------
    agent.update(objects)

    # -------------------- Depth (slow loop) --------------------
    if frame_count % DEPTH_SKIP == 0:
        try:
            small = cv2.resize(frame, (256, 192))
            cached_depth = depth_model.predict(small)
        except:
            pass

    depth = cached_depth

    if depth is None:
        render_frame = frame.copy()
        cv2.putText(render_frame, "INITIALIZING DEPTH...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Agent", render_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # -------------------- Risk zones --------------------
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

    # -------------------- YOLO boost --------------------
    for obj in objects.values():

        x1, y1, x2, y2 = obj["bbox"]

        area = (x2 - x1) * (y2 - y1)
        frame_area = CAM_WIDTH * CAM_HEIGHT

        ratio = area / frame_area
        cx = (x1 + x2) / 2

        if ratio > 0.10 and CAM_WIDTH * 0.33 < cx < CAM_WIDTH * 0.66:
            center_risk += 0.25

    center_risk = min(center_risk, 1.0)

    # -------------------- Policy --------------------
    raw_action, collision = policy.decide(
        left_risk, center_risk, right_risk
    )

    action = temporal.update(raw_action)

    # -------------------- FPS (smoothed) --------------------
    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    fps_hist.append(fps)
    avg_fps = sum(fps_hist) / len(fps_hist)

    # -------------------- Render --------------------
    render_frame = frame.copy()

    # draw objects
    for obj_id, obj in objects.items():

        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(render_frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        cv2.putText(render_frame,
                    f"{obj_id}:{obj['label']}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)

    # UI overlay
    cv2.putText(render_frame,
                f"FPS: {avg_fps:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0), 2)

    cv2.putText(render_frame,
                f"ACTION: {action}",
                (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if collision else (0, 255, 0),
                2)

    cv2.putText(render_frame,
                f"L:{left_risk:.2f} C:{center_risk:.2f} R:{right_risk:.2f}",
                (10, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255), 1)

    cv2.imshow("Agent", render_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()