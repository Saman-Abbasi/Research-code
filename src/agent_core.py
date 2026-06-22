import time
from collections import deque

class AgentCore:
    def __init__(self):
        # Stores events over time (scene memory)
        self.event_log = deque(maxlen=200)

        # Simple safety flags (expand later for navigation)
        self.safety_state = {
            "obstacle_near": False,
            "person_detected": False
        }

    
    def update(self, objects, frame_height=320):
        timestamp = time.time()
        event_batch = []

        # reset every frame — these must reflect only what's visible right now
        self.safety_state["person_detected"] = False
        self.safety_state["obstacle_near"] = False

        near_threshold = frame_height * 0.85  # bottom ~15% of frame counts as "near"

        for obj_id, obj in objects.items():
            label = obj["label"]
            x1, y1, x2, y2 = obj["bbox"]

            event_batch.append({"time": timestamp, "id": obj_id, "label": label, "bbox": obj["bbox"]})

            if label == "person":
                self.safety_state["person_detected"] = True
            if y2 > near_threshold:
                self.safety_state["obstacle_near"] = True

        self.event_log.extend(event_batch)
        return self.safety_state

    def get_recent_events(self, n=10):
        return list(self.event_log)[-n:]