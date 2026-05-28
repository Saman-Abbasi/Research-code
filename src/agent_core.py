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

    def update(self, objects):
        """
        objects = dict from IdentityManager:
        {id: {label, bbox, center}}
        """

        timestamp = time.time()

        event_batch = []

        for obj_id, obj in objects.items():
            label = obj["label"]
            x1, y1, x2, y2 = obj["bbox"]

            event = {
                "time": timestamp,
                "id": obj_id,
                "label": label,
                "bbox": obj["bbox"]
            }

            event_batch.append(event)

            # Basic safety heuristics (very early stage)
            if label in ["person"]:
                self.safety_state["person_detected"] = True

            # crude proximity check (bottom of frame = "near")
            if y2 > 350:
                self.safety_state["obstacle_near"] = True

        # reset safety each cycle (so it reflects current frame only)
        if len(objects) == 0:
            self.safety_state["person_detected"] = False
            self.safety_state["obstacle_near"] = False

        # store events
        self.event_log.extend(event_batch)

        return self.safety_state

    def get_recent_events(self, n=10):
        return list(self.event_log)[-n:]