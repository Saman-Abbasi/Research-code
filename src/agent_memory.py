from collections import defaultdict, deque
import time


class ObjectMemory:
    def __init__(self, max_history=50):
        # Stores a rolling history of detections for each object ID
        self.memory = defaultdict(lambda: deque(maxlen=max_history))

        # Stores the last time each object ID was seen
        self.last_seen = {}

    def update(self, object_id, label, bbox):
        # Current timestamp for temporal tracking
        timestamp = time.time()

        # Append latest detection info into object history
        self.memory[object_id].append({
            "label": label,
            "bbox": bbox,
            "time": timestamp
        })

        # Update last seen time for this object
        self.last_seen[object_id] = timestamp

    def get_recent(self, object_id):
        # Return full history for a specific object ID
        return list(self.memory[object_id])

    def get_active_objects(self, timeout=2.0):
        # Returns objects seen within the last N seconds
        now = time.time()

        active = []
        for obj_id, last_time in self.last_seen.items():
            if now - last_time < timeout:
                active.append(obj_id)

        return active

    def summarize(self):
        # Produces a lightweight summary of all tracked objects
        summary = {}

        for obj_id, history in self.memory.items():
            if len(history) > 0:
                latest = history[-1]

                summary[obj_id] = {
                    "label": latest["label"],
                    "seen_count": len(history)
                }

        return summary