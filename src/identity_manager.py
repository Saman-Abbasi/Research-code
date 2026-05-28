import time
import numpy as np

class IdentityManager:
    def __init__(self, max_lost_time=3.0):
        # Stores active identities
        self.objects = {}

        # Stores last seen time per global ID
        self.last_seen = {}

        # Simple counter for new IDs
        self.next_id = 0

        # How long we keep trying to reconnect an object
        self.max_lost_time = max_lost_time

    def _create_id(self):
        obj_id = self.next_id
        self.next_id += 1
        return obj_id

    def _bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def update(self, detections):
        """
        detections = list of:
        {
            "label": str,
            "bbox": [x1,y1,x2,y2]
        }
        """

        now = time.time()
        updated_ids = set()

        for det in detections:
            label = det["label"]
            bbox = det["bbox"]
            center = self._bbox_center(bbox)

            best_id = None
            best_score = float("inf")

            # Try to match with existing objects
            for obj_id, obj in self.objects.items():
                if obj["label"] != label:
                    continue

                dist = self._distance(center, obj["center"])

                if dist < best_score and dist < 150:
                    best_score = dist
                    best_id = obj_id

            # If match found → reuse ID
            if best_id is not None:
                self.objects[best_id]["center"] = center
                self.objects[best_id]["bbox"] = bbox
                self.last_seen[best_id] = now
                updated_ids.add(best_id)

            # If no match → create new identity
            else:
                new_id = self._create_id()
                self.objects[new_id] = {
                    "label": label,
                    "center": center,
                    "bbox": bbox
                }
                self.last_seen[new_id] = now
                updated_ids.add(new_id)

        # Cleanup old objects (forgotten ones)
        to_delete = []
        for obj_id, t in self.last_seen.items():
            if now - t > self.max_lost_time:
                to_delete.append(obj_id)

        for obj_id in to_delete:
            self.objects.pop(obj_id, None)
            self.last_seen.pop(obj_id, None)

        return self.objects