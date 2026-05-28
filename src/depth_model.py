import torch
import numpy as np
from PIL import Image
from transformers import pipeline

class DepthEstimator:
    def __init__(self):
        # Lightweight monocular depth model (MiDaS-style via HF pipeline)
        self.pipe = pipeline(
            "depth-estimation",
            model="Intel/dpt-hybrid-midas"
        )

    def predict(self, frame_bgr):
        # Convert OpenCV image (BGR) → RGB
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb)

        result = self.pipe(image)
        depth = np.array(result["depth"])

        # Normalize depth map (0 = near, 1 = far)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

        return depth