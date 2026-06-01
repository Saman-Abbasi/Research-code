# frame_buffer.py
# Stores ONLY the latest camera frame (no queue buildup)

import numpy as np

_latest_frame = None


def update_frame(frame):
    global _latest_frame
    _latest_frame = frame


def get_latest_frame():
    global _latest_frame
    return _latest_frame