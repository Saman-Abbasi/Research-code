# trigger_logic.py

import cv2
import numpy as np
import time

# Minimum time between automatic triggers
TRIGGER_COOLDOWN = 10.0

_last_trigger_time = 0


def is_uncertain(frame):
    """
    Returns True if image quality appears poor.
    """

    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    too_dark = mean_brightness < 40
    too_bright = mean_brightness > 220
    too_blurry = blur_score < 60

    return too_dark or too_bright or too_blurry


def can_trigger():
    """
    Global cooldown gate.

    Prevents VLM spam.
    """

    global _last_trigger_time

    current = time.time()

    if current - _last_trigger_time < TRIGGER_COOLDOWN:
        return False

    _last_trigger_time = current
    return True