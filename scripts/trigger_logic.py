# trigger_logic.py

import cv2
import numpy as np
import time

# Minimum time between automatic triggers
TRIGGER_COOLDOWN = 10.0

_last_trigger_time = 0
_was_uncertain     = False  # tracks previous uncertainty state


def is_uncertain(frame):
    """
    Rising-edge trigger: returns True only when transitioning
    from certain → uncertain. Persistent conditions (e.g. a
    consistently dark room) fire VLM once, not repeatedly.
    """
    global _was_uncertain

    if frame is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)
    blur_score      = cv2.Laplacian(gray, cv2.CV_64F).var()

    too_dark   = mean_brightness < 40
    too_bright = mean_brightness > 220
    too_blurry = blur_score < 60

    currently_uncertain = too_dark or too_bright or too_blurry

    # Only fire on the transition: certain → uncertain
    if currently_uncertain and not _was_uncertain:
        _was_uncertain = True
        return True

    # Reset when scene recovers so next transition fires again
    if not currently_uncertain:
        _was_uncertain = False

    return False


def can_trigger():
    """
    Global cooldown gate. Prevents VLM spam.
    """
    global _last_trigger_time

    current = time.time()

    if current - _last_trigger_time < TRIGGER_COOLDOWN:
        return False

    _last_trigger_time = current
    return True