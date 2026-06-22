# src/risk_weights.py
# Per-class risk multipliers. Weight 1.0 = baseline (same boost as an
# unweighted detection). >1.0 = higher priority hazard, <1.0 = lower priority.
# Keys must match model.names EXACTLY (case-sensitive).

CLASS_WEIGHTS = {
    # Static fall / drop hazards — highest priority, can't be talked down by speed
    "stair":              1.5,
    "stairs":             1.5,   # NOTE: duplicate of "stair" in your dataset — see flag above
    "pothole":            1.3,
    "railroad crossing":  1.4,
    "escalator":          1.3,

    # Moving vehicles — high speed, high consequence
    "Car":                1.4,
    "Bus":                1.4,
    "Truck":              1.4,
    "Ambulance":          1.4,
    "Motorcycle":         1.4,

    # Path obstructions / warnings
    "sidewalk closed":    1.2,
    "pole":               1.1,
    "caution":            1.0,
    "wet floor sign":     1.0,
    "cone":               0.9,
    "doors":              0.8,
    "trash can":          0.8,

    # Animate / mobile — present, but can move on their own
    "dog":                0.6,
    "cat":                0.5,
    "person":             0.7,

    # Furniture / low-injury static objects
    "elevator":           0.7,
    "bench":              0.7,
    "chair":              0.6,
    "table":              0.6,
    "package":            0.5,
    "window":             0.4,

    # Informational / landmark — not hazards themselves
    "stopsign":           0.4,
    "Exit":               0.3,
    "BusStop":             0.3,
    "crosswalk button":   0.3,
}

DEFAULT_WEIGHT = 0.6   # fallback for any class not listed above
BASE_BOOST     = 0.25  # same magnitude as the original flat boost