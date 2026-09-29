"""
wafer_orientation.py

Core rotation math for the Plassys PVD / Ion-Mill chamber wafer orientation tool.
"""

from dataclasses import dataclass
import numpy as np


ALPHA_MIN = -90.0
ALPHA_MAX = 180.0


@dataclass
class WaferOrientation:
    alpha_internal: float          # continuous internal alpha, degrees, in [-90, 90]
    theta: float                   # planetary angle, degrees, wrapped to [0, 360)
    alpha_input_raw: float         # exactly what the user typed (pre-autocorrect)
    theta_input_raw: float         # exactly what the user typed (pre-autocorrect)
    reference: str                 # 'load' or 'mill' — which display convention was used
    autocorrected: bool            # True if the +180 correction was applied
    rotation_matrix: np.ndarray    # 3x3, maps wafer-local coords -> chamber coords
    normal_vector: np.ndarray      # wafer face normal in chamber frame (unit vector)
    flat_vector: np.ndarray        # direction the wafer's flat edge points, in chamber frame


def wrap_theta(theta: float) -> float:
    """Wrap theta into [0, 360)."""
    return theta % 360.0


def _reflect_alpha_into_range(alpha_internal: float, theta: float):
    corrected = False
    a = alpha_internal
    t = theta

    a = a % 360.0
    if a > 180.0:
        a -= 360.0

    while a < ALPHA_MIN or a > ALPHA_MAX:
        a = 180.0 - a
        t = t + 180.0
        corrected = True

        a = a % 360.0
        if a > 180.0:
            a -= 360.0

    return a, wrap_theta(t), corrected


def alpha_display_to_internal(alpha_display: float, reference: str) -> float:
    if reference == "load":
        return alpha_display - 90.0
    elif reference == "mill":
        return alpha_display
    elif reference == "deposition":
        return 90.0 + alpha_display
    else:
        raise ValueError(f"Unknown reference '{reference}', must be 'load', 'mill', or 'deposition'")


def alpha_internal_to_display(alpha_internal: float, reference: str) -> float:
    if reference == "load":
        return alpha_internal + 90.0
    elif reference == "mill":
        return alpha_internal
    elif reference == "deposition":
        return alpha_internal - 90.0
    else:
        raise ValueError(f"Unknown reference '{reference}', must be 'load', 'mill', or 'deposition'")


def rotation_matrix_theta(theta_deg: float) -> np.ndarray:
    t = np.radians(-theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_matrix_alpha(alpha_internal_deg: float) -> np.ndarray:
    b = np.radians(alpha_internal_deg + 90.0)
    c, s = np.cos(b), np.sin(b)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def compute_orientation(alpha_display: float, theta_display: float, reference: str = "load") -> WaferOrientation:
    alpha_internal_raw = alpha_display_to_internal(alpha_display, reference)
    alpha_internal, theta_corrected, was_corrected = _reflect_alpha_into_range(
        alpha_internal_raw, theta_display
    )

    R_theta = rotation_matrix_theta(theta_corrected)
    R_alpha = rotation_matrix_alpha(alpha_internal)

    R = R_alpha @ R_theta

    local_normal = np.array([0.0, 0.0, 1.0])
    local_flat = np.array([0.0, -1.0, 0.0])

    normal_vector = R @ local_normal
    flat_vector = R @ local_flat

    return WaferOrientation(
        alpha_internal=alpha_internal,
        theta=theta_corrected,
        alpha_input_raw=alpha_display,
        theta_input_raw=theta_display,
        reference=reference,
        autocorrected=was_corrected,
        rotation_matrix=R,
        normal_vector=normal_vector,
        flat_vector=flat_vector,
    )


def preset_load() -> WaferOrientation:
    return compute_orientation(alpha_display=0.0, theta_display=0.0, reference="load")


def preset_mill() -> WaferOrientation:
    return compute_orientation(alpha_display=0.0, theta_display=0.0, reference="mill")


def preset_deposition() -> WaferOrientation:
    return compute_orientation(alpha_display=90.0, theta_display=0.0, reference="mill")
