"""
rotation_core.py

Core rotation math for the Plassys PVD / Ion-Mill chamber wafer orientation tool.

============================================================================
COORDINATE FRAME (fixed, chamber frame — see reference diagram)
============================================================================
    +x  -> points toward the ion mill
    +y  -> points away from the axis arm (axis arm sits at -y)
    +z  -> normal to the loading dock, pointing "up" out of the chamber
           at the load position

    At LOAD position (wafer flat, face up on loading dock):
        wafer normal    = +z
        wafer flat edge -> faces the axis arm, i.e. points along -y

============================================================================
DEGREES OF FREEDOM
============================================================================
Two rotations only, always applied in this order about the FIXED chamber
axes (not a gimbal — theta is about the fixed z, alpha is about the fixed
y, both defined in the chamber frame, matching the physical arm which
tilts about a fixed y-axis regardless of planetary spin):

    1. THETA  - planetary rotation about the chamber +z axis.
                theta = 0   -> flat edge faces axis arm (-y)
                Positive theta = CLOCKWISE when viewed top-down (the
                convention given: flat edge sweeps left -> top -> right
                for +theta).
                NOTE: standard right-hand-rule rotation about +z is
                counterclockwise for positive angle when viewed from
                +z looking down at the x-y plane. The user's convention
                is the OPPOSITE handedness, so we implement theta's
                rotation matrix with a negated sign relative to the
                standard right-hand rule to match clockwise-positive.

    2. ALPHA  - tilt about the chamber y-axis, applied AFTER theta.
                This models the physical tilt arm: the wafer is spun
                to theta first, then the whole assembly (including the
                already-spun wafer) tilts by alpha about the fixed
                y-axis. This matches "alpha and theta can be changed
                at the same time" / Manhattan-style junctions, where
                theta re-orients which part of the wafer is "uphill"
                before/while the alpha tilt is applied.

============================================================================
ALPHA REFERENCE: TWO CONVENTIONS
============================================================================
There are two ways alpha is expressed, and this module explicitly
distinguishes them:

  (A) INTERNAL / CONTINUOUS convention (used for all math):
        alpha_internal = -90 deg  -> LOAD position (wafer normal = +z)
        alpha_internal =   0 deg  -> MILL position (wafer normal = +x)
        alpha_internal = +90 deg  -> DEPOSITION position (wafer normal = -z)
      This is ONE continuous rotation, always in the same direction,
      about the fixed y-axis, hardware-limited to roughly [-90, +90].

  (B) MACHINE / DISPLAY convention (what the user types):
        The machine's own display resets alpha = 0 at LOAD, then resets
        AGAIN to alpha = 0 once the wafer reaches MILL. So a user-facing
        alpha is only meaningful together with a chosen reference preset:
            reference = 'load'  : alpha_display in [0, 90]   (0=load, 90=mill)
            reference = 'mill'  : alpha_display in [-90, 90]  (0=mill,
                                    -90=load, +90=deposition)
      This module converts (reference, alpha_display) <-> alpha_internal.

============================================================================
NEGATIVE-ANGLE AUTOCORRECT
============================================================================
If a typed (alpha, theta) pair is outside the physically reachable range,
apply the identity: tilting alpha the "wrong way" by some amount is
geometrically equivalent to tilting the correct way by (180 - alpha) with
theta spun 180 degrees. This is applied to BOTH alpha and theta together
in one pass: whenever the resolved alpha_internal falls outside
[-90, 90], reflect it back into range via:
        alpha_internal' = 180 - alpha_internal   (choose the reflection
                                                    that lands in range)
        theta'          = theta + 180
This is the same "add 180 and adjust" logic applied consistently.
============================================================================
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


# ─── REPLACEMENT OF _reflect_alpha_into_range ────────────────────────────
def _reflect_alpha_into_range(alpha_internal: float, theta: float):
    """
    If alpha_internal is outside [-90, 180], apply the 180-degree
    equivalence reflection until it lands in range. Returns
    (alpha_internal_corrected, theta_corrected, was_corrected).
    """
    corrected = False
    a = alpha_internal
    t = theta

    # Normalize alpha into (-180, 180] for a stable starting point
    a = a % 360.0
    if a > 180.0:
        a -= 360.0

    while a < ALPHA_MIN or a > ALPHA_MAX:
        a = 180.0 - a
        t = t + 180.0
        corrected = True
        
        # Renormalize inside the loop to ensure clean termination
        a = a % 360.0
        if a > 180.0:
            a -= 360.0

    return a, wrap_theta(t), corrected


def alpha_display_to_internal(alpha_display: float, reference: str) -> float:
    """
    Convert a user-typed alpha (in machine/display convention) to the
    internal continuous convention, WITHOUT range-checking/autocorrect
    (that happens later, jointly with theta).
    """
    if reference == "load":
        # display: 0 = load, 90 = mill  ->  internal: -90 = load, 0 = mill
        return alpha_display - 90.0
    elif reference == "mill":
        # display: -90 = load, 0 = mill, 90 = deposition -> already internal
        return alpha_display
    elif reference == "deposition":
        # display: 0 = deposition, -90 = mill, -180 = load.
        # UNVERIFIED: whether the Plassys display actually re-zeros alpha a
        # SECOND time at deposition (the way it's confirmed to re-zero once
        # at mill) has not been confirmed on the real hardware -- you asked
        # for this as a convention to test against, not a confirmed fact.
        # Mirrors the 'load' convention's offset (+90 there, so -90 here)
        # to keep the same continuous direction.
        return 90.0 + alpha_display
    else:
        raise ValueError(f"Unknown reference '{reference}', must be 'load', 'mill', or 'deposition'")


def alpha_internal_to_display(alpha_internal: float, reference: str) -> float:
    """Inverse of alpha_display_to_internal."""
    if reference == "load":
        return alpha_internal + 90.0
    elif reference == "mill":
        return alpha_internal
    elif reference == "deposition":
        return alpha_internal - 90.0
    else:
        raise ValueError(f"Unknown reference '{reference}', must be 'load', 'mill', or 'deposition'")


def rotation_matrix_theta(theta_deg: float) -> np.ndarray:
    """
    Rotation about chamber +z axis, CLOCKWISE-positive when viewed
    top-down from +z looking toward -z (the user's stated convention,
    opposite of the standard right-hand rule).

    Standard right-hand-rule CCW-positive rotation about z is:
        [ cos -sin  0 ]
        [ sin  cos  0 ]
        [  0    0   1 ]

    We negate the angle to flip handedness to CW-positive:
    """
    t = np.radians(-theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_matrix_alpha(alpha_internal_deg: float) -> np.ndarray:
    """
    Rotation about the fixed chamber +y axis, applied AFTER theta.

    Must satisfy:
        alpha_internal = -90 -> normal (0,0,1) maps to... wait, normal
        starts as +z at load ONLY when alpha_internal=-90. Let's verify:
        we want R(alpha=-90) applied to +z to give +z (load = identity
        on the normal, since load IS the +z-normal reference), and
        R(alpha=0) applied to +z to give +x (mill), and R(alpha=+90)
        applied to +z to give -z (deposition).

    Standard right-hand-rule rotation about +y by angle b:
        [ cos b   0   sin b ]
        [   0     1     0   ]
        [-sin b   0   cos b ]
    Applied to (0,0,1): -> (sin b, 0, cos b)

    We need a mapping f(alpha_internal) = b such that:
        f(-90) gives normal (0,0,1)   -> need sin b=0, cos b=1 -> b=0
        f(0)   gives normal (1,0,0)   -> need sin b=1, cos b=0 -> b=90
        f(90)  gives normal (0,0,-1)  -> need sin b=0, cos b=-1 -> b=180

    So b = alpha_internal + 90 satisfies all three. Use that.
    """
    b = np.radians(alpha_internal_deg + 90.0)
    c, s = np.cos(b), np.sin(b)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def compute_orientation(alpha_display: float, theta_display: float, reference: str = "load") -> WaferOrientation:
    """
    Main entry point. Takes user-typed alpha (in the given reference
    convention) and theta, applies autocorrect if needed, and returns
    the full resolved orientation including rotation matrix and key
    vectors for plotting.
    """
    alpha_internal_raw = alpha_display_to_internal(alpha_display, reference)
    alpha_internal, theta_corrected, was_corrected = _reflect_alpha_into_range(
        alpha_internal_raw, theta_display
    )

    R_theta = rotation_matrix_theta(theta_corrected)
    R_alpha = rotation_matrix_alpha(alpha_internal)

    # Theta is applied first (wafer spins on the platter), THEN alpha
    # tilts the whole already-spun assembly about the fixed y-axis.
    R = R_alpha @ R_theta

    # Wafer-local frame: normal starts along local +z, flat edge starts
    # pointing along local -y (toward axis arm at load, theta=0).
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


# ----------------------------------------------------------------------
# Preset positions
# ----------------------------------------------------------------------

def preset_load() -> WaferOrientation:
    return compute_orientation(alpha_display=0.0, theta_display=0.0, reference="load")


def preset_mill() -> WaferOrientation:
    return compute_orientation(alpha_display=0.0, theta_display=0.0, reference="mill")


def preset_deposition() -> WaferOrientation:
    return compute_orientation(alpha_display=90.0, theta_display=0.0, reference="mill")


if __name__ == "__main__":
    # Quick sanity checks against the three presets.
    def show(name, o: WaferOrientation):
        print(f"--- {name} ---")
        print(f"  alpha_internal = {o.alpha_internal:.2f}, theta = {o.theta:.2f}")
        print(f"  normal_vector  = {np.round(o.normal_vector, 4)}")
        print(f"  flat_vector    = {np.round(o.flat_vector, 4)}")
        print()

    show("LOAD", preset_load())
    show("MILL", preset_mill())
    show("DEPOSITION", preset_deposition())

    # Expected:
    # LOAD:       normal ~ (0,0,1),  flat ~ (0,-1,0)
    # MILL:       normal ~ (1,0,0),  flat ~ (0,-1,0)
    # DEPOSITION: normal ~ (0,0,-1), flat ~ (0,-1,0)  (flat direction unchanged
    #             since alpha rotates about y and flat starts along y-axis... )
