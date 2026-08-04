"""
plotting.py

Drawing routines for the top-down and side views of the Plassys chamber.
Shared by plassys_gui.py / plassys_master_gui.py (live Tkinter apps) and
test_render.py (headless Agg-backend sanity checks), so both use the
identical code path.

The wafer is rendered with simple Lambertian (directional-light) shading
and a visual thickness bevel so it reads as a solid 3D object that
visibly catches light differently as alpha/theta change, rather than a
flat painted shape. This is a lighting/shading trick on top of ordinary
2D matplotlib axes (not an interactive 3D scene) -- chosen because it's
fast, robust across matplotlib versions, and easy to verify headlessly,
while still giving a genuine "this is a solid disk in 3D space" cue.

Each function takes a matplotlib Axes and a rotation_core.WaferOrientation
and draws onto that axes in place.
"""

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Wedge, Ellipse

from rotation_core import WaferOrientation

# ----------------------------------------------------------------------
# Visual constants
# ----------------------------------------------------------------------
CHAMBER_RADIUS = 10.0
WAFER_RADIUS = 4.0
WAFER_HALF_THICKNESS = 0.35   # purely visual, for the 3D bevel effect
AXIS_ARM_LENGTH = 3.0
AXIS_ARM_WIDTH = 2.2
ION_MILL_X = CHAMBER_RADIUS + 0.3

FLAT_DEPTH_FRAC = 0.72

BG_COLOR = "#1e1f26"
CHAMBER_COLOR = "#2a2c38"
WAFER_EDGE_COLOR = "#f2f4f8"  # bright neutral edge stroke -- reads clearly against both face colors
WAFER_DARK = "#33455e"     # shaded / in-shadow FRONT face color
WAFER_LIGHT = "#d6ecff"    # lit FRONT face color
WAFER_RIM_DARK = "#c96f3a"  # warm rim/thickness color -- deliberately far from the blue face tones
BACK_DARK = "#5c3018"      # shaded / in-shadow BACK face color (warm, distinct from front)
BACK_LIGHT = "#f0b27a"     # lit BACK face color
RIM_FRONT_VISIBLE = WAFER_RIM_DARK   # sliver color when the FRONT face is toward the viewer
RIM_BACK_VISIBLE = "#25384f"          # sliver color when the BACK face is toward the viewer
FLAT_MARKER_COLOR = "#ff4d4d"
AXIS_ARM_COLOR = "#5a5d6e"
ION_MILL_COLOR = "#e08a5c"
DEP_CHAMBER_COLOR = "#3a5f8a"
TEXT_COLOR = "#e8e8ef"
GRID_COLOR = "#3a3c4a"
NORMAL_COLOR = "#ffd166"
LOCAL_X_COLOR = "#8fd694"
LOCAL_Y_COLOR = "#f2a65a"
SHADOW_COLOR = "#000000"

# Fixed light direction in chamber frame (upper-front-left), used for the
# simple Lambertian shading model. Not physically meaningful, just chosen
# to make tilt/rotation visually legible.
_LIGHT_DIR = np.array([-0.35, -0.55, 0.75])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)


def _shade_color(normal: np.ndarray, dark_color=WAFER_DARK, light_color=WAFER_LIGHT,
                  ambient=0.35, gain=0.75) -> tuple:
    """Lambertian brightness from a fixed light direction -> interpolated color,
    using the given (dark, light) palette -- pass the front or back palette
    depending on which physical face is currently visible (see _visible_face)."""
    brightness = ambient + gain * max(0.0, float(np.dot(normal, _LIGHT_DIR)))
    brightness = min(1.0, brightness)
    dark = np.array(mcolors.to_rgb(dark_color))
    light = np.array(mcolors.to_rgb(light_color))
    return tuple(dark + (light - dark) * brightness)


def _visible_face(o: WaferOrientation):
    """
    Determine which PHYSICAL face of the wafer -- the local +z ("front")
    face or the local -z ("back") face -- is oriented toward the viewer in
    world +z (i.e. what a camera looking down from above the chamber, or
    from the +x side for the profile view, would actually see).

    This matters because the wafer is opaque: past alpha=90 deg from the
    front-facing orientation, the wafer has tipped past edge-on and its
    BACK face is now the one pointing toward the viewer. A correct
    renderer must swap which local surface it paints as the "main face"
    and which palette (front blue vs. back warm-orange) it uses --
    otherwise the wafer always looks like it's showing its front side even
    when it has physically flipped away, which was the reported bug.

    Returns (visible_local_z, hidden_local_z, shading_normal, is_front):
      visible_local_z / hidden_local_z: the local z-height (+/- half
        thickness) of the outline that should be drawn as the main face /
        the thin hidden-side rim, respectively.
      shading_normal: the OUTWARD normal of the currently-visible face
        (o.normal_vector if front is visible, -o.normal_vector if back is
        visible) -- feed this to _shade_color.
      is_front: True if the local +z ("front", the face whose normal is
        o.normal_vector) is the one currently visible.
    """
    is_front = (-90.0 <= o.alpha_internal <= 90.0)
    if is_front:
        return WAFER_HALF_THICKNESS, -WAFER_HALF_THICKNESS, o.normal_vector, True
    else:
        return -WAFER_HALF_THICKNESS, WAFER_HALF_THICKNESS, -o.normal_vector, False


def _wafer_outline_local(radius=WAFER_RADIUS, flat_depth_frac=FLAT_DEPTH_FRAC, n=120, z=0.0):
    """
    Flat-edge ("D-shaped") wafer outline in the wafer's own local xy-plane,
    at local z-height `z`. Flat sits along local -y, matching
    rotation_core's local_flat = (0,-1,0) convention.
    Returns a (3, n) array of local (x, y, z) points.
    """
    d = flat_depth_frac * radius
    t0 = np.arcsin(np.clip(d / radius, -1.0, 1.0))
    t = np.linspace(-t0, np.pi + t0, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    zz = np.full_like(t, z)
    return np.stack([x, y, zz])


def draw_top_down(ax, o: WaferOrientation):
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.set_title("Top-Down View", color=TEXT_COLOR, fontsize=12, fontweight="bold")

    chamber = Circle((0, 0), CHAMBER_RADIUS, facecolor=CHAMBER_COLOR, edgecolor=GRID_COLOR, linewidth=1.5, zorder=1)
    ax.add_patch(chamber)

    arm = mpatches.FancyBboxPatch(
        (-AXIS_ARM_WIDTH / 2, -CHAMBER_RADIUS - AXIS_ARM_LENGTH * 0.3),
        AXIS_ARM_WIDTH, AXIS_ARM_LENGTH,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=AXIS_ARM_COLOR, edgecolor="none", zorder=2
    )
    ax.add_patch(arm)
    ax.text(0, -CHAMBER_RADIUS - AXIS_ARM_LENGTH * 0.9, "Axis Arm", color=TEXT_COLOR,
            fontsize=8, ha="center", va="top")

    mill = Wedge((ION_MILL_X, 0), 1.4, 60, 300, facecolor=ION_MILL_COLOR, edgecolor="none", zorder=2)
    ax.add_patch(mill)
    ax.text(ION_MILL_X + 1.8, 0, "Ion Mill", color=TEXT_COLOR, fontsize=9, ha="left", va="center")

    # Ground shadow: soft dark ellipse offset from the wafer footprint,
    # cheap depth cue independent of tilt.
    shadow = Ellipse((0.5, -0.5), WAFER_RADIUS * 2.05, WAFER_RADIUS * 1.7,
                      angle=0, facecolor=SHADOW_COLOR, alpha=0.18, zorder=2)
    ax.add_patch(shadow)

    # Wafer outline: figure out which PHYSICAL face (front or back) is
    # actually oriented toward the viewer right now, and draw THAT one as
    # the solid main face, with the hidden face's outline as a thin rim
    # behind it. Past alpha=90 deg from front-on, the back face becomes
    # visible and the fill switches to the warm back-face palette -- this
    # is what makes the wafer visibly "flip over" instead of always
    # looking front-facing.
    vis_z, hid_z, shading_normal, is_front = _visible_face(o)
    local_vis = _wafer_outline_local(z=vis_z)
    local_hid = _wafer_outline_local(z=hid_z)
    world_vis = o.rotation_matrix @ local_vis
    world_hid = o.rotation_matrix @ local_hid

    rim_color = RIM_FRONT_VISIBLE if is_front else RIM_BACK_VISIBLE
    ax.fill(world_hid[0], world_hid[1], color=rim_color, zorder=3)

    dark_pal, light_pal = (WAFER_DARK, WAFER_LIGHT) if is_front else (BACK_DARK, BACK_LIGHT)
    face_color = _shade_color(shading_normal, dark_pal, light_pal)
    ax.fill(world_vis[0], world_vis[1], color=face_color, zorder=4)
    ax.plot(world_vis[0], world_vis[1], color=WAFER_EDGE_COLOR, linewidth=1.3, zorder=4)

    # Mini orientation axes: short ticks from wafer center along the
    # wafer's own local +x and +y directions.
    tick_len = WAFER_RADIUS * 0.55
    local_x_tip = o.rotation_matrix @ np.array([tick_len, 0.0, 0.0])
    local_y_tip = o.rotation_matrix @ np.array([0.0, tick_len, 0.0])
    center = o.rotation_matrix @ np.array([0.0, 0.0, vis_z])
    ax.plot([center[0], local_x_tip[0]], [center[1], local_x_tip[1]], color=LOCAL_X_COLOR, linewidth=1.6, zorder=5)
    ax.plot([center[0], local_y_tip[0]], [center[1], local_y_tip[1]], color=LOCAL_Y_COLOR, linewidth=1.6, zorder=5)
    ax.plot(center[0], center[1], marker="o", color=TEXT_COLOR, markersize=3, zorder=6)

    # "Flat" label with leader line.
    flat_xy = o.flat_vector[:2]
    norm = np.linalg.norm(flat_xy)
    if norm > 1e-6:
        flat_dir = flat_xy / norm
        flat_mid = flat_dir * (FLAT_DEPTH_FRAC * WAFER_RADIUS)
        label_pt = flat_dir * (WAFER_RADIUS * 1.35)
        ax.annotate("flat", xy=(flat_mid[0], flat_mid[1]), xytext=(label_pt[0], label_pt[1]),
                    color=FLAT_MARKER_COLOR, fontsize=8, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=FLAT_MARKER_COLOR, lw=1.2), zorder=6)
    else:
        ax.text(0, 0, "(edge-on)", color=FLAT_MARKER_COLOR, fontsize=8, ha="center", va="center", zorder=6)

    ax.annotate("", xy=(CHAMBER_RADIUS * 0.95, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
    ax.text(CHAMBER_RADIUS * 0.98, -0.8, "+x", color=GRID_COLOR, fontsize=8)
    ax.annotate("", xy=(0, CHAMBER_RADIUS * 0.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
    ax.text(0.3, CHAMBER_RADIUS * 0.52, "+y", color=GRID_COLOR, fontsize=8)

    ax.set_xlim(-CHAMBER_RADIUS - 4, CHAMBER_RADIUS + 4)
    ax.set_ylim(-CHAMBER_RADIUS - 5, CHAMBER_RADIUS + 3)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(-CHAMBER_RADIUS - 3.5, CHAMBER_RADIUS + 2.2,
            "red = flat edge", color=FLAT_MARKER_COLOR, fontsize=8)
    ax.text(-CHAMBER_RADIUS - 3.5, CHAMBER_RADIUS + 1.0,
            "green/orange = wafer local x/y", color=TEXT_COLOR, fontsize=7)
    facing_text = "viewing: FRONT face (blue)" if is_front else "viewing: BACK face (orange)"
    facing_color = WAFER_LIGHT if is_front else BACK_LIGHT
    ax.text(-CHAMBER_RADIUS - 3.5, CHAMBER_RADIUS - 0.2, facing_text, color=facing_color, fontsize=7)


def draw_side_view(ax, o: WaferOrientation):
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.set_title("Side Profile (view along -y, from axis arm)", color=TEXT_COLOR, fontsize=11, fontweight="bold")

    chamber_top = CHAMBER_RADIUS
    chamber_bottom = -CHAMBER_RADIUS - 2
    ax.add_patch(mpatches.Rectangle((-CHAMBER_RADIUS - 2, chamber_bottom), 2 * (CHAMBER_RADIUS + 2),
                                      chamber_top - chamber_bottom, facecolor=CHAMBER_COLOR,
                                      edgecolor=GRID_COLOR, linewidth=1.5, zorder=1))

    dep_height = 2.5
    ax.add_patch(mpatches.Rectangle((-CHAMBER_RADIUS - 2, chamber_bottom), 2 * (CHAMBER_RADIUS + 2),
                                      dep_height, facecolor=DEP_CHAMBER_COLOR, edgecolor="none", zorder=1))
    ax.text(0, chamber_bottom + dep_height / 2, "Evap. Deposition Chamber", color=TEXT_COLOR,
            fontsize=9, ha="center", va="center")

    mill = Wedge((ION_MILL_X, 0), 1.4, 60, 300, facecolor=ION_MILL_COLOR, edgecolor="none", zorder=2)
    ax.add_patch(mill)
    ax.text(ION_MILL_X + 1.8, 0, "Ion\nMill", color=TEXT_COLOR, fontsize=9, ha="left", va="center")

    hub = Circle((0, 0), 1.0, facecolor=AXIS_ARM_COLOR, edgecolor="none", zorder=2)
    ax.add_patch(hub)
    ax.text(0, -1.8, "axis", color=TEXT_COLOR, fontsize=8, ha="center", va="top")

    # Wafer silhouette WITH thickness: fill the ribbon between the top-
    # surface and bottom-surface outlines (projected to x-z), which is
    # the actual 3D edge profile of a disk of finite thickness as it
    # tilts -- this is what gives the side view a genuine solid, 3D look
    # rather than a single flat line. Same front/back palette swap as the
    # top-down view, so the profile also shows the wafer visibly flipping
    # from blue (front) to orange (back) as it tilts past edge-on.
    vis_z, hid_z, shading_normal, is_front = _visible_face(o)
    local_vis = _wafer_outline_local(z=vis_z)
    local_hid = _wafer_outline_local(z=hid_z)
    world_vis = o.rotation_matrix @ local_vis
    world_hid = o.rotation_matrix @ local_hid

    ribbon_x = np.concatenate([world_vis[0], world_hid[0][::-1]])
    ribbon_z = np.concatenate([world_vis[2], world_hid[2][::-1]])
    dark_pal, light_pal = (WAFER_DARK, WAFER_LIGHT) if is_front else (BACK_DARK, BACK_LIGHT)
    face_color = _shade_color(shading_normal, dark_pal, light_pal)
    ax.fill(ribbon_x, ribbon_z, color=face_color, zorder=3)
    ax.plot(world_vis[0], world_vis[2], color=WAFER_EDGE_COLOR, linewidth=1.0, zorder=4)
    rim_color = RIM_FRONT_VISIBLE if is_front else RIM_BACK_VISIBLE
    ax.plot(world_hid[0], world_hid[2], color=rim_color, linewidth=1.0, zorder=4)

    # Flat marker
    flat_pt = o.rotation_matrix @ np.array([0.0, -FLAT_DEPTH_FRAC * WAFER_RADIUS, 0.0])
    ax.plot(flat_pt[0], flat_pt[2], marker="o", color=FLAT_MARKER_COLOR, markersize=7, zorder=5)

    # Mini orientation axes
    tick_len = WAFER_RADIUS * 0.55
    local_x_tip = o.rotation_matrix @ np.array([tick_len, 0.0, 0.0])
    local_y_tip = o.rotation_matrix @ np.array([0.0, tick_len, 0.0])
    ax.plot([0, local_x_tip[0]], [0, local_x_tip[2]], color=LOCAL_X_COLOR, linewidth=1.4, zorder=5)
    ax.plot([0, local_y_tip[0]], [0, local_y_tip[2]], color=LOCAL_Y_COLOR, linewidth=1.4, zorder=5)

    n = o.normal_vector * 3.0
    ax.annotate("", xy=(n[0], n[2]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=NORMAL_COLOR, lw=2), zorder=5)
    ax.text(n[0] * 1.15, n[2] * 1.15, "normal", color=NORMAL_COLOR, fontsize=8, ha="center")

    ax.annotate("", xy=(4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
    ax.text(4.2, -0.6, "+x", color=GRID_COLOR, fontsize=8)
    ax.annotate("", xy=(0, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
    ax.text(0.3, 4.2, "+z", color=GRID_COLOR, fontsize=8)

    ax.set_xlim(-CHAMBER_RADIUS - 4, CHAMBER_RADIUS + 5)
    ax.set_ylim(chamber_bottom - 1, chamber_top + 3)
    ax.set_aspect("equal")
    ax.axis("off")
