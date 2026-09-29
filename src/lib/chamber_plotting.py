"""
chamber_plotting.py

Drawing routines for the top-down and side views of the Plassys chamber.
"""

import math
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
from matplotlib.patches import Circle, Wedge, Ellipse

from wafer_orientation import WaferOrientation

CHAMBER_RADIUS = 10.0
WAFER_RADIUS = 4.0
WAFER_HALF_THICKNESS = 0.35
AXIS_ARM_LENGTH = 3.0
AXIS_ARM_WIDTH = 2.2
ION_MILL_X = CHAMBER_RADIUS + 0.3

FLAT_DEPTH_FRAC = 0.72

BG_COLOR = "#1e1f26"
CHAMBER_COLOR = "#2a2c38"
WAFER_EDGE_COLOR = "#f2f4f8"
WAFER_DARK = "#33455e"
WAFER_LIGHT = "#d6ecff"
WAFER_RIM_DARK = "#6d3b66"
BACK_DARK = "#4b2547"
BACK_LIGHT = "#e7bfe0"
RIM_FRONT_VISIBLE = WAFER_RIM_DARK
RIM_BACK_VISIBLE = "#25384f"
FLAT_MARKER_COLOR = "#ff4d4d"
AXIS_ARM_COLOR = "#5a5d6e"
ION_MILL_COLOR = "#e08a5c"
ION_MILL_HOUSING_COLOR = "#332821"  # NEW, "quantum" pass: a small dark warm housing block behind
                                      # the mill's own emission wedge (see _draw_ion_mill) -- reads
                                      # as a gun/emitter body rather than a flat colored pie slice.
                                      # Theme-invariant, matching ION_MILL_COLOR's own reasoning below.
DEP_CHAMBER_COLOR = "#3a5f8a"
TEXT_COLOR = "#e8e8ef"
GRID_COLOR = "#3a3c4a"
NORMAL_COLOR = "#ffd166"
LOCAL_X_COLOR = "#8fd694"
LOCAL_Y_COLOR = "#f2a65a"
SHADOW_COLOR = "#000000"
SIDE_LABEL_COLOR = "#cfd3dc"

SIDE_THRESHOLD_DEG = 10.0

# --- "Quantum" chrome pass: brings the chamber model's style/colorscheme
# in line with the rest of the app, plus a subtle idle animation. --------
# Everything below is purely decorative chrome layered ON TOP of the
# existing, physically-meaningful geometry (chamber body, wafer shading,
# ion mill/deposition-chamber position, local-axis colors) -- none of
# that core geometry, its position, or its meaning changes; see
# draw_top_down/draw_side_view's own docstrings for what's additive here.
# The glow/trace color always reads the CURRENT `TEXT_COLOR` global at
# draw time (the same "quantum blue" accent parametric_junction_view.py/
# main_gui.py use for graph typography and CTA buttons elsewhere),
# so this chrome flips with the light/dark theme automatically, exactly
# like every other theme-reactive element in this module.
PULSE_CYCLE_SECONDS = 5.2  # one full idle "breathe"/traveling-pulse loop


def _glow_stroke(ax, xs, ys, color, base_lw=1.1, glow_lw=3.2, n=3, base_alpha=70, zorder=4.0,
                  closed=False):
    """Strokes the given polyline/loop several times with increasing
    linewidth and decreasing alpha, then once more crisp on top -- the
    same layered-stroke 'soft outward glow' technique QuantumActionButton
    uses for its own hover border (see main_gui.py), translated to
    matplotlib so every chamber view (chamber rim, wafer rim, axis-arm
    trace) shares that same visual language instead of a plain flat
    stroke. `closed` appends the first point again so a loop has no seam."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if closed and len(xs) > 1:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
    rgb = mcolors.to_rgb(color)
    for i in range(n, 0, -1):
        frac = i / n
        lw = base_lw + (glow_lw - base_lw) * frac
        alpha = max(0.0, min(1.0, (base_alpha / 255.0) * (1.0 - 0.75 * frac)))
        ax.plot(xs, ys, color=rgb, linewidth=lw, alpha=alpha, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder, antialiased=True)
    ax.plot(xs, ys, color=rgb, linewidth=base_lw, alpha=min(1.0, base_alpha / 255.0 + 0.35),
            solid_capstyle="round", zorder=zorder + 0.05)


def _glow_dot(ax, x, y, color, r=0.09, glow_r=0.32, n=3, zorder=6.0, alpha=0.9):
    """Same layered-alpha glow technique as _glow_stroke, for a single
    point -- used for the axis-arm's traveling pulse and its via dot."""
    rgb = mcolors.to_rgb(color)
    for i in range(n, 0, -1):
        frac = i / n
        rr = r + (glow_r - r) * frac
        a = max(0.0, alpha * (1.0 - frac) * 0.8)
        ax.add_patch(Circle((x, y), rr, facecolor=rgb, edgecolor="none", alpha=a, zorder=zorder))
    ax.add_patch(Circle((x, y), r, facecolor=rgb, edgecolor="none", alpha=alpha, zorder=zorder + 0.05))


def _breathe(pulse_phase: float) -> float:
    """A slow 0..1 sine breathing value -- used to modulate glow alpha so
    the chamber reads as 'quietly alive' rather than static, without ever
    changing shape, position, or the actual orientation being shown."""
    return 0.55 + 0.45 * math.sin(pulse_phase)


def _draw_ion_mill(ax, x: float, y: float, r: float, pulse_phase: float, zorder_base: float = 1.0):
    """The ion mill emitter -- previously a single flat-colored Wedge with
    an arbitrary size/shape. Kept at the EXACT same anchor/radius/angular span (60-300
    degrees) so every caller's layout, axis limits, and the 'Mill'/'Ion
    Mill' label position right next to it are completely unaffected --
    only its own rendering is replaced: a small housing block behind it
    (reads as a gun/emitter body) plus the same wedge now drawn as a
    layered, softly pulsing emission glow instead of one flat pie slice."""
    housing = mpatches.FancyBboxPatch(
        (x - 0.55, y - 0.95), 1.05, 1.9,
        boxstyle="round,pad=0.05,rounding_size=0.22",
        facecolor=ION_MILL_HOUSING_COLOR, edgecolor=GRID_COLOR, linewidth=0.7, zorder=zorder_base)
    ax.add_patch(housing)
    breathe = _breathe(pulse_phase)
    for frac, alpha in ((1.55, 0.10 * breathe), (1.25, 0.16 * breathe)):
        ax.add_patch(Wedge((x, y), r * frac, 60, 300, facecolor=ION_MILL_COLOR,
                            edgecolor="none", alpha=alpha, zorder=zorder_base + 0.5))
    ax.add_patch(Wedge((x, y), r, 60, 300, facecolor=ION_MILL_COLOR, edgecolor="none",
                        zorder=zorder_base + 0.5))

# --- Light/Dark theme support --------------------------------------------
# Only the colors that actually need to flip for legibility against a
# LIGHT canvas are theme-reactive: the general background, axis/label
# text, the grid, and the side-view axis labels (all near-white in the
# dark palette above, which would be invisible on a light background).
# Everything else here -- the chamber body, the wafer's own shaded
# material colors, ion-mill/deposition-chamber markers, the local-axis
# arrows -- represents the physical chamber/wafer itself, not app chrome,
# and already has enough contrast against both a dark and a light canvas,
# so it's deliberately left unchanged in both themes (changing it would
# make it look like the theme were altering the hardware, not the UI).
# TEXT_COLOR ("quantum blue"): matches the same theme-reactive text
# color parametric_junction_view.py/main_gui.py now use for graph typography
# (see parametric_junction_view.py's own _DARK_VALUES/_LIGHT_VALUES comment for the
# full rationale/contrast numbers) -- a saturated blue rather than a
# near-black/near-white neutral, so the Chamber Reference Model's own
# top-down/side-view captions and titles read as the same consistent
# accent as every other matplotlib canvas in the app.
_DARK_VALUES = dict(BG_COLOR="#1e1f26", TEXT_COLOR="#8ecbff", GRID_COLOR="#3a3c4a",
                     SIDE_LABEL_COLOR="#cfd3dc")
_LIGHT_VALUES = dict(BG_COLOR="#f5f6fa", TEXT_COLOR="#12539e", GRID_COLOR="#d9dbe3",
                      SIDE_LABEL_COLOR="#3a3d4d")


def apply_theme(mode: str) -> None:
    """Switches this module's theme-reactive color constants between
    "dark" and "light" -- called from the Qt GUI's Settings > theme
    toggle. Every draw_* function below reads these as plain module-
    level globals AT DRAW TIME (never bakes them into a returned
    object), so mutating them here and then asking the GUI to redraw is
    sufficient; no chamber-view widget needs to be rebuilt."""
    values = _LIGHT_VALUES if mode == "light" else _DARK_VALUES
    globals().update(values)

_LIGHT_DIR = np.array([-0.35, -0.55, 0.75])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)


def _shade_color(normal: np.ndarray, dark_color=WAFER_DARK, light_color=WAFER_LIGHT,
                  ambient=0.35, gain=0.75) -> tuple:
    brightness = ambient + gain * max(0.0, float(np.dot(normal, _LIGHT_DIR)))
    brightness = min(1.0, brightness)
    dark = np.array(mcolors.to_rgb(dark_color))
    light = np.array(mcolors.to_rgb(light_color))
    return tuple(dark + (light - dark) * brightness)


def _visible_face(o: WaferOrientation):
    is_front = (o.alpha_internal <= 0.0)
    if is_front:
        return WAFER_HALF_THICKNESS, -WAFER_HALF_THICKNESS, o.normal_vector, True
    else:
        return -WAFER_HALF_THICKNESS, WAFER_HALF_THICKNESS, -o.normal_vector, False


def _wafer_outline_local(radius=WAFER_RADIUS, flat_depth_frac=FLAT_DEPTH_FRAC, n=120, z=0.0):
    d = flat_depth_frac * radius
    t0 = np.arcsin(np.clip(d / radius, -1.0, 1.0))
    t = np.linspace(-t0, np.pi + t0, n)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    zz = np.full_like(t, z)
    return np.stack([x, y, zz])


def draw_top_down(ax, o: WaferOrientation, compact: bool = False, pulse_phase: float = 0.0,
                   mini_junction=None):
    """compact=False (default) is the exact original rendering -- Old
    Tab 2's Live Angle Tester calls this with no `compact` argument at
    all, so it is completely unaffected by this parameter. compact=True
    is a second, deliberately sparser rendering for small preview
    widgets (New Tab 2's Wafer Orientation Preview, the Deposition
    Slideshow): title, axis-frame arrows, and the corner legend text
    are dropped, and the remaining labels shrink -- all real information
    (which face is showing, where the flat edge is, mill/arm position)
    stays, it's just not captioned as verbosely, since at a few inches
    wide the full-size captions overlapped each other instead of
    explaining anything.

    `pulse_phase` (new, optional, default 0.0 -- a no-op idle frame,
    never required) drives the purely decorative "quantum" glow/trace
    chrome added around the chamber rim, the ion mill, and the axis arm
    (see _draw_ion_mill/_glow_stroke/_glow_dot) -- none of it changes the
    actual chamber geometry, wafer shading, or orientation being shown.

    `mini_junction` (new, optional, default None), when given, is a list
    of {"points": [(x, y), ...], "color": "#hex"} dicts -- the CURRENT
    design's own finger/patch/plate polygons, already scaled down and
    centered on the wafer's own local origin by the caller (see
    main_gui.py's _compute_mini_junction_overlay) -- drawn directly
    on the wafer's currently-visible face using the EXACT SAME
    o.rotation_matrix transform this function already applies to its own
    local axis ticks, so the mini pattern rotates/flips in lockstep with
    the wafer for every alpha/theta/reference combination, never a
    separately-computed approximation of it."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    if not compact:
        ax.set_title("Top-Down View", color=TEXT_COLOR, fontsize=12, fontweight="bold")

    breathe = _breathe(pulse_phase)

    chamber = Circle((0, 0), CHAMBER_RADIUS, facecolor=CHAMBER_COLOR, edgecolor=GRID_COLOR, linewidth=1.5, zorder=1)
    ax.add_patch(chamber)
    _chamber_theta = np.linspace(0, 2 * math.pi, 140)
    _glow_stroke(ax, CHAMBER_RADIUS * np.cos(_chamber_theta), CHAMBER_RADIUS * np.sin(_chamber_theta),
                 TEXT_COLOR, base_lw=1.0, glow_lw=3.4, n=3, base_alpha=int(65 * breathe), zorder=1.4)

    arm_x0 = -AXIS_ARM_WIDTH / 2
    arm_y0 = -CHAMBER_RADIUS - AXIS_ARM_LENGTH * 0.3
    arm = mpatches.FancyBboxPatch(
        (arm_x0, arm_y0), AXIS_ARM_WIDTH, AXIS_ARM_LENGTH,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=AXIS_ARM_COLOR, edgecolor="none", zorder=2
    )
    ax.add_patch(arm)
    # Soft "drive signal" trace down the arm's own centerline, with one
    # glowing dot looping slowly along it -- the same circuit-trace/
    # traveling-pulse motif QuantumActionButton's CTAs use elsewhere in
    # the app (see main_gui.py), applied here to represent the
    # literal thing the axis arm does (drive the wafer's rotation),
    # rather than being generic decoration. Purely additive: the arm's
    # own real rect/position/fill color above is untouched.
    trace_y0, trace_y1 = arm_y0 + AXIS_ARM_LENGTH * 0.12, arm_y0 + AXIS_ARM_LENGTH * 0.92
    _glow_stroke(ax, [0.0, 0.0], [trace_y0, trace_y1], TEXT_COLOR, base_lw=0.8, glow_lw=2.2,
                 n=2, base_alpha=100, zorder=2.2)
    loop_t = (pulse_phase / (2 * math.pi)) % 1.0
    envelope = math.sin(math.pi * loop_t)
    if envelope > 0.03:
        pulse_y = trace_y0 + (trace_y1 - trace_y0) * loop_t
        _glow_dot(ax, 0.0, pulse_y, TEXT_COLOR, r=0.05 + 0.07 * envelope,
                  glow_r=0.34 * envelope, n=3, zorder=2.3, alpha=0.85 * envelope + 0.1)
    if not compact:
        ax.text(0, -CHAMBER_RADIUS - AXIS_ARM_LENGTH * 0.9, "Axis Arm", color=TEXT_COLOR,
                fontsize=8, ha="center", va="top")

    _draw_ion_mill(ax, ION_MILL_X, 0.0, 1.4, pulse_phase, zorder_base=1.0)
    ax.text(ION_MILL_X + 1.8, 0, "Mill" if compact else "Ion Mill", color=TEXT_COLOR,
            fontsize=7 if compact else 9, ha="left", va="center")

    shadow = Ellipse((0.5, -0.5), WAFER_RADIUS * 2.05, WAFER_RADIUS * 1.7,
                      angle=0, facecolor=SHADOW_COLOR, alpha=0.18, zorder=2)
    ax.add_patch(shadow)

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
    # Soft accent rim-glow hugging the wafer's OWN real silhouette (the
    # flat included -- stroked from its actual outline points, not a
    # circle) before the crisp edge on top, same "quiet halo" language as
    # the chamber rim above -- the wafer's own material shading/edge
    # color/outline shape are all completely unchanged.
    _glow_stroke(ax, world_vis[0], world_vis[1], TEXT_COLOR, base_lw=1.0, glow_lw=3.2, n=3,
                 base_alpha=int(55 * breathe), zorder=3.95)
    ax.plot(world_vis[0], world_vis[1], color=WAFER_EDGE_COLOR, linewidth=1.3, zorder=4)

    tick_len = WAFER_RADIUS * 0.55
    local_x_tip = o.rotation_matrix @ np.array([tick_len, 0.0, 0.0])
    local_y_tip = o.rotation_matrix @ np.array([0.0, tick_len, 0.0])
    center = o.rotation_matrix @ np.array([0.0, 0.0, vis_z])
    ax.plot([center[0], local_x_tip[0]], [center[1], local_x_tip[1]], color=LOCAL_X_COLOR, linewidth=1.6, zorder=5)
    ax.plot([center[0], local_y_tip[0]], [center[1], local_y_tip[1]], color=LOCAL_Y_COLOR, linewidth=1.6, zorder=5)
    ax.plot(center[0], center[1], marker="o", color=TEXT_COLOR, markersize=3, zorder=6)

    if mini_junction:
        # Draws a miniature version of the imported junction pattern
        # directly on the wafer face. Each polygon's points arrive
        # already normalized/centered by the caller (plain (x, y) pairs
        # in the SAME local frame the wafer's own tick vectors above
        # use), so lifting them through this face's own vis_z and the
        # SAME o.rotation_matrix used for every other local->world point
        # in this function is what guarantees the mini pattern reflects
        # the chosen orientation angles for real, in all three reference
        # points, rather than just sitting drawn flat on top irrespective
        # of orientation.
        for poly in mini_junction:
            pts = poly.get("points") or []
            if len(pts) < 3:
                continue
            local_pts = np.array([[px, py, vis_z] for (px, py) in pts]).T
            world_pts = o.rotation_matrix @ local_pts
            color = poly.get("color", "#bfe0ff")
            # A dark, fixed (theme/fill-independent) edge so the shape reads
            # against BOTH the pale front-face wafer and the purple back
            # face -- BG_COLOR blends into the dark chamber backdrop but
            # not reliably into the wafer's own (often pale) material tint.
            ax.fill(world_pts[0], world_pts[1], color=color,
                    edgecolor="#12151b", linewidth=0.5, alpha=1.0, zorder=6.4)
            # Real junction fingers are routinely thin slivers (width a tiny
            # fraction of their span) -- once normalized down to "really
            # small" on the wafer, the fill alone can shrink below a
            # rendered pixel and vanish. Stroking the closed outline at a
            # fixed on-screen linewidth (points, not data units) keeps a
            # visible trace of every shape regardless of how thin its true
            # footprint became, without changing the fill/shape itself.
            xs = np.append(world_pts[0], world_pts[0][0])
            ys = np.append(world_pts[1], world_pts[1][0])
            ax.plot(xs, ys, color=color, linewidth=1.15, solid_capstyle="round",
                    solid_joinstyle="round", alpha=1.0, zorder=6.45)

    flat_xy = o.flat_vector[:2]
    norm = np.linalg.norm(flat_xy)
    if norm > 1e-6:
        flat_dir = flat_xy / norm
        flat_mid = flat_dir * (FLAT_DEPTH_FRAC * WAFER_RADIUS)
        if compact:
            ax.plot(flat_mid[0], flat_mid[1], marker="o", color=FLAT_MARKER_COLOR, markersize=5, zorder=6)
        else:
            label_pt = flat_dir * (WAFER_RADIUS * 1.35)
            ax.annotate("flat", xy=(flat_mid[0], flat_mid[1]), xytext=(label_pt[0], label_pt[1]),
                        color=FLAT_MARKER_COLOR, fontsize=8, ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=FLAT_MARKER_COLOR, lw=1.2), zorder=6)
    else:
        ax.text(0, 0, "(edge-on)", color=FLAT_MARKER_COLOR, fontsize=7 if compact else 8,
                ha="center", va="center", zorder=6)

    if not compact:
        ax.annotate("", xy=(CHAMBER_RADIUS * 0.95, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
        ax.text(CHAMBER_RADIUS * 0.98, -0.8, "+x", color=GRID_COLOR, fontsize=8)
        ax.annotate("", xy=(0, CHAMBER_RADIUS * 0.5), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
        ax.text(0.3, CHAMBER_RADIUS * 0.52, "+y", color=GRID_COLOR, fontsize=8)

    ax.set_xlim(-CHAMBER_RADIUS - 4, CHAMBER_RADIUS + 4)
    ax.set_ylim(-CHAMBER_RADIUS - (2.2 if compact else 5), CHAMBER_RADIUS + (0.8 if compact else 3))
    ax.set_aspect("equal")
    ax.axis("off")

    facing_text = "FRONT" if is_front else "BACK"
    facing_color = WAFER_LIGHT if is_front else BACK_LIGHT
    if abs(o.alpha_internal) <= SIDE_THRESHOLD_DEG:
        facing_text = "SIDE (edge-on)"
        facing_color = SIDE_LABEL_COLOR
    # facing_color intentionally reuses the wafer's own (theme-invariant)
    # "hardware" tint so this label visually ties to the face currently in
    # view -- but that pale tint is only legible against the dark chamber
    # interior it was originally tuned for, and washes out against the
    # light theme's near-white canvas. A thin TEXT_COLOR stroke (already
    # proven legible on BG_COLOR everywhere else in this widget) keeps the
    # label's color-coding intact while guaranteeing contrast in both
    # themes, without touching facing_color/the hardware palette itself.
    _face_label_stroke = [mpatheffects.withStroke(linewidth=2.2, foreground=TEXT_COLOR, alpha=0.55)]
    if compact:
        ax.text(0, -CHAMBER_RADIUS - 1.9, facing_text, color=facing_color, fontsize=6.5,
                ha="center", va="top", path_effects=_face_label_stroke)
    else:
        ax.text(-CHAMBER_RADIUS - 3.5, CHAMBER_RADIUS + 2.2,
                "red = flat edge", color=FLAT_MARKER_COLOR, fontsize=8)
        ax.text(-CHAMBER_RADIUS - 3.5, CHAMBER_RADIUS + 1.0,
                "green/orange = wafer local x/y", color=TEXT_COLOR, fontsize=7)
        ax.text(-CHAMBER_RADIUS - 3.5, CHAMBER_RADIUS - 0.2,
                f"viewing: {facing_text} face" if facing_text != "SIDE (edge-on)" else "viewing: SIDE of wafer (edge-on)",
                color=facing_color, fontsize=7, path_effects=_face_label_stroke)


def draw_side_view(ax, o: WaferOrientation, compact: bool = False, pulse_phase: float = 0.0):
    """See draw_top_down's docstring -- same compact=False default
    (Old Tab 2 unaffected), same sparser compact=True rendering, and the
    same new optional `pulse_phase` (default 0.0, a no-op idle frame)
    driving the purely decorative glow/rotation chrome added below. No
    `mini_junction` here -- the top-down view is the one that actually
    shows the wafer face-on, so that's the only view the mini pattern is
    drawn on (see draw_top_down)."""
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    if not compact:
        ax.set_title("Side Profile (view along -y, from axis arm)", color=TEXT_COLOR, fontsize=11, fontweight="bold")

    breathe = _breathe(pulse_phase)

    chamber_top = CHAMBER_RADIUS
    chamber_bottom = -CHAMBER_RADIUS - 2
    ax.add_patch(mpatches.Rectangle((-CHAMBER_RADIUS - 2, chamber_bottom), 2 * (CHAMBER_RADIUS + 2),
                                      chamber_top - chamber_bottom, facecolor=CHAMBER_COLOR,
                                      edgecolor=GRID_COLOR, linewidth=1.5, zorder=1))

    # Same deposition-chamber band, same position/height/label -- now
    # drawn as a few stacked tints blending toward the "quantum blue"
    # accent near its top edge instead of one flat color, so it reads as
    # part of the same styled system as everything else in this file.
    dep_height = 2.5
    n_bands = 5
    base_rgb = np.array(mcolors.to_rgb(DEP_CHAMBER_COLOR))
    accent_rgb = np.array(mcolors.to_rgb(TEXT_COLOR))
    for i in range(n_bands):
        frac = i / (n_bands - 1)
        band_color = tuple(base_rgb + (accent_rgb - base_rgb) * 0.22 * frac)
        y0 = chamber_bottom + dep_height * (i / n_bands)
        h = dep_height / n_bands + 0.015
        ax.add_patch(mpatches.Rectangle((-CHAMBER_RADIUS - 2, y0), 2 * (CHAMBER_RADIUS + 2), h,
                                          facecolor=band_color, edgecolor="none", zorder=1))
    if not compact:
        ax.text(0, chamber_bottom + dep_height / 2, "Evap. Deposition Chamber", color=TEXT_COLOR,
                fontsize=9, ha="center", va="center")

    _draw_ion_mill(ax, ION_MILL_X, 0.0, 1.4, pulse_phase, zorder_base=1.0)
    ax.text(ION_MILL_X + 1.8, 0, "Mill" if compact else "Ion\nMill", color=TEXT_COLOR,
            fontsize=7 if compact else 9, ha="left", va="center")

    hub = Circle((0, 0), 1.0, facecolor=AXIS_ARM_COLOR, edgecolor="none", zorder=2)
    ax.add_patch(hub)
    _hub_theta = np.linspace(0, 2 * math.pi, 60)
    _glow_stroke(ax, 1.0 * np.cos(_hub_theta), 1.0 * np.sin(_hub_theta), TEXT_COLOR,
                 base_lw=0.8, glow_lw=2.1, n=2, base_alpha=int(85 * breathe), zorder=2.15)
    # A small spoke that slowly sweeps around the hub -- literally the
    # thing this hub does (drive the wafer's rotation), not generic
    # decoration; still purely additive to the hub circle above.
    spoke_x, spoke_y = 0.82 * math.cos(pulse_phase), 0.82 * math.sin(pulse_phase)
    ax.plot([0, spoke_x], [0, spoke_y], color=TEXT_COLOR, linewidth=1.1, alpha=0.75,
            solid_capstyle="round", zorder=2.2)
    ax.plot(spoke_x, spoke_y, marker="o", color=TEXT_COLOR, markersize=2.2, alpha=0.85, zorder=2.25)
    if not compact:
        ax.text(0, -1.8, "axis", color=TEXT_COLOR, fontsize=8, ha="center", va="top")

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
    _glow_stroke(ax, world_vis[0], world_vis[2], TEXT_COLOR, base_lw=0.9, glow_lw=2.6, n=2,
                 base_alpha=int(50 * breathe), zorder=3.95)
    ax.plot(world_vis[0], world_vis[2], color=WAFER_EDGE_COLOR, linewidth=1.0, zorder=4)
    rim_color = RIM_FRONT_VISIBLE if is_front else RIM_BACK_VISIBLE
    ax.plot(world_hid[0], world_hid[2], color=rim_color, linewidth=1.0, zorder=4)

    flat_pt = o.rotation_matrix @ np.array([0.0, -FLAT_DEPTH_FRAC * WAFER_RADIUS, 0.0])
    ax.plot(flat_pt[0], flat_pt[2], marker="o", color=FLAT_MARKER_COLOR, markersize=7, zorder=5)

    tick_len = WAFER_RADIUS * 0.55
    local_x_tip = o.rotation_matrix @ np.array([tick_len, 0.0, 0.0])
    local_y_tip = o.rotation_matrix @ np.array([0.0, tick_len, 0.0])
    ax.plot([0, local_x_tip[0]], [0, local_x_tip[2]], color=LOCAL_X_COLOR, linewidth=1.4, zorder=5)
    ax.plot([0, local_y_tip[0]], [0, local_y_tip[2]], color=LOCAL_Y_COLOR, linewidth=1.4, zorder=5)

    n = o.normal_vector * 3.0
    ax.annotate("", xy=(n[0], n[2]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=NORMAL_COLOR, lw=2), zorder=5)
    if not compact:
        ax.text(n[0] * 1.15, n[2] * 1.15, "normal", color=NORMAL_COLOR, fontsize=8, ha="center")

        ax.annotate("", xy=(4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
        ax.text(4.2, -0.6, "+x", color=GRID_COLOR, fontsize=8)
        ax.annotate("", xy=(0, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GRID_COLOR, lw=1))
        ax.text(0.3, 4.2, "+z", color=GRID_COLOR, fontsize=8)

    ax.set_xlim(-CHAMBER_RADIUS - (2.5 if compact else 4), CHAMBER_RADIUS + (2.5 if compact else 5))
    ax.set_ylim(chamber_bottom - (0.3 if compact else 1), chamber_top + (1.0 if compact else 3))
    ax.set_aspect("equal")
    ax.axis("off")


def draw_overlap_shadow_profile(ax, pmma_nm: float, pmgi_nm: float, alpha_deg: float,
                                 dx_um: float, dy_um: float):
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    pmma_um = pmma_nm / 1000.0
    pmgi_um = pmgi_nm / 1000.0
    h_um = pmma_um + pmgi_um
    gap = 1.2
    undercut = 0.45

    ax.add_patch(mpatches.Rectangle((-3, -0.3), 6, 0.3, facecolor="#9acaf8", edgecolor="none"))
    ax.text(-3.2, -0.15, "Sapphire Substrate", color=TEXT_COLOR, fontsize=8, ha="right", va="center")

    pmgi_color = "#1fb3b3"
    ax.add_patch(mpatches.Rectangle((-3, 0), 3 - (gap + undercut), pmgi_um, facecolor=pmgi_color, edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((gap + undercut, 0), 3 - (gap + undercut), pmgi_um, facecolor=pmgi_color, edgecolor="none"))
    ax.text(-3.2, pmgi_um / 2, "PMGI", color=TEXT_COLOR, fontsize=8, ha="right", va="center")

    pmma_color = "#58948a"
    ax.add_patch(mpatches.Rectangle((-3, pmgi_um), 3 - gap, pmma_um, facecolor=pmma_color, edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((gap, pmgi_um), 3 - gap, pmma_um, facecolor=pmma_color, edgecolor="none"))
    ax.text(-3.2, pmgi_um + pmma_um / 2, "PMMA", color=TEXT_COLOR, fontsize=8, ha="right", va="center")

    ang = math.radians(alpha_deg)
    arrow_top_y = h_um * 1.6 + 0.3
    arrow_top_x = -math.sin(ang) * arrow_top_y
    ax.annotate("", xy=(0, h_um * 1.05), xytext=(arrow_top_x, arrow_top_y),
                arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=2))
    ax.text(arrow_top_x, arrow_top_y + 0.15, f"α={alpha_deg:g}°", color="#ffd166",
           fontsize=9, ha="center", va="bottom")

    dx_disp = max(min(dx_um, 2.8), -2.8)
    ax.annotate("", xy=(dx_disp, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="#ff8f6b", lw=1.3))
    ax.text(dx_disp / 2, -0.55, f"dx={dx_um:.3f}µm\n(overlap-shortening shift)", color="#ff8f6b",
           fontsize=8, ha="center", va="top")

    dy_disp = max(min(dy_um, 2.8), -2.8)
    ax.annotate("", xy=(dy_disp, pmgi_um), xytext=(0, pmgi_um),
                arrowprops=dict(arrowstyle="<->", color="#8fd694", lw=1.3))
    ax.text(dy_disp / 2, pmgi_um + 0.15, f"dy={dy_um:.3f}µm (lengthening)", color="#8fd694",
           fontsize=8, ha="center", va="bottom")

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.0, h_um * 1.9 + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Side Profile: Resist Stack + Overlap Shadow", color=TEXT_COLOR, fontsize=10, fontweight="bold")


def draw_electrode_shadow_profile(ax, electrode_height_nm: float, alpha_deg: float, shadow_nm: float):
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    h_um = electrode_height_nm / 1000.0
    shadow_um = shadow_nm / 1000.0
    elec_w = 1.8

    ax.add_patch(mpatches.Rectangle((-4, -0.25), 8, 0.25, facecolor="#9acaf8", edgecolor="none"))
    ax.text(-4.2, -0.12, "Sapphire Substrate", color=TEXT_COLOR, fontsize=8, ha="right", va="center")

    ax.add_patch(mpatches.Rectangle((-elec_w, 0), elec_w, h_um, facecolor="#868686", edgecolor="none"))
    ax.text(-elec_w / 2, h_um / 2, "Nb\nElectrode", color="white", fontsize=8, ha="center", va="center")

    ang = math.radians(alpha_deg)
    arrow_h = h_um * 2.2 + 0.3
    arrow_x = -elec_w + arrow_h * math.tan(ang)
    ax.annotate("", xy=(-elec_w, h_um), xytext=(arrow_x, h_um + arrow_h),
                arrowprops=dict(arrowstyle="-|>", color="#ffd166", lw=2))
    ax.text(arrow_x, h_um + arrow_h + 0.1, f"α={alpha_deg:g}°", color="#ffd166",
           fontsize=9, ha="center", va="bottom")

    shadow_clamped = min(shadow_um, 4.0)
    ax.add_patch(mpatches.Rectangle((0, 0), shadow_clamped, 0.02, facecolor="none", edgecolor="none"))
    for hx in np.linspace(0, shadow_clamped, max(int(shadow_clamped * 12), 2)):
        ax.plot([hx, hx + 0.08], [0, 0.15], color="#ff8f6b", linewidth=0.8, alpha=0.7)
    ax.annotate("", xy=(shadow_clamped, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", color="#ff8f6b", lw=1.3))
    ax.text(shadow_clamped / 2, -0.45, f"shadow = {shadow_nm:.1f} nm\n(no metal lands here)",
           color="#ff8f6b", fontsize=8, ha="center", va="top")

    ax.add_patch(mpatches.Rectangle((shadow_clamped, 0), 4.0 - shadow_clamped, 0.12,
                                    facecolor="#856ca6", edgecolor="none"))
    ax.text(min(shadow_clamped + 1.0, 3.5), 0.3, "Al lands here", color="#856ca6", fontsize=8, ha="left", va="bottom")

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-0.9, h_um * 2.6 + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Side Profile: Electrode-Topography Shadow", color=TEXT_COLOR, fontsize=10, fontweight="bold")
