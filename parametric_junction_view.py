"""
parametric_junction_view.py
"""

import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle
import shapely.geometry as _sg  # used only by build_side_patches' exact
                                  # tip-margin calibration -- see its own
                                  # comment for why an approximate formula
                                  # isn't good enough there.

from recipe_generator import (
    DesignParameters, RecipeStep, electrode_geometry, finger_electrode_gap_um,
    electrode_safe_theta, self_shadow_safe_theta,
)

BG_COLOR = "#1e1f26"
NB_COLOR = "#3a5f8a"
NB_EDGE = "#5a7fb0"
NB_MILLED_EDGE = "#6be3c9"
AL_COLOR = "#d6ecff"
AL_EDGE = "#f2f4f8"
AL_DIM = "#5a6070"
HIGHLIGHT_COLOR = "#ffd166"
MILLING_HIGHLIGHT = "#ff8f6b"
OXIDE_HATCH_COLOR = "#9fb4c9"
TEXT_COLOR = "#e8e8ef"
SCALE_COLOR = "#e8e8ef"
GRID_COLOR = "#3a3c4a"
OUTLINE_COLOR = "#ffffff"  # the crossing-box highlight outline (see draw_top_down_layout /
                              # draw_side_view_layout below) -- near-white so it reads against
                              # the dark chamber fill; needs to flip dark for a light canvas

# --- Light/Dark theme support --------------------------------------------
# AL_COLOR and AL_EDGE (the deposited-aluminum fill/outline) are both
# near-white in the dark palette -- fine against a dark canvas, but
# nearly invisible against a light one, so both get real light-mode
# variants (a more saturated "metal" blue, and a dark outline stroke)
# rather than just leaving them unchanged. OXIDE_HATCH_COLOR is likewise
# pale enough to wash out on a light background. NB_COLOR and the other
# semantic/status colors already have enough contrast against both a
# dark and a light canvas and are left unchanged in both themes.
# TEXT_COLOR/SCALE_COLOR ("quantum blue"): fixes a real bug where graph
# text (titles, captions, the scale bar) went from near-white to
# near-black on switching to light mode, and canvases could appear stuck
# in dark mode until a manual view reset (see main_gui.py's
# _apply_theme -- now forces a synchronous .draw() instead of
# .draw_idle() so every canvas repaints immediately on toggle, not just
# eventually). Both values below use a saturated "technological/quantum"
# blue rather than a near-black/near-white neutral -- a light, clearly
# legible blue against the dark canvas, and a deep, strongly-contrasting
# blue against the light one -- so text stays readable against either
# background (matching main_gui.py's GRAPH_TEXT_COLOR so every
# matplotlib canvas in the app reads as one consistent typographic
# accent).
_DARK_VALUES = dict(BG_COLOR="#1e1f26", TEXT_COLOR="#8ecbff", SCALE_COLOR="#8ecbff",
                     GRID_COLOR="#3a3c4a", AL_COLOR="#d6ecff", AL_EDGE="#f2f4f8",
                     OXIDE_HATCH_COLOR="#9fb4c9", OUTLINE_COLOR="#ffffff")
_LIGHT_VALUES = dict(BG_COLOR="#f5f6fa", TEXT_COLOR="#12539e", SCALE_COLOR="#12539e",
                      GRID_COLOR="#d9dbe3", AL_COLOR="#6fa8dc", AL_EDGE="#2a2c38",
                      OXIDE_HATCH_COLOR="#5f7a91", OUTLINE_COLOR="#1c1d24")


def apply_theme(mode: str) -> None:
    """Switches this module's theme-reactive color constants between
    "dark" and "light". See chamber_plotting.apply_theme's docstring -- same
    pattern: every draw function here reads these as plain module
    globals at draw time, so mutating them is sufficient on its own."""
    values = _LIGHT_VALUES if mode == "light" else _DARK_VALUES
    globals().update(values)

def _rot(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])

def _dir_vec(angle_deg):
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])

class FingerAssembly:
    def __init__(self, angle_deg, length_um, lw_um, plate_w_um, plate_h_um, has_adhesion=True, overlap_um=0.0):
        self.angle_deg = angle_deg
        self.lw = lw_um
        self.has_adhesion = has_adhesion
        R = _rot(angle_deg)

        def to_world(local_xy): return R @ np.array(local_xy)
        self.to_world = to_world

        start_x = -overlap_um
        end_x = length_um - overlap_um

        self.finger_corners_local = [
            (start_x, -lw_um / 2), (end_x, -lw_um / 2),
            (end_x, lw_um / 2), (start_x, lw_um / 2),
        ]

        p1 = end_x + plate_h_um if has_adhesion else end_x
        self.plate_corners_local = [
            (end_x, -plate_w_um / 2), (p1, -plate_w_um / 2),
            (p1, plate_w_um / 2), (end_x, plate_w_um / 2),
        ] if has_adhesion else self.finger_corners_local

        self.seam_point_local = (p1, 0.0)
        self.tip_point_local = (p1, 0.0)

    def world_polygon(self, local_corners): return np.array([self.to_world(c) for c in local_corners])
    def world_point(self, local_xy): return self.to_world(local_xy)


class WorldFingerGeometry:
    """A FingerAssembly-like adapter for a finger whose geometry is NOT the
    parametric default -- e.g. a hand-drawn "finger"-role custom shape that
    has replaced one side's finger (see junction_import.from_design_parameters).

    build_side_patches() (and the adhesion-plate construction alongside it)
    only ever need four things from a "finger geometry" object: its world
    angle, its linewidth, its seam/tip point in world coordinates, and an
    "available length" scalar, as established when build_side_patches was
    extracted from what used to be FingerAssembly-only logic. This wraps
    exactly those four values so plates/patches can attach to ANY finger
    geometry, custom or parametric, through the identical code path -- this
    is the fix for a bug where adhesion pads/patches were always built from
    the parametric FingerAssembly even after a custom finger shape had
    replaced it.

    Unlike FingerAssembly (whose own to_world has no translation -- it's
    implicitly anchored at the world origin, matching every parametric
    finger pivoting on the same shared crossing point), a custom finger can
    sit anywhere, so `world_point()` here simply returns the precomputed
    world seam point regardless of what local coordinate is passed in --
    the only caller (build_side_patches) always passes this object's own
    `seam_point_local`, so this is a safe, deliberate simplification."""
    def __init__(self, angle_deg, lw, seam_world, available_length_um):
        self.angle_deg = angle_deg
        self.lw = lw
        self._seam_world = np.asarray(seam_world, dtype=float)
        self.seam_point_local = (0.0, 0.0)
        self.finger_corners_local = [(0.0, 0.0), (available_length_um, 0.0)]

    def world_point(self, local_xy):
        return self._seam_world


def build_plate_polygon_world(seam_world, angle_deg, plate_w_um, plate_h_um):
    """World-space adhesion-plate rectangle attached at `seam_world`,
    extending `plate_h_um` further outward along `angle_deg` and
    `plate_w_um` wide -- the exact same rectangle FingerAssembly's own
    plate_corners_local builds in its local frame (seam -> seam+plate_h,
    centered, width plate_w), generalized to an arbitrary world anchor/angle
    so it works identically for a parametric finger or a hand-drawn
    "finger"-role custom shape. Shared by JunctionLayout (via the
    FingerAssembly path) and junction_import.from_design_parameters (via a
    WorldFingerGeometry override) so both attach adhesion pads with
    identical math."""
    ax, ay = _dir_vec(angle_deg)
    px, py = -ay, ax
    sx, sy = seam_world
    ex, ey = sx + ax * plate_h_um, sy + ay * plate_h_um
    hw = plate_w_um / 2.0
    return [
        (sx + px * hw, sy + py * hw),
        (ex + px * hw, ey + py * hw),
        (ex - px * hw, ey - py * hw),
        (sx - px * hw, sy - py * hw),
    ]


def build_side_patches(asm, side, patch_angle, p):
    """Builds the patch dicts (cx, cy, w, h, angle, side) + placement-log
    entries for ONE finger side, from any object exposing `.angle_deg`,
    `.lw`, `.seam_point_local` + `.world_point()`, and `.finger_corners_local`
    (a FingerAssembly, or a WorldFingerGeometry wrapping a custom
    "finger"-role override -- see that class's own docstring). Extracted
    from what used to be JunctionLayout._layout_patches's own per-side loop
    body so it can be reused per-side from junction_import.from_design_parameters
    too, once that function has resolved which finger geometry is actually
    active for this side (custom override or parametric default) -- this is
    what lets patches attach to a hand-drawn finger's real position/
    rotation/length instead of always the parametric one.

    Returns (patches: List[dict], log_entries: List[dict])."""
    n = max(1, int(p.patch_count))
    w_thin, spacing = p.patch_width_um, p.patch_spacing_um
    long_ext, short_ext = p.patch_long_extension_um, p.patch_short_extension_um
    tip_margin = getattr(p, "patch_tip_margin_um", 2.0)

    seam = asm.world_point(asm.seam_point_local)
    finger_hat = _rot(asm.angle_deg) @ np.array([1.0, 0.0])
    finger_perp = np.array([-finger_hat[1], finger_hat[0]])
    patch_hat = _rot(patch_angle) @ np.array([1.0, 0.0])
    patch_perp = np.array([-patch_hat[1], patch_hat[0]])

    delta_deg = patch_angle - asm.angle_deg
    sin_delta = math.sin(math.radians(delta_deg))
    clamped = abs(sin_delta) < 0.05
    if clamped:
        sin_delta = 0.05 if sin_delta >= 0 else -0.05
    u = (asm.lw / 2.0) / abs(sin_delta)

    patch_length = 2 * u + long_ext + short_ext
    half_len, half_w = patch_length / 2.0, w_thin / 2.0
    center_shift = (long_ext - short_ext) / 2.0

    # Bug fix: the measured spacing shown in the Measurements panel did not
    # match the value typed into the spacing field -- ideally every
    # typed-in value should stay in sync with its measured counterpart,
    # including the minimum tip margin. Both the old placement pitch
    # (w_thin + spacing) and the old tip-margin baseline (w_thin / 2)
    # silently assumed patches sit AXIS-ALIGNED with the finger -- true
    # only when patch_angle equals the finger's own angle. Once patches
    # are rotated by delta_deg (patch_angle - finger angle, the normal
    # case), the real edge-to-edge gap Shapely measures between two
    # adjacent, identically-rotated patches (compute_measurements'
    # edge_to_edge_spacings_um) works out to pitch * |sin(delta)| -
    # w_thin, not the raw typed spacing -- confirmed to numerically
    # reproduce a reported case where a 1.82 typed spacing measured back
    # as 1.2. Fixed by solving that equation for the pitch that makes the
    # real measured gap equal the typed spacing:
    #     pitch * |sin(delta)| - w_thin == spacing
    #     pitch == (spacing + w_thin) / |sin(delta)|
    pitch = (spacing + w_thin) / abs(sin_delta)
    total_span = (n - 1) * pitch + w_thin

    # The "minimum tip margin past last patch" field turned out NOT to
    # need the same |sin(delta)| SCALE correction (a rigid translation of
    # the whole patch group along the finger's own axis shifts the real
    # measured margin -- an intersection with the finger's own straight
    # edge, itself parallel to that axis -- by exactly the translation
    # amount, regardless of the patches' rotation; verified directly
    # against junction_import.compute_measurements() across a spread of
    # tip-margin values, which came back with a clean slope of exactly 1).
    # What it DID need is a corrected additive BASELINE: the old code
    # assumed the outermost patch's near edge sits exactly w_thin/2 short
    # of the seam when tip_margin == 0 -- true only for an axis-aligned
    # patch. For a rotated patch, its finite WIDTH (perpendicular to its
    # own long axis) shifts exactly where its rotated boundary crosses the
    # finger's own edge line, by an amount that depends on delta_deg and
    # the patch's own long/short extension asymmetry. `_tip_exit_s` below
    # computes that crossing point EXACTLY (the same Shapely line/polygon
    # intersection junction_import._finger_past_last_patch_geometry uses
    # for the real measurement, just evaluated on a candidate patch placed
    # with tip_margin == 0), so the corrected baseline always agrees with
    # what compute_measurements() will actually report.
    def _tip_exit_s(sign):
        """Along-finger-axis position, relative to `seam`, of the
        farthest-toward-the-tip point where a tip_margin==0 candidate
        outermost patch would cross the finger's own edge line on one
        side (sign=+1/-1, offset asm.lw/2 from the finger centerline)."""
        hw = (asm.lw / 2.0) * sign
        center0 = seam + patch_hat * center_shift  # crossing point == seam here (tip_margin == 0)
        probe_half = (patch_length + w_thin) * 2.0 + 10.0
        probe_a = seam + finger_perp * hw - finger_hat * probe_half
        probe_b = seam + finger_perp * hw + finger_hat * probe_half
        probe = _sg.LineString([tuple(probe_a), tuple(probe_b)])
        corners = [
            center0 + patch_hat * half_len + patch_perp * half_w,
            center0 + patch_hat * half_len - patch_perp * half_w,
            center0 - patch_hat * half_len - patch_perp * half_w,
            center0 - patch_hat * half_len + patch_perp * half_w,
        ]
        patch_poly = _sg.Polygon([tuple(c) for c in corners])
        inter = probe.intersection(patch_poly)
        if inter.is_empty:
            return half_w  # degenerate fallback -- matches the old, approximate baseline
        if inter.geom_type == "LineString":
            candidates = list(inter.coords)
        elif inter.geom_type == "MultiLineString":
            candidates = [pt for geom in inter.geoms for pt in geom.coords]
        elif hasattr(inter, "x"):
            candidates = [(inter.x, inter.y)]
        else:
            candidates = []
        if not candidates:
            return half_w
        best = max(candidates, key=lambda pt: (pt[0] - seam[0]) * finger_hat[0] + (pt[1] - seam[1]) * finger_hat[1])
        return (best[0] - seam[0]) * finger_hat[0] + (best[1] - seam[1]) * finger_hat[1]

    # "Min. tip margin past last patch" is a single shared value guarding
    # the SHORTER of the two sides (left/right) -- matching its own name
    # and the Measurements panel's "shortest=" figure -- so the baseline
    # must be anchored to whichever side's real crossing point sits
    # farthest toward the tip (the tighter constraint).
    tip_offset = max(_tip_exit_s(+1.0), _tip_exit_s(-1.0))
    off_last = -(tip_offset + tip_margin)
    offsets = [off_last - (n - 1 - i) * pitch for i in range(n)]

    available_length = asm.finger_corners_local[1][0] - asm.finger_corners_local[0][0]
    fits = (total_span + tip_margin) <= available_length
    patches, log_entries = [], []
    for i, off in enumerate(offsets):
        crossing = seam + finger_hat * off
        center = crossing + patch_hat * center_shift
        patches.append(dict(cx=center[0], cy=center[1], w=patch_length, h=w_thin,
                            angle=patch_angle, side=side))
        log_entries.append({
            "side": side, "index": i,
            "finger_angle_deg": asm.angle_deg, "finger_width_um": asm.lw,
            "patch_angle_deg": patch_angle, "delta_deg": delta_deg,
            "sin_delta_clamped": clamped,
            "projected_half_width_u_um": u,
            "long_extension_um": long_ext, "short_extension_um": short_ext,
            "along_finger_offset_um": off,
            "seam_point": tuple(seam), "crossing_point": tuple(crossing),
            "patch_center": tuple(center), "patch_length_um": patch_length, "patch_thickness_um": w_thin,
            "total_span_requested_um": total_span, "available_finger_length_um": available_length,
            "tip_margin_um": tip_margin, "fits_within_finger": fits,
            "pitch_um": pitch, "tip_offset_um": tip_offset,
            "formula": ("u = (finger_width/2) / |sin(patch_angle - finger_angle)|; "
                       "patch_length = 2u + long_extension + short_extension; "
                       "pitch (along-finger spacing between crossing points) = "
                       "(spacing + patch_width) / |sin(patch_angle - finger_angle)|, so the real "
                       "Shapely edge-to-edge gap between patches comes back out to the typed "
                       "spacing; patch_center = crossing_point + patch_hat * (long_extension - "
                       "short_extension) / 2; the outermost patch's crossing point is anchored "
                       "tip_offset + tip_margin back from the seam, where tip_offset is computed "
                       "by exact polygon/line intersection so the real measured 'finger tip past "
                       "last patch' (shortest side) comes back out to the typed tip margin"),
        })
    return patches, log_entries


class Electrode:
    def __init__(self, cx, cy, angle_deg, w_um, h_um, auto_placed=True):
        self.cx, self.cy, self.angle_deg = cx, cy, angle_deg
        self.w, self.h = w_um, h_um
        self.auto_placed = auto_placed
        R = _rot(angle_deg)
        origin = np.array([cx, cy])

        def to_world(local_xy): return R @ np.array(local_xy) + origin
        self.to_world = to_world

        self.corners_local = [
            (0.0, -h_um / 2), (w_um, -h_um / 2), (w_um, h_um / 2), (0.0, h_um / 2),
        ]
        self.near_edge_local = [(0.0, -h_um / 2), (0.0, h_um / 2)]

    def corners_world(self): return np.array([self.to_world(c) for c in self.corners_local])
    def near_edge_midpoint_world(self): return self.to_world((0.0, 0.0))


class JunctionLayout:
    def __init__(self, p: DesignParameters):
        self.p = p
        lw_h_um = (p.horizontal_lw_nm or 200.0) / 1000.0
        lw_v_um = (p.vertical_lw_nm or 200.0) / 1000.0
        self.lw_h = lw_h_um
        self.lw_v = lw_v_um

        overlap_um = getattr(p, "overlap_target_um", 1.5) or 1.5

        self.h_asm = FingerAssembly(
            p.horizontal_finger_angle, p.horizontal_length_um, lw_h_um,
            p.adhesion_plate_width_um, p.adhesion_plate_height_um, p.has_adhesion_pads, overlap_um=overlap_um
        )
        self.v_asm = FingerAssembly(
            p.vertical_finger_angle, p.vertical_length_um, lw_v_um,
            p.adhesion_plate_width_um, p.adhesion_plate_height_um, p.has_adhesion_pads, overlap_um=overlap_um
        )
        self.asm = {"horizontal": self.h_asm, "vertical": self.v_asm}

        self.electrodes = {}
        self.gap_um = {}
        for side in ("horizontal", "vertical"):
            eg = electrode_geometry(p, side)
            self.electrodes[side] = Electrode(eg.cx, eg.cy, eg.angle_deg, eg.w_um, eg.h_um, eg.auto_placed)
            self.gap_um[side] = finger_electrode_gap_um(p, side)

        self.patches = []
        if p.patch_integrated: self._layout_patches()

    def _layout_patches(self):
        p = self.p
        self.patch_placement_log = []
        for side, asm, patch_angle in (("horizontal", self.h_asm, p.patch_angle), ("vertical", self.v_asm, p.patch_angle + 180.0)):
            patches, log_entries = build_side_patches(asm, side, patch_angle, p)
            self.patches.extend(patches)
            self.patch_placement_log.extend(log_entries)

    def jj_area_um2(self): return self.lw_h * self.lw_v

    def bounds(self):
        pts = []
        for side, asm in self.asm.items():
            pts.append(self.electrodes[side].corners_world())
            pts.append(asm.world_polygon(asm.finger_corners_local))
        pts = np.concatenate(pts, axis=0)
        for patch in self.patches: pts = np.concatenate([pts, [[patch["cx"], patch["cy"]]]], axis=0)
        return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()

def _finger_side_for_feature(feature):
    if feature is None: return None
    if feature.startswith("horizontal"): return "horizontal"
    if feature.startswith("vertical"): return "vertical"
    return None

class RecipeState:
    def __init__(self, steps, current_index):
        self.steps = steps
        self.current_index = current_index
        self.current_step = steps[current_index]

        self.deposited_sides = set()
        self.overlapped_sides = set()
        self.patch_a_done = False
        self.patch_b_done = False
        self.milled_top_done = False
        self.milled_sidewall_thetas = []
        self.last_deposition_group = None
        self._oxidation_after = {}

        last_group = None
        for i, s in enumerate(steps):
            if i > current_index: break
            f = s.feature
            if f in ("horizontal_finger", "horizontal_connect"):
                self.deposited_sides.add("horizontal")
                last_group = "horizontal"
            elif f in ("vertical_finger", "vertical_connect"):
                self.deposited_sides.add("vertical")
                last_group = "vertical"
            elif f == "horizontal_overlap":
                self.overlapped_sides.add("horizontal")
                last_group = "horizontal"
            elif f == "vertical_overlap":
                self.overlapped_sides.add("vertical")
                last_group = "vertical"
            elif f == "patch_a":
                self.patch_a_done = True
                last_group = "patches"
            elif f == "patch_b":
                self.patch_b_done = True
                last_group = "patches"
            elif f == "mill_top": self.milled_top_done = True
            elif f == "mill_sidewall" and s.theta is not None: self.milled_sidewall_thetas.append(s.theta)
            elif s.category == "misc" and s.title == "Oxidation step": self._oxidation_after[i] = last_group

        self.last_deposition_group = last_group
        self.current_oxidation_group = self._oxidation_after.get(current_index)

def _draw_poly(ax, pts, facecolor, edgecolor, zorder, alpha=1.0, lw=1.2, hatch=None, linestyle="-"):
    poly = mpatches.Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder, alpha=alpha, hatch=hatch, linestyle=linestyle)
    ax.add_patch(poly)

def _draw_patch_rect(ax, patch, facecolor, edgecolor, zorder, alpha=1.0):
    w, h = patch["w"], patch["h"]
    t = mtransforms.Affine2D().rotate_deg(patch["angle"] + 90.0).translate(patch["cx"], patch["cy"])
    rect = Rectangle((-w / 2, -h / 2), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, zorder=zorder, alpha=alpha)
    rect.set_transform(t + ax.transData)
    ax.add_patch(rect)

def _highlight_electrode_edge(ax, electrode: Electrode, theta, color, zorder, lw=4.0):
    mill_dir = _dir_vec(theta)
    corners = electrode.corners_world()
    n = len(corners)
    best_edge, best_dot = None, -1e9
    for i in range(n):
        a, b = corners[i], corners[(i + 1) % n]
        outward = np.array([b[1] - a[1], -(b[0] - a[0])])
        outward = outward / (np.linalg.norm(outward) + 1e-9)
        dot = float(np.dot(outward, mill_dir))
        if dot > best_dot: best_dot, best_edge = dot, (a, b)
    if best_edge is not None:
        a, b = best_edge
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, zorder=zorder, solid_capstyle="round")

def draw_junction_step(ax, p: DesignParameters, steps, current_index: int):
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    layout = JunctionLayout(p)
    state = RecipeState(steps, current_index)
    current_feature = state.current_step.feature
    current_side = _finger_side_for_feature(current_feature)

    for side, asm in layout.asm.items():
        edge_color = NB_MILLED_EDGE if state.milled_top_done else NB_EDGE
        _draw_poly(ax, layout.electrodes[side].corners_world(), NB_COLOR, edge_color, zorder=1, lw=1.4)

    if current_feature == "mill_top":
        for side, asm in layout.asm.items(): _draw_poly(ax, layout.electrodes[side].corners_world(), "none", MILLING_HIGHLIGHT, zorder=2, lw=3.5)
    elif current_feature == "mill_sidewall" and state.current_step.theta is not None:
        for side, asm in layout.asm.items(): _highlight_electrode_edge(ax, layout.electrodes[side], state.current_step.theta, MILLING_HIGHLIGHT, zorder=2)
    for theta in state.milled_sidewall_thetas:
        for side, asm in layout.asm.items(): _highlight_electrode_edge(ax, layout.electrodes[side], theta, NB_MILLED_EDGE, zorder=1, lw=2.5)

    for side, asm in layout.asm.items():
        deposited = side in state.deposited_sides
        is_current = (current_side == side) and current_feature in (f"{side}_finger", f"{side}_connect", f"{side}_overlap")
        finger_pts = asm.world_polygon(asm.finger_corners_local)

        if deposited:
            color = HIGHLIGHT_COLOR if is_current else AL_COLOR
            oxidized = state.current_oxidation_group == side and current_feature is None
            hatch = "///" if (oxidized or (side in state.overlapped_sides and state.last_deposition_group != side and current_feature is None and state.current_step.category == "misc")) else None
            _draw_poly(ax, finger_pts, color, AL_EDGE, zorder=3, lw=1.6 if is_current else 1.2)
            if p.has_adhesion_pads: _draw_poly(ax, asm.world_polygon(asm.plate_corners_local), color, AL_EDGE, zorder=3, lw=1.6 if is_current else 1.2)
        else:
            _draw_poly(ax, finger_pts, "none", AL_DIM, zorder=2, alpha=0.6, lw=1.0)
            if p.has_adhesion_pads: _draw_poly(ax, asm.world_polygon(asm.plate_corners_local), "none", AL_DIM, zorder=2, alpha=0.6, lw=1.0)

    if p.patch_integrated:
        for patch in layout.patches:
            done = (patch["side"] == "horizontal" and state.patch_a_done) or (patch["side"] == "vertical" and state.patch_b_done)
            is_current = (patch["side"] == "horizontal" and current_feature == "patch_a") or (patch["side"] == "vertical" and current_feature == "patch_b")
            color = HIGHLIGHT_COLOR if is_current else AL_COLOR
            _draw_patch_rect(ax, patch, color if done else "none", AL_EDGE if done else AL_DIM, zorder=4, alpha=1.0 if done else 0.6)

    if "horizontal" in state.deposited_sides and "vertical" in state.deposited_sides:
        jj_highlight = current_feature in ("horizontal_overlap", "vertical_overlap")
        ax.add_patch(Rectangle((-layout.lw_h / 2, -layout.lw_v / 2), layout.lw_h, layout.lw_v, facecolor="none", edgecolor=HIGHLIGHT_COLOR if jj_highlight else "#ffffff", linewidth=1.8, zorder=6, linestyle="--"))
        ax.annotate(f"JJ area ≈ {layout.jj_area_um2():.4f} µm²", xy=(0, 0), xytext=(0, -1.4), color=TEXT_COLOR, fontsize=8, ha="center", zorder=7)

    first_side = p.deposit_first
    if ("vertical" if first_side == "horizontal" else "horizontal") in state.deposited_sides:
        x0, x1, y0, y1 = layout.bounds()
        ax.text(x0, y1 + 0.3, f"bottom layer: {first_side} finger   |   top layer: {'vertical' if first_side == 'horizontal' else 'horizontal'} finger", color=TEXT_COLOR, fontsize=8, ha="left", va="bottom")

    if state.current_step.category == "misc" and state.current_step.title == "Oxidation step":
        group = state.last_deposition_group
        if group in ("horizontal", "vertical"):
            seam = layout.asm[group].world_point(layout.asm[group].seam_point_local)
            ax.annotate("(I) oxidized", xy=seam, xytext=(seam[0], seam[1] + 0.5), color=OXIDE_HATCH_COLOR, fontsize=8, ha="center", zorder=7)
        elif group == "patches":
            ax.annotate("(I) patches oxidized", xy=(0, 0), xytext=(0, -1.9), color=OXIDE_HATCH_COLOR, fontsize=8, ha="center", zorder=7)

    x0, x1, y0, y1 = layout.bounds()
    span = max(x1 - x0, y1 - y0)
    bar_len = min([0.1, 0.2, 0.5, 1, 2, 5, 10, 20], key=lambda v: abs(v - (span * 0.2)))
    bar_x0, bar_y = x0 + span * 0.03, y0 - span * 0.10
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y, bar_y], color=SCALE_COLOR, linewidth=2.5, zorder=8)
    ax.text(bar_x0 + bar_len / 2, bar_y - span * 0.05, f"{bar_len:g} µm", color=SCALE_COLOR, fontsize=8, ha="center", va="top", zorder=8)
    ax.set_xlim(x0 - span * 0.15, x1 + span * 0.10)
    ax.set_ylim(bar_y - span * 0.20, y1 + span * 0.20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Junction Formation (schematic, proportional only)", color=TEXT_COLOR, fontsize=11, fontweight="bold")

def _draw_background_grid(ax, x0, x1, y0, y1, step_um=1.0):
    import math as _m
    gx0, gx1 = _m.floor(x0 / step_um) * step_um, _m.ceil(x1 / step_um) * step_um
    gy0, gy1 = _m.floor(y0 / step_um) * step_um, _m.ceil(y1 / step_um) * step_um
    x = gx0
    while x <= gx1 + 1e-9:
        ax.plot([x, x], [gy0, gy1], color=GRID_COLOR, linewidth=0.5, zorder=0, alpha=0.5)
        x += step_um
    y = gy0
    while y <= gy1 + 1e-9:
        ax.plot([gx0, gx1], [y, y], color=GRID_COLOR, linewidth=0.5, zorder=0, alpha=0.5)
        y += step_um


def draw_junction_preview(ax, p: DesignParameters):
    layout = JunctionLayout(p)
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    for side, asm in layout.asm.items():
        _draw_poly(ax, layout.electrodes[side].corners_world(), NB_COLOR, NB_EDGE, zorder=1)
        _draw_poly(ax, asm.world_polygon(asm.finger_corners_local), AL_COLOR, AL_EDGE, zorder=3)
        if p.has_adhesion_pads: _draw_poly(ax, asm.world_polygon(asm.plate_corners_local), AL_COLOR, AL_EDGE, zorder=3)

    if p.patch_integrated:
        for patch in layout.patches: _draw_patch_rect(ax, patch, AL_COLOR, AL_EDGE, zorder=4)

    ax.add_patch(Rectangle((-layout.lw_h / 2, -layout.lw_v / 2), layout.lw_h, layout.lw_v, facecolor="none", edgecolor="#ffffff", linewidth=1.8, zorder=6, linestyle="--"))

    overlap_target_um = getattr(p, "overlap_target_um", 1.5) or 1.5
    is_safe = True
    min_overlap_um = max_overlap_um = None
    try:
        from recipe_generator import compute_overlap_shadow_bounds
        dx_um, dy_um, min_overlap_um, max_overlap_um, is_safe = compute_overlap_shadow_bounds(
            getattr(p, "pmma_thickness_nm", None) or 875.0,
            getattr(p, "pmgi_thickness_nm", None) or 600.0,
            getattr(p, "alpha_deposition", None) if getattr(p, "alpha_deposition", None) is not None else 30.0,
            chosen_overlap_nm=overlap_target_um * 1000.0,
        )
    except Exception:
        dx_um, dy_um = 0.85, 0.35

    x0, x1, y0, y1 = layout.bounds()
    span = max(x1 - x0, y1 - y0, 1e-6)

    _draw_background_grid(ax, x0 - span * 0.15, x1 + span * 0.10, y0 - span * 0.20, y1 + span * 0.20,
                           step_um=min([0.1, 0.2, 0.5, 1, 2, 5, 10], key=lambda v: abs(v - span / 8)))

    second_angle = p.second_finger_angle()
    second_side = "vertical" if p.deposit_first == "horizontal" else "horizontal"
    overlap_dir = _dir_vec(second_angle)
    perp_dir = _dir_vec(second_angle + 90.0)
    tick_len = max(layout.asm[second_side].lw * 2.0, span * 0.03)

    def _overlap_point(depth_um):
        return -overlap_dir * depth_um

    def _draw_overlap_marker(depth_um, color, label, label_side=1.0):
        pt = _overlap_point(depth_um)
        t0, t1 = pt - perp_dir * tick_len / 2, pt + perp_dir * tick_len / 2
        ax.plot([t0[0], t1[0]], [t0[1], t1[1]], color=color, linewidth=1.6, zorder=9, linestyle=":")
        ax.plot([0, pt[0]], [0, pt[1]], color=color, linewidth=0.8, zorder=5, linestyle=":", alpha=0.7)
        label_pt = pt + perp_dir * tick_len * 1.1 * label_side
        ax.text(label_pt[0], label_pt[1], label, color=color, fontsize=7, ha="center", va="center", zorder=9)

    _draw_overlap_marker(min_overlap_um if min_overlap_um is not None else overlap_target_um - dx_um,
                          "#ff4d4d", f"−dx worst-case\n{dx_um:.3f} µm shrink", label_side=1.0)
    _draw_overlap_marker(max_overlap_um if max_overlap_um is not None else overlap_target_um + dy_um,
                          "#8fd694", f"+dy worst-case\n{dy_um:.3f} µm growth", label_side=-1.0)

    caption = (f"JJ Target Area ≈ {layout.jj_area_um2():.4f} µm²\n"
               f"Target Overlap: {overlap_target_um:.2f} µm")
    if min_overlap_um is not None:
        caption += f"  (worst-case remaining: {min_overlap_um:.3f} µm)"
    ax.text(0.02, 0.02, caption, transform=ax.transAxes, color=TEXT_COLOR, fontsize=8, ha="left", va="bottom", zorder=7)

    banner_y = 0.98
    if not is_safe:
        shortfall = -(min_overlap_um if min_overlap_um is not None else 0.0)
        ax.text(0.5, banner_y,
                f"⚠ INSUFFICIENT OVERLAP: worst-case shadow shift leaves "
                f"{min_overlap_um:.3f} µm ({'open junction risk' if min_overlap_um <= 0 else 'marginal'}) "
                f"-- increase target overlap by ≥{shortfall:.3f} µm or reduce alpha.",
                transform=ax.transAxes, color="#1e1f26", fontsize=8, fontweight="bold", ha="center", va="top",
                zorder=10, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ff4d4d", edgecolor="none"))
        banner_y -= 0.07

    for side, asm in layout.asm.items():
        electrode = layout.electrodes[side]
        gap = layout.gap_um[side]
        tip_world = asm.world_point(asm.tip_point_local)
        edge_world = electrode.near_edge_midpoint_world()
        if gap < 0:
            color = "#c98bff" if p.allow_electrode_overlap else "#ff4d4d"
        elif gap < 0.3:
            color = "#ffd166"
        else:
            color = "#8fd694"
        perp = _dir_vec(electrode.angle_deg + 90.0)
        d_tick = max(electrode.h * 0.35, span * 0.02)
        for pt in (tip_world, edge_world):
            a, b = pt - perp * d_tick / 2, pt + perp * d_tick / 2
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.1, zorder=9)
        ax.annotate("", xy=edge_world, xytext=tip_world,
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.0), zorder=9)
        mid = (tip_world + edge_world) / 2.0
        label_pt = mid + perp * d_tick * 1.3
        label = f"{gap:.3f} µm gap" if gap >= 0 else f"{-gap:.3f} µm OVERLAP"
        ax.text(label_pt[0], label_pt[1], label, color=color, fontsize=7, ha="center", va="center", zorder=9)

    short_lines = []
    for side in ("horizontal", "vertical"):
        gap = layout.gap_um[side]
        if gap >= 0:
            continue
        safe_t, unsafe_t = electrode_safe_theta(p, side)
        if not p.allow_electrode_overlap:
            short_lines.append(f"{side}: overlaps electrode by {-gap:.2f}µm -- not allowed")
        else:
            short_lines.append(f"{side}: overlaps electrode {-gap:.2f}µm -- use θ={safe_t:.0f}°, NOT {unsafe_t:.0f}°")
    if short_lines:
        ax.text(0.5, banner_y, "⚠ " + "\n".join(short_lines),
                transform=ax.transAxes, color="#1e1f26", fontsize=7, fontweight="bold", ha="center", va="top",
                zorder=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffb454", edgecolor="none"))

    for side, asm in layout.asm.items():
        x_lo = asm.finger_corners_local[0][0]
        x_hi = asm.finger_corners_local[1][0]
        center_local = ((x_lo + x_hi) / 2.0, asm.lw / 2.0 + max(0.12, span * 0.012))
        center_world = asm.world_point(center_local)
        half_len = abs(x_hi - x_lo) * 0.42
        fwd = _dir_vec(asm.angle_deg)
        for sign in (1.0, -1.0):
            tip = center_world + fwd * sign * half_len
            ax.annotate("", xy=tip, xytext=center_world,
                        arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT_COLOR, lw=1.1,
                                         shrinkA=0, shrinkB=0, mutation_scale=9),
                        zorder=9)
            theta_val = (asm.angle_deg if sign > 0 else asm.angle_deg + 180.0) % 360.0
            label_pt = tip + fwd * sign * max(0.10, span * 0.01)
            ax.text(label_pt[0], label_pt[1], f"θ={theta_val:.0f}°", color=HIGHLIGHT_COLOR,
                    fontsize=6.5, ha="center", va="center", zorder=9)

    bar_y = y0 - span * 0.10
    bar_len = min([0.1, 0.2, 0.5, 1, 2, 5, 10, 20], key=lambda v: abs(v - (span * 0.2)))
    ax.plot([x0 + span * 0.03, x0 + span * 0.03 + bar_len], [bar_y, bar_y], color=SCALE_COLOR, linewidth=2.5, zorder=8)
    ax.text(x0 + span * 0.03 + bar_len / 2, bar_y - span * 0.05, f"{bar_len:g} µm", color=SCALE_COLOR, fontsize=8, ha="center", va="top", zorder=8)

    ax.set_xlim(x0 - span * 0.15, x1 + span * 0.10)
    ax.set_ylim(bar_y - span * 0.20, y1 + span * 0.20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Junction Design Preview & Tolerance Bounds", color=TEXT_COLOR, fontsize=11, fontweight="bold")
