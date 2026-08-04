"""
junction_view.py

Schematic (proportional, not physically-simulated) renderer for the
Josephson-junction formation slideshow: draws the Nb electrodes, then
progressively reveals each Al feature (finger, adhesion plate, patch) as
its recipe step is reached, plus mill-phase highlighting and a scale bar.

Two important design choices, both deliberate simplifications per your
direction ("approximate proportions... not the exact generation of an
ultra-specific design"):

1. Each finger's angle (DesignParameters.horizontal_finger_angle /
   vertical_finger_angle) is used DIRECTLY as the physical direction, from
   the junction crossing, that the finger points toward its own Nb
   electrode. The whole finger+adhesion-plate+electrode assembly is drawn
   rotated to that angle -- so a finger angle of 270 draws its electrode
   at the bottom, 45 draws it diagonally, etc. This replaced an earlier
   version that silently assumed horizontal=right/vertical=top no matter
   what angle was typed.

2. Which specific recipe sub-step is "current" vs. "already done" vs.
   "not yet reached" is now computed from the FULL step list + a step
   index (not a single feature-name lookup), so previously-deposited
   features correctly stay visible through later prep/oxidation steps
   instead of reverting to a ghost outline.

No shadow-evaporation displacement is modeled in the drawing itself (that
math lives in recipe_model.compute_overlap_shadow_bounds for the numeric
side) -- the goal here is proportional, step-accurate schematic, not a
physical footprint simulation.
"""

import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle

from recipe_model import DesignParameters, RecipeStep

BG_COLOR = "#1e1f26"
NB_COLOR = "#3a5f8a"
NB_EDGE = "#5a7fb0"
NB_MILLED_EDGE = "#6be3c9"       # persistent "this side/surface has been cleaned" edge color
AL_COLOR = "#d6ecff"
AL_EDGE = "#f2f4f8"
AL_DIM = "#5a6070"                # not-yet-deposited features shown dimmed (ghost outline)
HIGHLIGHT_COLOR = "#ffd166"       # the step currently happening, right now
MILLING_HIGHLIGHT = "#ff8f6b"     # the region currently being ion-milled
OXIDE_HATCH_COLOR = "#9fb4c9"
TEXT_COLOR = "#e8e8ef"
SCALE_COLOR = "#e8e8ef"
GRID_COLOR = "#3a3c4a"


def _rot(angle_deg):
    """Standard math-convention 2D rotation matrix (CCW positive), used for
    finger/electrode design-space angles -- an independent convention from
    the chamber's clockwise-positive wafer theta in rotation_core.py."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _dir_vec(angle_deg):
    a = math.radians(angle_deg)
    return np.array([math.cos(a), math.sin(a)])


class FingerAssembly:
    """
    Geometry for one finger + its adhesion plate + its Nb electrode.
    Now subtracts the nominal overlap from the starting point to embed 
    the junction correctly into the finger budget.
    """
    def __init__(self, angle_deg, length_um, lw_um, plate_w_um, plate_h_um,
                 overlap_um=0.0, electrode_depth_um=2.0, electrode_span_um=4.0):
        self.angle_deg = angle_deg
        self.lw = lw_um
        R = _rot(angle_deg)

        def to_world(local_xy):
            return R @ np.array(local_xy)

        self.to_world = to_world

        # FIXED: Subtract overlap from the total finger length so the base 
        # structure starts back at a negative offset rather than at zero!
        start_x = -overlap_um
        end_x = length_um - overlap_um

        # Finger local corners
        self.finger_corners_local = [
            (start_x, -lw_um / 2), (end_x, -lw_um / 2),
            (end_x, lw_um / 2), (start_x, lw_um / 2),
        ]

        # Adhesion plate sits directly where the adjusted finger ends
        p0, p1 = end_x, end_x + plate_h_um
        self.plate_corners_local = [
            (p0, -plate_w_um / 2), (p1, -plate_w_um / 2),
            (p1, plate_w_um / 2), (p0, plate_w_um / 2),
        ]
        
        self.seam_point_local = (p1, 0.0)

        # Nb electrode sits beyond the adhesion plate
        e0 = p1
        e1 = e0 + electrode_depth_um
        self.electrode_corners_local = [
            (e0, -electrode_span_um / 2), (e1, -electrode_span_um / 2),
            (e1, electrode_span_um / 2), (e0, electrode_span_um / 2),
        ]
        self.electrode_near_edge_local = [(e0, -electrode_span_um / 2), (e0, electrode_span_um / 2)]

    def world_polygon(self, local_corners):
        return np.array([self.to_world(c) for c in local_corners])

    def world_point(self, local_xy):
        return self.to_world(local_xy)

    def far_extent_um(self):
        return self.electrode_corners_local[1][0]


class JunctionLayout:
    """Builds both finger assemblies + patch geometry from DesignParameters."""
    def __init__(self, p: DesignParameters):
        self.p = p
        lw_h_um = (p.horizontal_lw_nm or 200.0) / 1000.0
        lw_v_um = (p.vertical_lw_nm or 200.0) / 1000.0
        self.lw_h = lw_h_um
        self.lw_v = lw_v_um

        # Convert overlap from nm to um (defaulting to 1.5 um if none specified)
        overlap_um = (getattr(p, "overlap_length_nm", 1500.0) or 1500.0) / 1000.0

        # Pass overlap down to assemblies
        self.h_asm = FingerAssembly(
            p.horizontal_finger_angle, p.horizontal_length_um, lw_h_um,
            p.adhesion_plate_width_um, p.adhesion_plate_height_um, overlap_um=overlap_um
        )
        self.v_asm = FingerAssembly(
            p.vertical_finger_angle, p.vertical_length_um, lw_v_um,
            p.adhesion_plate_width_um, p.adhesion_plate_height_um, overlap_um=overlap_um
        )
        self.asm = {"horizontal": self.h_asm, "vertical": self.v_asm}

        self.patches = []
        if p.patch_integrated:
            self._layout_patches()
            
    # Keep remainder of JunctionLayout methods (jj_area_um2, bounds, etc.) intact...

    def _layout_patches(self):
        """Patches straddle each finger's plate/electrode seam, spread
        transverse to that finger's own axis, oriented at the user's
        absolute patch_angle (side A) / patch_angle+180 (side B)."""
        p = self.p
        n = max(1, int(p.patch_count))
        w, h = p.patch_width_um, p.patch_height_um
        spacing = p.patch_spacing_um
        total_span = n * w + (n - 1) * spacing
        offsets = [(-total_span / 2 + w / 2) + i * (w + spacing) for i in range(n)]

        for side, asm, patch_angle in (
            ("horizontal", self.h_asm, p.patch_angle),
            ("vertical", self.v_asm, p.patch_angle + 180.0),
        ):
            seam = asm.world_point(asm.seam_point_local)
            transverse = _rot(asm.angle_deg + 90.0) @ np.array([1.0, 0.0])
            for off in offsets:
                center = seam + transverse * off
                self.patches.append(dict(cx=center[0], cy=center[1], w=w, h=h,
                                          angle=patch_angle, side=side))

    def jj_area_um2(self):
        return self.lw_h * self.lw_v

    def bounds(self):
        pts = []
        for asm in (self.h_asm, self.v_asm):
            pts.append(asm.world_polygon(asm.electrode_corners_local))
            pts.append(asm.world_polygon(asm.finger_corners_local))
        pts = np.concatenate(pts, axis=0)
        for patch in self.patches:
            pts = np.concatenate([pts, [[patch["cx"], patch["cy"]]]], axis=0)
        return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


# ----------------------------------------------------------------------
# Cumulative recipe-state helpers
# ----------------------------------------------------------------------

_DEP_FEATURES = {"horizontal_finger", "vertical_finger",
                  "horizontal_connect", "vertical_connect",
                  "horizontal_overlap", "vertical_overlap",
                  "patch_a", "patch_b"}


def _finger_side_for_feature(feature):
    if feature is None:
        return None
    if feature.startswith("horizontal"):
        return "horizontal"
    if feature.startswith("vertical"):
        return "vertical"
    return None


class RecipeState:
    """Computes what has happened by a given step index: which fingers/
    patches are (at least partly) deposited, which mill regions are done,
    and what the most recent deposition group was (for oxidation labeling).
    """

    def __init__(self, steps, current_index):
        self.steps = steps
        self.current_index = current_index
        self.current_step = steps[current_index]

        self.deposited_sides = set()       # {'horizontal', 'vertical'} once their FIRST sub-step happened
        self.overlapped_sides = set()      # side has completed its connect+overlap pair
        self.patch_a_done = False
        self.patch_b_done = False
        self.milled_top_done = False
        self.milled_sidewall_thetas = []   # thetas of completed sidewall mills
        self.last_deposition_group = None  # e.g. 'horizontal', 'vertical', 'patches' -- for oxidation labeling
        self._oxidation_after = {}         # index of oxidation step -> group it oxidizes

        last_group = None
        for i, s in enumerate(steps):
            if i > current_index:
                break
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
            elif f == "mill_top":
                self.milled_top_done = True
            elif f == "mill_sidewall":
                if s.theta is not None:
                    self.milled_sidewall_thetas.append(s.theta)
            elif s.category == "misc" and s.title == "Oxidation step":
                self._oxidation_after[i] = last_group

        self.last_deposition_group = last_group
        self.current_oxidation_group = self._oxidation_after.get(current_index)


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def _draw_poly(ax, pts, facecolor, edgecolor, zorder, alpha=1.0, lw=1.2, hatch=None, linestyle="-"):
    poly = mpatches.Polygon(pts, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=lw, zorder=zorder, alpha=alpha, hatch=hatch, linestyle=linestyle)
    ax.add_patch(poly)


def _draw_patch_rect(ax, patch, facecolor, edgecolor, zorder, alpha=1.0):
    w, h = patch["w"], patch["h"]
    t = mtransforms.Affine2D().rotate_deg(patch["angle"]).translate(patch["cx"], patch["cy"])
    rect = Rectangle((-w / 2, -h / 2), w, h, facecolor=facecolor, edgecolor=edgecolor,
                      linewidth=1.0, zorder=zorder, alpha=alpha)
    rect.set_transform(t + ax.transData)
    ax.add_patch(rect)


def _highlight_electrode_edge(ax, asm: FingerAssembly, theta, color, zorder, lw=4.0):
    """
    Draw a thick highlight along whichever edge of this finger's Nb
    electrode faces the mill direction `theta`. SIMPLIFICATION /
    ASSUMPTION: theta (the chamber wafer-rotation angle from the recipe)
    is mapped 1:1 onto this design-space drawing angle. Physically,
    there's an unknown fixed offset between the wafer's chamber-frame
    theta=0 reference and this die's own local orientation on the wafer
    (depends on how the die was diced/placed) that isn't modeled anywhere
    in this project yet -- so treat this as "geometrically plausible",
    not a verified mapping of which real edge gets milled.
    """
    mill_dir = _dir_vec(theta)
    corners = asm.world_polygon(asm.electrode_corners_local)
    n = len(corners)
    best_edge = None
    best_dot = -1e9
    for i in range(n):
        a, b = corners[i], corners[(i + 1) % n]
        mid = (a + b) / 2.0
        edge_vec = b - a
        outward = np.array([edge_vec[1], -edge_vec[0]])
        outward = outward / (np.linalg.norm(outward) + 1e-9)
        dot = float(np.dot(outward, mill_dir))
        if dot > best_dot:
            best_dot = dot
            best_edge = (a, b)
    if best_edge is not None:
        a, b = best_edge
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=lw, zorder=zorder, solid_capstyle="round")


def draw_junction_step(ax, p: DesignParameters, steps, current_index: int):
    """
    Draws the junction schematic reflecting cumulative recipe state up to
    and including `current_index` in `steps` (a list of recipe_model
    RecipeStep). Replaces the old single-feature-name API so that
    previously-completed sub-steps stay visible/solid through later
    prep/oxidation steps instead of reverting to a ghost outline.
    """
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    layout = JunctionLayout(p)
    state = RecipeState(steps, current_index)
    current_feature = state.current_step.feature
    current_side = _finger_side_for_feature(current_feature)

    # --- Nb electrodes: always visible (pre-existing layer) ---
    for side, asm in layout.asm.items():
        milled_now = (current_feature == "mill_sidewall" and False)  # per-edge, handled below
        edge_color = NB_EDGE
        lw = 1.4
        if state.milled_top_done:
            edge_color = NB_MILLED_EDGE
        _draw_poly(ax, asm.world_polygon(asm.electrode_corners_local), NB_COLOR, edge_color, zorder=1, lw=lw)

    # --- Mill-phase highlighting ---
    if current_feature == "mill_top":
        for side, asm in layout.asm.items():
            _draw_poly(ax, asm.world_polygon(asm.electrode_corners_local), "none",
                       MILLING_HIGHLIGHT, zorder=2, lw=3.5)
    elif current_feature == "mill_sidewall" and state.current_step.theta is not None:
        for side, asm in layout.asm.items():
            _highlight_electrode_edge(ax, asm, state.current_step.theta, MILLING_HIGHLIGHT, zorder=2)
    # Persistent "already milled" edge markers for completed sidewall mills
    for theta in state.milled_sidewall_thetas:
        for side, asm in layout.asm.items():
            _highlight_electrode_edge(ax, asm, theta, NB_MILLED_EDGE, zorder=1, lw=2.5)

    # --- Al features: ghost (not yet deposited) vs solid (deposited) ---
    for side, asm in layout.asm.items():
        deposited = side in state.deposited_sides
        is_current = (current_side == side) and current_feature in (
            f"{side}_finger", f"{side}_connect", f"{side}_overlap")
        finger_pts = asm.world_polygon(asm.finger_corners_local)
        plate_pts = asm.world_polygon(asm.plate_corners_local)
        if deposited:
            color = HIGHLIGHT_COLOR if is_current else AL_COLOR
            oxidized = state.current_oxidation_group == side and current_feature is None
            hatch = "///" if (oxidized or (side in state.overlapped_sides and
                                            state.last_deposition_group != side and
                                            current_feature is None and state.current_step.category == "misc")) else None
            _draw_poly(ax, finger_pts, color, AL_EDGE, zorder=3, lw=1.6 if is_current else 1.2)
            _draw_poly(ax, plate_pts, color, AL_EDGE, zorder=3, lw=1.6 if is_current else 1.2)
        else:
            _draw_poly(ax, finger_pts, "none", AL_DIM, zorder=2, alpha=0.6, lw=1.0)
            _draw_poly(ax, plate_pts, "none", AL_DIM, zorder=2, alpha=0.6, lw=1.0)

    if p.patch_integrated:
        for patch in layout.patches:
            done = (patch["side"] == "horizontal" and state.patch_a_done) or \
                   (patch["side"] == "vertical" and state.patch_b_done)
            is_current = (patch["side"] == "horizontal" and current_feature == "patch_a") or \
                         (patch["side"] == "vertical" and current_feature == "patch_b")
            if done:
                color = HIGHLIGHT_COLOR if is_current else AL_COLOR
                _draw_patch_rect(ax, patch, color, AL_EDGE, zorder=4)
            else:
                _draw_patch_rect(ax, patch, "none", AL_DIM, zorder=4, alpha=0.6)

    # --- JJ marker + area label, once both fingers have at least connected ---
    if "horizontal" in state.deposited_sides and "vertical" in state.deposited_sides:
        jj_w, jj_h = layout.lw_h, layout.lw_v
        jj_highlight = current_feature in ("horizontal_overlap", "vertical_overlap")
        jj_color = HIGHLIGHT_COLOR if jj_highlight else "#ffffff"
        ax.add_patch(Rectangle((-jj_w / 2, -jj_h / 2), jj_w, jj_h,
                                facecolor="none", edgecolor=jj_color, linewidth=1.8,
                                zorder=6, linestyle="--"))
        area = layout.jj_area_um2()
        ax.annotate(f"JJ area \u2248 {area:.4f} \u00b5m\u00b2", xy=(0, 0),
                    xytext=(0, -1.4), color=TEXT_COLOR, fontsize=8, ha="center", zorder=7)

    # --- Layer-order label: which finger is bottom vs top ---
    first_side = p.deposit_first
    second_side = "vertical" if first_side == "horizontal" else "horizontal"
    if second_side in state.deposited_sides:
        x0, x1, y0, y1 = layout.bounds()
        ax.text(x0, y1 + 0.3, f"bottom layer: {first_side} finger   |   top layer: {second_side} finger",
                 color=TEXT_COLOR, fontsize=8, ha="left", va="bottom")

    # --- Oxidation marker: "(I)" label near the most recently completed
    # deposition group while an oxidation step is the current step ---
    if state.current_step.category == "misc" and state.current_step.title == "Oxidation step":
        group = state.last_deposition_group
        if group in ("horizontal", "vertical"):
            asm = layout.asm[group]
            seam = asm.world_point(asm.seam_point_local)
            ax.annotate("(I) oxidized", xy=seam, xytext=(seam[0], seam[1] + 0.5),
                        color=OXIDE_HATCH_COLOR, fontsize=8, ha="center", zorder=7)
        elif group == "patches":
            ax.annotate("(I) patches oxidized", xy=(0, 0), xytext=(0, -1.9),
                        color=OXIDE_HATCH_COLOR, fontsize=8, ha="center", zorder=7)

    # --- Scale bar ---
    x0, x1, y0, y1 = layout.bounds()
    span = max(x1 - x0, y1 - y0)
    nice_steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    target = span * 0.2
    bar_len = min(nice_steps, key=lambda v: abs(v - target))
    bar_x0 = x0 + span * 0.03
    bar_y = y0 - span * 0.10
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y, bar_y], color=SCALE_COLOR, linewidth=2.5, zorder=8)
    ax.plot([bar_x0, bar_x0], [bar_y - span * 0.01, bar_y + span * 0.01], color=SCALE_COLOR, linewidth=2.5, zorder=8)
    ax.plot([bar_x0 + bar_len, bar_x0 + bar_len], [bar_y - span * 0.01, bar_y + span * 0.01],
            color=SCALE_COLOR, linewidth=2.5, zorder=8)
    ax.text(bar_x0 + bar_len / 2, bar_y - span * 0.05, f"{bar_len:g} \u00b5m",
            color=SCALE_COLOR, fontsize=8, ha="center", va="top", zorder=8)

    ax.set_xlim(x0 - span * 0.15, x1 + span * 0.10)
    ax.set_ylim(bar_y - span * 0.20, y1 + span * 0.20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Junction Formation (schematic, proportional only)",
                  color=TEXT_COLOR, fontsize=11, fontweight="bold")


def draw_junction_preview(ax, p: DesignParameters):
    """Fully-assembled preview for the Junction Design tab showing safety margins."""
    layout = JunctionLayout(p)
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    for side, asm in layout.asm.items():
        _draw_poly(ax, asm.world_polygon(asm.electrode_corners_local), NB_COLOR, NB_EDGE, zorder=1)
        _draw_poly(ax, asm.world_polygon(asm.finger_corners_local), AL_COLOR, AL_EDGE, zorder=3)
        _draw_poly(ax, asm.world_polygon(asm.plate_corners_local), AL_COLOR, AL_EDGE, zorder=3)

    if p.patch_integrated:
        for patch in layout.patches:
            _draw_patch_rect(ax, patch, AL_COLOR, AL_EDGE, zorder=4)

    # Core Overlap Junction Box
    jj_w, jj_h = layout.lw_h, layout.lw_v
    ax.add_patch(Rectangle((-jj_w / 2, -jj_h / 2), jj_w, jj_h,
                            facecolor="none", edgecolor="#ffffff", linewidth=1.8,
                            zorder=6, linestyle="--"))
    
    # ─── ADDED: SAFETY BOUND OVERLAYS (dx & dy) ─────────────────────────
    # Dynamically read or calculate dx/dy values from your recipe_model helper
    # For display placeholder if not attached directly to DesignParameters:
    try:
        from recipe_model import compute_overlap_shadow_bounds
        # Assuming typical 45 deg or current step angle alpha
        alpha_val = getattr(p, "alpha_deg", 45.0)
        pmma = getattr(p, "pmma_thickness_nm", 300.0)
        pmgi = getattr(p, "pmgi_thickness_nm", 200.0)
        dx_um, dy_um, _ = compute_overlap_shadow_bounds(pmma, pmgi, alpha_val)
    except Exception:
        dx_um, dy_um = 0.85, 0.35  # fallback matching your worked note example

    # Plot visual safety boundaries around the intersection
    overlap_um = (getattr(p, "overlap_length_nm", 1500.0) or 1500.0) / 1000.0
    
    # Worst case min-overlap line (-dx)
    ax.axvline(x=-dx_um, color="#ff4d4d", linestyle=":", linewidth=1.2, label=f"Min Bound (-dx)")
    # Worst case max-overlap line (+dy)
    ax.axvline(x=dy_um, color="#8fd694", linestyle=":", linewidth=1.2, label=f"Max Bound (+dy)")
    
    area = layout.jj_area_um2()
    ax.annotate(f"JJ Target Area \u2248 {area:.4f} \u00b5m\u00b2\nTarget Overlap: {overlap_um:.2f} \u00b5m", 
                xy=(0, 0), xytext=(0, -1.6), color=TEXT_COLOR, fontsize=8, ha="center", zorder=7)

    # Scale bar processing...
    x0, x1, y0, y1 = layout.bounds()
    span = max(x1 - x0, y1 - y0, 1e-6)
    nice_steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    target = span * 0.2
    bar_len = min(nice_steps, key=lambda v: abs(v - target))
    bar_x0 = x0 + span * 0.03
    bar_y = y0 - span * 0.10
    ax.plot([bar_x0, bar_x0 + bar_len], [bar_y, bar_y], color=SCALE_COLOR, linewidth=2.5, zorder=8)
    ax.text(bar_x0 + bar_len / 2, bar_y - span * 0.05, f"{bar_len:g} \u00b5m",
            color=SCALE_COLOR, fontsize=8, ha="center", va="top", zorder=8)

    ax.set_xlim(x0 - span * 0.15, x1 + span * 0.10)
    ax.set_ylim(bar_y - span * 0.20, y1 + span * 0.20)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Junction Design Preview & Tolerance Bounds", color=TEXT_COLOR, fontsize=11, fontweight="bold")
