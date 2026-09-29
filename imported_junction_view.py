"""
imported_junction_view.py

Renders an ImportedJunction (from junction_import.py) as a preview,
cropped tightly to the junction geometry -- with the electrode layer
clipped to a padded region around it and drawn faded/truncated ("torn
edge" style) so you can SEE whether/where it overlaps, without letting a
huge pad drag the whole view out of scale.

Angle arrows show a confident preferred/risky pair only when a finger's
electrode_status is actually known ("overlaps"); otherwise both
directions are shown neutrally, since geometry alone can't safely assert
a verdict for a finger that's clear of (or has unknown status vs) the
electrode -- see junction_import.explain_finger_angle_choice.

get_click_targets() returns hit-test regions so the embedding GUI can
wire a matplotlib button_press_event to a "why is this flagged" panel.
"""

import math
from typing import Optional
import shapely.geometry as sg
from matplotlib.patches import Polygon as MplPolygon

import parametric_junction_view as _jv
from parametric_junction_view import BG_COLOR, AL_COLOR, TEXT_COLOR, HIGHLIGHT_COLOR, GRID_COLOR, SCALE_COLOR
from junction_import import ImportedJunction, MeasurementReport, finger_display_label

OUTLINE_COLOR = "#ffffff"  # default/selection edge stroke for patch/plate/finger polygons below --
                              # near-white so it reads against the dark chamber fill in dark mode;
                              # flips dark for a light canvas (see apply_theme below)

PATCH_COLOR = "#8fd694"
PLATE_COLOR = "#f2c14e"
STUB_MARK_COLOR = "#6b93c9"
PREFERRED_COLOR = "#8fd694"
RISKY_COLOR = "#ffb454"       # Warning #1's risky color -- yellow/orange
NEUTRAL_COLOR = "#8ea3c2"     # used when electrode_status doesn't support a confident verdict
SHADOW_RISKY_COLOR = "#ff5c5c"  # Warning #2's risky color -- red. Same caution symbol as Warning
                                  # #1 (not a different icon) -- color alone carries the distinction,
                                  # since both now share one merged arrow set per finger axis.
ELECTRODE_COLOR = "#c98bff"
MATERIAL_COLOR = "#6b7280"
SUBSTRATE_COLOR = "#4a4d5e"

# Role-assignment override colors (Qt GUI's "click a shape, mark it as
# finger/patch/electrode/substrate" feature) -- purely cosmetic, applied
# on top of the normal kind-based coloring below when a shape's
# role_override is set.
ROLE_OVERRIDE_COLORS = {
    "finger": AL_COLOR,
    "patch": PATCH_COLOR,
    "plate": PLATE_COLOR,
    "electrode": ELECTRODE_COLOR,
    "substrate": SUBSTRATE_COLOR,
}


def apply_theme(mode: str) -> None:
    """Switches this module's theme-reactive color constants between
    "dark" and "light". Unlike chamber_plotting.py/parametric_junction_view.py, this module
    doesn't own its BG_COLOR/AL_COLOR/TEXT_COLOR/HIGHLIGHT_COLOR/
    GRID_COLOR/SCALE_COLOR values -- it copied them once at import time
    from parametric_junction_view (`from parametric_junction_view import ...`), which does NOT
    stay in sync when parametric_junction_view's globals are mutated later. So this
    re-triggers parametric_junction_view's own theme switch first, then re-copies
    those 6 names plus re-derives OUTLINE_COLOR to match.

    ROLE_OVERRIDE_COLORS["finger"] is likewise a value baked at dict-
    construction time from the old AL_COLOR, so it's refreshed here too
    -- otherwise a themed finger-role override would silently keep
    showing the previous theme's aluminum color.
    """
    _jv.apply_theme(mode)
    globals()["BG_COLOR"] = _jv.BG_COLOR
    globals()["AL_COLOR"] = _jv.AL_COLOR
    globals()["TEXT_COLOR"] = _jv.TEXT_COLOR
    globals()["HIGHLIGHT_COLOR"] = _jv.HIGHLIGHT_COLOR
    globals()["GRID_COLOR"] = _jv.GRID_COLOR
    globals()["SCALE_COLOR"] = _jv.SCALE_COLOR
    globals()["OUTLINE_COLOR"] = _jv.OUTLINE_COLOR
    ROLE_OVERRIDE_COLORS["finger"] = _jv.AL_COLOR


def _draw_poly(ax, points, facecolor, edgecolor, zorder=3, alpha=1.0, lw=1.0):
    patch = MplPolygon(points, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                        linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(patch)


_MAX_GRID_LINES_PER_AXIS = 200  # safety cap, see _draw_grid below


def _draw_grid(ax, x0, x1, y0, y1, step, artists_out=None):
    """Draws vertical lines every `step` from x0 to x1, and horizontal
    lines every `step` from y0 to y1.

    Repeated mouse-wheel zooming could previously drive the view to an
    extreme span and freeze or crash the app: the Qt GUI's own zoom
    handling (see main_gui.py's _zoom_step/_on_canvas_scroll) is
    now bounded to a sane [0.02, 20000] um span, but this function had no
    independent floor of its own -- every caller picks `step` from a
    small candidate list capped at 100 um (see refresh_grid_and_scale/
    draw_imported_junction_preview), so an extreme-enough span (reachable
    from a burst of many queued scroll-wheel events, each compounding a
    fixed zoom factor with no ceiling) could still ask this function for
    tens or hundreds of thousands of grid lines on a single redraw --
    creating that many Line2D artists is what actually froze/crashed the
    app. Defense in depth: this function now coarsens `step` on its own
    whenever the requested extent would need more than
    _MAX_GRID_LINES_PER_AXIS lines along either axis, so it can never
    draw an unbounded number of artists regardless of what a caller
    passes in -- independent of, and in addition to, the GUI-level fix
    that keeps the view from ever reaching such an extreme span in the
    first place."""
    span_x = x1 - x0
    span_y = y1 - y0
    if step > 0 and math.isfinite(step):
        worst_axis_span = max(span_x, span_y)
        if worst_axis_span > 0 and worst_axis_span / step > _MAX_GRID_LINES_PER_AXIS:
            step = worst_axis_span / _MAX_GRID_LINES_PER_AXIS
    if not (step > 0 and math.isfinite(step)):
        return
    x = math.floor(x0 / step) * step
    while x <= x1 + 1e-9:
        ln = ax.plot([x, x], [y0, y1], color=GRID_COLOR, linewidth=0.5, zorder=0, alpha=0.5)[0]
        if artists_out is not None:
            artists_out.append(ln)
        x += step
    y = math.floor(y0 / step) * step
    while y <= y1 + 1e-9:
        ln = ax.plot([x0, x1], [y, y], color=GRID_COLOR, linewidth=0.5, zorder=0, alpha=0.5)[0]
        if artists_out is not None:
            artists_out.append(ln)
        y += step


def refresh_grid_and_scale(ax):
    """Repositions/rescales the grid lines and scale bar to match the
    axes' CURRENT live xlim/ylim -- e.g. right after a pure scroll/pinch
    zoom (Qt GUI's _zoom_step), which only calls ax.set_xlim/set_ylim
    and redraws the canvas without ever re-invoking
    draw_imported_junction_preview.

    Fixes a bug where the grid lines and scale bar would stay fixed at
    whatever zoom level was active when they were last drawn, instead of
    tracking subsequent zoom in/out. Root cause: _draw_grid's Line2D
    segments and the scale bar's
    Line2D + Text are drawn ONCE, as fixed-length artists anchored to
    whatever crop extent (crop_x0..crop_y1 / full_x0/y0/full_span) was
    current at the LAST FULL redraw. A full redraw (draw_imported_
    junction_preview) always ends by calling ax.set_xlim/set_ylim to
    that same extent, so the grid/scale bar look correct immediately
    after any full redraw -- but ordinary mouse-wheel/pinch zooming
    calls ax.set_xlim/set_ylim directly, with no full redraw in between,
    so these fixed artists never track it: they stay put at their
    original drawn extent while the design geometry (real polygons,
    which have no "drawn extent" of their own -- only a transform)
    appears to zoom correctly around them.

    Fixed by calling this from the Axes' own xlim_changed callback (the
    same callback already used to rescale overlay-label fontsizes, e.g.
    _on_create_import_xlim_changed) so the grid/scale bar are removed
    and redrawn at the axes' new live view on every xlim change, without
    needing (or triggering) a full redraw.

    draw_imported_junction_preview tags the artists it draws for the
    grid + scale bar onto `ax._live_grid_scale_artists` so this function
    can find and replace them. If that view mode never draws a grid/
    scale bar at all (e.g. the Simulate tab's Post-Shadow/Simulated
    Design views), or the axes were cleared since without a fresh grid/
    scale-bar draw, the tracked artists are detached (ax.clear() sets
    artist.axes to None) and this bails out rather than resurrecting a
    grid into a view that never asked for one.

    Deliberately never calls ax.set_xlim/set_ylim itself -- doing so
    would re-fire this same xlim_changed callback and recurse forever.
    """
    old = getattr(ax, "_live_grid_scale_artists", None)
    if not old:
        return False
    if any(getattr(artist, "axes", None) is not ax for artist in old):
        return False
    crop_x0, crop_x1 = sorted(ax.get_xlim())
    crop_y0, crop_y1 = sorted(ax.get_ylim())
    if not all(math.isfinite(v) for v in (crop_x0, crop_x1, crop_y0, crop_y1)):
        return False
    full_span = min(crop_x1 - crop_x0, crop_y1 - crop_y0)
    if not (full_span > 1e-9):
        return False
    for artist in old:
        try:
            artist.remove()
        except Exception:
            pass
    new_artists = []
    _draw_grid(ax, crop_x0, crop_x1, crop_y0, crop_y1,
               step=min([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], key=lambda v: abs(v - full_span / 8)),
               artists_out=new_artists)
    bar_y = crop_y0 - full_span * 0.06
    bar_len = min([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], key=lambda v: abs(v - full_span * 0.2))
    new_artists.append(ax.plot([crop_x0, crop_x0 + bar_len], [bar_y, bar_y],
                               color=SCALE_COLOR, linewidth=2.5, zorder=8)[0])
    new_artists.append(ax.text(crop_x0 + bar_len / 2, bar_y - full_span * 0.02, f"{bar_len:g} µm",
                               color=SCALE_COLOR, fontsize=8, ha="center", va="top", zorder=8))
    ax._live_grid_scale_artists = new_artists
    return True


def get_click_targets(junction: ImportedJunction, span: float):
    """Per-finger warning-badge positions + explanation text, for hit-testing
    a button_press_event in the embedding GUI. Kept as a standalone function
    (not baked into the draw call) so the GUI can recompute it without
    redrawing, e.g. right after the user overrides warning_source.
    Registers up to TWO targets per finger: the merged finger-axis badge
    (Warning #1 + Warning #2 combined onto one arrow set) and the patch
    badge (when patches exist) -- matching the two-badge drawing. Both
    map to the same explanation text since that text already covers
    every applicable concern."""
    from junction_import import explain_finger_angle_choice
    targets = []
    for i, f in enumerate(junction.fingers):
        explanation = explain_finger_angle_choice(f)
        radius = max(span * 0.035, 0.15)

        mx, my = (f.stub_end[0] + f.patch_end[0]) / 2.0, (f.stub_end[1] + f.patch_end[1]) / 2.0
        fx, fy = f.patch_end[0] - f.stub_end[0], f.patch_end[1] - f.stub_end[1]
        flen = (fx ** 2 + fy ** 2) ** 0.5 or 1.0
        ux, uy = fx / flen, fy / flen
        perp_x, perp_y = -uy, ux
        off = max(f.width_um * 2.2, span * 0.022)
        targets.append(dict(
            finger_index=i, x=mx + perp_x * off, y=my + perp_y * off, radius=radius,
            explanation=explanation, status=f.warning_source,
        ))

        if f.patch_outward_end is not None:
            pmx = (f.patch_inward_end[0] + f.patch_outward_end[0]) / 2.0
            pmy = (f.patch_inward_end[1] + f.patch_outward_end[1]) / 2.0
            pfx = f.patch_outward_end[0] - f.patch_inward_end[0]
            pfy = f.patch_outward_end[1] - f.patch_inward_end[1]
            pflen = (pfx ** 2 + pfy ** 2) ** 0.5 or 1.0
            pux, puy = pfx / pflen, pfy / pflen
            pperp_x, pperp_y = -puy, pux
            poff = -max(f.width_um * 2.2, span * 0.022)  # negative -- other side of the axis,
                                                            # matching offset_sign=-1 in the drawing
            targets.append(dict(
                finger_index=i, x=pmx + pperp_x * poff, y=pmy + pperp_y * poff, radius=radius,
                explanation=explanation, status=f.warning_source,
            ))
    return targets



def _draw_merged_finger_badge(ax, f, width_um, span, warning1_enabled, warning2_enabled, offset_sign=1):
    """Exactly ONE arrow-pair for a finger's own axis (stub<->patch),
    merging Warning #1 (electrode crossing) and Warning #2 (self-
    shadowing) onto the SAME set instead of two separate, overlapping
    ones -- showing both warnings' own full arrow sets on the same axis
    was redundant clutter rather than useful detail.
    Each of the two physical directions gets its own independent
    verdict: risky per Warning #1 only (orange), risky per Warning #2
    only (red, same caution symbol as #1 -- color alone distinguishes
    them), risky per BOTH at once (a real dashed two-color line,
    alternating orange/red), or not risky (green, with a checkmark only
    if the OTHER direction on this axis IS flagged -- otherwise plain,
    matching how a fully-safe axis looked before)."""
    stub_theta = f.theta_toward_stub_deg
    patch_theta = f.theta_toward_patch_deg
    w1_risky_theta = stub_theta if (warning1_enabled and f.warning_source in ("finger", "both")) else None
    w2_risky_theta = patch_theta if (warning2_enabled and f.deposited_second
                                     and f.self_shadow_safe_theta_deg is not None) else None

    def verdict_for(theta):
        if not warning1_enabled and not warning2_enabled:
            return "neutral"
        is_w1 = w1_risky_theta is not None and abs(theta - w1_risky_theta) < 1e-6
        is_w2 = w2_risky_theta is not None and abs(theta - w2_risky_theta) < 1e-6
        if is_w1 and is_w2:
            return "dual"
        if is_w1:
            return "risky1"
        if is_w2:
            return "risky2"
        return "safe"

    patch_v, stub_v = verdict_for(patch_theta), verdict_for(stub_theta)
    has_any_risk = "dual" in (patch_v, stub_v) or "risky1" in (patch_v, stub_v) or "risky2" in (patch_v, stub_v)

    anchor_a, anchor_b = f.stub_end, f.patch_end
    mx, my = (anchor_a[0] + anchor_b[0]) / 2.0, (anchor_a[1] + anchor_b[1]) / 2.0
    fx, fy = anchor_b[0] - anchor_a[0], anchor_b[1] - anchor_a[1]
    flen = (fx ** 2 + fy ** 2) ** 0.5 or 1.0
    ux, uy = fx / flen, fy / flen
    perp_x, perp_y = -uy, ux
    off = max(width_um * 2.2, span * 0.022) * offset_sign
    ox, oy = mx + perp_x * off, my + perp_y * off
    half = flen * 0.22

    COLORS = {"risky1": RISKY_COLOR, "risky2": SHADOW_RISKY_COLOR, "safe": PREFERRED_COLOR, "neutral": NEUTRAL_COLOR}

    for sign, theta, v in ((1, patch_theta, patch_v), (-1, stub_theta, stub_v)):
        tip = (ox + ux * sign * half, oy + uy * sign * half)
        if v == "dual":
            ax.plot([ox, tip[0]], [oy, tip[1]], color=RISKY_COLOR, linewidth=1.3,
                    linestyle=(0, (4, 4)), zorder=9, solid_capstyle="butt")
            ax.plot([ox, tip[0]], [oy, tip[1]], color=SHADOW_RISKY_COLOR, linewidth=1.3,
                    linestyle=(4, (4, 4)), zorder=9, solid_capstyle="butt")
            ax.annotate("", xy=tip, xytext=(ox + ux * sign * half * 0.85, oy + uy * sign * half * 0.85),
                       arrowprops=dict(arrowstyle="-|>", color=SHADOW_RISKY_COLOR, lw=1.0, mutation_scale=6), zorder=9)
            color, mark = RISKY_COLOR, "⚠"
        else:
            color = COLORS[v]
            mark = "⚠" if v in ("risky1", "risky2") else ("✓" if (v == "safe" and has_any_risk) else "")
            ax.annotate("", xy=tip, xytext=(ox, oy),
                       arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=6), zorder=9)
        lx, ly = tip[0] + ux * sign * max(0.05, span * 0.008), tip[1] + uy * sign * max(0.05, span * 0.008)
        label = f"{mark} θ={theta:.0f}°" if mark else f"θ={theta:.0f}°"
        ax.text(lx, ly, label, color=color, fontsize=6, ha="center", va="center", zorder=9,
                fontweight="bold" if mark == "⚠" else "normal")

    if has_any_risk:
        badge_color = SHADOW_RISKY_COLOR if "risky2" in (patch_v, stub_v) and "risky1" not in (patch_v, stub_v) else RISKY_COLOR
        ax.text(ox, oy, "⚠", color=badge_color, fontsize=7, ha="center", va="center",
                zorder=11, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="#3a2f14", edgecolor=badge_color, linewidth=0.9))
    elif warning1_enabled or warning2_enabled:
        ax.text(ox, oy, "✓", color=PREFERRED_COLOR, fontsize=7, ha="center", va="center",
                zorder=11, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="#1e2e22", edgecolor=PREFERRED_COLOR, linewidth=0.9))


def _draw_warning_badge(ax, anchor_a, anchor_b, theta_out, theta_in, badge_kind, width_um, span, offset_sign,
                        safe_color=None, risky_color=None, symbol="⚠", offset_scale=1.0):
    safe_color = safe_color or PREFERRED_COLOR
    risky_color = risky_color or RISKY_COLOR
    mx, my = (anchor_a[0] + anchor_b[0]) / 2.0, (anchor_a[1] + anchor_b[1]) / 2.0
    fx, fy = anchor_b[0] - anchor_a[0], anchor_b[1] - anchor_a[1]
    flen = (fx ** 2 + fy ** 2) ** 0.5 or 1.0
    ux, uy = fx / flen, fy / flen
    perp_x, perp_y = -uy, ux
    off = max(width_um * 2.2, span * 0.022) * offset_sign * offset_scale
    ox, oy = mx + perp_x * off, my + perp_y * off
    half = flen * 0.22

    if badge_kind == "risky":
        pairs = ((1, theta_out, safe_color, "✓"), (-1, theta_in, risky_color, symbol))
    elif badge_kind == "safe":
        pairs = ((1, theta_out, safe_color, ""), (-1, theta_in, safe_color, ""))
    else:
        pairs = ((1, theta_out, NEUTRAL_COLOR, ""), (-1, theta_in, NEUTRAL_COLOR, ""))

    for sign, theta, color, mark in pairs:
        tip = (ox + ux * sign * half, oy + uy * sign * half)
        ax.annotate("", xy=tip, xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=6), zorder=9)
        lx, ly = tip[0] + ux * sign * max(0.05, span * 0.008), tip[1] + uy * sign * max(0.05, span * 0.008)
        label = f"{mark} θ={theta:.0f}°" if mark else f"θ={theta:.0f}°"
        ax.text(lx, ly, label, color=color, fontsize=6, ha="center", va="center", zorder=9,
                fontweight="bold" if mark == symbol else "normal")

    if badge_kind == "risky":
        ax.text(ox, oy, symbol, color=risky_color, fontsize=7, ha="center", va="center",
                zorder=11, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="#3a2f14", edgecolor=risky_color, linewidth=0.9))
    elif badge_kind == "safe":
        ax.text(ox, oy, "✓", color=safe_color, fontsize=7, ha="center", va="center",
                zorder=11, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="#1e2e22", edgecolor=safe_color, linewidth=0.9))


def _layer_class_matches(category: Optional[str], shape_category: str) -> bool:
    if category is None:
        return True
    if category == "__jj_all__":
        return shape_category in ("finger", "patch", "plate")
    return category == shape_category


def _build_plot_title(source_label: str) -> str:
    if source_label.startswith("Parametric"):
        return "Junction Creator Preview"
    elif source_label:
        return f"Imported Junction: {source_label}"
    return "Junction Preview"


def build_measurement_caption(report: MeasurementReport) -> str:
    lines = [f"Source: {report.source_label}"]
    if report.jj_crossing_area_um2 is not None:
        lines.append(f"JJ crossing area (tunnel barrier) ≈ {report.jj_crossing_area_um2:.5f} µm²")
    if report.total_exposed_area_um2 is not None:
        lines.append(f"Total write area (union, no double-count) ≈ {report.total_exposed_area_um2:.4f} µm²")
    for i, (L, W) in enumerate(zip(report.finger_lengths_um, report.finger_widths_nm)):
        src = report.finger_warning_source[i]
        status_txt = {"finger": "finger crosses electrode", "patches": "patches cross electrode",
                      "both": "finger AND patches cross", "none": "no electrode boundary nearby",
                      "unknown": "electrode unknown"}[src]
        suffix = " -- click ⚠ badge for why" if src in ("finger", "patches", "both") else ""
        lines.append(f"Finger[{i}]: L={L:.3f} µm, W={W:.1f} nm ({status_txt}){suffix}")
    for pg in report.patch_groups:
        spacings = ", ".join(f"{s:.3f}" for s in pg.edge_to_edge_spacings_um)
        lines.append(f"Finger[{pg.finger_index}] patches: n={pg.patch_count}, "
                     f"edge-to-edge spacing(s)=[{spacings}] µm")
        if pg.electrode_overlap_depth_um is not None:
            lines.append(f"  → outermost patch overlap depth into electrode ≈ "
                         f"{pg.electrode_overlap_depth_um:.3f} µm")
    return "\n".join(lines)


def build_detailed_measurement_caption(report: MeasurementReport) -> str:
    lines = [f"Source: {report.source_label}"]
    if report.jj_crossing_area_um2 is not None:
        lines.append(f"JJ crossing area (tunnel barrier) ≈ {report.jj_crossing_area_um2:.5f} µm²")
    if report.total_exposed_area_um2 is not None:
        lines.append(f"Total write area (union, no double-count) ≈ {report.total_exposed_area_um2:.4f} µm²")

    for i, (L, W) in enumerate(zip(report.finger_lengths_um, report.finger_widths_nm)):
        label = finger_display_label(i)
        lines.append(f"Finger {label}: length={L:.3f} µm, linewidth={W:.1f} nm")
        overlap_depth = report.finger_electrode_overlap_depth_um[i] if i < len(report.finger_electrode_overlap_depth_um) else None
        if overlap_depth is not None:
            lines.append(f"  → finger itself overlaps electrode by ≈ {overlap_depth:.3f} µm")
        tip_shortest = report.finger_tip_shortest_overhang_um[i] if i < len(report.finger_tip_shortest_overhang_um) else None
        if tip_shortest is not None:
            lines.append(f"  → shortest tip corner overhang into electrode ≈ {tip_shortest:.3f} µm")

    for pg in report.patch_groups:
        label = finger_display_label(pg.finger_index)
        lines.append(f"Finger {label} patches (n={pg.patch_count}):")
        for k in range(pg.patch_count):
            w_nm = pg.individual_widths_um[k] * 1000.0
            parts = [f"linewidth={w_nm:.1f} nm"]
            if k < len(pg.individual_near_overhang_um):
                parts.append(f"near overhang={pg.individual_near_overhang_um[k]:.3f} µm")
            if k < len(pg.individual_far_overhang_um):
                parts.append(f"far overhang={pg.individual_far_overhang_um[k]:.3f} µm")
            if k < len(pg.individual_overlap_depths_um) and pg.individual_overlap_depths_um[k] is not None:
                parts.append(f"electrode overlap={pg.individual_overlap_depths_um[k]:.3f} µm")
            if k < len(pg.individual_overlap_area_um2):
                parts.append(f"contact area={pg.individual_overlap_area_um2[k]:.4f} µm²")
            lines.append(f"  patch[{k}]: " + ", ".join(parts))
        if pg.edge_to_edge_spacings_um:
            spacings = ", ".join(f"{s:.3f}" for s in pg.edge_to_edge_spacings_um)
            lines.append(f"  edge-to-edge spacing(s)=[{spacings}] µm")
        if pg.individual_far_overhang_um:
            smallest = min(pg.individual_far_overhang_um + pg.individual_near_overhang_um)
            largest = max(pg.individual_far_overhang_um + pg.individual_near_overhang_um)
            lines.append(f"  → smallest overhang ≈ {smallest:.3f} µm, largest overhang ≈ {largest:.3f} µm")
        if pg.total_overlap_area_um2 is not None:
            lines.append(f"  → total patch-to-finger/plate contact area ≈ {pg.total_overlap_area_um2:.4f} µm²")
        if pg.finger_tip_past_last_patch_um is not None:
            left, right, shortest = pg.finger_tip_past_last_patch_um
            lines.append(f"  → finger tip extends past last patch: left={left:.3f} µm, "
                         f"right={right:.3f} µm (shortest side ≈ {shortest:.3f} µm)")
    return "\n".join(lines)


def draw_imported_junction_preview(ax, junction: ImportedJunction, report: MeasurementReport = None,
                                    electrode_view: str = "clip", zoom_half_width_um: Optional[float] = None,
                                    center_override: Optional[tuple] = None,
                                    selected_stable_id: Optional[str] = None,
                                    layer_highlight: Optional[str] = None,
                                    shape_class_fn=None,
                                    show_caption: bool = True,
                                    warning1_enabled: bool = True,
                                    warning2_enabled: bool = False,
                                    show_badges: bool = True):
    ax.clear()
    ax.set_facecolor(BG_COLOR)

    x0, y0, x1, y1 = junction.bounds()
    span = max(x1 - x0, y1 - y0, 1e-6)
    jcx, jcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if center_override is not None:
        jcx, jcy = center_override

    # Fills the available canvas area instead of always cropping to a
    # square, similar to how layout-viewer tools like KLayout use the
    # full widget: this used to crop a SQUARE region
    # (same half-extent on both axes) and hand it to set_aspect("equal"),
    # which then pillarboxes/letterboxes that square into whatever
    # rectangle the widget actually is -- dead, unused space on the two
    # long sides for any non-square canvas (which New Tab 1's landscape
    # preview always is). Fix: widen the crop's SHORTER axis (in data
    # space) so the crop rectangle's aspect ratio matches the actual
    # widget's aspect ratio -- set_aspect("equal") then has no leftover
    # space to pad with, since the data range already fills the widget.
    # The tighter axis keeps its original (zoom_half_width_um, or the
    # default 1.2x-margin) extent, so the junction still fits exactly as
    # before; only the OTHER axis grows to use the space that used to be
    # blank. get_position() x figure size (inches) needs no renderer/
    # live draw, unlike get_window_extent(), so this works on the very
    # first draw too, not just after a resize event.
    try:
        fig = ax.figure
        # original=True is load-bearing, not decorative: set_aspect("equal")
        # below (default adjustable="box") makes matplotlib PERMANENTLY
        # shrink the axes' rendered box to force the box's own aspect to
        # match whatever data limits were set, the very next time this
        # Axes is drawn -- ax.get_position() (without original=True) then
        # returns that already-shrunk box on every SUBSEQUENT call, not
        # the widget's true available rectangle, which silently re-derives
        # a too-small aspect_ratio here and reintroduces the exact
        # pillarboxing this fix exists to remove. original=True reads the
        # stable layout position (set once when the Axes was created /
        # last explicitly repositioned) instead, immune to that feedback
        # loop: without it, aspect_ratio converges back toward 1.0
        # (square) after the very first draw, no matter how wide the
        # actual widget is.
        #
        # The view must also fill the full height, not just the full
        # width, so the preview doesn't leave blank space above and
        # below. fig.get_size_inches() is only as fresh as the LAST time something
        # actually called fig.set_size_inches() -- normally the Qt
        # canvas's own resizeEvent does that, but this preview is
        # explicitly re-drawn on tab-switch/nav-change (New Tab 1's own
        # refresh-on-show fix), which can run before that resize event
        # has actually landed, leaving fig_w_in/fig_h_in stale relative
        # to the canvas widget's REAL current on-screen size -- a wrong
        # (too-square, or otherwise off) aspect_ratio then bakes a
        # letterboxed/pillarboxed crop in on that draw. The canvas
        # widget's own .width()/.height() (plain QWidget properties, no
        # draw-cycle or resize-event dependency at all) are ALWAYS the
        # true current pixel size, so prefer those -- divided by dpi to
        # match fig_w_in/fig_h_in's units -- whenever a live canvas is
        # actually attached (a throwaway scratch Figure used for image
        # export, with no canvas, still falls back to get_size_inches(),
        # unchanged from before).
        canvas = getattr(fig, "canvas", None)
        cw = getattr(canvas, "width", None)
        ch = getattr(canvas, "height", None)
        if canvas is not None and callable(cw) and callable(ch) and cw() > 0 and ch() > 0:
            fig_w_in, fig_h_in = cw() / fig.dpi, ch() / fig.dpi
        else:
            fig_w_in, fig_h_in = fig.get_size_inches()
        bbox = ax.get_position(original=True)
        ax_w_in, ax_h_in = bbox.width * fig_w_in, bbox.height * fig_h_in
        aspect_ratio = (ax_w_in / ax_h_in) if ax_h_in > 1e-6 else 1.0
    except Exception:
        aspect_ratio = 1.0
    if not (aspect_ratio > 0):
        aspect_ratio = 1.0

    def _widen_to_aspect(half):
        # Returns (half_x, half_y) with half_x/half_y == aspect_ratio and
        # min(half_x, half_y) == half (the original tight-fit extent).
        if aspect_ratio >= 1.0:
            return half * aspect_ratio, half
        return half, half / aspect_ratio

    if zoom_half_width_um is not None:
        half_x, half_y = _widen_to_aspect(zoom_half_width_um)
        crop_x0, crop_y0 = jcx - half_x, jcy - half_y
        crop_x1, crop_y1 = jcx + half_x, jcy + half_y
        full_x0, full_y0, full_x1, full_y1 = crop_x0, crop_y0, crop_x1, crop_y1
        full_span = zoom_half_width_um * 2.0
    else:
        default_half = max((span / 2.0) * 1.2, 3.0)
        half_x, half_y = _widen_to_aspect(default_half)
        crop_x0, crop_y0 = jcx - half_x, jcy - half_y
        crop_x1, crop_y1 = jcx + half_x, jcy + half_y
        full_x0, full_y0, full_x1, full_y1 = crop_x0, crop_y0, crop_x1, crop_y1
        full_span = default_half * 2.0
    crop = sg.box(crop_x0, crop_y0, crop_x1, crop_y1)

    # Fixes a bug where zooming in on part of the junction and then
    # zooming back out would leave the background missing everywhere
    # except the region that had been in view while zoomed in. Root
    # cause: this used to clip electrode_material/electrode_polygons to `crop` -- which,
    # via main_gui.py's own _create_import_view_kwargs (added to fix
    # an earlier bug, the drag-shape feedback loop), is exactly whatever
    # the CURRENT on-screen view happens to be, so that a real redraw
    # keeps showing whatever the user was already looking at instead of
    # re-fitting to the whole design every time. That's the right thing to
    # preserve for the CAMERA (crop_x0..crop_y1, still used below for
    # ax.set_xlim/ylim), but using the SAME small transient window to
    # decide which electrode geometry even gets DRAWN means a real redraw
    # that happens to fire while zoomed in (e.g. a ruler click, which
    # calls _update_preview()) permanently bakes that moment's small crop
    # into the drawn geometry itself -- not just the displayed view. Zoom
    # back out afterward and matplotlib faithfully shows more of the same
    # Axes, but nothing was ever actually drawn outside that one small
    # crop, so everything past it reads as background having vanished.
    # Fixed by clipping the electrode layers to a box that's anchored to
    # the DESIGN's own stable extent (jcx/jcy/span -- never the transient
    # view), padded generously so ordinary zooming/panning around the
    # design never runs past its edge, and additionally unioned with
    # whatever the current view crop is (so a deliberate zoom-out wider
    # than that padded box is still never itself clipped). This is a
    # boundary electrode geometry gets DRAWN within, independent of
    # whatever redraw happened to be in flight when the view was last
    # zoomed -- the camera (crop/xlim/ylim) is untouched.
    ELECTRODE_CLIP_PAD = 5.0
    design_half_x, design_half_y = _widen_to_aspect(max((span / 2.0) * 1.2, 3.0))
    electrode_crop = sg.box(
        min(jcx - design_half_x * ELECTRODE_CLIP_PAD, crop_x0),
        min(jcy - design_half_y * ELECTRODE_CLIP_PAD, crop_y0),
        max(jcx + design_half_x * ELECTRODE_CLIP_PAD, crop_x1),
        max(jcy + design_half_y * ELECTRODE_CLIP_PAD, crop_y1),
    )

    if electrode_view == "clip" and junction.electrode_material:
        for pts in junction.electrode_material:
            if len(pts) < 3:
                continue
            try:
                clipped = sg.Polygon(pts).intersection(electrode_crop)
            except Exception:
                continue
            if clipped.is_empty:
                continue
            geoms = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
            for g in geoms:
                if g.geom_type != "Polygon" or g.area < 1e-9:
                    continue
                _draw_poly(ax, list(g.exterior.coords), MATERIAL_COLOR, "none", zorder=1, alpha=0.35)

    if electrode_view == "clip" and junction.electrode_polygons:
        for pts in junction.electrode_polygons:
            if len(pts) < 3:
                continue
            try:
                # Fixes a bug where a parametrically-created design's
                # electrodes rendered in the plain background material
                # color (grey) instead of the same purple used for an
                # imported design's electrodes. Root cause: this used to subtract
                # material_union from every electrode polygon before
                # drawing it -- correct for a REAL GDS import, where
                # electrode_polygons are HOLES (absent-space) that are
                # genuinely disjoint from electrode_material's drawn
                # metal, so the subtraction was always a no-op there.
                # But for a parametric (Created) design, from_design_
                # parameters deliberately sets electrode_material equal
                # to electrode_polygons (there's no separate real
                # background-material layer in that model -- see its own
                # comment) -- so here the "subtraction" was poly minus
                # ITSELF, i.e. always the empty polygon. Every Created
                # design's electrode therefore silently rendered as
                # nothing at all in this purple layer, leaving only the
                # grey MATERIAL_COLOR layer drawn behind it visible,
                # matching the reported symptom. Simply not subtracting
                # fixes both cases at once: a no-op for real imports
                # (unchanged appearance), and restores the purple
                # electrode layer for Created designs so it renders with
                # the same visual recipe as an imported design's
                # electrode.
                poly = sg.Polygon(pts)
                clipped = poly.intersection(electrode_crop)  # see the electrode_crop fix
                                                              # described above -- the same
                                                              # fix applies here, not just electrode_material
            except Exception:
                continue
            if clipped.is_empty:
                continue
            geoms = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
            for g in geoms:
                if g.geom_type != "Polygon" or g.area < 1e-9:
                    continue
                _draw_poly(ax, list(g.exterior.coords), ELECTRODE_COLOR, "none", zorder=2,
                          alpha=0.35 if _layer_class_matches(layer_highlight, "electrode") else 0.08)
                if layer_highlight == "electrode":
                    _draw_poly(ax, list(g.exterior.coords), "none", OUTLINE_COLOR, zorder=6, lw=2.0)

    for s in junction.other_shapes:
        color = PLATE_COLOR if s.kind == "plate" else PATCH_COLOR
        if s.role_override:
            color = ROLE_OVERRIDE_COLORS.get(s.role_override, color)
        s_class = shape_class_fn(s) if shape_class_fn else (s.role_override or s.kind)
        matches = _layer_class_matches(layer_highlight, s_class)
        _draw_poly(ax, s.points, color, OUTLINE_COLOR, zorder=3, alpha=0.9 if matches else 0.15)
        if (s.stable_id and s.stable_id == selected_stable_id) or (layer_highlight is not None and matches):
            _draw_poly(ax, s.points, "none", OUTLINE_COLOR, zorder=6, lw=2.6)

    for i, f in enumerate(junction.fingers):
        color = ROLE_OVERRIDE_COLORS.get(f.role_override, AL_COLOR) if f.role_override else AL_COLOR
        f_class = shape_class_fn(f) if shape_class_fn else (f.role_override or "finger")
        matches = _layer_class_matches(layer_highlight, f_class)
        _draw_poly(ax, f.points, color, OUTLINE_COLOR, zorder=4, alpha=1.0 if matches else 0.15)
        if (f.stable_id and f.stable_id == selected_stable_id) or (layer_highlight is not None and matches):
            _draw_poly(ax, f.points, "none", OUTLINE_COLOR, zorder=6, lw=2.6)
        label_x = f.stub_end[0] * 0.35 + f.patch_end[0] * 0.65
        label_y = f.stub_end[1] * 0.35 + f.patch_end[1] * 0.65
        ax.text(label_x, label_y, finger_display_label(i), color="#111214", fontsize=9, fontweight="bold",
               ha="center", va="center", zorder=7,
               bbox=dict(boxstyle="circle,pad=0.25", facecolor="#ffffff", edgecolor="#111214", linewidth=1.0))

    # ax._live_grid_scale_artists lets refresh_grid_and_scale (see its own
    # docstring -- it keeps the grid/scale bar from staying stuck under
    # pure scroll-zoom) find and replace exactly these
    # artists later, without touching anything else on the axes.
    _live_grid_scale_artists = []
    _draw_grid(ax, crop_x0, crop_x1, crop_y0, crop_y1,
               step=min([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100], key=lambda v: abs(v - full_span / 8)),
               artists_out=_live_grid_scale_artists)

    for i, f in enumerate(junction.fingers):
        if not show_badges:
            continue
        patches_cross = f.warning_source in ("patches", "both")
        has_patches = f.patch_outward_end is not None

        _draw_merged_finger_badge(ax, f, f.width_um, span, warning1_enabled, warning2_enabled, offset_sign=1)

        if has_patches:
            p_theta_out, p_theta_in = f.patch_theta_outward_deg, f.patch_theta_inward_deg
            p_badge_kind = "risky" if patches_cross else "safe"
            if not warning1_enabled:
                p_badge_kind = "unknown"
            _draw_warning_badge(ax, f.patch_inward_end, f.patch_outward_end, p_theta_out, p_theta_in,
                                p_badge_kind, f.width_um, span, offset_sign=-1)

        ax.plot(*f.stub_end, marker="o", markersize=3.5, color=STUB_MARK_COLOR, zorder=8)

    if report is not None and show_caption:
        caption = build_measurement_caption(report)
        ax.text(0.02, 0.02, caption, transform=ax.transAxes, color=TEXT_COLOR, fontsize=7.5,
                ha="left", va="bottom", zorder=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2c38", edgecolor="#4a4d5e", alpha=0.92))

    bar_y = full_y0 - full_span * 0.06
    bar_len = min([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], key=lambda v: abs(v - full_span * 0.2))
    _live_grid_scale_artists.append(
        ax.plot([full_x0, full_x0 + bar_len], [bar_y, bar_y], color=SCALE_COLOR, linewidth=2.5, zorder=8)[0])
    _live_grid_scale_artists.append(
        ax.text(full_x0 + bar_len / 2, bar_y - full_span * 0.02, f"{bar_len:g} µm", color=SCALE_COLOR,
                fontsize=8, ha="center", va="top", zorder=8))
    ax._live_grid_scale_artists = _live_grid_scale_artists

    ax.set_xlim(crop_x0, crop_x1)
    ax.set_ylim(crop_y0, crop_y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(_build_plot_title(junction.source_label), color=TEXT_COLOR, fontsize=10, fontweight="bold")
