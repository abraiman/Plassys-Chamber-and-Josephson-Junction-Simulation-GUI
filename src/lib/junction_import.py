"""
junction_import.py

Turns a real design -- either a .gds file or the actual KLayout Python
script that generated one -- into a measured, analyzable junction
geometry, instead of the hand-entered parametric model in recipe_generator.py.

Two import paths, one shared downstream representation (ImportedJunction):
  - import_gds_cell(...)        real .gds file, parsed with gdstk
  - run_klayout_script(...)     the actual .py generator script, executed
                                 against a gdstk-backed pya shim so it runs
                                 UNMODIFIED rather than being pattern-matched

Core measurement logic (validated against real fabrication failure data,
see notes on `classify_fingers`): which end of a finger is the "electrode
end" is determined from the finger's OWN construction asymmetry -- the
stub (short, bare) end vs. the patch/adhesion-plate (long, reinforced)
end -- NOT from boolean-overlapping the finger against the electrode
layer's geometry, which turned out to give the WRONG answer on designs
where the electrode layer blankets almost the entire chip, so "which end
overlaps more" is a noisy signal; the construction asymmetry is not.
"""

import io
import os
import sys
import math
import hashlib
import contextlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gdstk
import shapely.geometry as sg
import shapely.ops as so
import shapely.affinity as affinity

import klayout_pya_shim

from electrode_convention import ElectrodeConvention, DEFAULT_ELECTRODE_CONVENTION  # noqa: F401 -- re-exported
                                                                                      # for backward compatibility;
                                                                                      # extracted to its own module
                                                                                      # so recipe_generator.py can also
                                                                                      # import it without a
                                                                                      # circular import.


# Overlap-fraction bounds shared across the electrode-crossing classifier
# (_assign_warning_sources) AND the mill-contact check in
# build_simulation_result. Below CROSS_LO: fully clear of the electrode,
# never touches it. Between CROSS_LO and CROSS_HI: straddles a boundary
# ("crosses", the shadowing-relevant case -- direction matters). Above
# CROSS_HI: fully buried IN electrode material -- doesn't straddle a
# boundary (no shadowing direction to worry about) but DOES make direct
# electrode contact, which matters for a completely different question:
# whether this shape needs a mill step before its deposition. Kept as one
# shared pair of constants specifically so those two questions ("does
# this shape cross a boundary" vs "does this shape touch the electrode at
# all") can never silently drift to different thresholds.
CROSS_LO, CROSS_HI = 0.02, 0.98

# A very slight deviation between the as-drawn and post-shadow crossing
# area is expected from floating-point/polygon-intersection noise in the
# geometry pipeline and shouldn't, on its own, be classified as a real
# "area reduced" case -- many crossings show a negligible deviation even
# when shadowing has no real effect on the junction area. A JJ crossing
# whose real (post-shadow OR as-drawn) area is at or above this fraction
# of its nominal (as-designed, width x width) area counts as a clean,
# full crossing -- the shortfall below 100% is treated as pipeline noise,
# not a genuine partial overlap. Shared by two call sites that both need
# this exact same judgment call and must never silently drift apart: this
# module's own compute_measurements (the as-drawn check, pre-existing) and
# main_gui.py's _compute_postshadow_report (the post-shadow "AREA
# REDUCED" check -- see that function's own comment for why a second,
# geometry-based "did the fingers actually still fully overlap" check is
# required alongside this ratio, not just the ratio alone).
JJ_CROSSING_FULL_OVERLAP_RATIO = 0.995


# ============================================================================
# Shared geometry helpers
# ============================================================================

def _polygon_axis_and_ends(points: List[Tuple[float, float]]):
    """For a 4-point rectangle (possibly rotated), return:
      axis_angle_deg: direction of the LONG edge, mod 360 (one of two
                       equivalent directions -- disambiguated later)
      end_a, end_b:   midpoints of the two SHORT edges (the true tip
                       and crossing ends of a finger)
      width:          the true perpendicular width (short-edge length) --
                       NOT a bounding-box estimate, which is wrong for any
                       rectangle rotated off-axis (e.g. the 45-degree
                       Manhattan fingers everywhere in these designs: their
                       axis-aligned bbox is roughly SQUARE regardless of
                       the finger's real width, since bbox side scales with
                       (length+width)/sqrt(2) at 45 degrees).
    Falls back gracefully for >4-point polygons by using the min-area
    bounding rectangle's long axis instead.

    Fixed during a measurement-precision audit prompted by imported
    designs consistently measuring a few decimal places short of their
    known KLayout ruler values: `width` used to be
    `min(lens[e0], lens[e1])` -- the SMALLER of the two width-direction
    edges. For a mathematically perfect rectangle the two are identical
    and it makes no difference, but a REAL imported GDS shape essentially
    never is: GDS coordinates are quantized to the file's own database
    unit, boolean/merge/fracturing tools used somewhere upstream of the
    exported file can leave the two nominally-equal edges a hair apart,
    and any of that tips the two width edges slightly out of exact
    agreement. Taking the MINIMUM of the two turns that ordinary,
    symmetric quantization noise into a ONE-DIRECTIONAL, systematic
    UNDER-measurement (never over) -- exactly the direction of the
    observed symptom (this app reporting LESS than KLayout's own ruler
    measurement of the identical shape, never more). The AVERAGE of the
    two edges is the unbiased estimate of the shape's real design width
    (matches min/max exactly for a perfect rectangle, so this changes
    nothing for the common case -- only for a genuinely imperfect real
    shape, where it removes the built-in negative bias)."""
    pts = list(points)
    n = len(pts)
    if n == 4:
        lens = [((pts[(j + 1) % 4][0] - pts[j][0]) ** 2 + (pts[(j + 1) % 4][1] - pts[j][1]) ** 2) ** 0.5
                for j in range(4)]
        j_long, j_short = _long_short_edge_indices(lens)
        dx = pts[(j_long + 1) % 4][0] - pts[j_long][0]
        dy = pts[(j_long + 1) % 4][1] - pts[j_long][1]
        axis_angle = math.degrees(math.atan2(dy, dx)) % 360
        e0, e1 = j_short, j_short + 2
        end_a = ((pts[e0][0] + pts[(e0 + 1) % 4][0]) / 2, (pts[e0][1] + pts[(e0 + 1) % 4][1]) / 2)
        end_b = ((pts[e1][0] + pts[(e1 + 1) % 4][0]) / 2, (pts[e1][1] + pts[(e1 + 1) % 4][1]) / 2)
        width = (lens[e0] + lens[e1]) / 2.0
        return axis_angle, end_a, end_b, width

    # Fallback for non-rectangular fingers: shapely's minimum rotated
    # rectangle gives a robust long axis even for slightly irregular shapes.
    shp = sg.Polygon(pts)
    mrr = shp.minimum_rotated_rectangle
    mrr_pts = list(mrr.exterior.coords)[:4]
    lens = [((mrr_pts[(j + 1) % 4][0] - mrr_pts[j][0]) ** 2 + (mrr_pts[(j + 1) % 4][1] - mrr_pts[j][1]) ** 2) ** 0.5
            for j in range(4)]
    j_long, j_short = _long_short_edge_indices(lens)
    dx = mrr_pts[(j_long + 1) % 4][0] - mrr_pts[j_long][0]
    dy = mrr_pts[(j_long + 1) % 4][1] - mrr_pts[j_long][1]
    axis_angle = math.degrees(math.atan2(dy, dx)) % 360
    e0, e1 = j_short, j_short + 2
    end_a = ((mrr_pts[e0][0] + mrr_pts[(e0 + 1) % 4][0]) / 2, (mrr_pts[e0][1] + mrr_pts[(e0 + 1) % 4][1]) / 2)
    end_b = ((mrr_pts[e1][0] + mrr_pts[(e1 + 1) % 4][0]) / 2, (mrr_pts[e1][1] + mrr_pts[(e1 + 1) % 4][1]) / 2)
    # Same avg-not-min fix as the 4-point branch above, for consistency
    # (mrr's own two width edges are already equal to floating-point
    # precision by construction, so this makes no measurable difference
    # here -- kept identical purely so both branches agree on convention).
    width = (lens[e0] + lens[e1]) / 2.0
    return axis_angle, end_a, end_b, width


def _long_short_edge_indices(lens: List[float]) -> Tuple[int, int]:
    """Which of a 4-edge rectangle's two opposite-edge PAIRS (0&2, or
    1&3) is the long one, and a stable representative index within each
    pair -- (j_long, j_short), always 0-or-1 and 1-or-0 respectively.

    Fixed after observing the "Margin before finger's end" edit drift
    further away with every identical, repeated application, until the
    patch was no longer even touching the design: a proper rectangle's
    two edges within the SAME opposite pair are equal by construction,
    but only up to floating-point precision once the polygon has been
    REBUILT by an edit (apply_patch_extension_edit et al. compute each
    of the 4 corners via a different arithmetic expression, so two
    "identical" edge lengths differ by a sub-ULP rounding residual). The
    previous code picked the single overall longest edge via
    lens.index(max(lens)) across all 4 -- for a near-tie between the two
    long edges, that residual alone decided which one "won," flipping
    axis_angle by 180 deg between successive calls on the very same
    physical shape. That flip silently swaps _measure_local_width_and_
    overhang's near/far (p_min/p_max swap exactly under a 180-deg
    direction flip), so re-applying the SAME near/far values read back
    from a flipped call reassigned them to the OPPOSITE ends -- moving
    the patch -- and the very next identical edit had a fresh chance to
    flip AGAIN, compounding into runaway drift instead of settling.

    Fixed by deciding which PAIR is long by comparing ACROSS pairs
    (edge 0's pair vs edge 1's pair -- long side vs short side, always
    well-separated in magnitude for any real finger/patch, never a
    near-tie) and then always using a FIXED index within that pair
    (whichever is numerically first) for direction -- this removes the
    unstable same-pair, near-tied comparison entirely rather than trying
    to make it more precise, and reproduces the exact same result as
    the old code for every already-correct (well-separated) case."""
    long_pair_is_02 = (lens[0] + lens[2]) >= (lens[1] + lens[3])
    return (0, 1) if long_pair_is_02 else (1, 0)


@dataclass
class ImportedFinger:
    points: List[Tuple[float, float]]
    length_um: float
    width_um: float
    axis_angle_deg: float          # one of the two equivalent directions, informational only
    stub_end: Tuple[float, float]  # short/bare end -- the JJ-crossing side
    patch_end: Tuple[float, float]  # end nearest the attached patches/plate -- the electrode side
    theta_toward_patch_deg: float   # points stub -> patch (substrate -> electrode-facing end)
    theta_toward_stub_deg: float    # points patch -> stub (the reverse direction)
    nearby_patch_count: int = 0

    # warning_source: WHICH shape actually crosses the electrode boundary
    # and therefore needs Warning #1 checked against ITS OWN deposition
    # angle -- "finger" (non-patch-integrated designs), "patches" (the
    # common patch-integrated case), "both", "none" (fully clear or fully
    # buried, no boundary in footprint), or "unknown" (no electrode found).
    warning_source: str = "unknown"

    finger_overlap_frac: Optional[float] = None
    patch_group_overlap_frac: Optional[float] = None

    # Patch-group's own axis + preferred/risky theta. Fixed after a ruler
    # cross-check found these used to be set ONLY
    # when warning_source is "patches" or "both" (i.e. only when the
    # patch group actually straddles the electrode boundary). Whenever a
    # patch group is fully buried in electrode material or fully clear of
    # it (no boundary crossing at all -- common for "OnElectrode" style
    # designs), these stayed None forever, which cascaded into: no patch
    # entry in compute_recommended_angles' output, no angle-diagnostic
    # badge, and the Deposition Angles tab silently defaulting the
    # patches to the fingers' own angles (physically wrong -- patches are
    # ADDITIVE per recipe_generator.py's own documented convention and need
    # their own axis-derived angle regardless of whether they cross an
    # electrode edge). Now populated whenever a patch group exists at
    # all, not just when it crosses.
    patch_outward_end: Optional[Tuple[float, float]] = None
    patch_inward_end: Optional[Tuple[float, float]] = None
    patch_theta_outward_deg: Optional[float] = None
    patch_theta_inward_deg: Optional[float] = None

    # Self-shadowing (Warning #2) -- only meaningful once a deposition
    # order is known (not derivable from geometry alone; set via
    # apply_deposit_order()). None until then.
    deposited_second: Optional[bool] = None
    self_shadow_safe_theta_deg: Optional[float] = None

    # User-assigned role override (Qt GUI's "click a shape, mark it as
    # finger/patch/electrode/substrate" feature) -- purely cosmetic
    # (changes displayed color only), does NOT affect warning_source
    # classification or measurements. None means "use the default
    # finger coloring." stable_id lets the GUI re-apply an override
    # after this object gets rebuilt on the next live-preview redraw
    # (fresh ImportedFinger objects are constructed on every edit, so
    # overrides can't be stored ON the object across redraws -- the GUI
    # keeps its own {stable_id: role} map and re-applies it each time).
    role_override: Optional[str] = None
    stable_id: str = ""

    @property
    def preferred_theta_deg(self) -> Optional[float]:
        return self.theta_toward_patch_deg if self.warning_source in ("finger", "both") else None

    @property
    def risky_theta_deg(self) -> Optional[float]:
        return self.theta_toward_stub_deg if self.warning_source in ("finger", "both") else None


@dataclass
class ImportedShape:
    points: List[Tuple[float, float]]
    kind: str   # "patch" or "plate"
    width_um: float
    length_um: float
    role_override: Optional[str] = None
    stable_id: str = ""
    # Fixed while investigating a patch spacing/tip-margin sync issue:
    # _assign_shapes_to_fingers' only way to tell which
    # finger a patch belongs to is "nearest finger tip point/polygon" --
    # a real ambiguity once a patch group's own long/short extension
    # asymmetry or tip margin pushes a patch's real position closer to the
    # OTHER finger than its own (observed with patches from a
    # perfectly ordinary 0/270-degree, patch_angle=45 design -- the
    # DesignParameters defaults -- landing in the wrong finger's group).
    # For a parametric design, which finger a patch belongs to is known
    # EXACTLY at construction time (from_design_parameters builds each
    # side's own patches in its own per-finger loop iteration) -- this
    # records that here so _assign_shapes_to_fingers can use it directly
    # instead of falling back to the ambiguous geometric guess. None for
    # any patch with no known construction-time owner (a real GDS import,
    # or a hand-placed "patch"-role shape from the shape editor), where
    # the geometric fallback is still the best available option.
    owner_finger_hint: Optional[int] = None


@dataclass
class ImportedJunction:
    fingers: List[ImportedFinger]
    other_shapes: List[ImportedShape]
    source_label: str
    jj_layer: Optional[int] = None
    electrode_polygons: List[List[Tuple[float, float]]] = field(default_factory=list)  # holes/absent-space (for warnings)
    electrode_material: List[List[Tuple[float, float]]] = field(default_factory=list)  # drawn material (for background display)
    import_notes: List[str] = field(default_factory=list)
    patch_placement_log: List[dict] = field(default_factory=list)  # exact math behind every
                                                                      # patch's cx/cy/w/h/angle --
                                                                      # for traceability and for
                                                                      # the future Slideshow tab

    def bounds(self):
        xs, ys = [], []
        for f in self.fingers:
            for x, y in f.points:
                xs.append(x); ys.append(y)
        for s in self.other_shapes:
            for x, y in s.points:
                xs.append(x); ys.append(y)
        if not xs and self.electrode_polygons:
            # Electrode-only view (no finger/patch drawn yet) -- fall back
            # to the electrode's own extent instead of collapsing to a
            # zero-size point at the origin, which showed nothing at all.
            for poly in self.electrode_polygons:
                for x, y in poly:
                    xs.append(x); ys.append(y)
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xs), max(ys))


def _assign_stable_ids(fingers: List["ImportedFinger"], others: List["ImportedShape"]) -> None:
    """Index-based stable ids ('finger_0', 'other_1', ...) so the Qt
    GUI's role-color overrides (keyed by these ids) can be re-applied
    after fresh ImportedFinger/ImportedShape objects get rebuilt on
    every live-preview redraw. Limitation, noted honestly: if the
    NUMBER of patches/plates changes between edits, index-based ids can
    shift which shape an override lands on -- acceptable for now given
    this is a cosmetic-only feature, not a measurement one."""
    for i, f in enumerate(fingers):
        f.stable_id = f"finger_{i}"
    for i, o in enumerate(others):
        o.stable_id = f"other_{i}"


def _assign_warning_sources(fingers: List["ImportedFinger"], others: List["ImportedShape"], electrode_union) -> None:
    """Shared classification math: given already-identified fingers and
    patch/plate shapes, determines each finger's warning_source against
    electrode_union (mutates fingers in place). Used by BOTH the GDS/
    script import path (where fingers/others come from geometric
    detection in classify_fingers) and the parametric design adapter
    (from_design_parameters, where they're already known exactly from
    DesignParameters) -- same classification math either way, just a
    different (and each individually correct) way of identifying which
    polygon is which finger/patch/plate."""

    def _overlap_frac(poly):
        if poly.area < 1e-12 or electrode_union is None:
            return None
        return poly.intersection(electrode_union).area / poly.area

    def _classify_frac(frac):
        if frac is None:
            return "unknown"
        if frac < CROSS_LO or frac > CROSS_HI:
            return "no_edge"  # fully clear OR fully buried -- no boundary in this shape's own footprint
        return "crosses"

    if electrode_union is None:
        return

    for finger in fingers:
        assigned = []
        for o in others:
            # Fixed while re-verifying against a real four-finger crossing
            # design after the ruler-cross-check fix above: this used to
            # assign EVERY nearby "other" shape -- patches AND adhesion
            # plates alike -- into the same patch group, then compute
            # that group's axis from their UNION. A plate is a
            # differently-shaped, often differently-rotated rectangle
            # (recipe_generator.py's own docstring: the plate is deposited
            # WITH THE FINGER, "same layer, same sub-step, no separate
            # angle" -- it is NOT part of the patches' own deposition).
            # On a real patch-integrated import, mixing
            # in the plate (whose largest-by-area polygon dominated
            # unary_union's result) skewed the computed patch axis from a
            # clean 0deg to a real, wrong, off-axis angle
            # that never showed up before because this whole computation
            # was previously gated behind patches_cross and never ran for
            # this design. _assign_shapes_to_fingers (used elsewhere, for
            # measurements) already filters to kind=="patch" for exactly
            # this reason; this loop never had the same filter. Now it does.
            if o.kind != "patch":
                continue
            op = sg.Polygon(o.points)
            d_this = op.distance(sg.Point(finger.patch_end))
            d_other_fingers = [op.distance(sg.Point(f2.patch_end)) for f2 in fingers if f2 is not finger]
            if not d_other_fingers or d_this <= min(d_other_fingers):
                assigned.append(op)

        finger_poly = sg.Polygon(finger.points)
        finger.finger_overlap_frac = _overlap_frac(finger_poly)
        finger_crosses = _classify_frac(finger.finger_overlap_frac) == "crosses"

        patches_cross = False
        if assigned:
            patch_group_poly = so.unary_union(assigned)
            finger.patch_group_overlap_frac = _overlap_frac(patch_group_poly)
            patches_cross = _classify_frac(finger.patch_group_overlap_frac) == "crosses"
            # Fixed after a ruler cross-check:
            # this axis/theta computation used to be gated on
            # "if patches_cross:" -- so a patch group that's fully buried
            # in electrode material or fully clear of it (no boundary
            # crossing found by the fraction test) never got an axis or
            # theta computed at all, even though it still physically
            # needs a deposition angle. Patches are ADDITIVE (per
            # recipe_generator.py's own documented convention) and always
            # need their own axis-derived angle -- independent of
            # whether the coarse electrode-crossing fraction test finds
            # a boundary in their footprint. Now computed unconditionally
            # whenever a patch group exists (assigned is non-empty).
            #
            # patch group's OWN axis -- typically axis-aligned
            # rectangles with a totally different natural direction
            # than the finger's own 45-degree-family angle
            #
            # Fixed here, reproduced
            # exactly on a synthetic staircase patch group before
            # trusting it: a multi-patch group is a set of SEPARATE,
            # non-touching rectangles (patch_spacing_um deliberately
            # keeps them apart -- see compute_measurements' own
            # edge_gaps), so unary_union(assigned) is almost always a
            # MultiPolygon, not one Polygon. This used to pick out just
            # the LARGEST single patch by area (multi-patch groups
            # aren't all the same size) and derive the "outward"/
            # "inward" ends from ONLY that one patch's own two short
            # edges -- discarding every other patch in the group,
            # including whichever one is actually the outermost/
            # electrode-facing one if it isn't also the largest. When
            # the largest patch happens to be an INNER one that never
            # reaches the electrode, neither of its own ends lies
            # inside electrode_union, so ea_in below was always False
            # and outward/inward fell back to an arbitrary tie-break
            # ordering instead of the group's true electrode-facing
            # side -- reproduced this exactly: two mirrored finger
            # groups (one facing left, one facing right) both came out
            # reporting the SAME "outward" angle, which is exactly the
            # "only one deposition angle needed" symptom directly
            # reported (the other side's patches would then be open at
            # that angle). Using the CONVEX HULL of the whole group
            # (covers every patch's true extent, not just one member)
            # fixes this -- a hull of several separated rectangles is
            # essentially never itself a clean 4-point rectangle, so
            # this naturally exercises _polygon_axis_and_ends' own
            # minimum-rotated-rectangle fallback path, which is exactly
            # the robust "long axis + true extreme short-edge midpoints"
            # behavior this needs.
            hull = patch_group_poly.convex_hull
            pg_pts = list(hull.exterior.coords) if hull.geom_type == "Polygon" \
                else list(max(patch_group_poly.geoms, key=lambda g: g.area).exterior.coords)  # degenerate fallback (e.g. collinear patches)
            _axis, end_a, end_b, _w = _polygon_axis_and_ends(pg_pts)
            # whichever end sits (mostly) inside the electrode is "outward";
            # when the patch group doesn't cross a boundary at all (fully
            # buried or fully clear), electrode_union.contains() still
            # gives a well-defined answer (True if fully inside, False if
            # fully outside), so "outward"/"inward" remain meaningful even
            # without a crossing.
            ea_in = electrode_union.contains(sg.Point(end_a))
            outward, inward = (end_a, end_b) if ea_in else (end_b, end_a)
            finger.patch_outward_end, finger.patch_inward_end = outward, inward
            finger.patch_theta_outward_deg = math.degrees(
                math.atan2(outward[1] - inward[1], outward[0] - inward[0])) % 360
            finger.patch_theta_inward_deg = (finger.patch_theta_outward_deg + 180.0) % 360

        if finger_crosses and patches_cross:
            finger.warning_source = "both"
        elif finger_crosses:
            finger.warning_source = "finger"
        elif patches_cross:
            finger.warning_source = "patches"
        else:
            finger.warning_source = "none"


# ======================================================================
# Editing an already-imported design --
# the geometry the Qt GUI shows for an imported (GDS/KLayout) junction
# was, until now, always a static, read-only snapshot of exactly what
# was extracted from the file. These three functions are the missing
# piece that lets the GUI mutate an ImportedFinger/ImportedShape's own
# length/width/angle in place and have every downstream measurement,
# warning classification, and recommended-angle computation catch up --
# WITHOUT re-running the raw GDS/script extraction pipeline (which could
# reclassify shapes differently just from resizing one of them).
# ======================================================================

def axis_line_intersection(point_a, angle_a_deg, point_b, angle_b_deg):
    """World-space intersection of two infinite lines, each given as a
    point plus a direction angle (direction SIGN doesn't matter -- a line
    and its 180-degree-flipped direction describe the exact same line).
    Returns None if the two lines are parallel (no unique intersection)."""
    a1, a2 = math.radians(angle_a_deg), math.radians(angle_b_deg)
    d1x, d1y = math.cos(a1), math.sin(a1)
    d2x, d2y = math.cos(a2), math.sin(a2)
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None
    dx, dy = point_b[0] - point_a[0], point_b[1] - point_a[1]
    t = (dx * d2y - dy * d2x) / denom
    return (point_a[0] + d1x * t, point_a[1] + d1y * t)


def _find_crossing_finger_pair(indices, shapely_polys, by_index, require_perpendicular=False):
    """Returns (i, j) for the pair of raw-polygon indices that most
    plausibly form a Manhattan JJ's actual two crossing fingers, using a
    signature that's INDEPENDENT of shape length -- unlike
    classify_fingers' own default "pick the 2 longest shapes," which
    silently breaks for a patch-integrated design whose patches happen to
    have a longer geometric overhang than the finger's own bare stub-to-
    crossing length. This was found on a real patch-integrated GDS
    import, where the program picked
    the two longest PATCHES as the fingers, and the real, shorter fingers
    as patches.

    require_perpendicular=True was added after a real GDS import
    reproduced the issue: it restricts the search
    to pairs whose axes cross at (near) exactly 90 degrees -- the
    hallmark of the Manhattan junction design. Without this, "any non-near-parallel crossing" (the
    MIN_CROSS_ANGLE_DEG>=20 test alone) can still pick a coincidentally-
    overlapping pair of shapes that cross at some OTHER oblique angle.
    A dense patch-integrated design was observed with two
    patches, one on each finger's own axis, crossing each other at 45
    degrees with a LARGER real overlap area than the two true fingers'
    own 90-degree crossing -- so the old "biggest tier-2 overlap wins"
    rule picked the wrong, non-perpendicular pair outright, even though
    the true fingers demonstrably crossed too. See classify_fingers for
    how this is tried FIRST, before any length-based or unrestricted-
    angle fallback.

    Two signals, tried in order of confidence:
      1. REAL physical polygon overlap -- this is literally what a JJ
         crossing IS, and essentially never happens between a finger and
         its own patch/plate (patches only ever extend OUTWARD past the
         finger's own patch_end, never back across it) or between two
         patches on the same or opposite sides of the design.
      2. For a deliberate negative-overlap "gap" design (the two fingers
         point at each other without quite touching): the pair whose own
         axis LINES cross at a point that (a) actually lies close to BOTH
         real segments -- not some wild extrapolation of two unrelated
         lines that happen to not be parallel -- and (b) is more
         ISOLATED from every other shape on the layer than any other
         candidate pair's crossing point would be. Patches/plates
         cluster tightly near their own electrode; the true JJ crossing
         sits alone out in the open gap between the two electrodes.

    Returns None if no pair plausibly crosses at all (the caller falls
    back to the old length-based default, so this can never make a
    design that used to import fine suddenly fail to produce 2 fingers)."""
    best = None  # ((tier, score), i, j) -- Python tuple comparison naturally
                   # prefers a higher tier (a confirmed real overlap always
                   # beats a mere near-crossing), then a higher score within it
    MIN_CROSS_ANGLE_DEG = 20.0  # below this, two shapes are near-parallel/
                                  # colinear -- exactly what a patch/plate
                                  # overlapping or continuing its OWN
                                  # finger's axis looks like (a normal,
                                  # expected construction -- compute_
                                  # measurements' own comments call out
                                  # "a patch overlapping its finger" as a
                                  # routine case), not two independent
                                  # fingers crossing at a real angle. A
                                  # genuine Manhattan crossing pair always
                                  # approaches from meaningfully different
                                  # directions -- without this guard, a
                                  # patch's own overlap into its parent
                                  # finger would score as tier-2 "real
                                  # overlap" and get mistaken for the JJ
                                  # crossing itself.
    MIN_FINGER_ASPECT_RATIO = 4.0  # Fixed after a report on a real
                                  # imported cell: "biggest real overlap
                                  # wins" alone isn't enough -- a real GDS
                                  # design can include a small, wide
                                  # reinforcement/contact pad positioned
                                  # right where two fingers cross, and that
                                  # pad's own overlap with one of the real
                                  # fingers can be geometrically LARGER than
                                  # the two real fingers' own crossing
                                  # overlap with each other, so it won the
                                  # "finger" slot outright (reported as a
                                  # finger with an implausible ~1.2um width
                                  # -- plate-like, not finger-like). A real
                                  # Josephson finger is, by its physical
                                  # role, always a THIN elongated lead --
                                  # never anywhere close to as wide as it is
                                  # long -- so any candidate whose own
                                  # length:width ratio doesn't clear this
                                  # bar is disqualified from the crossing-
                                  # pair search entirely, regardless of how
                                  # much overlap area it has with anything.
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            key = _pair_crossing_key(i, j, shapely_polys, by_index,
                                      MIN_FINGER_ASPECT_RATIO, MIN_CROSS_ANGLE_DEG,
                                      require_perpendicular=require_perpendicular)
            if key is None:
                continue
            if best is None or key > best[0]:
                best = (key, i, j)
    if best is None:
        return None
    return (best[1], best[2])


PERPENDICULAR_TOL_DEG = 15.0  # Crossing another
                                # feature at exactly 90 degrees is the hallmark
                                # of the Manhattan junction design. This is how far
                                # from exactly 90 degrees a crossing may sit and
                                # still count as "perpendicular" for finger-pair
                                # identification. Generous enough for ordinary
                                # GDS/e-beam rounding (every real design measured
                                # so far lands within a hundredth of a degree of
                                # 90), tight enough to exclude a design's other
                                # diagonal-family shapes (e.g. two patches each
                                # sharing their own parent finger's 45/135-degree
                                # axis, which are also "not near-parallel" to
                                # each other under the old >=20-degree guard
                                # alone, but are never actually perpendicular).


def _pair_crossing_key(i, j, shapely_polys, by_index,
                        min_aspect_ratio=4.0, min_cross_angle_deg=20.0,
                        require_perpendicular=False):
    """The scoring core shared by _find_crossing_finger_pair's full search
    AND classify_fingers' own sanity check on its length-based default
    pair (see classify_fingers for why that second use matters -- fixed
    after a real patch-integrated design showed that raw overlap AREA is not by
    itself a reliable "this is the true JJ crossing" signal, because a
    design can legitimately have reinforcement/contact shapes deliberately
    overlapping a finger's tip by MORE area than the real, intentionally-
    minimized finger-to-finger crossing itself -- so a blanket "biggest
    overlap wins" search can clobber an already-correct default with a
    wrong pair. Returns None if (i, j) doesn't plausibly cross at all;
    otherwise a tuple that Python's tuple ordering ranks a real overlap
    above a mere near-miss crossing, and a bigger one above a smaller one
    within the same tier.

    require_perpendicular=True additionally rejects any pair whose
    crossing angle isn't within PERPENDICULAR_TOL_DEG of exactly 90
    degrees -- see _find_crossing_finger_pair's own docstring for the
    real reported failure this closes (a design with several 45/135-
    degree-family shapes, not all of them real fingers, where the old
    ">=20 degrees, not exactly 90" guard alone let a non-perpendicular
    pair with more overlap area win the finger slot over the true,
    perpendicular crossing pair)."""
    _len_i, _i, angle_i, end_a_i, end_b_i, width_i = by_index[i]
    _len_j, _j, angle_j, end_a_j, end_b_j, width_j = by_index[j]
    if _len_i / max(width_i, 1e-9) < min_aspect_ratio or \
       _len_j / max(width_j, 1e-9) < min_aspect_ratio:
        return None  # not shaped like a real finger
    raw_diff = abs(angle_i - angle_j) % 180.0
    angle_diff = min(raw_diff, 180.0 - raw_diff)
    if angle_diff < min_cross_angle_deg:
        return None
    if require_perpendicular and abs(angle_diff - 90.0) > PERPENDICULAR_TOL_DEG:
        return None
    real_overlap = shapely_polys[i].intersection(shapely_polys[j]).area
    if real_overlap > 1e-9:
        return (2, real_overlap)
    crossing = axis_line_intersection(end_a_i, angle_i, end_a_j, angle_j)
    if crossing is None:
        return None  # parallel axes -- can't be a Manhattan crossing pair
    cp = sg.Point(crossing)
    dist_i = shapely_polys[i].distance(cp)
    dist_j = shapely_polys[j].distance(cp)
    tol = max(width_i, width_j) * 3.0 + 0.05  # generous but bounded -- a real
                                                 # near-miss crossing sits
                                                 # within a few linewidths of
                                                 # both shapes, not arbitrarily far
    if dist_i > tol or dist_j > tol:
        return None  # the lines only cross far from where the real shapes are
    others = [shapely_polys[k].distance(cp) for k in range(len(shapely_polys)) if k != i and k != j]
    isolation = min(others) if others else float("inf")
    return (1, isolation)


def _derive_finger_ends_and_angles(end_a, end_b, length, other_shape_polys, other_finger_polys):
    """Shared math for 'which end of a finger candidate is the bare stub
    (JJ-crossing side) and which is the patch/plate end (electrode
    side)': proximity to a REAL patch/plate shape means patch end; with
    none available, falls back to distance from the OTHER finger
    (inverted -- the crossing side is NEAR it). Also derives the
    resulting theta_toward_patch/theta_toward_stub and a nearby-shape
    count. Extracted so classify_fingers' own detection loop and
    promote_shape_to_finger (the Mark-as-Finger correction tool, for when
    classify_fingers picked the wrong shapes) always agree exactly on how
    a finger's ends get assigned -- one formula, not two that could
    drift apart."""
    def nearest_dist(pt):
        p = sg.Point(pt)
        dists = [poly.distance(p) for poly in other_shape_polys]
        return min(dists) if dists else None

    da, db = nearest_dist(end_a), nearest_dist(end_b)
    if da is None and db is None:
        if other_finger_polys:
            da2 = min(poly.distance(sg.Point(end_a)) for poly in other_finger_polys)
            db2 = min(poly.distance(sg.Point(end_b)) for poly in other_finger_polys)
            patch_end, stub_end = (end_a, end_b) if da2 > db2 else (end_b, end_a)
        else:
            patch_end, stub_end = end_a, end_b  # nothing at all to compare against -- arbitrary
    else:
        da = da if da is not None else float("inf")
        db = db if db is not None else float("inf")
        patch_end, stub_end = (end_a, end_b) if da < db else (end_b, end_a)

    theta_toward_patch = math.degrees(math.atan2(patch_end[1] - stub_end[1], patch_end[0] - stub_end[0])) % 360
    theta_toward_stub = (theta_toward_patch + 180.0) % 360
    nearby_count = sum(1 for poly in other_shape_polys if poly.distance(sg.Point(patch_end)) < length * 0.5)
    return patch_end, stub_end, theta_toward_patch, theta_toward_stub, nearby_count


def promote_shape_to_finger(junction: "ImportedJunction", shape: "ImportedShape") -> "ImportedFinger":
    """Converts a patch/plate ImportedShape into a real ImportedFinger --
    for when classify_fingers()'s automatic heuristic picked the wrong
    shapes. This was observed on a real GDS import: a
    patch-integrated design whose patches happened to be geometrically
    LONGER than the true fingers got its patches and fingers swapped.
    Called when the user explicitly marks a real junction member as
    "Finger" via the Mark-as toolbar on an IMPORTED design -- a genuine
    correction, not the purely cosmetic recolor role_override used to be
    (see ImportedFinger/ImportedShape's own role_override docstrings,
    predating this function). Preserves the shape's own stable_id across
    the conversion so the GUI's selection/scale-base/measurement caches
    (all keyed by stable_id) don't lose track of it. Mutates
    junction.fingers/junction.other_shapes in place (removes shape from
    other_shapes, appends the new ImportedFinger) and returns the new
    finger; caller is responsible for calling recompute_derived_fields()
    afterward (not done here, to match every other single-shape edit
    function in this module, which also leave that to the caller)."""
    axis_angle, end_a, end_b, width = _polygon_axis_and_ends(shape.points)
    length = math.hypot(end_a[0] - end_b[0], end_a[1] - end_b[1])
    other_shape_polys = [sg.Polygon(s.points) for s in junction.other_shapes if s is not shape]
    other_finger_polys = [sg.Polygon(f.points) for f in junction.fingers]
    patch_end, stub_end, theta_toward_patch, theta_toward_stub, nearby_count = _derive_finger_ends_and_angles(
        end_a, end_b, length, other_shape_polys, other_finger_polys)
    new_finger = ImportedFinger(
        points=list(shape.points), length_um=length, width_um=width, axis_angle_deg=axis_angle,
        stub_end=stub_end, patch_end=patch_end,
        theta_toward_patch_deg=theta_toward_patch, theta_toward_stub_deg=theta_toward_stub,
        nearby_patch_count=nearby_count, stable_id=shape.stable_id,
    )
    junction.other_shapes.remove(shape)
    junction.fingers.append(new_finger)
    return new_finger


def demote_finger_to_shape(junction: "ImportedJunction", finger: "ImportedFinger", kind: str) -> "ImportedShape":
    """The reverse of promote_shape_to_finger(): converts a real
    ImportedFinger into a patch/plate ImportedShape, for when
    classify_fingers() picked a real patch/plate as one of its two
    "fingers" (the other half of the same swapped-classification failure mode --
    swapping one pair of shapes always means BOTH directions need
    correcting). kind must be "patch" or "plate". Preserves the finger's
    own stable_id across the conversion, same reasoning as
    promote_shape_to_finger. Mutates junction.fingers/junction.other_shapes
    in place; caller calls recompute_derived_fields() afterward."""
    axis_angle, end_a, end_b, width = _polygon_axis_and_ends(finger.points)
    length = math.hypot(end_a[0] - end_b[0], end_a[1] - end_b[1])
    new_shape = ImportedShape(points=list(finger.points), kind=kind, width_um=width, length_um=length,
                               stable_id=finger.stable_id)
    junction.fingers.remove(finger)
    junction.other_shapes.append(new_shape)
    return new_shape


def finger_crossing_point(finger: "ImportedFinger", other_finger: "ImportedFinger"):
    """The point where this finger's own axis line crosses the OTHER
    finger's axis line -- generalizes the parametric layout's shared world
    origin (both default fingers' axes pass through it by construction,
    see parametric_junction_view.FingerAssembly -- every parametric finger pivots on
    the same un-translated local frame) to a finger placed or rotated
    anywhere. This is the reference point "overlap with the next finger"
    is measured against for ANY finger -- hand-drawn, GDS-imported, or
    parametric -- not just the default centered layout. Returns None if
    the two fingers' axes are parallel (no crossing point exists)."""
    return axis_line_intersection(finger.stub_end, finger.theta_toward_patch_deg,
                                   other_finger.stub_end, other_finger.theta_toward_patch_deg)


def finger_overlap_start_point(finger: "ImportedFinger", other_finger: "ImportedFinger"):
    """The point 'the overlap' (finger_overlap_um) measures FROM: the near
    edge of the two fingers' REAL polygon crossing footprint, on this
    finger's own stub-tip side -- i.e. the point right after the physical
    junction, walking from the bare tip (stub_end) in toward the
    electrode. This was found on a real design where the KLayout
    script's own overlap constant is
    1.5 um: using the idealized axis-LINE crossing point
    (finger_crossing_point) as this start point is wrong whenever the two
    fingers have any real width, because that point sits INSIDE the
    junction's own crossing footprint (roughly its geometric middle, not
    its edge) -- so measuring the 'overlap' from there over-counts by
    however much of the junction's own footprint lies between its near
    edge and that center line, silently inflating every reported overlap
    by roughly half the other finger's width along this axis. The real
    quantity a fabricator means by 'overlap' is the bare stub margin
    AFTER the tip has fully cleared the junction, not a distance that
    still runs partway through it.

    Falls back to the idealized axis-line crossing point when there's no
    real polygon overlap at all yet -- a deliberate negative-overlap
    'gap' design, or a finger that doesn't yet reach the other one -- in
    which case there's no real footprint edge to measure from and the
    axis crossing point remains the only meaningful reference.

    A follow-up fix, prompted by feedback against the on-screen
    overlap dimension line on a real import: the near-edge point used to
    be picked as the closest VERTEX of the two fingers' full 2D polygon-
    intersection footprint. For a perpendicular crossing that vertex
    happens to land at the right AXIAL distance (so finger_overlap_um's
    number was already correct -- see the fix above), but the vertex
    itself sits OFF this finger's own centerline, offset sideways by
    roughly half of THIS finger's own width. Since stub_end IS on the
    centerline (by construction -- see _polygon_axis_and_ends), drawing
    the overlap dimension as a straight line between the two points
    (main_gui._draw_measurement_overlay's add_dimension) produced a
    visibly diagonal segment whose on-screen length didn't match the
    printed number: exactly a "center of one end, side of the other"
    mismatch -- the overlap should be measured consistently from either
    the centers of both the beginning and the end, or the side of both,
    not a mix of the two.

    Fixed to use the point where THIS finger's own CENTERLINE (the
    segment stub_end -> patch_end) first enters the other finger's real
    polygon boundary -- always exactly on-axis, so the drawn segment is a
    straight line along the finger's own axis whose length equals the
    printed value exactly. For the ordinary rectangular-finger case this
    is numerically IDENTICAL to the old vertex-based value (both reduce
    to the same "other finger's near boundary along this axis" distance),
    so this only changes WHICH POINT is returned, not any previously
    verified overlap_um number. Falls back to the old vertex-based
    approach (then to the idealized axis crossing) for the rare case
    where a non-rectangular real import's centerline doesn't cleanly
    cross the other polygon's boundary even though the two shapes do
    overlap."""
    theta_rad = math.radians(finger.theta_toward_patch_deg)
    ax, ay = math.cos(theta_rad), math.sin(theta_rad)
    sx, sy = finger.stub_end

    def t_of(pt):
        return (pt[0] - sx) * ax + (pt[1] - sy) * ay

    finger_poly = sg.Polygon(finger.points)
    other_poly = sg.Polygon(other_finger.points)
    inter = finger_poly.intersection(other_poly)
    if not inter.is_empty and inter.area > 1e-9:
        # Prefer the on-axis entry point: where this finger's own
        # centerline first crosses INTO the other finger's real boundary.
        try:
            centerline = sg.LineString([finger.stub_end, finger.patch_end])
            boundary_hit = centerline.intersection(other_poly.boundary)
        except Exception:
            boundary_hit = None
        pts_on_axis = []
        if boundary_hit is not None and not boundary_hit.is_empty:
            if hasattr(boundary_hit, "geoms"):
                pts_on_axis = [(g.x, g.y) for g in boundary_hit.geoms if hasattr(g, "x")]
            elif hasattr(boundary_hit, "x"):
                pts_on_axis = [(boundary_hit.x, boundary_hit.y)]
        if pts_on_axis:
            return min(pts_on_axis, key=t_of)
        # Fallback: the finger's centerline doesn't cleanly cross the
        # other polygon's boundary (an unusual/non-rectangular real
        # import) -- fall back to the nearest vertex of the actual
        # overlap footprint, same as before this fix.
        geoms = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
        pts = [pt for g in geoms if hasattr(g, "exterior") for pt in g.exterior.coords]
        if pts:
            return min(pts, key=t_of)
    return finger_crossing_point(finger, other_finger)


def finger_overlap_um(finger: "ImportedFinger", other_finger: "ImportedFinger") -> Optional[float]:
    """How far this finger's stub end (its bare tip) sits PAST the near
    edge of its real physical crossing with the other finger
    (finger_overlap_start_point) -- i.e. the bare stub margin left AFTER
    the tip has fully cleared the junction, measured along this finger's
    own outward axis. Positive means the tip extends past the junction by
    that much (the intended, same sign convention as
    DesignParameters.overlap_target_um); negative means the tip doesn't
    even reach the junction's near edge -- an incomplete crossing. None
    if the two axes are parallel (see finger_crossing_point)."""
    start = finger_overlap_start_point(finger, other_finger)
    if start is None:
        return None
    theta_rad = math.radians(finger.theta_toward_patch_deg)
    ax, ay = math.cos(theta_rad), math.sin(theta_rad)
    sx, sy = finger.stub_end
    return (start[0] - sx) * ax + (start[1] - sy) * ay


def apply_finger_edit(finger: ImportedFinger, *, length_um: Optional[float] = None,
                       width_um: Optional[float] = None,
                       theta_toward_patch_deg: Optional[float] = None,
                       overlap_um: Optional[float] = None,
                       crossing_point: Optional[Tuple[float, float]] = None) -> None:
    """Regenerates one finger's rectangle in place from an edited length,
    width, and/or axis angle -- any argument left None keeps that field's
    current value. stub_end (the short/bare, JJ-crossing end) is normally
    held FIXED: physically, that's the end that doesn't move when you
    resize a finger from the electrode-facing side, and it's also the
    anchor the rest of the design (the JJ crossing point itself) is
    measured from. patch_end moves to stub_end + new_length in the
    (possibly also newly-edited) axis direction; theta_toward_stub_deg is
    kept as the exact opposite of theta_toward_patch_deg, same convention
    classify_fingers itself uses.

    overlap_um (new): repositions stub_end so this finger sits exactly
    overlap_um past `crossing_point` (the intersection of this finger's own
    axis with the OTHER finger's axis -- see finger_crossing_point()) along
    its own outward axis, instead of holding the old stub_end fixed --
    length_um still applies independently on top (patch_end is always
    recomputed as stub_end + length in the axis direction). This function
    has no access to "the other finger" on its own (by design -- like
    every other function in this section, it edits ONE shape at a time),
    so the caller must resolve crossing_point itself via
    finger_crossing_point() first; overlap_um is silently ignored if
    crossing_point is None.

    Does NOT touch warning_source, overlap fractions, or any of the
    patch-group axis fields -- those depend on the finger's relationship
    to the electrode and to nearby patches, which can genuinely change
    as a result of this edit (e.g. a lengthened finger now reaches an
    electrode it didn't before). Call recompute_derived_fields() on the
    whole junction right after this (or after a batch of these edits) to
    refresh all of that -- kept as a separate step so the caller can
    batch several shape edits before paying for one recompute pass."""
    new_length = max(0.01, float(length_um)) if length_um is not None else finger.length_um
    new_width = max(0.005, float(width_um)) if width_um is not None else finger.width_um
    new_theta = float(theta_toward_patch_deg) % 360.0 if theta_toward_patch_deg is not None else finger.theta_toward_patch_deg

    theta_rad = math.radians(new_theta)
    ux, uy = math.cos(theta_rad), math.sin(theta_rad)
    if overlap_um is not None and crossing_point is not None:
        sx = crossing_point[0] - ux * float(overlap_um)
        sy = crossing_point[1] - uy * float(overlap_um)
    else:
        sx, sy = finger.stub_end
    new_patch_end = (sx + ux * new_length, sy + uy * new_length)

    perp_x, perp_y = -uy, ux
    hw = new_width / 2.0
    finger.points = [
        (sx + perp_x * hw, sy + perp_y * hw),
        (new_patch_end[0] + perp_x * hw, new_patch_end[1] + perp_y * hw),
        (new_patch_end[0] - perp_x * hw, new_patch_end[1] - perp_y * hw),
        (sx - perp_x * hw, sy - perp_y * hw),
    ]
    finger.length_um = new_length
    finger.width_um = new_width
    finger.stub_end = (sx, sy)
    finger.patch_end = new_patch_end
    finger.axis_angle_deg = new_theta % 360.0
    finger.theta_toward_patch_deg = new_theta
    finger.theta_toward_stub_deg = (new_theta + 180.0) % 360.0


def apply_shape_edit(shape: ImportedShape, *, length_um: Optional[float] = None,
                      width_um: Optional[float] = None) -> None:
    """Resizes a patch/plate rectangle in place from an edited length
    and/or width, keeping its current CENTER and ORIENTATION fixed --
    unlike a finger, a patch/plate has no inherent stub/tip distinction
    to anchor on, so 'resize in place, centered' is the natural edit.
    Reuses _polygon_axis_and_ends (the same long-axis/short-axis math
    already used to derive width_um/length_um from raw GDS polygons in
    the first place) so this stays consistent with how those fields were
    originally computed, rather than introducing a second, possibly
    inconsistent notion of a shape's orientation."""
    axis_angle_deg, end_a, end_b, cur_width = _polygon_axis_and_ends(shape.points)
    cur_length = math.hypot(end_b[0] - end_a[0], end_b[1] - end_a[1])
    cx, cy = (end_a[0] + end_b[0]) / 2.0, (end_a[1] + end_b[1]) / 2.0

    new_length = max(0.005, float(length_um)) if length_um is not None else cur_length
    new_width = max(0.005, float(width_um)) if width_um is not None else cur_width

    theta_rad = math.radians(axis_angle_deg)
    ux, uy = math.cos(theta_rad), math.sin(theta_rad)
    perp_x, perp_y = -uy, ux
    hl, hw = new_length / 2.0, new_width / 2.0
    shape.points = [
        (cx - ux * hl - perp_x * hw, cy - uy * hl - perp_y * hw),
        (cx + ux * hl - perp_x * hw, cy + uy * hl - perp_y * hw),
        (cx + ux * hl + perp_x * hw, cy + uy * hl + perp_y * hw),
        (cx - ux * hl + perp_x * hw, cy - uy * hl + perp_y * hw),
    ]
    shape.length_um = new_length
    shape.width_um = new_width


def apply_patch_extension_edit(shape: ImportedShape, finger: "ImportedFinger", patch_angle_deg: float, *,
                                near_overhang_um: Optional[float] = None,
                                far_overhang_um: Optional[float] = None,
                                width_um: Optional[float] = None,
                                connector_union=None,
                                current_near_overhang_um: float = 0.0,
                                current_far_overhang_um: float = 0.0) -> None:
    """Resizes ONE existing patch shape from an edited near/far overhang
    (the real, measured equivalent of the parametric "short/long extension"
    fields -- see MeasurementReport.patch_groups' individual_near_overhang_um/
    individual_far_overhang_um) and/or thickness, keeping the patch's
    ALONG-FINGER position (its crossing point relative to the finger) fixed
    -- only how far it extends past that point on each side, and how thick
    it is, change.

    connector_union (preferred): the real solid material this patch
    attaches to -- the finger's own polygon, plus its adhesion plate's if
    one exists (see _find_finger_plate / compute_measurements' own
    connector_union construction). When given, the crossing point and
    local clearance are measured directly from real geometry via
    _measure_local_width_and_overhang -- the EXACT SAME function that
    computes the near/far overhang values shown in the Shape Properties
    card (junction_import.measurements_by_stable_id) -- guaranteeing an
    exact round-trip. Fixed here: the original version
    always derived the clearance from `finger.width_um` alone (mirroring
    parametric_junction_view.build_side_patches()'s IDEALIZED placement formula),
    which is only correct when a patch crosses the bare finger --
    exactly the same class of bug _measure_local_width_and_overhang's
    own docstring already documents and fixes for the read side ("patches
    near the tip commonly cross the much wider adhesion plate instead").
    Without connector_union, a patch crossing its plate reported a
    self-inconsistent near/far after this edit (observed as a
    0.25um discrepancy on a design with has_adhesion_pads=True and a
    finger not aligned to the default 0/270 axes).

    `current_near_overhang_um`/`current_far_overhang_um` are used ONLY
    as a fallback when connector_union isn't given (kept for backward
    compatibility with callers that don't have it handy -- exact for the
    common case where a patch crosses the bare finger with no plate, or
    the default axis-aligned parametric layout, but can drift by the
    same margin connector_union fixes otherwise)."""
    axis_angle_deg, end_a, end_b, cur_width = _polygon_axis_and_ends(shape.points)
    cx, cy = (end_a[0] + end_b[0]) / 2.0, (end_a[1] + end_b[1]) / 2.0

    theta_rad = math.radians(patch_angle_deg)
    px, py = math.cos(theta_rad), math.sin(theta_rad)

    if connector_union is not None:
        local_width, cur_near, cur_far, _geo = _measure_local_width_and_overhang(
            finger.stub_end, finger.patch_end, connector_union, shape.points, patch_angle_deg, (cx, cy))
        # Recover the exact same crossing point
        # _measure_local_width_and_overhang derives internally (it
        # doesn't expose the point itself, so this is the identical
        # line-intersection math, duplicated in closed form rather than
        # via a second call).
        fx, fy = finger.patch_end[0] - finger.stub_end[0], finger.patch_end[1] - finger.stub_end[1]
        flen = math.hypot(fx, fy) or 1.0
        f_hat = (fx / flen, fy / flen)
        finger_angle_deg = math.degrees(math.atan2(fy, fx))
        denom = f_hat[0] * py - f_hat[1] * px
        if abs(denom) < 1e-12:
            crossing_x, crossing_y = cx, cy
        else:
            dx, dy = cx - finger.stub_end[0], cy - finger.stub_end[1]
            t = (dx * py - dy * px) / denom
            crossing_x, crossing_y = finger.stub_end[0] + t * f_hat[0], finger.stub_end[1] + t * f_hat[1]
        sin_delta = math.sin(math.radians(patch_angle_deg - finger_angle_deg))
        if abs(sin_delta) < 0.05:
            sin_delta = 0.05 if sin_delta >= 0 else -0.05
        u = (local_width / 2.0) / abs(sin_delta)
        current_near_overhang_um, current_far_overhang_um = cur_near, cur_far
    else:
        # Fallback: reconstruct the crossing point from the passed-in
        # "current" values and the IDEALIZED (bare-finger) clearance --
        # center = crossing + patch_hat * ((far - near) / 2), matching
        # build_side_patches' own "center = crossing + patch_hat *
        # ((long_ext - short_ext) / 2)" (far == long/outward extension,
        # near == short/inward extension), inverted to recover crossing.
        half_span = (current_far_overhang_um - current_near_overhang_um) / 2.0
        crossing_x = cx - px * half_span
        crossing_y = cy - py * half_span
        delta_deg = patch_angle_deg - finger.axis_angle_deg
        sin_delta = math.sin(math.radians(delta_deg))
        if abs(sin_delta) < 0.05:
            sin_delta = 0.05 if sin_delta >= 0 else -0.05
        u = (finger.width_um / 2.0) / abs(sin_delta)

    new_near = float(near_overhang_um) if near_overhang_um is not None else current_near_overhang_um
    new_far = float(far_overhang_um) if far_overhang_um is not None else current_far_overhang_um
    new_width = max(0.005, float(width_um)) if width_um is not None else cur_width

    new_length = 2 * u + new_far + new_near
    new_cx = crossing_x + px * ((new_far - new_near) / 2.0)
    new_cy = crossing_y + py * ((new_far - new_near) / 2.0)

    perp_x, perp_y = -py, px
    hl, hw = new_length / 2.0, new_width / 2.0
    shape.points = [
        (new_cx - px * hl - perp_x * hw, new_cy - py * hl - perp_y * hw),
        (new_cx + px * hl - perp_x * hw, new_cy + py * hl - perp_y * hw),
        (new_cx + px * hl + perp_x * hw, new_cy + py * hl + perp_y * hw),
        (new_cx - px * hl + perp_x * hw, new_cy - py * hl + perp_y * hw),
    ]
    shape.length_um = new_length
    shape.width_um = new_width


def finger_display_label(index: int) -> str:
    """Canonical display label for a finger by its 0-based index in
    `junction.fingers`: 'A', 'B', 'C', 'D', ... -- the single shared
    source of truth every UI surface (Measurements dock, Predicted
    Recipe Sequence, Simulate/Slideshow warnings, tree/table views) now
    calls, replacing an ad hoc `"AB"[i] if i < 2 else str(i)` idiom that
    had been copy-pasted at roughly 25 separate call sites across
    main_gui.py and imported_junction_view.py.

    Fixed after a report against a real
    4-finger series-chain design: that idiom labels index
    0/1 as "A"/"B" (matching this app's own pre-existing, test-asserted
    convention -- many existing tests check for literal "Finger A" text,
    which is why this extends the LETTER scheme rather than switching to
    plain numbers) but falls back to a
    bare digit ("2", "3", ...) for every finger beyond the first two --
    an inconsistent, unintentional scheme, not a deliberate one. Beyond
    'Z' this continues in spreadsheet-column style ('AA', 'AB', ...) so
    an extreme finger count degrades gracefully instead of raising,
    though no real design is expected to ever reach it."""
    if index < 0:
        return str(index)
    label = ""
    n = index
    while True:
        label = chr(65 + (n % 26)) + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


def translate_finger(finger: ImportedFinger, dx: float, dy: float) -> None:
    """Rigid translation of one finger by a fixed (dx, dy) -- shifts its
    polygon, stub_end, patch_end, and (if already computed) its patch-
    group outward/inward ends, all by the same offset. Length, width,
    and every angle are unchanged by construction, since a pure
    translation can't affect them. Used for the Layers dock's batch
    Nudge, which moves every shape in a layer together; call
    recompute_derived_fields() on the whole junction afterward (once,
    after the whole batch) to refresh anything that depends on position
    relative to the electrode or to other shapes."""
    finger.points = [(x + dx, y + dy) for x, y in finger.points]
    finger.stub_end = (finger.stub_end[0] + dx, finger.stub_end[1] + dy)
    finger.patch_end = (finger.patch_end[0] + dx, finger.patch_end[1] + dy)
    if finger.patch_outward_end is not None:
        finger.patch_outward_end = (finger.patch_outward_end[0] + dx, finger.patch_outward_end[1] + dy)
    if finger.patch_inward_end is not None:
        finger.patch_inward_end = (finger.patch_inward_end[0] + dx, finger.patch_inward_end[1] + dy)


def translate_shape(shape: ImportedShape, dx: float, dy: float) -> None:
    """Rigid translation of one patch/plate by a fixed (dx, dy) -- see
    translate_finger's docstring; same idea, simpler object (just a
    points list, no stub/patch ends to carry along)."""
    shape.points = [(x + dx, y + dy) for x, y in shape.points]


def apply_patch_group_margin_edit(patches: List[ImportedShape], finger: "ImportedFinger",
                                   delta_margin_um: float) -> None:
    """Shifts every patch in one finger's patch group along that finger's
    own axis by a uniform amount, changing the group's shared "tip margin"
    (how much finger/plate material survives beyond the outermost patch)
    by exactly delta_margin_um -- matches DesignParameters.patch_tip_margin_um
    being one shared value per finger, not a per-patch one, so an edit
    here always applies to the whole group at once, never a single patch.
    Increasing the margin moves the whole group AWAY from the finger's
    outward end (back toward the crossing), matching
    parametric_junction_view.build_side_patches()'s own offsets formula, which
    subtracts tip_margin uniformly from every patch's along-finger offset."""
    theta_rad = math.radians(finger.axis_angle_deg)
    fx, fy = math.cos(theta_rad), math.sin(theta_rad)
    dx, dy = -fx * float(delta_margin_um), -fy * float(delta_margin_um)
    for shape in patches:
        translate_shape(shape, dx, dy)


def measurements_by_stable_id(junction: "ImportedJunction", report: Optional["MeasurementReport"]) -> Dict[str, dict]:
    """Maps each finger's/patch's/plate's stable_id to a small dict of its
    own REAL, already-computed measurements from `report` -- the exact
    same numbers already shown when "Measurements: On" is toggled -- so
    the Shape Properties card (main_gui.py) can pre-fill its
    role-specific fields with real values the moment a shape is clicked,
    instead of design defaults. Built once per compute_measurements()
    call (see _update_preview's call sites), not recomputed per click.

    For a finger: length_um, width_um, theta_toward_patch_deg,
    warning_source, and overlap_um -- this finger's overlap with the
    OTHER finger via finger_overlap_um(), meaningful only when exactly 2
    fingers exist (None otherwise).

    For a patch: near_overhang_um / far_overhang_um (the short/long-
    extension equivalents -- PatchGroupMeasurement's own docstring
    confirms these are pure geometry, so they work identically for a
    real GDS import with no DesignParameters at all), its own width_um/
    length_um, the stable_id of the finger it belongs to, and (on the
    OUTERMOST patch of each group only, matching
    finger_tip_past_last_patch_um's own convention) group_tip_past_last_patch_um.

    For a plate: length_um/width_um straight off the ImportedShape, plus
    the stable_id of the finger it's attached to (auto-detected by
    nearest patch_end, same as _find_finger_plate).

    Returns {} if report is None."""
    out: Dict[str, dict] = {}
    if report is None:
        return out

    fingers = junction.fingers
    for i, f in enumerate(fingers):
        overlap_um = finger_overlap_um(f, fingers[1 - i]) if len(fingers) == 2 else None
        out[f.stable_id] = dict(
            kind="finger", length_um=f.length_um, width_um=f.width_um,
            theta_toward_patch_deg=f.theta_toward_patch_deg,
            warning_source=f.warning_source, overlap_um=overlap_um,
        )

    # Re-derive the SAME per-group shape ordering compute_measurements()
    # itself used (identical inputs, identical sort key) so this can zip
    # each shape's stable_id against report.patch_groups' already-computed
    # individual_near_overhang_um/individual_far_overhang_um arrays
    # without re-deriving those numbers itself.
    groups = _assign_shapes_to_fingers(junction)
    for pg in report.patch_groups:
        f = fingers[pg.finger_index]
        shapes = groups.get(pg.finger_index, [])
        fx, fy = f.patch_end[0] - f.stub_end[0], f.patch_end[1] - f.stub_end[1]
        flen = math.hypot(fx, fy) or 1.0
        ux, uy = fx / flen, fy / flen

        def pos_along_axis(shape):
            cx = sum(pt[0] for pt in shape.points) / len(shape.points)
            cy = sum(pt[1] for pt in shape.points) / len(shape.points)
            return (cx - f.stub_end[0]) * ux + (cy - f.stub_end[1]) * uy

        shapes_sorted = sorted(shapes, key=pos_along_axis)
        n = len(shapes_sorted)
        for j, shp in enumerate(shapes_sorted):
            out[shp.stable_id] = dict(
                kind="patch", finger_stable_id=f.stable_id,
                near_overhang_um=pg.individual_near_overhang_um[j] if j < len(pg.individual_near_overhang_um) else None,
                far_overhang_um=pg.individual_far_overhang_um[j] if j < len(pg.individual_far_overhang_um) else None,
                width_um=shp.width_um, length_um=shp.length_um,
                group_tip_past_last_patch_um=pg.finger_tip_past_last_patch_um if j == n - 1 else None,
            )

    for f in fingers:
        plate = _find_finger_plate(junction, f)
        if plate is not None:
            out[plate.stable_id] = dict(kind="plate", finger_stable_id=f.stable_id,
                                        length_um=plate.length_um, width_um=plate.width_um)
    return out


def recompute_derived_fields(junction: ImportedJunction) -> None:
    """Call after apply_finger_edit()/apply_shape_edit() (once per batch
    of edits, not once per shape) to refresh every field that depends on
    the edited geometry's relationship to the electrode and to nearby
    patches: warning_source, finger_overlap_frac, patch_group_overlap_frac,
    and the patch group's own outward/inward axis + theta. Deliberately
    reuses _assign_warning_sources exactly as-is (the same function the
    GDS-import and parametric-design paths both already call) rather than
    duplicating its classification math -- it's already a pure function
    of the junction's current fingers/other_shapes/electrode geometry, not
    of how that geometry was produced, so it's safe to re-run after an
    in-place edit. compute_measurements() (which independently re-derives
    patch groupings and every overhang/contact-area/spacing measurement
    from the current geometry on every call) still needs to be called
    separately by the caller, same as always."""
    electrode_union = None
    if junction.electrode_polygons:
        electrode_union = so.unary_union([sg.Polygon(p) for p in junction.electrode_polygons if len(p) >= 3])
    _assign_warning_sources(junction.fingers, junction.other_shapes, electrode_union)


def compute_deposit_group(fingers: List["ImportedFinger"], first_index: int) -> set:
    """Pure computation, factored out of apply_deposit_order (see its own
    docstring for the full physical reasoning this implements): the set
    of finger indices that deposit in the SAME physical pass as
    `fingers[first_index]` -- same axis (mod 180 deg, tolerance 1 deg)
    AND no direct JJ crossing with another member of that axis group
    (two same-axis fingers that actually cross, e.g. a classic
    anti-parallel Dolan-bridge pair, must never share one un-isolated
    pass regardless of axis).

    Factored out (no behavior change) so
    enumerate_deposit_order_options below can reuse the identical logic
    to ENUMERATE every possible "which group deposits first" choice for
    the Deposition Order panel, not just apply one already-chosen
    index's consequences."""
    if not fingers or not (0 <= first_index < len(fingers)):
        return set()
    first_axis_deg = fingers[first_index].theta_toward_patch_deg % 180.0

    def _same_axis(theta_deg):
        d = abs((theta_deg % 180.0) - first_axis_deg)
        d = min(d, 180.0 - d)
        return d < 1.0

    group_idxs = {i for i, f in enumerate(fingers)
                  if i == first_index or _same_axis(f.theta_toward_patch_deg)}

    crossing_set = set()
    for a, b in find_all_finger_crossings(fingers):
        crossing_set.add((a, b))
        crossing_set.add((b, a))

    # Drop any non-first_index group member that actually crosses
    # another member currently in the group -- two fingers that
    # physically overlap form a real JJ and must never be merged
    # into one shared, unisolated pass, even when they lie on the
    # same axis line (the classic anti-parallel 2-finger case above).
    for i in list(group_idxs):
        if i == first_index:
            continue
        if any((i, j) in crossing_set for j in group_idxs if j != i):
            group_idxs.discard(i)
    return group_idxs


def enumerate_deposit_order_options(fingers: List["ImportedFinger"]) -> List[Tuple[int, set]]:
    """Enumerates the distinct "which finger(s) deposit first" choices
    for the Deposition Order panel, generalized for series-chain /
    layered-Manhattan designs where more than one finger genuinely
    shares a single physical deposition pass.

    For Manhattan-in-series junctions,
    multiple fingers parallel to each other deposit in a given
    deposition step, so the button needs to reflect that: it can't
    just be "finger A deposits first", it needs to detect the fingers
    grouped together in the same deposition step, and give the user the
    option of what to deposit first from that group. Before this, the panel
    always offered exactly "Finger A" / "Finger B" regardless of how
    many fingers actually existed, so a 3rd/4th imported finger sharing
    Finger A's own axis (per apply_deposit_order's own "layered
    Manhattan junction array" generalization) was silently dragged
    along with no indication in the UI at all, AND had no way to be
    picked independently if the user actually wanted to start from it.

    Returns a list of (representative_index, member_index_set) pairs,
    one per DISTINCT group (deduped by member set, since e.g. a 4-finger
    chain's Finger A and Finger D share one axis and produce the exact
    same group either way -- see compute_deposit_group). `representative_
    index` is simply the smallest index in the group; picking it (or any
    other member -- the result is identical either way, which is exactly
    why the groups can be deduped in the first place) as apply_deposit_
    order's own `first_index` argument reproduces this identical group.

    A classic 2-finger design (perpendicular OR anti-parallel) always
    produces exactly the original two singleton options ("Finger A" /
    "Finger B"), byte-for-byte the same choices the panel already
    offered before this generalization -- this only ever adds options
    or merges an existing one across more fingers, never removes the
    classic 2-finger behavior."""
    seen_order: List[frozenset] = []
    seen_reps = {}
    for i in range(len(fingers)):
        group = frozenset(compute_deposit_group(fingers, i))
        if not group:
            continue
        if group not in seen_reps:
            seen_reps[group] = min(group)
            seen_order.append(group)
    return [(seen_reps[g], set(g)) for g in seen_order]


def apply_deposit_order(junction: ImportedJunction, first_index: int) -> None:
    """Sets Warning #2 (self-shadowing) info once the deposition order is
    known -- this can't be inferred from geometry alone, unlike Warning
    #1, so it's only computed when explicitly told which finger goes
    first (mutates junction.fingers in place).

    Physics: the finger deposited SECOND crosses over the first finger's
    already-deposited step. Depositing toward its own stub end (away
    from the patch/far end) lands any resulting thinning on the
    harmless safety-overlap side rather than the current-carrying side
    at the actual JJ crossing -- same reasoning already used for the
    parametric design's self_shadow_safe_theta (recipe_generator.py), now
    available for imports too.

    Generalized for series-chain designs (a layered Manhattan junction
    array): a finger genuinely shares
    `first_index`'s own deposit group -- and therefore its own physical
    deposition pass -- whenever it's drawn on the SAME axis (parallel,
    mod 180 degrees), following the design rule that any new
    junction finger added must be either parallel to finger A or
    perpendicular to finger B. Before this fix, EVERY finger except
    `first_index` itself was unconditionally marked deposited_second=True,
    regardless of its own axis -- correct for a classic 2-finger design
    (the other finger is always perpendicular anyway, so this changes
    nothing there), but wrong for a 3rd+ finger sharing `first_index`'s
    own axis, which physically deposits in the SAME pass, not a second,
    separate one.

    Fixed via a regression sweep against the
    classic 2-finger case: "same axis, mod 180" alone is NOT enough to
    say two fingers share one physical pass. A classic Dolan-bridge
    2-finger design is built from exactly two ANTI-PARALLEL fingers
    (theta 0 deg / 180 deg) that meet and physically CROSS/overlap in the
    middle -- that's what makes it a junction at all. Anti-parallel
    fingers sit on the exact same drawn line, so the naive axis test
    above wrongly recognized them as "parallel, share one pass," merging
    them into a single group with a single shared angle and silently
    dropping the required second pass and the oxide barrier between them
    (verified to turn a genuine, intact 2-finger design's
    circuit-continuity verdict from "OK" into "OPEN + SHORTED"). The
    real, physically-correct rule: two same-axis fingers may only share
    one simultaneous pass if they do NOT themselves form a JJ crossing
    with each other -- fingers that actually cross must always get
    separate passes with isolation between them, on axis or not. A
    series-chain's own axis-mates (e.g. Finger B and Finger C, both
    vertical but at different x positions) never cross each other
    directly (they only connect via their perpendicular chain
    neighbors), so they're unaffected by this exclusion and still
    correctly merge into one shared pass.

    Now a thin wrapper around compute_deposit_group (see its
    own docstring) -- identical logic, factored out so
    enumerate_deposit_order_options can enumerate every possible choice,
    not just apply one already-chosen index's consequences. No behavior
    change here."""
    fingers = junction.fingers
    group_idxs = compute_deposit_group(fingers, first_index)

    for i, f in enumerate(fingers):
        f.deposited_second = not (i in group_idxs)
        f.self_shadow_safe_theta_deg = f.theta_toward_stub_deg if f.deposited_second else None


def classify_fingers(raw_polygons: List[List[Tuple[float, float]]], electrode_union=None) -> ImportedJunction:
    """
    raw_polygons: list of point-lists, all on the JJ layer, for ONE junction.

    Splits them into fingers (long, roughly-square bounding box because
    they're usually drawn diagonally -- OR clearly the two longest/widest
    shapes) vs. patches/plates (everything else), then for each finger
    determines which end is the stub (bare, short -- the JJ-crossing side)
    and which is the patch/plate end (electrode side), using proximity of
    OTHER shapes to each end -- validated against real fabrication
    failure data (this heuristic correctly gives safe=225/unsafe=45 deg,
    matching a documented open-junction failure; a raw electrode-layer
    boolean overlap check gave the WRONG answer on the same data, because
    the electrode layer blankets almost the whole chip).
    """
    shapely_polys = [sg.Polygon(p) for p in raw_polygons if len(p) >= 3]

    # Default: the shapes with the largest long-axis length. In every
    # BARE (non-patch-integrated) design seen so far there are exactly 2
    # (a Manhattan crossing), and this default doesn't hardcode that --
    # it picks the 2 longest, and anything else is a patch/plate.
    scored = []
    for i, pts in enumerate(raw_polygons):
        axis_angle, end_a, end_b, width = _polygon_axis_and_ends(pts)
        length = ((end_a[0] - end_b[0]) ** 2 + (end_a[1] - end_b[1]) ** 2) ** 0.5
        scored.append((length, i, axis_angle, end_a, end_b, width))
    scored.sort(reverse=True, key=lambda t: t[0])

    finger_count = 2 if len(scored) >= 2 else len(scored)
    finger_idxs = {s[1] for s in scored[:finger_count]}

    # Fixed after a report from a real, patch-integrated GDS import: the
    # length-based default above
    # silently breaks for any patch-integrated design whose patches'
    # geometric overhang happens to be LONGER than the finger's own bare
    # stub-to-crossing length -- exactly what patch integration does on
    # purpose. It picked the two longest PATCHES as "the fingers," and
    # the real, shorter fingers as patches -- a completely wrong result
    # with no warning. Cross-checked against a length-INDEPENDENT
    # physical signature instead: the real fingers are the one pair of
    # shapes that actually crosses (or, for a deliberate negative-overlap
    # "gap" design, nearly crosses) each other -- see
    # _find_crossing_finger_pair's own docstring for why no patch/plate
    # ever does this. Only OVERRIDES the length-based default when the
    # two disagree, so an already-correct design (the common case, where
    # the longest-2 pick and the crossing pick agree) is completely
    # unaffected -- this is a targeted correction for the reported
    # failure mode, not a wholesale rewrite of an already-validated
    # heuristic (kept as the fallback whenever no plausible crossing pair
    # is found at all, e.g. a single isolated finger with nothing to
    # cross).
    if finger_count == 2:
        by_index = {s[1]: s for s in scored}
        # Fixed after being reported and reproduced against a
        # real GDS import with deliberately narrower adhesion plates
        # than its own fingers (a deliberate test of whether the
        # connection between the patches and the adhesion plates was
        # creating false resistances): the length-based default AND the old any-angle
        # crossing-pair override both picked a coincidentally-
        # overlapping, NON-perpendicular pair of patches (one on each
        # real finger's own 45/135-degree axis, crossing each other at
        # 45 degrees with a bigger overlap area than the true fingers'
        # own crossing) as "the fingers" -- the real, thinner, actually-
        # perpendicular fingers got filed as patches. Since a shape
        # crossing another feature at exactly 90 degrees is the
        # signature of both being junction fingers,
        # a Manhattan crossing's perpendicularity
        # is a STRONGER, more specific physical signature than "crosses
        # at some non-parallel angle" -- checked FIRST, before the
        # length-based default or the old unrestricted-angle override,
        # so it can never lose to a coincidental non-perpendicular
        # overlap. Falls through to the existing logic unchanged whenever
        # no plausible perpendicular pair exists at all (e.g. a single
        # isolated finger, or -- deliberately unsupported here -- a
        # genuinely non-Manhattan, non-perpendicular junction design),
        # so an already-correct design is completely unaffected.
        perpendicular_pair = _find_crossing_finger_pair(
            list(range(len(raw_polygons))), shapely_polys, by_index, require_perpendicular=True)
        if perpendicular_pair is not None:
            finger_idxs = set(perpendicular_pair)
        default_pair = tuple(sorted(finger_idxs))
        default_key = _pair_crossing_key(default_pair[0], default_pair[1], shapely_polys, by_index)
        # Fixed after testing against a real, densely-packed patch-
        # integrated array design: the crossing-pair search used to run
        # unconditionally and swap in whichever pair scored highest by
        # raw overlap area -- but a design can have a reinforcement/
        # contact shape deliberately overlapping a finger's tip by MORE
        # area than the real fingers' own (intentionally minimized)
        # crossing with each other, so the "better-scoring" pair it found
        # was, in that file, wrong -- it clobbered an ALREADY-CORRECT
        # length-based default (the two longest shapes there really were
        # the two real fingers, and they really do cross). Only run the
        # search -- and only accept its answer -- when the default pair
        # itself fails to demonstrate a real crossing at all (that's what
        # the patch-integrated bug this was originally built for
        # looks like: the two longest shapes are patches that share their
        # parent finger's own axis, not two independently-crossing
        # shapes). An already-valid default is trusted and left alone.
        if default_key is None:
            crossing_pair = _find_crossing_finger_pair(list(range(len(raw_polygons))), shapely_polys, by_index)
            if crossing_pair is not None:
                finger_idxs = set(crossing_pair)

    # Build every finger FIRST, in its own
    # pass, before touching any non-finger shape -- so the "is this shape
    # on the same axis as, and directly attached to, an already-known
    # finger" check below (the spec for what makes something
    # an adhesion plate, as opposed to a patch) always has the COMPLETE
    # finger list available, regardless of which shape happens to sort
    # first by length. A single combined loop (the previous structure)
    # only worked by the coincidence that fingers are usually the two
    # longest shapes and so usually got appended first; a design where a
    # patch happens to be longer than a finger would have silently seen
    # an empty `fingers` list at that point instead.
    fingers, others = [], []
    finger_entries = [s for s in scored if s[1] in finger_idxs]
    other_entries = [s for s in scored if s[1] not in finger_idxs]

    for length, i, axis_angle, end_a, end_b, width in finger_entries:
        pts = raw_polygons[i]
        shp = shapely_polys[i]
        bb = shp.bounds

        # Fixed previously: only consider patches/
        # plates (not the OTHER FINGER) when locating the patch-
        # facing end. Proximity to the other finger specifically
        # means "near the crossing" -- the stub side, by definition
        # -- not the patch side. For a design with no patches/plates
        # at all (bare 2-finger crossing), falls back to distance
        # from the OTHER FINGER, INVERTED (the stub/crossing side is
        # NEAR it). Extracted into _derive_finger_ends_and_angles so
        # promote_shape_to_finger (the Mark-as-Finger correction
        # tool) uses this exact same formula, not a re-derived one
        # that could silently disagree.
        other_shape_polys = [shapely_polys[j] for j in range(len(shapely_polys)) if j not in finger_idxs]
        other_finger_polys = [shapely_polys[j] for j in finger_idxs if j != i]
        patch_end, stub_end, theta_toward_patch, theta_toward_stub, nearby_count = \
            _derive_finger_ends_and_angles(end_a, end_b, length, other_shape_polys, other_finger_polys)

        fingers.append(ImportedFinger(
            points=pts, length_um=length, width_um=width, axis_angle_deg=axis_angle,
            stub_end=stub_end, patch_end=patch_end,
            theta_toward_patch_deg=theta_toward_patch, theta_toward_stub_deg=theta_toward_stub,
            nearby_patch_count=nearby_count,
        ))

    # Validated against a real design with deliberately narrower
    # adhesion plates than its own fingers, so
    # width alone can no longer distinguish a plate from a patch: the
    # adhesion plates are classified as the shapes directly attached to
    # the junction fingers that use the same axial angles as the fingers
    # (thus are deposited simultaneously with the fingers). Checked
    # FIRST, ahead of the old width threshold -- a shape sharing its own
    # finger's exact rectangle axis (within ANGLE_TOL_DEG) AND physically
    # touching it (within TOUCH_TOL_UM) is a plate regardless of width;
    # verified against the real file's ground truth (true plates: axis
    # matches their finger to within 0.01 degrees, distance 0 -- true
    # patches: a wholly unrelated axis, and never touching). The old
    # width>0.6um rule is kept as the fallback for shapes with no such
    # attachment (e.g. a wide plate on its own separate feed line, not
    # touching any finger) so existing designs are unaffected.
    ANGLE_TOL_DEG = 2.0
    TOUCH_TOL_UM = 0.05
    for length, i, axis_angle, end_a, end_b, width in other_entries:
        pts = raw_polygons[i]
        shp = shapely_polys[i]
        bb = shp.bounds

        # Fixed here: this recomputed width/length from
        # the axis-aligned bounding box (bb), completely ignoring the
        # ALREADY-CORRECT length/width this same loop iteration
        # computed above via _polygon_axis_and_ends -- which is the
        # right method (used for fingers, verified repeatedly
        # elsewhere) since it works in the shape's own rotated frame.
        # A bounding box is only correct for axis-aligned rectangles;
        # for anything rotated (patches/plates on a diagonal finger,
        # the common case) it mixes both true side lengths together
        # via trig. This was observed on a real plate with true sides
        # 1.199 x 4.000, rotated 45 degrees, which produced a bounding box
        # width of EXACTLY length*cos(45)+width*sin(45) = 3.677 for
        # BOTH reported dimensions -- matching neither real side
        # length at all, and only being caught because the drawn
        # dimension line's odd diagonal position in the overlay made
        # it visibly wrong rather than silently wrong.
        same_axis_attached = False
        for f in fingers:
            raw_diff = abs(axis_angle - f.axis_angle_deg) % 180.0
            angle_diff = min(raw_diff, 180.0 - raw_diff)
            if angle_diff <= ANGLE_TOL_DEG:
                finger_poly = sg.Polygon(f.points)
                if shp.distance(finger_poly) <= TOUCH_TOL_UM:
                    same_axis_attached = True
                    break
        if same_axis_attached:
            kind = "plate"
        else:
            kind = "plate" if width > 0.6 else "patch"  # thin (<0.6um) = patch; wider = adhesion plate
        others.append(ImportedShape(points=pts, kind=kind, width_um=width, length_um=length))

    _assign_stable_ids(fingers, others)
    _assign_warning_sources(fingers, others, electrode_union)

    return ImportedJunction(fingers=fingers, other_shapes=others, source_label="")


# ============================================================================
# Path 1: real .gds file import
# ============================================================================

def inspect_gds_layers(path: str):
    """Returns [(layer, datatype, polygon_count, avg_size_um), ...] across
    the WHOLE library (all cells, un-flattened polygon counts) so the user
    can pick which layer is the junction layer -- there's no universal
    constant (confirmed: it's whatever the e-beam tool happens to be
    configured to), so this is always a user choice, never auto-assumed."""
    lib = gdstk.read_gds(path)
    counts = {}
    sizes = {}
    for cell in lib.cells:
        for p in cell.polygons:
            key = (p.layer, p.datatype)
            counts[key] = counts.get(key, 0) + 1
            bb = p.bounding_box()
            sizes.setdefault(key, []).append(max(bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]))
    out = []
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        avg = sum(sizes[key]) / len(sizes[key])
        out.append((key[0], key[1], n, avg))
    return out


def list_gds_cells_on_layer(path: str, layer: int, datatype: int = 0, min_shapes=2, max_shapes=12):
    """Cells that plausibly hold ONE junction on the given layer: a small
    number of shapes, all on that layer (content-based, not name-based --
    naming conventions vary a lot across these files, e.g.
    'PatchInt_Row0_Col0_80nm' vs 'JJ_Matrix_r0_c0_L60_w1_80nm')."""
    lib = gdstk.read_gds(path)
    out = []
    for cell in lib.cells:
        polys = [p for p in cell.polygons if p.layer == layer and p.datatype == datatype]
        if min_shapes <= len(polys) <= max_shapes and len(polys) == len(cell.polygons):
            bb = cell.bounding_box()
            if bb is not None:
                out.append((cell.name, len(polys), bb))
    return out


class ElectrodeBase:
    """Cached, whole-chip electrode geometry in TRUE WORLD/ABSOLUTE
    coordinates -- computed ONCE per file, then reused for every junction
    cell imported from it. This is what "automatically import Pads_v4_
    Manhattan (or its equivalent) first" means in practice: found and
    loaded the moment the GDS file is opened, with NO layer number or any
    other input from the user.

    Two geometry sets are kept, both in world coordinates:
      - material_world: the actual drawn/filled shapes (for background
        display, so you can see the real Pads layout behind the junction)
      - holes_world: the ABSENT-SPACE regions -- the hollow rectangle
        NOT covered by each bracket-shaped piece of material (this is
        the electrode, for warning/overlap purposes)

    Also tracks the OVERALL bounding box of all material -- at the outer
    edge of the whole array, there's no next bracket to form a hole
    against, but everything beyond the pattern's outer edge is still
    electrode (on the outside of a Pads_v4_Manhattan-style array, it is
    all electrode, so the outermost junctions also have a complete
    connection), so that region needs to be synthesized rather than
    found as a hole in any single polygon.
    """
    def __init__(self, material_world: List["sg.Polygon"], holes_world: List["sg.Polygon"],
                 hole_bearing_material: Optional[List["sg.Polygon"]] = None,
                 convention: Optional[ElectrodeConvention] = None):
        self.material_world = material_world
        self.holes_world = holes_world
        self.convention = convention or DEFAULT_ELECTRODE_CONVENTION
        self._material_union_cache = None  # lazily computed once, reused across every
                                             # nearby_holes() call on this same base --
                                             # unary_union over potentially thousands of
                                             # whole-chip material polygons is expensive
                                             # enough that recomputing it per-junction-
                                             # import would be a real performance hit

        if not self.convention.array_edge_extension:
            self.overall_bounds = None
            self._main_cluster_windows = []
            return

        # overall_bounds must come from only the LARGE electrode windows
        # (area > min_electrode_area_um2) that form the actual repeating
        # array -- there are two separate contamination sources: (1) thin
        # vertical connector strips between sites have their own tiny
        # real notch, but THEIR OWN polygon bounding box spans nearly the
        # full chip height (they're tall by design); (2) corner
        # alignment marks (e.g. EBL registration marks) can coincidentally
        # have a bracket-like notch close in size to a real window. Both
        # get excluded here: the area filter handles (1), and connected-
        # components clustering on hole centroid position handles (2) --
        # keeps the dense uniform grid and drops isolated far-flung marks.
        big_holes = [h for h in holes_world if h.area >= self.convention.min_electrode_area_um2]
        main_cluster = self._find_main_cluster(big_holes) if big_holes else []
        self._main_cluster_windows = main_cluster  # cleaned real windows -- used for the per-direction edge check
        bounds_source = main_cluster if main_cluster else (hole_bearing_material or material_world)
        if bounds_source:
            xs0, ys0, xs1, ys1 = zip(*(m.bounds for m in bounds_source))
            self.overall_bounds = (min(xs0), min(ys0), max(xs1), max(ys1))
        else:
            self.overall_bounds = None

    def _material_union(self):
        if self._material_union_cache is None:
            self._material_union_cache = so.unary_union(self.material_world) if self.material_world else None
        return self._material_union_cache

    @staticmethod
    def _find_main_cluster(polys, connect_dist: float = 1000.0):
        """Connected-components clustering on polygon centroids (any two
        within connect_dist of each other are linked) -- returns only the
        LARGEST connected cluster. This is what actually distinguishes
        "the real repeating array" from isolated contamination:
        corner alignment marks can coincidentally be bracket-shaped with
        a big-enough notch to pass an area filter, but they sit in
        isolation, far from the dense main body -- an IQR/percentile
        check on position doesn't reliably catch this when the real array
        itself already spans nearly the full chip, but a density/
        connectivity check does regardless of overall spread)."""
        n = len(polys)
        if n == 0:
            return []
        centroids = [p.centroid for p in polys]
        visited = [False] * n
        best_cluster: List[int] = []
        for i in range(n):
            if visited[i]:
                continue
            cluster = []
            queue = deque([i])
            visited[i] = True
            while queue:
                idx = queue.popleft()
                cluster.append(idx)
                for j in range(n):
                    if not visited[j] and centroids[idx].distance(centroids[j]) <= connect_dist:
                        visited[j] = True
                        queue.append(j)
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        return [polys[i] for i in best_cluster]

    def nearby_holes(self, world_pt, radius: Optional[float] = None, min_area: float = 10.0):
        """Hole polygons (world coords) within a generous FIXED radius of
        world_pt, PLUS the electrode "shell" -- everywhere OUTSIDE the
        array's clean overall bounds (self.overall_bounds, computed from
        the main connected cluster of real windows -- see
        _find_main_cluster), like a donut around the array. Only added
        when self.convention.array_edge_extension is True -- off by
        default for non-array (single-design) files, since "outside the
        array" is meaningless without a repeating array to be outside of.

        Computed as crop.difference(array_bounds) -- a direct polygon
        subtraction -- so the shell's inner edge is ALWAYS exactly the
        array's true outer boundary, never anchored to world_pt itself.
        An earlier version anchored the extension AT world_pt, which let
        it reach inward past the true boundary and paint large stretches
        of the array's own interior as electrode -- verified to be wrong."""
        radius = radius if radius is not None else self.convention.search_radius_um
        candidates = [h for h in self.holes_world if h.area >= min_area and h.distance(world_pt) <= radius]

        if self.convention.array_edge_extension and self.overall_bounds is not None:
            ox0, oy0, ox1, oy1 = self.overall_bounds
            # Pad the array's own window bounds outward by a small,
            # fixed standoff before treating anything beyond it as
            # electrode. Edge-column JJ CROSSINGS
            # legitimately sit ~10um outside the real windows' own tight
            # bounding box (every crossing keeps a standoff from the
            # nearest electrode structure, same as interior columns --
            # it's just that for edge columns, "nearest structure" is the
            # array's own outer boundary). Without this pad, the raw
            # window bbox swallowed the crossing point itself as
            # "electrode" for every edge column. The patches/finger tip
            # reach further out than this pad, so they still correctly
            # register a partial crossing.
            pad = self.convention.standoff_pad_um
            ox0, oy0 = ox0 - pad, oy0 - pad
            ox1, oy1 = ox1 + pad, oy1 + pad
            crop = sg.box(world_pt.x - radius * 2, world_pt.y - radius * 2,
                          world_pt.x + radius * 2, world_pt.y + radius * 2)
            array_box = sg.box(ox0, oy0, ox1, oy1)
            outside = crop.difference(array_box)
            # Fixed here: this shell was claiming "outside
            # the array = electrode" without checking whether that space
            # is actually covered by real drawn chip material -- for an
            # edge-column junction whose finger tip genuinely lands ON
            # that material, this produced a false "crosses the
            # electrode" classification (a real, non-zero overlap
            # fraction) even though NO empty electrode space is actually
            # there. In practice the finger tip sits on real
            # material, not absent-space, and the drawn preview already
            # correctly subtracted material before rendering, which is
            # why the calculation and the picture disagreed. The hollow
            # convention means electrode = absent-space; material can
            # never be electrode, regardless of which region it's in.
            material_union = self._material_union()
            if material_union is not None and not outside.is_empty:
                outside = outside.difference(material_union)
            if not outside.is_empty:
                sub = outside.geoms if hasattr(outside, "geoms") else [outside]
                candidates.extend(g for g in sub if g.geom_type == "Polygon" and g.area > 1.0)

        return candidates

    def nearby_material(self, world_pt, radius: float = 600.0):
        """Material polygons (world coords) within `radius` of world_pt --
        for background display only, not for warning calculations."""
        return [m for m in self.material_world if m.distance(world_pt) <= radius]


def find_electrode_cell_name(lib, convention: Optional[ElectrodeConvention] = None) -> Optional[str]:
    """Finds the electrode-defining cell by name, per the given
    convention (or DEFAULT_ELECTRODE_CONVENTION, which reproduces the
    original "Pads_v4_Manhattan, or any cell with 'pad' in its name"
    search exactly)."""
    conv = convention or DEFAULT_ELECTRODE_CONVENTION
    names = [c.name for c in lib.cells]
    if conv.cell_name_exact and conv.cell_name_exact in names:
        return conv.cell_name_exact
    if conv.cell_name_fallback_substring:
        needle = conv.cell_name_fallback_substring.lower()
        for n in names:
            if needle in n.lower():
                return n
    return None


def load_electrode_base(path: str, convention: Optional[ElectrodeConvention] = None) -> Optional[ElectrodeBase]:
    """Flattens the WHOLE chip once and collects the electrode cell's
    geometry in true world coordinates, per the given ElectrodeConvention
    (or DEFAULT_ELECTRODE_CONVENTION, reproducing the exact original
    behavior for 'Pads_v4_Manhattan'-style files -- no layer number
    needed, hollow absent-space is electrode). Also picks up geometry
    drawn directly in the top cell on those same layers -- real designs
    are known to have this (dozens of shapes drawn straight into the top
    cell), invisible to any search that only looks at sub-cell instances.
    Returns None if no electrode-like content is found (not an error --
    some imports genuinely won't have one, or need convention.layer_override
    set explicitly if auto-detection by name doesn't apply)."""
    conv = convention or DEFAULT_ELECTRODE_CONVENTION
    lib = gdstk.read_gds(path)

    if conv.layer_override is not None:
        layers_used = {conv.layer_override}
    else:
        cell_name = find_electrode_cell_name(lib, conv)
        if cell_name is None:
            return None
        electrode_cell = next(c for c in lib.cells if c.name == cell_name)
        layers_used = {(p.layer, p.datatype) for p in electrode_cell.polygons}
        if not layers_used:
            return None

    top = lib.top_level()[0] if lib.top_level() else lib.cells[0]
    flat = top.copy("flat_electrode_base")
    flat.flatten()

    material = []
    holes = []
    hole_bearing_material = []  # only shapes that actually have a notch -- excludes any
                                 # large simple-rectangle decoration sharing the same layer
                                 # (e.g. a chip outline), which would otherwise blow up the
                                 # "overall pattern footprint" to the whole chip and make
                                 # the edge-of-array extension never trigger
    for p in flat.polygons:
        if (p.layer, p.datatype) not in layers_used:
            continue
        poly = sg.Polygon(p.points)
        material.append(poly)
        if conv.convention == "hollow":
            hole = poly.envelope.difference(poly)
            if hole.is_empty:
                continue
            hole_bearing_material.append(poly)
            sub = hole.geoms if hasattr(hole, "geoms") else [hole]
            holes.extend(h for h in sub if h.geom_type == "Polygon")

    if conv.convention == "filled":
        # the drawn material itself IS the electrode -- no hole
        # extraction, unlike this project's own "hollow" convention
        holes = material
        hole_bearing_material = material

    return ElectrodeBase(material, holes, hole_bearing_material, convention=conv)


def _find_cell_world_origin(top_cell, target_name: str, base_x: float = 0.0, base_y: float = 0.0,
                             _visited=None) -> Optional[Tuple[float, float]]:
    """Recursively searches the ENTIRE cell hierarchy (arbitrary nesting
    depth) for an instance of `target_name`, composing translations at
    each level, and returns its true world-space origin. A shallow
    "only check top's direct references" search was found to be wrong for
    files that wrap cells multiple levels deep (e.g. a top-level chip
    cell that references a junction-array hierarchy cell, which in turn
    references the actual JJ
    cell) -- it silently found nothing and left electrode geometry
    unresolved for every junction in those files, even though the
    electrode base itself loaded fine."""
    if _visited is None:
        _visited = set()
    if id(top_cell) in _visited:
        return None
    _visited.add(id(top_cell))
    for ref in getattr(top_cell, "references", []):
        rx, ry = ref.origin
        wx, wy = base_x + rx, base_y + ry
        if getattr(ref.cell, "name", None) == target_name:
            return (wx, wy)
        found = _find_cell_world_origin(ref.cell, target_name, wx, wy, _visited)
        if found is not None:
            return found
    return None


def from_design_parameters(p, include_fingers: bool = True) -> ImportedJunction:
    """Converts a Tab 1 DesignParameters object (recipe_generator.py) into
    the SAME ImportedJunction/ImportedFinger/ImportedShape model the
    GDS/script import paths produce -- built by directly reusing
    JunctionLayout (parametric_junction_view.py), the exact geometry Tab 1 already
    draws, so this can never silently drift out of sync with what's
    shown there. Once converted, a parametric design gets the identical
    warning_source classification, angle arrows, and click-to-explain
    panel that imported designs already have -- one shared engine, not
    two independently-maintained ones.

    Unlike the GDS/script paths, nothing here is guessed from raw
    polygon shapes: length, width, angle, and which polygon is the
    finger/plate/patch are all already known exactly from
    DesignParameters, so this is MORE deterministic than import, not
    less.

    include_fingers=False builds ONLY the electrode geometry, skipping
    finger/patch/plate construction entirely -- for the "electrode must
    be designed first" workflow stage, where no junction should be
    drawn (even a default one) until the electrode is confirmed."""
    from parametric_junction_view import JunctionLayout, WorldFingerGeometry, build_plate_polygon_world, build_side_patches, _rot
    from recipe_generator import electrode_geometry
    import numpy as np

    layout = JunctionLayout(p)

    fingers: List[ImportedFinger] = []
    others: List[ImportedShape] = []
    patch_placement_log: List[dict] = []

    if include_fingers:
        # Resolve any custom "finger"-role shapes FIRST, one per side,
        # before ANY finger/plate/patch geometry gets built. Previously
        # JunctionLayout(p) built plates/patches
        # from the parametric FingerAssembly at `layout = JunctionLayout(p)`
        # above, and a custom finger override only ever replaced the
        # already-finished fingers[] list entry afterward, so adhesion
        # plates/patches never actually followed a hand-drawn finger's
        # real position/rotation/length at all. Resolving overrides here,
        # before layout.asm is consulted for finger/plate/patch geometry,
        # is what makes the fix possible.
        finger_override_by_side: Dict[str, dict] = {}
        for shp in getattr(p, "electrode_shapes", None) or []:
            if shp.role != "finger":
                continue
            world_pts = shp.to_world_polygon()
            axis_angle, end_a, end_b, width = _polygon_axis_and_ends(world_pts)
            shp_cx = sum(pt[0] for pt in world_pts) / len(world_pts)
            shp_cy = sum(pt[1] for pt in world_pts) / len(world_pts)
            # Which side (horizontal/vertical finger) this shape replaces --
            # nearest parametric finger polygon by centroid distance, same
            # heuristic as before, just resolved against the UNTOUCHED
            # parametric geometry instead of an already-partially-replaced
            # fingers[] list.
            side = min(
                ("horizontal", "vertical"),
                key=lambda s: sg.Polygon(layout.asm[s].world_polygon(layout.asm[s].finger_corners_local))
                    .distance(sg.Point(shp_cx, shp_cy)),
            )
            # Classify stub_end (JJ-crossing side) vs patch_end
            # (electrode-facing side) by distance to THIS side's real
            # electrode anchor (electrode_geometry(p, side).cx/cy), not
            # the world origin. Origin-distance was only ever a proxy
            # that happened to work because a purely parametric layout is
            # always centered there -- it breaks the moment a finger can
            # be placed and rotated freely.
            eg = electrode_geometry(p, side)
            d_a = (end_a[0] - eg.cx) ** 2 + (end_a[1] - eg.cy) ** 2
            d_b = (end_b[0] - eg.cx) ** 2 + (end_b[1] - eg.cy) ** 2
            stub_end, patch_end = (end_a, end_b) if d_a >= d_b else (end_b, end_a)
            theta_patch = math.degrees(math.atan2(patch_end[1] - stub_end[1], patch_end[0] - stub_end[0])) % 360
            length_um = math.hypot(patch_end[0] - stub_end[0], patch_end[1] - stub_end[1])
            finger_override_by_side[side] = dict(
                points=world_pts, length_um=length_um, width_um=width, axis_angle_deg=axis_angle,
                stub_end=stub_end, patch_end=patch_end,
                theta_toward_patch_deg=theta_patch, theta_toward_stub_deg=(theta_patch + 180.0) % 360,
            )
            # If two custom shapes both resolve nearest the same side,
            # the later one (in electrode_shapes order) wins -- matches
            # the old list-overwrite-by-index behavior for that edge case.

        for side, asm in layout.asm.items():
            override = finger_override_by_side.get(side)
            if override is not None:
                # Build a WorldFingerGeometry seam that
                # mirrors FingerAssembly's own seam_point_local/
                # tip_point_local exactly -- past the adhesion plate's
                # outer edge when adhesion pads are enabled, otherwise
                # right at this finger's own (real, hand-drawn) patch_end
                # -- so build_side_patches (below) attaches patches to
                # THIS finger's true position/rotation/length, not the
                # hidden parametric one.
                theta_rad = math.radians(override["theta_toward_patch_deg"])
                axis_hat = np.array([math.cos(theta_rad), math.sin(theta_rad)])
                patch_end_arr = np.array(override["patch_end"], dtype=float)
                if p.has_adhesion_pads:
                    seam_world = tuple(patch_end_arr + axis_hat * p.adhesion_plate_height_um)
                else:
                    seam_world = tuple(patch_end_arr)
                geom = WorldFingerGeometry(override["theta_toward_patch_deg"], override["width_um"],
                                           seam_world, override["length_um"])
                finger_pts = override["points"]
                stub_end, finger_patch_end = override["stub_end"], override["patch_end"]
                theta_toward_patch = override["theta_toward_patch_deg"]
                theta_toward_stub = override["theta_toward_stub_deg"]
                axis_angle_deg = override["axis_angle_deg"]
                length_um = override["length_um"]
                width_um = override["width_um"]
                role_override = "finger"
            else:
                geom = asm
                finger_pts = [tuple(pt) for pt in asm.world_polygon(asm.finger_corners_local)]
                stub_end = tuple(asm.world_point((asm.finger_corners_local[0][0], 0.0)))   # inner/bare end (JJ crossing side)
                finger_patch_end = tuple(asm.world_point(asm.tip_point_local))              # outer end (electrode-facing side)
                theta_toward_patch = math.degrees(math.atan2(finger_patch_end[1] - stub_end[1], finger_patch_end[0] - stub_end[0])) % 360
                theta_toward_stub = (theta_toward_patch + 180.0) % 360
                axis_angle_deg = asm.angle_deg
                length_um = getattr(p, f"{side}_length_um")
                width_um = asm.lw
                role_override = None

            patch_angle = p.patch_angle if side == "horizontal" else p.patch_angle + 180.0
            side_patches, log_entries = ([], [])
            if p.patch_integrated:
                # Built via the SAME geometry (override or parametric)
                # this finger uses everywhere else, so patches attach to a hand-drawn finger's
                # real geometry instead of the invisible parametric one.
                side_patches, log_entries = build_side_patches(geom, side, patch_angle, p)

            fingers.append(ImportedFinger(
                points=finger_pts, length_um=length_um, width_um=width_um, axis_angle_deg=axis_angle_deg,
                stub_end=stub_end, patch_end=finger_patch_end,
                theta_toward_patch_deg=theta_toward_patch, theta_toward_stub_deg=theta_toward_stub,
                nearby_patch_count=len(side_patches),
                role_override=role_override,
            ))
            this_finger_idx = len(fingers) - 1

            if p.has_adhesion_pads:
                if override is not None:
                    plate_pts = build_plate_polygon_world(override["patch_end"], theta_toward_patch,
                                                           p.adhesion_plate_width_um, p.adhesion_plate_height_um)
                else:
                    plate_pts = [tuple(pt) for pt in asm.world_polygon(asm.plate_corners_local)]
                others.append(ImportedShape(points=plate_pts, kind="plate",
                                            width_um=p.adhesion_plate_width_um, length_um=p.adhesion_plate_height_um))

            for patch in side_patches:
                cx, cy, w, h, angle = patch["cx"], patch["cy"], patch["w"], patch["h"], patch["angle"]
                R = _rot(angle)
                local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
                pts = [tuple(R @ np.array(c) + np.array([cx, cy])) for c in local]
                # Fixed here: patch["w"] is the LONG dimension
                # (patch_length = 2u + long_extension + short_extension) and
                # patch["h"] is the THIN dimension (the actual linewidth) --
                # this was assigning them backwards (width_um=w, length_um=h),
                # silently wrong since patches were first built this way. It
                # never surfaced because nothing displayed the raw width_um
                # value directly, and the overlap-depth approximation
                # (area/width_um) just produced a differently-scaled number
                # rather than an obviously-wrong one. width_um must be the
                # THIN dimension to match every other convention in this
                # codebase (ImportedFinger.width_um is the finger's own
                # linewidth, not its length).
                others.append(ImportedShape(points=pts, kind="patch", width_um=h, length_um=w,
                                            owner_finger_hint=this_finger_idx))
            patch_placement_log.extend(log_entries)

    # Custom-placed shapes (Rectangle/Circle/Polygon tools + role
    # assignment) become real ImportedShape entries here for every role
    # EXCEPT "finger" -- that role is now fully resolved above, before
    # any finger/plate/patch geometry is built at all (see
    # finger_override_by_side), which is required for custom finger
    # geometry to be respected.
    # Placed BEFORE _assign_warning_sources so plate/patch ones are
    # correctly factored into whichever finger they're nearest to, same
    # as auto-generated ones.
    for shp in getattr(p, "electrode_shapes", None) or []:
        if shp.role in ("plate", "patch", "substrate"):
            others.append(ImportedShape(points=shp.to_world_polygon(), kind=shp.role,
                                        width_um=shp.w, length_um=shp.h, role_override=shp.role))

    # Electrode: prefer the free-form shape-canvas list (electrode_shapes)
    # when populated -- it isn't limited to exactly one horizontal + one
    # vertical rectangle the way the old Electrode Designer fields are.
    # Falls back to the old fixed 2-electrode model exactly as before
    # when electrode_shapes is empty (fully backward compatible).
    if getattr(p, "electrode_shapes", None):
        electrode_polys = [sg.Polygon(shp.to_world_polygon()) for shp in p.electrode_shapes if shp.role == "electrode"]
    else:
        # (Electrode class, recipe_generator.electrode_geometry) -- this is
        # Tab 1's own OLD electrode model, analogous to electrode_polygons/
        # holes on the import side.
        electrode_polys = [sg.Polygon(layout.electrodes[side].corners_world()) for side in ("horizontal", "vertical")]
    electrode_union = so.unary_union(electrode_polys) if electrode_polys else None

    _assign_stable_ids(fingers, others)
    _assign_warning_sources(fingers, others, electrode_union)

    result = ImportedJunction(fingers=fingers, other_shapes=others, source_label="Parametric design (Tab 1)",
                              patch_placement_log=patch_placement_log)
    if electrode_union is not None:
        geoms = electrode_union.geoms if hasattr(electrode_union, "geoms") else [electrode_union]
        result.electrode_polygons = [list(g.exterior.coords) for g in geoms if g.geom_type == "Polygon"]
        result.electrode_material = result.electrode_polygons  # same shapes serve as both here -- there's
                                                                  # no separate "background substrate" layer
                                                                  # in the parametric model the way there is
                                                                  # for an imported chip's Pads_v4_Manhattan
    return result


def import_gds_cell(path: str, layer: int, datatype: int, cell_name: str,
                     electrode_base: Optional[ElectrodeBase] = None,
                     electrode_convention: Optional[ElectrodeConvention] = None) -> ImportedJunction:
    """electrode_base: pass a pre-loaded ElectrodeBase (from
    load_electrode_base, called once per file) to reuse across many
    junction imports from the same file. If not given, it's loaded once
    here using electrode_convention (or DEFAULT_ELECTRODE_CONVENTION,
    which reproduces the original 'Pads_v4_Manhattan' behavior exactly)
    -- no layer number needed for that default case."""
    lib = gdstk.read_gds(path)
    cell = next((c for c in lib.cells if c.name == cell_name), None)
    if cell is None:
        raise ValueError(f"No cell named {cell_name!r} in {path}")

    top = lib.top_level()[0] if lib.top_level() else cell
    origin = _find_cell_world_origin(top, cell_name)
    ox, oy = origin if origin is not None else (None, None)

    # Junction geometry is kept in the SAME true world/absolute
    # coordinates as everything else in the GDS (shift local shape points
    # by this cell's own placement origin) -- this is what makes the
    # electrode's real position line up automatically with NO matching/
    # guessing needed: GDS placement already encodes the true relative
    # geometry.
    shift_x, shift_y = (ox, oy) if ox is not None else (0.0, 0.0)
    raw = [[(pt[0] + shift_x, pt[1] + shift_y) for pt in p.points]
           for p in cell.polygons if p.layer == layer and p.datatype == datatype]
    if not raw:
        raise ValueError(f"Cell {cell_name!r} has no shapes on layer {layer}/{datatype}")

    electrode_union = None
    electrode_polys_out = []
    electrode_material_out = []
    electrode_coords_note = None

    base = electrode_base if electrode_base is not None else load_electrode_base(path, electrode_convention)
    if base is None:
        electrode_coords_note = "No electrode-like cell was found in this file."
    elif ox is None:
        electrode_coords_note = (
            f"Could not resolve {cell_name!r}'s placement -- electrode geometry left as 'unknown'."
        )
    else:
        n_pts = sum(len(poly) for poly in raw)
        jj_world_pt = sg.Point(
            sum(pt[0] for poly in raw for pt in poly) / n_pts,
            sum(pt[1] for poly in raw for pt in poly) / n_pts,
        )
        SEARCH_RADIUS_UM = 600.0
        nearby_holes = base.nearby_holes(jj_world_pt, radius=SEARCH_RADIUS_UM)
        if not nearby_holes:
            electrode_coords_note = "No nearby electrode (absent-space) region was found near this junction."
        else:
            electrode_union = so.unary_union(nearby_holes)
            geoms = electrode_union.geoms if hasattr(electrode_union, "geoms") else [electrode_union]
            electrode_polys_out = [list(g.exterior.coords) for g in geoms if g.geom_type == "Polygon"]

        # background material for display -- same generous radius so the
        # surrounding Pads structure is genuinely visible in context, not
        # just the bare minimum needed for the overlap calculation
        nearby_mat = base.nearby_material(jj_world_pt, radius=SEARCH_RADIUS_UM)
        electrode_material_out = [list(m.exterior.coords) for m in nearby_mat if m.geom_type == "Polygon"]

    result = classify_fingers(raw, electrode_union=electrode_union)
    result.source_label = f"{os.path.basename(path)} :: {cell_name}"
    result.jj_layer = layer
    result.electrode_polygons = electrode_polys_out
    result.electrode_material = electrode_material_out
    if electrode_coords_note:
        result.import_notes = [electrode_coords_note]
    return result


# ============================================================================
# Path 2: run the real KLayout Python script via the pya shim
# ============================================================================

def _patched_exists_false(path):
    return False


def run_klayout_script(source_code: str, function_name: str, call_kwargs: dict,
                        jj_layer: Optional[int] = None) -> ImportedJunction:
    """
    Executes `source_code` in a sandboxed namespace with sys.modules['pya']
    replaced by the gdstk-backed shim, then calls `function_name(**call_kwargs)`
    directly -- bypassing the script's own 100-cell grid loop and template
    import entirely, since only ONE junction's geometry is needed.

    os.path.exists is patched to always return False for the duration of
    exec() so a script's own `if not os.path.exists(template_path): ...`
    guard (present in most of these scripts) skips the missing-template
    read gracefully instead of raising, without needing to modify the
    script at all.
    """
    previous_pya = klayout_pya_shim.install_pya_shim()
    original_exists = os.path.exists
    os.path.exists = _patched_exists_false
    try:
        namespace = {"__name__": "__junction_script__"}
        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(source_code, "<klayout_script>", "exec"), namespace)
        except Exception as e:
            # The 100-cell grid loop / template import may legitimately
            # fail for reasons unrelated to the generator function itself
            # (e.g. a hard os.path.exists check with no guard, or a
            # FileNotFoundError raised explicitly) -- that's fine, we only
            # need `function_name` to have been DEFINED by that point,
            # which happens before the grid loop runs in every script seen
            # so far. Only re-raise if the function never got defined.
            if function_name not in namespace:
                raise RuntimeError(
                    f"Script raised before defining {function_name}(): {type(e).__name__}: {e}"
                ) from e

        if function_name not in namespace:
            raise RuntimeError(f"Script never defined a function named {function_name!r}. "
                               f"Top-level names found: {[k for k in namespace if not k.startswith('__')]}")

        fn = namespace[function_name]

        # Some of these generator functions take the script's own `layout`
        # object as their first argument (e.g. create_jj_cell(layout, name,
        # ...)). Auto-supply it from whatever the script itself created at
        # top level, if the caller didn't already pass one -- the caller
        # can't know that object exists before exec() runs.
        import inspect
        sig_params = list(inspect.signature(fn).parameters)
        effective_kwargs = dict(call_kwargs)
        if "layout" in sig_params and "layout" not in effective_kwargs:
            effective_kwargs["layout"] = namespace.get("layout") or sys.modules["pya"].Layout()

        result_cell = fn(**effective_kwargs)
        if not hasattr(result_cell, "polygons"):
            raise RuntimeError(f"{function_name}() didn't return a Cell-like object with .polygons")

        if jj_layer is not None:
            raw = [pts for (layer, datatype, pts) in result_cell.polygons if layer == jj_layer]
        else:
            # No layer specified -- if everything landed on one layer (the
            # common case for these scripts), use it; otherwise the caller
            # needs to pick.
            layers_used = sorted(set(layer for (layer, _dt, _pts) in result_cell.polygons))
            if len(layers_used) != 1:
                raise ValueError(
                    f"Script wrote shapes to multiple layers {layers_used} -- "
                    f"pass jj_layer explicitly to pick one."
                )
            raw = [pts for (_layer, _dt, pts) in result_cell.polygons]
            jj_layer = layers_used[0]

        result = classify_fingers(raw, electrode_union=None)
        result.source_label = f"script:{function_name}({', '.join(f'{k}={v}' for k, v in call_kwargs.items())})"
        result.jj_layer = jj_layer
        return result
    finally:
        os.path.exists = original_exists
        klayout_pya_shim.uninstall_pya_shim(previous_pya)


def find_generator_functions(source_code: str) -> List[str]:
    """Best-effort list of top-level function names in the script that look
    like they might be the junction-cell generator (returns a Cell), for
    populating a picker if the caller doesn't already know the name."""
    import ast
    tree = ast.parse(source_code)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
    # Prefer ones that look purpose-named, but return all -- the caller decides.
    names.sort(key=lambda n: (0 if "jj" in n.lower() or "junction" in n.lower() else 1, n))
    return names


def get_function_signature(source_code: str, function_name: str) -> List[Tuple[str, Optional[str]]]:
    """Returns [(param_name, default_literal_repr_or_None), ...] for a
    top-level function, via `ast` -- WITHOUT executing the script. Used to
    build parameter-entry fields in the UI before the user commits to
    running anything. Only simple literal defaults (numbers/strings/bools)
    are rendered as text; anything else shows as None (blank field)."""
    import ast
    tree = ast.parse(source_code)
    fn_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name), None)
    if fn_node is None:
        raise ValueError(f"No top-level function named {function_name!r} found.")

    args = fn_node.args.args
    defaults = fn_node.args.defaults  # aligned to the LAST len(defaults) args
    n_no_default = len(args) - len(defaults)
    out = []
    for i, a in enumerate(args):
        default_repr = None
        if i >= n_no_default:
            d = defaults[i - n_no_default]
            try:
                default_repr = repr(ast.literal_eval(d))
            except Exception:
                default_repr = None
        out.append((a.arg, default_repr))
    return out


# ============================================================================
# Measurements -- the specific numbers requested: patch spacing, patch/plate
# overlap into the electrode, finger linewidths, finger/patch lengths.
# ============================================================================

@dataclass
class PatchGroupMeasurement:
    finger_index: int
    patch_count: int
    edge_to_edge_spacings_um: List[float]   # consecutive gaps, in position order along the finger
    individual_lengths_um: List[float]
    individual_widths_um: List[float]
    max_reach_um: Optional[float] = None     # distance from the finger's patch_end to the farthest patch tip
    electrode_overlap_depth_um: Optional[float] = None  # outermost patch only -- kept for backward compat
    individual_overlap_depths_um: List[Optional[float]] = field(default_factory=list)  # EVERY patch,
                                                            # not just the outermost -- a staircase
                                                            # patch layout means each one can overlap
                                                            # the electrode by a genuinely different
                                                            # amount, not just whichever is furthest out
    individual_near_overhang_um: List[float] = field(default_factory=list)  # how far each patch
                                                            # extends past the finger's crossing-side
                                                            # edge (measured from real geometry, not
                                                            # the design parameter -- works identically
                                                            # for imported designs, which have no such
                                                            # parameter at all)
    individual_far_overhang_um: List[float] = field(default_factory=list)   # same, but past the
                                                            # finger's electrode-facing edge
    finger_tip_past_last_patch_um: Optional[Tuple[float, float, float]] = None  # (left_side,
                                                            # right_side, shortest) -- how far the
                                                            # FINGER's own tip extends past the
                                                            # outermost patch, measured separately
                                                            # on each of the finger's two long edges,
                                                            # since the outermost patch's own crossing
                                                            # edge is rarely perpendicular to the
                                                            # finger's axis (patches are typically at
                                                            # 45 deg while fingers are at 0/90), so the
                                                            # two sides commonly differ -- verified
                                                            # against hand
                                                            # ruler measurements on a real import
                                                            # (0.663 / 0.528 um on the two sides).
                                                            # Set only on the OUTERMOST patch's own
                                                            # PatchGroupMeasurement entry conceptually,
                                                            # but stored at the group level since it's
                                                            # a property of "the last patch", not each one.
    individual_overlap_area_um2: List[float] = field(default_factory=list)  # actual contact
                                                            # AREA between each patch and whatever it
                                                            # attaches to (finger and/or plate) -- the
                                                            # physically relevant quantity for contact
                                                            # resistance, distinct from the LINEAR
                                                            # overhang measurements above. Too small
                                                            # (or too little relative to the junction's
                                                            # own area) is a real resistance concern.
    total_overlap_area_um2: Optional[float] = None  # sum across every patch in this group


@dataclass
class MeasurementReport:
    source_label: str
    jj_crossing_area_um2: Optional[float]
    total_exposed_area_um2: Optional[float]        # real write/exposed area (union of ALL shapes -- no double-counting)
    naive_summed_area_um2: Optional[float]          # sum of each shape's own area, WOULD double-count overlaps
    finger_lengths_um: List[float]
    finger_widths_nm: List[float]
    finger_theta_toward_patch_deg: List[float]
    finger_theta_toward_stub_deg: List[float]
    finger_warning_source: List[str]                 # "finger" | "patches" | "both" | "none" | "unknown", per finger
    finger_explanations: List[str]                  # human-readable, ready for a click-to-expand UI
    patch_groups: List[PatchGroupMeasurement]
    finger_electrode_overlap_depth_um: List[Optional[float]] = field(default_factory=list)  # per
                                                            # finger -- how far THIS finger itself
                                                            # (not its patches) penetrates the
                                                            # electrode, when it's the one making
                                                            # direct contact rather than patches
    finger_tip_shortest_overhang_um: List[Optional[float]] = field(default_factory=list)  # per
                                                            # finger -- the finger's own tip has TWO
                                                            # corners; if the electrode's boundary
                                                            # isn't perpendicular to the finger's own
                                                            # axis, those corners penetrate the
                                                            # electrode by genuinely different depths.
                                                            # This is the smaller (weaker) of the two --
                                                            # the corner most at risk of not actually
                                                            # connecting even when the aggregate/area
                                                            # overlap looks fine.
    finger_crossing_overlap_um: List[Optional[float]] = field(default_factory=list)  # per
                                                            # finger -- how far THIS finger's own stub
                                                            # end extends PAST its crossing point with
                                                            # the OTHER finger (see finger_overlap_um's
                                                            # docstring), i.e. the JJ overlap length
                                                            # measured along this finger's own axis.
                                                            # Meaningful only when exactly 2 fingers
                                                            # exist (None otherwise, or if the two
                                                            # fingers' axes are parallel). This is the
                                                            # SAME value/formula already surfaced per-
                                                            # finger in measurements_by_stable_id's
                                                            # "overlap_um" key (used by the Shape
                                                            # Properties card) -- stored here as well so
                                                            # the bottom Measurements tree and the
                                                            # canvas dimension-line overlay can show the
                                                            # identical number without recomputing it
                                                            # via a different code path.
    warnings: List[str] = field(default_factory=list)
    jj_crossings: List["JJCrossingMeasurement"] = field(default_factory=list)  # NEW,
                                                            # generalizes jj_crossing_area_um2 above
                                                            # to a series-chain design with more than
                                                            # 2 fingers -- the measurements need to
                                                            # reflect that a design like this can have
                                                            # several SIS junctions at once, with the
                                                            # dimensions of each one shown.
                                                            # One entry per REAL pairwise finger-polygon
                                                            # crossing found by find_all_finger_crossings
                                                            # -- a classic 2-finger design still gets
                                                            # exactly one entry here, identical in every
                                                            # field to what jj_crossing_area_um2 alone
                                                            # used to report (jj_crossing_area_um2 is
                                                            # kept, unchanged, as jj_crossings[0]'s area,
                                                            # for every existing caller that only reads
                                                            # the old scalar field).



@dataclass
class JJCrossingMeasurement:
    """One real pairwise finger-to-finger JJ crossing (see
    find_all_finger_crossings), generalizing the old assumption that a
    junction has exactly one crossing between finger 0 and finger 1. A
    4-finger series chain (A-B-2-3) produces 3 of these -- one per real
    overlap -- matching the rule that an N-finger chain forms N-1 SIS
    junctions."""
    finger_i: int
    finger_j: int
    label_i: str
    label_j: str
    area_um2: Optional[float]
    nominal_area_um2: Optional[float]


def find_all_finger_crossings(fingers: List["ImportedFinger"]) -> List[Tuple[int, int]]:
    """Returns every pair of finger INDICES (i < j) whose polygons
    physically overlap -- i.e. every real Manhattan-style JJ crossing in
    the design. Generalizes the old hardcoded assumption (baked into
    compute_measurements, the disconnection-warning checks, and the
    shadow-simulation recipe grouping) that a junction has exactly 2
    fingers with exactly one crossing between them.

    A "layered Manhattan junction array"
    -- several fingers chained end-to-end, each new finger's tip
    overlapping the next finger in the chain instead of the electrode
    directly -- models N-1 SIS junctions in series across N fingers (a
    4-finger chain -> 3 junctions). A classic 2-finger design still
    returns exactly one pair, [(0, 1)] (when they actually cross), so
    this is a strict superset of the prior behavior -- nothing that
    already worked for 2 fingers changes."""
    pairs = []
    polys = []
    for f in fingers:
        try:
            polys.append(sg.Polygon(f.points))
        except Exception:
            polys.append(None)
    for i in range(len(fingers)):
        if polys[i] is None:
            continue
        for j in range(i + 1, len(fingers)):
            if polys[j] is None:
                continue
            try:
                inter_area = polys[i].intersection(polys[j]).area
            except Exception:
                continue
            if inter_area > 1e-9:
                pairs.append((i, j))
    return pairs


def compute_recommended_angles(junction: ImportedJunction) -> dict:
    """Recommended deposition angle(s) for each finger and for the
    patches, following this hierarchy: a direction with NO warning at
    all ("perfect") always wins; failing that, a direction flagged only
    by Warning #2 (self-shadowing) beats one flagged by Warning #1
    (electrode crossing), since Warning #1 is the more serious failure
    mode. If a finger's two directions are equally good, both are
    recommended. Patches are reconciled ACROSS both fingers' patch
    groups (all patches share one physical deposition angle): if the
    two fingers' patch groups agree on a safe angle, that's the single
    recommendation; if they don't (one needs 0 deg, the other needs 180
    deg), BOTH are recommended, since that genuinely requires two
    separate patch depositions to correctly cover both sides.

    Requires apply_deposit_order() to have been called first -- returns
    a dict with a "not_set" flag if deposit order is unknown."""
    fingers = junction.fingers
    if not fingers or any(f.deposited_second is None for f in fingers):
        return {"not_set": True}

    def tiers_for(theta_a, theta_b, w1_risky_theta, w2_risky_theta):
        out = {}
        for theta in (theta_a, theta_b):
            if w1_risky_theta is not None and theta == w1_risky_theta:
                out[theta] = "w1"
            elif w2_risky_theta is not None and theta == w2_risky_theta:
                out[theta] = "w2"
            else:
                out[theta] = "perfect"
        return out

    def recommend(tiers: dict):
        by_tier = {"perfect": [], "w2": [], "w1": []}
        for theta, tier in tiers.items():
            by_tier[tier].append(theta)
        for tier in ("perfect", "w2", "w1"):
            if by_tier[tier]:
                return sorted(by_tier[tier]), tier
        return [], "perfect"  # unreachable given tiers always has >=1 entry per key checked above

    def _risky_thetas(f):
        w1_risky = f.theta_toward_stub_deg if f.warning_source in ("finger", "both") else None
        w2_risky = f.theta_toward_patch_deg if (f.deposited_second and f.self_shadow_safe_theta_deg is not None) else None
        return w1_risky, w2_risky

    def finger_recommendation(f):
        w1_risky, w2_risky = _risky_thetas(f)
        tiers = tiers_for(f.theta_toward_stub_deg, f.theta_toward_patch_deg, w1_risky, w2_risky)
        angles, tier = recommend(tiers)
        # Fixed while investigating a translated-
        # design test scenario (post-shadow JJ-crossing-area check):
        # when a finger has NO warning in either direction
        # (the good, common case -- both candidates land in the "perfect"
        # tier and get returned TOGETHER, tied), `recommend()` above sorts
        # them purely numerically. Every consumer that seeds a single
        # default angle from this list (_populate_sim_recommendations_
        # tree's `entry["angles"][0]`) then gets whichever candidate
        # happens to be the smaller NUMBER -- with no regard for which
        # direction is this finger's own natural, as-drawn extending
        # direction (theta_toward_patch_deg, the direction the finger
        # actually points to reach its own crossing/patch). For a finger
        # whose stub angle happens to be numerically smaller than its
        # patch angle (e.g. Finger B in a classic 2-finger crossing:
        # toward_patch=180, toward_stub=0), the seeded default silently
        # became toward_stub -- a finger deposited "backwards", no longer
        # reaching its own crossing at all. Reproduced directly: a design
        # translated in place (a real fingerprint-mismatch reseed, not
        # just a fresh import) picked toward_stub as Finger B's default
        # and its post-shadow JJ crossing area with Finger A dropped to
        # exactly 0 -- a real, silent connectivity break, not a display
        # glitch. Fixed by re-ordering (never re-filtering -- "both
        # recommended when tied" stays true) so theta_toward_patch_deg
        # sorts first whenever it's one of the tied candidates, matching
        # the single most natural default: the direction this finger was
        # actually drawn to extend toward.
        if f.theta_toward_patch_deg in angles and angles[0] != f.theta_toward_patch_deg:
            angles = [f.theta_toward_patch_deg] + [a for a in angles if a != f.theta_toward_patch_deg]
        return {"finger_index": fingers.index(f), "angles": angles, "tier": tier,
                "explanation": _tier_explanation(tier, angles, is_patch=False)}

    # For series-chain designs (a "layered Manhattan junction array"):
    # two fingers that share an axis
    # (apply_deposit_order's own deposited_second grouping) deposit in
    # the exact SAME physical pass -- one wafer rotation, one angle --
    # so they cannot be given independently "optimal" angles the way
    # finger_recommendation above picks one in isolation. Reproduced
    # directly: a middle chain finger with no electrode
    # contact of its own (e.g. Finger A) has BOTH its candidate angles
    # equally "perfect" in isolation and arbitrarily tie-breaks to the
    # lower one, while its axis-mate (e.g. Finger D) genuinely NEEDS the
    # other angle to reach its own electrode contact safely -- two
    # fingers on the same real wafer rotation ending up with two
    # different "recommended" angles is physically meaningless. Fixed by
    # reconciling every candidate angle's tier as the WORST (most
    # restrictive) tier found across every member of the group, so the
    # group's shared angle is whichever one is safe for every member
    # simultaneously -- exactly the group-mate with the strictest real
    # constraint decides for the whole group.
    def group_recommendation(members):
        if len(members) == 1:
            return finger_recommendation(members[0])
        tier_rank = {"perfect": 0, "w2": 1, "w1": 2}
        ref = members[0]
        candidates = sorted({round(ref.theta_toward_stub_deg, 6), round(ref.theta_toward_patch_deg, 6)})

        def _closest_tier(theta, w1_risky, w2_risky):
            if w1_risky is not None and abs(theta - w1_risky) < 0.5:
                return "w1"
            if w2_risky is not None and abs(theta - w2_risky) < 0.5:
                return "w2"
            return "perfect"

        combined = {}
        for theta in candidates:
            worst = "perfect"
            for f in members:
                w1_risky, w2_risky = _risky_thetas(f)
                this_tier = _closest_tier(theta, w1_risky, w2_risky)
                if tier_rank[this_tier] > tier_rank[worst]:
                    worst = this_tier
            combined[theta] = worst
        angles, tier = recommend(combined)
        names = ", ".join(f.stable_id for f in members)
        explanation = _tier_explanation(tier, angles, is_patch=False) + (
            f" (Shared physical pass across {len(members)} parallel fingers -- {names} -- so this angle is "
            f"whichever one is safe for every one of them, not just this finger alone.)"
        )
        return {"angles": angles, "tier": tier, "explanation": explanation}

    first_group = [f for f in fingers if f.deposited_second is False]
    second_group = [f for f in fingers if f.deposited_second is True]
    first_finger = first_group[0] if first_group else None
    second_finger = second_group[0] if second_group else None
    result = {"not_set": False}

    per_finger = {}
    for group in (first_group, second_group):
        if not group:
            continue
        group_rec = group_recommendation(group)
        for f in group:
            entry = dict(finger_recommendation(f))
            if len(group) > 1:
                # Group reconciliation overrides the angle(s)/tier/
                # explanation (the physically-real shared-pass answer),
                # but keeps this finger's own finger_index intact -- a
                # 2-finger design's group always has exactly 1 member, so
                # this whole branch is a no-op there (byte-identical to
                # the old per-finger-only behavior).
                entry["angles"] = group_rec["angles"]
                entry["tier"] = group_rec["tier"]
                entry["explanation"] = group_rec["explanation"]
            per_finger[fingers.index(f)] = entry

    if first_finger is not None:
        result["first_finger"] = per_finger[fingers.index(first_finger)]
    if second_finger is not None:
        result["second_finger"] = per_finger[fingers.index(second_finger)]
    # Generalized for series-chain designs:
    # first_finger/second_finger above are kept EXACTLY as before (still
    # just one representative finger per deposit group) so every existing
    # 2-finger caller keeps working unchanged -- but a 3rd+ finger sharing
    # either group (see apply_deposit_order's own generalization) was
    # previously invisible to this function entirely. result["fingers"]
    # is every finger's own recommendation (angle/tier reconciled across
    # its whole shared-pass group when it has axis-mates), keyed by its
    # real index, so build_simulation_result can build a real recipe
    # step for EVERY finger instead of just the two representatives.
    result["fingers"] = per_finger

    # Patches: reconcile across every finger that has any, since they all
    # share ONE physical deposition layer -- every patch on the design
    # receives metal from every active patch-deposition pass, regardless
    # of which finger it happens to sit nearest to. This is NOT a
    # per-finger-group independent calculation, and it is NOT solved
    # shape-by-shape.
    #
    # Fixed after a prior
    # "patches always need both directions" attempt produced bogus
    # decimal angles on real validation data: patches are simple axis-aligned
    # rectangles with exactly ONE shared axial pair of possible
    # directions (e.g. 0 deg and 180 deg) -- never a finger-derived
    # diagonal angle, never an arbitrary per-shape value. The ONLY
    # reason two passes (rather than one) are ever needed is a genuine
    # Warning #1 (electrode-crossing) DISAGREEMENT between the two
    # fingers' own patch groups: each finger's patch group has its own
    # independent verdict on which single axial direction is safe (the
    # direction that reaches ITS OWN electrode "uphill"), and that
    # verdict only exists when that group actually crosses a boundary
    # (patches_cross True). A patch group that does NOT cross a boundary
    # contributes no constraint at all -- it does not force two passes.
    # If every crossing group's required angle agrees (or nothing
    # crosses), one pass suffices; if two crossing groups disagree, both
    # angles are required -- never more, never some other value.
    patch_recs = []          # required angles, one per finger whose patch group actually crosses
    any_patches = False
    ref_angles = None        # the shared axial pair, from any one finger's own (patch-only) group
    for f in fingers:
        if f.patch_outward_end is None:
            continue
        any_patches = True
        if ref_angles is None:
            ref_angles = sorted({f.patch_theta_outward_deg, f.patch_theta_inward_deg})
        if f.warning_source in ("patches", "both"):
            patch_recs.append(f.patch_theta_outward_deg)

    if not any_patches:
        result["patches"] = None
    elif not patch_recs:
        # No finger's patch group crosses an electrode boundary -- there is
        # no Warning-1 constraint anywhere on the patch layer, so a single
        # pass at either shared axial direction is safe. Report both as
        # informational options, mirroring how a non-crossing finger's own
        # "perfect" tier already reports both of its directions.
        angles = ref_angles
        result["patches"] = {"angles": angles, "needs_two_depositions": False,
                             "explanation": _tier_explanation("perfect", angles, is_patch=True)}
    elif len(set(patch_recs)) == 1:
        angles = [patch_recs[0]]
        result["patches"] = {"angles": angles, "needs_two_depositions": False,
                             "explanation": f"Every finger's patch group that crosses an electrode boundary agrees θ={angles[0]:.0f}° is the safe direction."}
    else:
        angles = sorted(set(patch_recs))
        explanation = (
            f"The two fingers' patch groups disagree: one needs θ={angles[0]:.0f}°, the other "
            f"θ={angles[-1]:.0f}°. All patches share one layer, so this requires two separate "
            "patch depositions, not a compromise angle."
        )
        result["patches"] = {"angles": angles, "needs_two_depositions": True, "explanation": explanation}
    return result


def _tier_explanation(tier: str, angles: List[float], is_patch: bool) -> str:
    subject = "patch group" if is_patch else "finger"
    angle_text = "/".join(f"{a:.0f}°" for a in angles)
    if tier == "perfect":
        if len(angles) > 1:
            return f"Both directions are clear of any warning -- θ={angle_text} either works."
        return f"θ={angle_text} is completely clear of any warning."
    elif tier == "w2":
        return (f"θ={angle_text} only trips Warning #2 (self-shadowing) -- accepted here since "
                f"Warning #1 (electrode crossing) is the more serious failure mode.")
    else:
        return f"θ={angle_text} is the only option, but it trips Warning #1 (electrode crossing)."


def explain_finger_angle_choice(finger: ImportedFinger) -> str:
    """Human-readable explanation, built from TWO independent concerns --
    Warning #1 (does the finger or its patches cross the electrode
    boundary?) and Warning #2 (self-shadowing: is this finger deposited
    SECOND, crossing over the other finger's already-deposited step?).
    These are genuinely separate physical events -- a finger crossing an
    electrode boundary doesn't make its own step-crossing shadow risk
    disappear -- so both get computed and reported whenever they apply,
    not one as a fallback for the other."""
    lines = []

    # --- Warning #1: electrode boundary crossing ---
    if finger.warning_source in ("finger", "both"):
        lines.append(
            f"Warning #1 (electrode crossing): the finger crosses the electrode edge "
            f"({finger.finger_overlap_frac*100:.0f}% overlap).\n"
            f"  ✓ θ={finger.theta_toward_patch_deg:.0f}°: reaches the electrode.\n"
            f"  ⚠ θ={finger.theta_toward_stub_deg:.0f}°: shadows the step edge -- likely opens the junction."
        )
        if finger.warning_source == "both":
            lines.append("  The patch group also crosses separately -- check its own angle too.")
    elif finger.warning_source == "patches":
        lines.append(
            f"Warning #1 (electrode crossing): finger is clear "
            f"({(finger.finger_overlap_frac or 0)*100:.0f}%); its patch group crosses the electrode "
            f"instead ({finger.patch_group_overlap_frac*100:.0f}%).\n"
            f"  ✓ θ={finger.patch_theta_outward_deg:.0f}°: reaches the electrode.\n"
            f"  ⚠ θ={finger.patch_theta_inward_deg:.0f}°: shadows the patch edge -- disconnect risk."
        )
    elif finger.warning_source == "none":
        lines.append(
            f"Warning #1 (electrode crossing): no electrode boundary crosses this finger or its "
            f"patches -- safe at either θ={finger.theta_toward_patch_deg:.0f}° or "
            f"{finger.theta_toward_stub_deg:.0f}°."
        )
    else:
        lines.append("Warning #1 (electrode crossing): electrode geometry unknown -- not checked.")

    # --- Warning #2: self-shadowing (independent of Warning #1's result) ---
    if finger.deposited_second and finger.self_shadow_safe_theta_deg is not None:
        lines.append(
            f"Warning #2 (self-shadowing): deposited second, crossing the other finger's step.\n"
            f"  ✓ θ={finger.self_shadow_safe_theta_deg:.0f}°: thinning lands on the harmless side.\n"
            f"  ⚠ θ={finger.theta_toward_patch_deg:.0f}°: thinning lands on the current-carrying "
            f"side -- avoid if possible, or increase deposition thickness for this step."
        )
    elif finger.deposited_second is False:
        lines.append("Warning #2 (self-shadowing): deposited FIRST -- no step exists yet for it to "
                     "shadow across, safe either direction.")
    else:
        lines.append("Warning #2 (self-shadowing): set which finger deposits first (Deposition Order) "
                     "to check this.")

    return "\n".join(lines)


def _measure_local_width_and_overhang(stub_end, patch_end, connector_union, patch_points, patch_axis_angle_deg, patch_centroid):
    """connector_union = whatever solid material this patch actually
    connects to (the finger's own polygon, its adhesion plate's polygon,
    or their union) -- the TRUE local width is measured directly at the
    crossing point via a perpendicular probe line, rather than assumed
    from the finger's own width_um.

    Fixed here: the earlier version always used the
    finger's width for the clearance calculation (u = width/2 /
    |sin(delta)|), which is only correct when the patch actually crosses
    the bare finger. Patches near the tip commonly cross the (much
    wider) adhesion plate instead -- verified with a synthetic
    case (0.2um finger vs 1.5um plate) that this produces a 0.65um
    error, not a rounding-level discrepancy. Probing the actual local
    cross-section at the real crossing point, rather than assuming
    which structure is there, fixes this for both cases automatically
    without needing to know in advance whether a given patch is over
    the finger or the plate.

    Returns (local_width_um, near_overhang_um, far_overhang_um)."""
    fx, fy = patch_end[0] - stub_end[0], patch_end[1] - stub_end[1]
    flen = math.hypot(fx, fy) or 1.0
    f_hat = (fx / flen, fy / flen)
    finger_angle_deg = math.degrees(math.atan2(fy, fx))

    ang = math.radians(patch_axis_angle_deg)
    p_hat = (math.cos(ang), math.sin(ang))
    denom = f_hat[0] * p_hat[1] - f_hat[1] * p_hat[0]
    if abs(denom) < 1e-12:
        crossing = patch_centroid
    else:
        dx, dy = patch_centroid[0] - stub_end[0], patch_centroid[1] - stub_end[1]
        t = (dx * p_hat[1] - dy * p_hat[0]) / denom
        crossing = (stub_end[0] + t * f_hat[0], stub_end[1] + t * f_hat[1])

    perp = (-f_hat[1], f_hat[0])
    probe_len = 50.0  # generous -- comfortably longer than any real finger/plate width
    probe = sg.LineString([
        (crossing[0] - perp[0] * probe_len, crossing[1] - perp[1] * probe_len),
        (crossing[0] + perp[0] * probe_len, crossing[1] + perp[1] * probe_len),
    ])
    inter = probe.intersection(connector_union) if connector_union is not None else None
    local_width = inter.length if (inter is not None and not inter.is_empty) else 0.0
    if local_width < 1e-9 and connector_union is not None:
        # Fixed here: if the crossing point happens to land
        # exactly ON a boundary edge (collinear with the probe), shapely
        # returns a degenerate zero-length intersection instead of the
        # real cross-section. The tip-probe version
        # of this same technique below hit exactly this case at a
        # finger's tip corner. A tiny inset along the finger's own axis
        # (toward stub_end) moves the probe safely into the interior
        # without meaningfully changing the measurement at this scale.
        inset = 0.01
        cx, cy = crossing[0] - f_hat[0] * inset, crossing[1] - f_hat[1] * inset
        probe = sg.LineString([
            (cx - perp[0] * probe_len, cy - perp[1] * probe_len),
            (cx + perp[0] * probe_len, cy + perp[1] * probe_len),
        ])
        inter = probe.intersection(connector_union)
        local_width = inter.length if not inter.is_empty else 0.0

    delta = math.radians(patch_axis_angle_deg - finger_angle_deg)
    sin_delta = math.sin(delta)
    if abs(sin_delta) < 0.05:
        sin_delta = 0.05 if sin_delta >= 0 else -0.05
    u = (local_width / 2.0) / abs(sin_delta)

    ox, oy = crossing
    ux, uy = p_hat

    def proj(pt):
        return (pt[0] - ox) * ux + (pt[1] - oy) * uy

    patch_proj = [proj(p) for p in patch_points]
    p_min, p_max = min(patch_proj), max(patch_proj)
    far_overhang = p_max - u
    near_overhang = (-u) - p_min

    def unproj(s):
        return (ox + s * ux, oy + s * uy)

    # Boundary points (where the finger/plate's own edge sits, projected
    # onto the patch's axis) and the patch's own near/far end points --
    # exposed so the overlay can draw the ACTUAL span for each overhang
    # instead of a bare number with no accompanying line. Added
    # alongside the analogous fix for the finger-tip
    # measurement (see _finger_past_last_patch_geometry) -- same
    # category of problem, same fix approach.
    geometry = {
        "near_boundary_pt": unproj(-u), "near_patch_pt": unproj(p_min),
        "far_boundary_pt": unproj(u), "far_patch_pt": unproj(p_max),
    }
    return local_width, near_overhang, far_overhang, geometry


def _find_finger_plate(junction: ImportedJunction, finger: ImportedFinger):
    """The plate (if any) belonging to this specific finger -- same
    nearest-to-patch_end logic as _assign_shapes_to_fingers, but for
    kind=='plate' specifically since that function now deliberately
    excludes plates."""
    best, best_d = None, 1e18
    for shape in junction.other_shapes:
        if shape.kind != "plate":
            continue
        d = sg.Polygon(shape.points).distance(sg.Point(finger.patch_end))
        if d < best_d:
            best_d, best = d, shape
    return best


def finger_connector_union(junction: ImportedJunction, finger: ImportedFinger):
    """The real solid material a patch on this finger actually attaches
    to: the finger's own polygon, unioned with its adhesion plate's if
    one exists (_find_finger_plate). Extracted from compute_measurements'
    own per-finger loop (unchanged behavior there) so
    apply_patch_extension_edit can build the identical connector_union a
    caller needs to pass in for its exact-round-trip path, without
    duplicating this construction at every call site."""
    finger_poly = sg.Polygon(finger.points)
    plate_shape = _find_finger_plate(junction, finger)
    if plate_shape is not None:
        return so.unary_union([finger_poly, sg.Polygon(plate_shape.points)])
    return finger_poly


def _finger_past_last_patch_geometry(stub_end, patch_end, connector_union, last_patch_points):
    """Pure geometry, returning enough to both measure AND draw this
    correctly: for each of the finger's two long edges, the point where
    that edge exits the outermost patch (walking toward the tip) and
    the finger's own tip-corner point on that same side. Distance
    between them is the measured value.

    Found and fixed here: the overlay used to draw a
    small, FIXED-length (0.2um) tick at the tip corner regardless of the
    actual measured distance, in the WRONG direction (extending further
    past the tip, away from the patch, into empty space, rather than
    back toward the patch it's actually measuring the gap to) --
    visible in screenshots showing dotted lines running off into
    empty space with no visible connection to the patch the number was
    supposedly measuring against. Returning the real entry/exit points
    here lets the overlay draw the ACTUAL span, so the visual and the
    printed number are the same computation, not just a symbolic
    placeholder near the right general area.

    Returns [(exit_point, tip_point, distance_um), (exit_point, tip_point, distance_um)]
    for the two sides, in the same order _measure_finger_past_last_patch
    used to return (left, right)."""
    fx, fy = patch_end[0] - stub_end[0], patch_end[1] - stub_end[1]
    flen = math.hypot(fx, fy) or 1.0
    f_hat = (fx / flen, fy / flen)
    perp = (-f_hat[1], f_hat[0])

    probe_len = 50.0
    inset = 0.01
    probe_pt = (patch_end[0] - f_hat[0] * inset, patch_end[1] - f_hat[1] * inset)
    probe = sg.LineString([
        (probe_pt[0] - perp[0] * probe_len, probe_pt[1] - perp[1] * probe_len),
        (probe_pt[0] + perp[0] * probe_len, probe_pt[1] + perp[1] * probe_len),
    ])
    inter = probe.intersection(connector_union) if connector_union is not None else None
    tip_width = inter.length if (inter is not None and not inter.is_empty) else 0.0
    hw = tip_width / 2.0

    patch_poly = sg.Polygon(last_patch_points)
    results = []
    for sign in (+1, -1):
        ox, oy = stub_end[0] + perp[0] * hw * sign, stub_end[1] + perp[1] * hw * sign
        ex, ey = patch_end[0] + perp[0] * hw * sign, patch_end[1] + perp[1] * hw * sign
        edge_line = sg.LineString([(ox, oy), (ex, ey)])
        inter2 = edge_line.intersection(patch_poly)
        if inter2.is_empty:
            results.append(((ex, ey), (ex, ey), 0.0))
            continue
        if inter2.geom_type == "LineString":
            candidates = list(inter2.coords)
        elif inter2.geom_type == "MultiLineString":
            candidates = [pt for geom in inter2.geoms for pt in geom.coords]
        elif hasattr(inter2, "x"):
            candidates = [(inter2.x, inter2.y)]
        else:
            candidates = []
        if not candidates:
            results.append(((ex, ey), (ex, ey), 0.0))
            continue
        exit_pt = max(candidates, key=lambda p: (p[0] - ox) * f_hat[0] + (p[1] - oy) * f_hat[1])
        dist = math.hypot(ex - exit_pt[0], ey - exit_pt[1])
        results.append((exit_pt, (ex, ey), dist))
    return results


def _measure_finger_past_last_patch(stub_end, patch_end, connector_union, last_patch_points):
    """Thin wrapper over _finger_past_last_patch_geometry for callers
    that only need the three numbers (the measurement pipeline) -- the
    overlay drawing code calls the geometry function directly instead,
    so the two can never show different numbers for the same design.
    Returns (left_um, right_um, shortest_um)."""
    geo = _finger_past_last_patch_geometry(stub_end, patch_end, connector_union, last_patch_points)
    dists = [d for _e, _t, d in geo]
    return dists[0], dists[1], min(dists)


def _assign_shapes_to_fingers(junction: ImportedJunction):
    """Groups each PATCH (specifically -- not plates, not any other
    shape kind) with whichever finger's patch_end it's closest to --
    this is how patches were actually built in every macro seen (each
    finger gets its own local group).

    Fixed here: this used to group ANY other_shape
    (patches AND adhesion plates alike) regardless of kind, and its only
    caller (compute_measurements, building patch_groups specifically)
    blindly treated everything in a group as a patch. A design with
    adhesion plates but genuinely zero patches (observed on a
    real design with no patches) would have its plate grouped in and reported
    as "Finger A patches (n=1)" -- a real patch measurement section for
    a patch that does not exist. Filtering to kind=="patch" here, at the
    one and only place this grouping happens, fixes it for every caller
    at once rather than patching around it downstream.

    Fixed here too: the "nearest patch_end point" geometric
    heuristic is genuinely ambiguous once a patch group's own long/short
    extension asymmetry or tip margin pushes a patch far enough from its
    own finger's tip -- observed with a perfectly ordinary
    0/270-degree, patch_angle=45 design (the DesignParameters defaults),
    where a patch legitimately built for one finger landed closer to the
    OTHER finger's tip point and was silently grouped there instead,
    corrupting that finger's own spacing/tip-margin/overhang numbers with
    a patch that was never really part of its group. For a parametric
    design (from_design_parameters), which finger a patch belongs to is
    known EXACTLY at construction time and recorded on the shape itself
    (ImportedShape.owner_finger_hint) -- used directly here when present,
    with the geometric distance-to-patch_end guess kept ONLY as the
    fallback for shapes with no known construction-time owner (a real GDS
    import, or a hand-placed "patch"-role shape from the shape editor)."""
    groups = {i: [] for i in range(len(junction.fingers))}
    for shape in junction.other_shapes:
        if shape.kind != "patch":
            continue
        hint = getattr(shape, "owner_finger_hint", None)
        if hint is not None and 0 <= hint < len(junction.fingers):
            groups[hint].append(shape)
            continue
        shp = sg.Polygon(shape.points)
        best_i, best_d = None, 1e18
        for i, f in enumerate(junction.fingers):
            d = shp.distance(sg.Point(f.patch_end))
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            groups[best_i].append(shape)
    return groups


def _measure_finger_electrode_entry_depth(stub_end, patch_end, finger_width_um, electrode_union):
    """For each of the finger's two long edges, walk from stub_end toward
    patch_end and find where that specific edge line first ENTERS the
    electrode -- the true per-side entry point, which can differ
    between the two sides whenever the entry edge isn't perpendicular
    to the finger's own axis. Returns how far the tip corner sits past
    that entry point on each side.

    Replaced here: the previous version measured distance
    from each tip corner to the NEAREST point on the electrode's
    boundary, using shapely's boundary.distance(). Verified
    against a real design that this does not measure entry depth at
    all -- for the default design it returned 0.65um, which turned out
    to exactly equal the corner's distance to the electrode's own
    lateral/far edge (an accidental artifact of how large that specific
    electrode polygon happens to be drawn), not the ~1.5um actual
    overlap depth. Re-verified this fix against the same real design
    afterward: it now returns exactly 1.500, matching the true overlap.

    Returns (left_um, right_um, shortest_um)."""
    fx, fy = patch_end[0] - stub_end[0], patch_end[1] - stub_end[1]
    flen = math.hypot(fx, fy) or 1.0
    f_hat = (fx / flen, fy / flen)
    perp = (-f_hat[1], f_hat[0])
    hw = finger_width_um / 2.0

    results = []
    for sign in (+1, -1):
        ox, oy = stub_end[0] + perp[0] * hw * sign, stub_end[1] + perp[1] * hw * sign
        ex, ey = patch_end[0] + perp[0] * hw * sign, patch_end[1] + perp[1] * hw * sign
        edge_line = sg.LineString([(ox, oy), (ex, ey)])
        inter = edge_line.intersection(electrode_union) if electrode_union is not None else None
        if inter is None or inter.is_empty:
            results.append(0.0)  # this side never reaches the electrode at all
            continue
        if inter.geom_type == "LineString":
            candidates = list(inter.coords)
        elif inter.geom_type == "MultiLineString":
            candidates = [pt for geom in inter.geoms for pt in geom.coords]
        elif hasattr(inter, "x"):
            candidates = [(inter.x, inter.y)]
        else:
            candidates = []
        if not candidates:
            results.append(0.0)
            continue
        entry_pt = min(candidates, key=lambda p: (p[0] - ox) * f_hat[0] + (p[1] - oy) * f_hat[1])
        results.append(math.hypot(ex - entry_pt[0], ey - entry_pt[1]))
    return results[0], results[1], min(results)


def compute_measurements(junction: ImportedJunction) -> MeasurementReport:
    warnings = []

    finger_lengths = [f.length_um for f in junction.fingers]
    finger_widths_nm = [f.width_um * 1000.0 for f in junction.fingers]
    theta_toward_patch = [f.theta_toward_patch_deg for f in junction.fingers]
    theta_toward_stub = [f.theta_toward_stub_deg for f in junction.fingers]
    warning_sources = [f.warning_source for f in junction.fingers]
    explanations = [explain_finger_angle_choice(f) for f in junction.fingers]

    # The measured JJ crossing area should equal the nominal crossing
    # area unless the overlap is genuinely insufficient -- if the
    # junction fingers only partially overlap or do not overlap at all,
    # that shortfall needs to be visible rather than masked. Previously
    # this used to always report the naive linewidth
    # product (width_A x width_B) with no check that the two fingers
    # actually cross cleanly. For two perpendicular strips crossing
    # fully, the REAL geometric intersection of their polygons equals
    # that product exactly, so a well-formed design reports the same
    # number either way -- but a design where the fingers only
    # partially overlap (or miss each other) silently reported the
    # same (too-large, nominal-only) product regardless, giving no way
    # to see that the junction is actually smaller (or nonexistent).
    # Now computed from the real polygon intersection, falling back to
    # the naive product only when there's no polygon data at all; a
    # meaningful shortfall vs. the naive product is surfaced as an
    # explicit warning rather than silently under-reporting.
    # Generalized so the measurements reflect that a series-chain design
    # can have several SIS junctions occurring at once, with the
    # dimensions of each shown: a series-chain design (N fingers, each overlapping the
    # next) forms N-1 real JJ crossings, not just one between finger 0
    # and finger 1. find_all_finger_crossings finds every REAL pairwise
    # finger-polygon overlap; a classic 2-finger design still returns
    # exactly one pair, so this is byte-identical to the old behavior
    # for 2 fingers -- except when they don't actually cross at all, in
    # which case (0, 1) is still evaluated (as it always was) so the "do
    # NOT overlap" warning below still fires for a broken 2-finger design.
    jj_area = None
    jj_crossings: List[JJCrossingMeasurement] = []
    crossing_pairs = find_all_finger_crossings(junction.fingers)
    if len(junction.fingers) == 2 and not crossing_pairs:
        crossing_pairs = [(0, 1)]
    for ci, cj in crossing_pairs:
        finger_i, finger_j = junction.fingers[ci], junction.fingers[cj]
        label_i, label_j = finger_display_label(ci), finger_display_label(cj)
        nominal_jj_area = finger_i.width_um * finger_j.width_um
        try:
            finger_a_poly = sg.Polygon(finger_i.points)
            finger_b_poly = sg.Polygon(finger_j.points)
            real_crossing_area = finger_a_poly.intersection(finger_b_poly).area
        except Exception:
            real_crossing_area = None
        # 2-finger designs keep the exact old warning text ("the two
        # fingers") so existing substring-matching tests/UI keep working;
        # a chain of 3+ fingers names which pair, since "the two fingers"
        # would be ambiguous once there's more than one crossing.
        pair_desc = "the two fingers" if len(junction.fingers) == 2 else f"Finger {label_i} and Finger {label_j}"
        if real_crossing_area is None or nominal_jj_area < 1e-12:
            area = nominal_jj_area
        elif real_crossing_area >= nominal_jj_area * JJ_CROSSING_FULL_OVERLAP_RATIO:
            # Clean full crossing -- report the exact nominal (designed)
            # value rather than polygon-intersection floating-point noise.
            area = nominal_jj_area
        else:
            area = real_crossing_area
            shortfall_pct = 100.0 * (1.0 - real_crossing_area / nominal_jj_area) if nominal_jj_area > 0 else 100.0
            if real_crossing_area < 1e-9:
                warnings.append(
                    f"JJ crossing area ({label_i}×{label_j}): {pair_desc} do NOT overlap -- no junction is formed "
                    f"(nominal area would have been {nominal_jj_area:.4f} µm²)."
                )
            else:
                warnings.append(
                    f"JJ crossing area ({label_i}×{label_j}): {pair_desc} only partially overlap -- {real_crossing_area:.4f} µm² "
                    f"actual vs. {nominal_jj_area:.4f} µm² nominal ({shortfall_pct:.0f}% smaller)."
                )
        jj_crossings.append(JJCrossingMeasurement(
            finger_i=ci, finger_j=cj, label_i=label_i, label_j=label_j,
            area_um2=area, nominal_area_um2=nominal_jj_area,
        ))
    if jj_crossings:
        # Backward-compat scalar -- unchanged for the 2-finger case
        # (jj_crossings has exactly one entry there), a reasonable
        # "primary crossing" summary for older callers that only read
        # this one field on an N-finger chain.
        jj_area = jj_crossings[0].area_um2

    # Real write/exposed area for e-beam cost estimation: the UNION of every
    # shape (fingers + patches + plates), so overlapping regions (e.g. a
    # patch overlapping its finger) are counted once, not once per shape.
    # naive_summed_area is the WRONG way to do it (sum of individual shape
    # areas) -- reported alongside so the double-counting is visible as an
    # actual number, not just asserted.
    all_polys = [sg.Polygon(f.points) for f in junction.fingers] + [sg.Polygon(o.points) for o in junction.other_shapes]
    total_exposed_area = None
    naive_summed_area = None
    if all_polys:
        total_exposed_area = so.unary_union(all_polys).area
        naive_summed_area = sum(p.area for p in all_polys)
        if naive_summed_area - total_exposed_area > 1e-9:
            pct = 100.0 * (naive_summed_area - total_exposed_area) / naive_summed_area
            warnings.append(
                f"Shapes overlap by {naive_summed_area - total_exposed_area:.4f} µm² "
                f"({pct:.1f}% of the summed area) -- total_exposed_area_um2 already accounts for this."
            )

    electrode_union = None
    if junction.electrode_polygons:
        electrode_union = so.unary_union([sg.Polygon(p) for p in junction.electrode_polygons if len(p) >= 3])

    groups = _assign_shapes_to_fingers(junction)
    patch_groups = []
    for i, f in enumerate(junction.fingers):
        shapes = groups.get(i, [])
        if not shapes:
            continue
        # order along the finger's own axis (distance from the stub end)
        fx, fy = f.patch_end[0] - f.stub_end[0], f.patch_end[1] - f.stub_end[1]
        flen = (fx ** 2 + fy ** 2) ** 0.5 or 1.0
        ux, uy = fx / flen, fy / flen

        def pos_along_axis(shape):
            cx = sum(p[0] for p in shape.points) / len(shape.points)
            cy = sum(p[1] for p in shape.points) / len(shape.points)
            return (cx - f.stub_end[0]) * ux + (cy - f.stub_end[1]) * uy

        shapes_sorted = sorted(shapes, key=pos_along_axis)
        positions = [pos_along_axis(s) for s in shapes_sorted]
        polys_sorted = [sg.Polygon(s.points) for s in shapes_sorted]
        # True edge-to-edge gap via real polygon distance -- NOT a
        # projection onto the finger's own diagonal axis. These patches are
        # spaced by a Y-offset in the macros (edge_gap_y), independent of
        # the finger's own 45-degree axis, so "distance along the finger
        # axis minus half-extents" doesn't recover the real gap; asking
        # shapely for the actual nearest-point distance between the two
        # polygons does, regardless of what axis convention was used to
        # place them.
        edge_gaps = [polys_sorted[k].distance(polys_sorted[k + 1]) for k in range(len(polys_sorted) - 1)]

        half_extents = []
        for s in shapes_sorted:
            proj = [(p[0]) * ux + (p[1]) * uy for p in s.points]
            half_extents.append((max(proj) - min(proj)) / 2.0)
        max_reach = (positions[-1] + half_extents[-1]) if positions else None

        # Connector union: whatever solid material these patches actually
        # attach to -- the finger's own polygon, plus its adhesion plate
        # if one exists. Built ONCE per finger and reused for the local-
        # width probing below and for the new overlap-area measurement,
        # rather than assuming the finger's own width_um applies
        # everywhere (wrong wherever a wider plate is actually present).
        connector_union = finger_connector_union(junction, f)

        # Per-patch electrode overlap -- fixed here: this
        # used to only check the OUTERMOST patch. A staircase-style patch
        # layout means each patch can overlap the electrode by a
        # genuinely different amount, not just whichever is furthest out.
        individual_overlap_depths = []
        for s in shapes_sorted:
            depth = None
            if electrode_union is not None:
                shp = sg.Polygon(s.points)
                inter = shp.intersection(electrode_union)
                if inter.area > 1e-9:
                    depth = inter.area / max(s.width_um, 1e-6)  # approximate penetration depth
            individual_overlap_depths.append(depth)
        overlap_depth = individual_overlap_depths[-1] if individual_overlap_depths else None  # kept
                                                            # for backward compat with existing callers

        # Per-patch overhang -- pure geometry, works for imported designs
        # too (no long_extension/short_extension parameter to fall back
        # on there at all). Uses connector_union (not a fixed finger
        # width) so patches crossing the adhesion plate get the correct
        # local clearance, not the bare finger's narrower one.
        individual_near_overhangs, individual_far_overhangs = [], []
        individual_local_widths = []
        individual_overlap_areas = []
        for s in shapes_sorted:
            axis_angle, _end_a, _end_b, _w = _polygon_axis_and_ends(s.points)
            centroid = (sum(p[0] for p in s.points) / len(s.points), sum(p[1] for p in s.points) / len(s.points))
            local_w, near_oh, far_oh, _geo = _measure_local_width_and_overhang(
                f.stub_end, f.patch_end, connector_union, s.points, axis_angle, centroid)
            individual_local_widths.append(local_w)
            individual_near_overhangs.append(near_oh)
            individual_far_overhangs.append(far_oh)
            # Overlap AREA: actual contact area between this patch and
            # whatever it attaches to (finger and/or plate) -- the
            # physically relevant quantity for contact resistance,
            # distinct from the LINEAR overhang measurements above.
            patch_poly = sg.Polygon(s.points)
            inter_area = patch_poly.intersection(connector_union).area
            individual_overlap_areas.append(inter_area)
        total_overlap_area = sum(individual_overlap_areas) if individual_overlap_areas else None

        tip_past_last_patch = None
        if shapes_sorted:
            outer = shapes_sorted[-1]
            left, right, shortest = _measure_finger_past_last_patch(
                f.stub_end, f.patch_end, connector_union, outer.points)
            tip_past_last_patch = (left, right, shortest)

        patch_groups.append(PatchGroupMeasurement(
            finger_index=i, patch_count=len(shapes_sorted),
            edge_to_edge_spacings_um=edge_gaps,
            individual_lengths_um=[s.length_um for s in shapes_sorted],
            individual_widths_um=[s.width_um for s in shapes_sorted],
            max_reach_um=max_reach, electrode_overlap_depth_um=overlap_depth,
            individual_overlap_depths_um=individual_overlap_depths,
            individual_near_overhang_um=individual_near_overhangs,
            individual_far_overhang_um=individual_far_overhangs,
            finger_tip_past_last_patch_um=tip_past_last_patch,
            individual_overlap_area_um2=individual_overlap_areas,
            total_overlap_area_um2=total_overlap_area,
        ))

    # Per-finger electrode overlap depth -- when the finger ITSELF makes
    # direct electrode contact (warning_source "finger" or "both"), not
    # just its patches. Same area/width approximation as the per-patch
    # version above, applied to the finger's own polygon.
    finger_overlap_depths = []
    finger_tip_shortest_overhangs = []
    finger_crossing_overlaps = []
    for i, f in enumerate(junction.fingers):
        depth = None
        if electrode_union is not None and f.warning_source in ("finger", "both"):
            shp = sg.Polygon(f.points)
            inter = shp.intersection(electrode_union)
            if inter.area > 1e-9:
                depth = inter.area / max(f.width_um, 1e-6)
        finger_overlap_depths.append(depth)

        # The finger's own tip has two corners; if the electrode's entry
        # edge isn't perpendicular to the finger's axis, they can
        # penetrate to genuinely different depths -- the shorter one is
        # the weaker connection point, not captured by the area-based
        # depth above.
        left_depth, right_depth, tip_shortest = _measure_finger_electrode_entry_depth(
            f.stub_end, f.patch_end, f.width_um, electrode_union)
        finger_tip_shortest_overhangs.append(tip_shortest if electrode_union is not None else None)

        # JJ crossing overlap -- how far THIS finger's stub extends past
        # its crossing point with (one of) the OTHER finger(s) it crosses,
        # i.e. the same finger_overlap_um() formula already used by
        # measurements_by_stable_id for the Shape Properties card. Kept
        # here too so the bottom Measurements tree and the canvas overlay
        # both read this from the report, guaranteeing all three surfaces
        # (card, tree, overlay) always agree.
        #
        # GENERALIZED for series-chain designs: a
        # 2-finger design's "the other finger" is now "this finger's
        # first real crossing partner", found from the same
        # crossing_pairs list used for jj_crossings above -- identical
        # result for exactly 2 fingers (there's only ever one possible
        # partner), and no longer None for finger indices >= 2 in a
        # longer chain.
        partners = [cj if ci == i else ci for ci, cj in crossing_pairs if i in (ci, cj)]
        other = junction.fingers[partners[0]] if partners else None
        finger_crossing_overlaps.append(finger_overlap_um(f, other) if other is not None else None)

    # Generalized: the old blanket
    # "Expected 2 fingers" warning fired for EVERY design with more than
    # 2 fingers, even a genuinely valid, fully-connected series chain
    # (exactly the "layered Manhattan junction array" design this was
    # built for). Replaced with an actual topology check for chains of 3+
    # fingers: every finger must be reachable, by real JJ crossings, from
    # a finger that itself makes direct electrode/patch contact -- i.e.
    # the chain's two ends are grounded and nothing is floating. A
    # classic 2-finger design is untouched (this whole block is skipped
    # for it, exactly as before -- it never got this warning either way).
    if len(junction.fingers) < 2:
        # Too few fingers to form ANY junction at all -- not a chain-
        # topology question, just a plainly incomplete design. Keeps the
        # old wording's intent (and its "finger" substring, still relied
        # on by existing degraded-state checks) for this genuinely
        # different, degenerate case.
        warnings.append(f"Expected at least 2 fingers for a Manhattan crossing, found {len(junction.fingers)} -- "
                         "JJ crossing area and per-finger angle analysis may not be meaningful.")
    elif len(junction.fingers) > 2:
        electrode_contact_idxs = [i for i, ws in enumerate(warning_sources) if ws in ("finger", "patches", "both")]
        if electrode_contact_idxs:  # only meaningful once electrode classification has actually run
            n = len(junction.fingers)
            adjacency = {i: set() for i in range(n)}
            for ci, cj in crossing_pairs:
                adjacency[ci].add(cj)
                adjacency[cj].add(ci)
            reached = set(electrode_contact_idxs)
            frontier = list(electrode_contact_idxs)
            while frontier:
                cur = frontier.pop()
                for nb in adjacency[cur]:
                    if nb not in reached:
                        reached.add(nb)
                        frontier.append(nb)
            if len(reached) != n or len(crossing_pairs) != n - 1:
                unreached = [finger_display_label(i) for i in range(n) if i not in reached]
                detail = f" Finger(s) not reachable from an electrode contact: {', '.join(unreached)}." if unreached else ""
                warnings.append(
                    f"Finger chain topology looks incomplete: found {len(crossing_pairs)} finger crossing(s) among "
                    f"{n} fingers, with {len(electrode_contact_idxs)} finger(s) in direct electrode/patch contact "
                    f"(a valid {n}-finger series chain needs {n - 1} crossings and every finger grounded through "
                    f"them).{detail}"
                )

    return MeasurementReport(
        source_label=junction.source_label, jj_crossing_area_um2=jj_area,
        total_exposed_area_um2=total_exposed_area, naive_summed_area_um2=naive_summed_area,
        finger_lengths_um=finger_lengths, finger_widths_nm=finger_widths_nm,
        finger_theta_toward_patch_deg=theta_toward_patch, finger_theta_toward_stub_deg=theta_toward_stub,
        finger_warning_source=warning_sources, finger_explanations=explanations,
        patch_groups=patch_groups, finger_electrode_overlap_depth_um=finger_overlap_depths,
        finger_tip_shortest_overhang_um=finger_tip_shortest_overhangs,
        jj_crossings=jj_crossings,
        finger_crossing_overlap_um=finger_crossing_overlaps,
        warnings=warnings,
    )


# ============================================================================
# New Tab 3 (Recipe) bridge: turning a New Tab 2 simulation into an ordered,
# per-connector list of recipe-relevant features. See qt_gui_full_reference.md
# Part 6/10 -- this is the "New Tab 3" piece that was previously agreed-but-
# unbuilt. Deliberately produces HIGH-LEVEL features (one per connector
# event: a mill, a deposition, an oxidation), NOT literal machine steps --
# recipe_generator.generate_recipe_steps already owns turning a feature list
# into the full literal step sequence (Ti gather, mill gas/discharge/beam
# boilerplate, shutter cycles, pump/wait), and that machinery is reused
# completely unchanged rather than duplicated here.
# ============================================================================

@dataclass
class SimulationFeature:
    """One high-level, per-connector recipe-relevant event derived from a
    New Tab 2 simulation run. This is the unit New Tab 3's timeline shows
    one block per, and the unit that gets expanded into literal RecipeStep
    machine steps."""
    kind: str                          # "mill" | "deposition" | "oxidation"
    alpha: Optional[float]
    theta: Optional[float]
    thickness_nm: Optional[float]
    label: str                          # e.g. "Deposit Finger[0]"
    provenance: str                     # human explanation of WHY this feature exists, for the timeline block
    mill_duration_min: Optional[float] = None  # per-pass ion-mill duration override (item #108) -- only
                                            # meaningful for kind == "mill"; None means "use the Settings-
                                            # wide default (DesignParameters.mill_duration_min)". Set from
                                            # the Recipe Parameters panel's dynamic per-pass duration
                                            # fields (main_gui._rebuild_recipe_panel_dynamic_fields),
                                            # since different mill passes in a real recipe are not always
                                            # the same duration.
    warning_source: Optional[str] = None   # drives the timeline block's color -- matches ImportedFinger.warning_source
    source_id: Optional[str] = None        # stable_id of the originating finger/patch group, if any --
                                            # or the sentinel "__manual_insert__" for a step the user
                                            # inserted by hand (main_gui._insert_recipe_feature),
                                            # which has no originating geometry at all.
    ambiguous_choice: bool = False         # True if multiple angles were equally valid and one was picked arbitrarily
    group_source_ids: Optional[List[str]] = None  # Having two of the
                                            # same step is redundant -- each parallel-axis shape is
                                            # treated as a pair that both get
                                            # deposited in the same step: a deposition feature that
                                            # represents a SHARED physical pass covering more than one
                                            # finger at once (every member's own committed thetas AND
                                            # per-pass thicknesses are identical, so it's genuinely one
                                            # real wafer rotation, not a coincidence) stores its OTHER
                                            # member fingers' stable_ids here, keeping `source_id` as the
                                            # single primary/representative finger for this feature.
                                            # None (the overwhelmingly common case, and every existing
                                            # feature built before this fix) means "this feature belongs
                                            # to source_id alone" -- byte-identical to today's behavior.
                                            # See build_simulation_result's own comment for the merge rule,
                                            # and main_gui._attributed_to for the matching helper
                                            # every consumer should use instead of comparing source_id alone.
    manual_override: bool = False          # True once the user has hand-edited this feature's alpha/theta
                                            # on Simulate Design's "Mill / Oxidation Placement" card -- the
                                            # generated recipe can be wrong for a real design, so that
                                            # card lets alpha/theta be overridden per-
                                            # feature; this flag is what drives the "Manual Override"
                                            # warning badge on that block. Superseding an earlier
                                            # design assumption, an override now DELIBERATELY SURVIVES a
                                            # fresh simulation run instead of being reset -- a new
                                            # SimulationFeature list is still built from scratch on every
                                            # regenerate, but main_gui.MainWindow.
                                            # _reapply_manual_angle_overrides re-applies any override
                                            # (matched by (kind, label), or re-inserts a whole hand-created
                                            # "__manual_insert__" step) onto that fresh list immediately
                                            # afterward, so a hand-typed value keeps applying until the
                                            # user changes something that actually invalidates it.


@dataclass
class SimulationResult:
    """The full bridge object between New Tab 2 and New Tab 3. `fingerprint`
    is what New Tab 3's staleness banner compares against the CURRENT
    simulation state -- if they differ, the geometry/process parameters
    have changed since this recipe was generated and it should be
    re-generated before being trusted."""
    fingerprint: str
    finger_summaries: List[dict]        # [{stable_id, warning_source, deposited_second, active_theta_deg}]
    patches_needs_two_depositions: bool
    patches_thetas: List[float]
    alpha_deposition: float
    has_disconnections: bool
    disconnection_notes: List[str]
    features: List[SimulationFeature]
    not_set: bool = False               # True if deposit order was never set -- features will be empty
    not_set_reason: str = ""


def _simulation_fingerprint(junction: "ImportedJunction", p,
                             finger_layer_overrides: Optional[Dict[int, List[float]]] = None,
                             patch_layer_overrides: Optional[List[float]] = None,
                             finger_thickness_overrides: Optional[Dict[int, List[Optional[float]]]] = None,
                             patch_thickness_overrides: Optional[List[Optional[float]]] = None) -> str:
    """A short hash of exactly the inputs that determine the recipe's
    content -- per-finger warning_source/deposit-order/active-theta, the
    process parameters that feed the shadow formulas, and whatever the
    user has committed in New Tab 2's
    per-feature Deposition Angles tree (self._sim_finger_layers/
    _sim_patch_layers). Two calls with identical fingerprints are
    guaranteed to produce the identical feature list; any change to
    geometry, deposit order, these process parameters, OR a committed
    angle/layer-count edit changes the fingerprint -- which is exactly
    the staleness signal New Tab 3's banner needs. Before this fix, an
    edit made only to the override layers (with geometry/params
    otherwise unchanged) was invisible to the fingerprint, so an
    already-generated recipe never got flagged stale even though the
    user's edit was silently never used to build it."""
    parts = [
        f"{f.stable_id}:{f.warning_source}:{f.deposited_second}:{f.preferred_theta_deg}:{f.self_shadow_safe_theta_deg}"
        for f in junction.fingers
    ]
    parts.append(f"alpha={getattr(p, 'alpha_deposition', None)}")
    parts.append(f"alpha_mill={getattr(p, 'alpha_mill', None)}")
    parts.append(f"pmma={getattr(p, 'pmma_thickness_nm', None)}")
    parts.append(f"pmgi={getattr(p, 'pmgi_thickness_nm', None)}")
    parts.append(f"eh={getattr(p, 'electrode_height_nm', None)}")
    parts.append(f"al1={getattr(p, 'al_thickness_1_nm', None)}")
    parts.append(f"al2={getattr(p, 'al_thickness_2_nm', None)}")
    parts.append(f"al2w2={getattr(p, 'al_thickness_2_warn2_nm', None)}")
    parts.append(f"patch_t={getattr(p, 'patch_thickness_nm', None)}")
    if finger_layer_overrides:
        for idx in sorted(finger_layer_overrides):
            parts.append(f"fov{idx}=" + ",".join(f"{t:g}" for t in finger_layer_overrides[idx]))
    if patch_layer_overrides is not None:
        parts.append("pov=" + ",".join(f"{t:g}" for t in patch_layer_overrides))
    # Per-pass thickness overrides also
    # determine the recipe's content now (target thickness echoes right
    # into each deposition step's text) -- must be part of the staleness
    # signal too, same reasoning as the theta overrides above.
    if finger_thickness_overrides:
        for idx in sorted(finger_thickness_overrides):
            parts.append(f"fth{idx}=" + ",".join(
                ("" if t is None else f"{t:g}") for t in finger_thickness_overrides[idx]))
    if patch_thickness_overrides is not None:
        parts.append("pth=" + ",".join(("" if t is None else f"{t:g}") for t in patch_thickness_overrides))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def default_finger_pass_thickness(f: "ImportedFinger", active_theta: float, p) -> Optional[float]:
    """The auto-computed per-pass thickness DEFAULT -- factored out so
    both build_simulation_result's fallback path AND main_gui.py's
    Deposition Angles tree (which seeds every new pass row with this
    value the moment it's added, so the user always sees and edits a
    real number, never an invisible auto-placed one -- deposition
    thickness should not simply be auto-placed into the generated
    recipe; the user needs a place to determine the thickness of each
    deposition step) use the identical selection
    logic. See DesignParameters' own field comments for the underlying
    process convention (thin first finger, thicker second finger,
    thicker still on a Warning #2-risky pass)."""
    if not f.deposited_second:
        return getattr(p, "al_thickness_1_nm", None)
    is_w2_risky_pass = (
        f.self_shadow_safe_theta_deg is not None
        and abs((active_theta - f.theta_toward_patch_deg + 180.0) % 360.0 - 180.0) < 0.5
    )
    if is_w2_risky_pass:
        thickness = getattr(p, "al_thickness_2_warn2_nm", None)
        if thickness is None:
            thickness = getattr(p, "al_thickness_2_nm", None)
        return thickness
    return getattr(p, "al_thickness_2_nm", None)


def default_patch_pass_thickness(p) -> Optional[float]:
    return getattr(p, "patch_thickness_nm", None)


def build_simulation_result(junction: "ImportedJunction", p,
                             disconnection_notes: Optional[List[str]] = None,
                             finger_layer_overrides: Optional[Dict[int, List[float]]] = None,
                             patch_layer_overrides: Optional[List[float]] = None,
                             finger_thickness_overrides: Optional[Dict[int, List[Optional[float]]]] = None,
                             patch_thickness_overrides: Optional[List[Optional[float]]] = None) -> SimulationResult:
    """The main entry point New Tab 3 calls. Requires apply_deposit_order()
    to have already been called on `junction` (matching compute_recommended_
    angles' own requirement) -- if deposit order was never set, returns a
    result with not_set=True and an empty feature list rather than guessing
    an order, since deposition order is a genuine process CHOICE that
    cannot be inferred from geometry alone (see Part 3.5 of the reference
    doc).

    finger_layer_overrides / patch_layer_overrides -- fixed to accept
    {finger_index: [theta, ...]} / [theta, ...],
    exactly the shape of main_gui.py's self._sim_finger_layers /
    self._sim_patch_layers (New Tab 2's per-feature Deposition Angles
    tree). Before this fix, this function ALWAYS recomputed a single
    angle per finger fresh from compute_recommended_angles, completely
    ignoring any edit, added/removed deposition pass, or angle-order
    change the user made in that tree -- so "Generate Recipe from
    Simulation" silently regenerated the ORIGINAL recommendation no
    matter what the user had actually committed to on screen. When an
    override is given for a finger/patches, its list of thetas is used
    VERBATIM, in the given order (so reordering two thetas in the list
    reorders the resulting deposition/mill steps too) -- falling back to
    the original single-recommended-angle behavior only when no override
    is given for that finger/for patches (keeps every existing caller,
    including tests, working unchanged)."""
    finger_layer_overrides = finger_layer_overrides or {}
    fingerprint = _simulation_fingerprint(junction, p, finger_layer_overrides, patch_layer_overrides,
                                           finger_thickness_overrides, patch_thickness_overrides)
    disconnection_notes = disconnection_notes or []

    if not junction.fingers or any(f.deposited_second is None for f in junction.fingers):
        return SimulationResult(
            fingerprint=fingerprint, finger_summaries=[], patches_needs_two_depositions=False,
            patches_thetas=[], alpha_deposition=getattr(p, "alpha_deposition", 0.0),
            has_disconnections=bool(disconnection_notes), disconnection_notes=disconnection_notes,
            features=[], not_set=True,
            not_set_reason="Deposition order isn't set yet -- run the Angle Diagnostic on New Tab 2 "
                           "and choose which finger deposits first before generating a recipe.",
        )

    rec = compute_recommended_angles(junction)
    if rec.get("not_set"):
        # Should not happen given the check above, but stay defensive rather
        # than crash if compute_recommended_angles' own precondition ever
        # diverges from this function's.
        return SimulationResult(
            fingerprint=fingerprint, finger_summaries=[], patches_needs_two_depositions=False,
            patches_thetas=[], alpha_deposition=getattr(p, "alpha_deposition", 0.0),
            has_disconnections=bool(disconnection_notes), disconnection_notes=disconnection_notes,
            features=[], not_set=True, not_set_reason="Deposit order not resolved.",
        )

    def _pick_angle(angles: List[float]) -> Tuple[float, bool]:
        """Recipes need exactly ONE concrete angle per feature. When
        compute_recommended_angles reports multiple equally-valid angles
        (a genuine 'perfect' tier with both directions fine), pick the
        lower one deterministically (sorted() already guarantees this
        ordering) and flag it as an arbitrary choice so the timeline block
        can say so honestly rather than implying only one option existed."""
        return angles[0], len(angles) > 1

    # Restructured after a real 2-finger direct-
    # contact design report, then scoped down after a real
    # patch-integrated design report: mill and deposition
    # features are appended below to a single `features` list in their
    # natural, chronological discovery order (mill(s) for a finger/patch
    # group immediately followed by its own deposit(s) -- exactly the
    # original, pre-restructuring order). For a NO-patch/no-plate design
    # where every finger contacts its electrode directly, that natural
    # order is reshuffled once at the very end (mills first, since a
    # milled surface doesn't reoxidize under vacuum, so there's no reason
    # to interleave when nothing else needs protecting in between) -- see
    # the post-processing block below. For a patch/plate-integrated
    # design, the natural order is left exactly as generated: the
    # milling happens right before the patches are deposited, not before
    # any deposition happens -- the ion milling steps for
    # patch-integrated designs happen after the junction fingers have
    # been deposited. I.e. the fingers (with their own tightly-
    # sequenced oxidation timing) deposit first, and only the patch
    # contact mill sits immediately before the patch deposition.
    features: List[SimulationFeature] = []
    finger_summaries: List[dict] = []
    fingers_in_order = sorted(junction.fingers, key=lambda f: bool(f.deposited_second))
    alpha_dep = getattr(p, "alpha_deposition", 0.0)
    # Fixed after a report: mill features used to
    # always reuse alpha_dep, the DEPOSITION alpha, silently inheriting
    # whatever that happened to be set to. Per shadowing_design_rules.md
    # section 2.1, mill alpha is its own fixed, chosen convention --
    # steep enough to hit the electrode's SIDEWALLS specifically rather
    # than its top surface -- NOT geometry-derived and NOT the same
    # number as the deposition angle. Falls back to alpha_dep only if
    # alpha_mill was never set on p at all (keeps older DesignParameters
    # objects working), not as a silent "same as deposition" default.
    alpha_mill = getattr(p, "alpha_mill", None)
    if alpha_mill is None:
        alpha_mill = alpha_dep

    oxidation_added = False
    any_first_group = any(not f.deposited_second for f in junction.fingers)
    # Having two of the same step is
    # redundant -- each
    # parallel-axis shape is treated as a pair that both get deposited in the same
    # step. Two (or more) fingers in the SAME deposited_second group
    # whose own committed deposition pass lands at the exact same theta
    # AND thickness are genuinely one real wafer rotation -- reusing the
    # SAME SimulationFeature (via group_source_ids) instead of emitting
    # one duplicate feature per finger. Keyed by (deposited_second,
    # rounded theta, rounded thickness) so floating-point noise from
    # per-finger thickness defaults doesn't prevent an otherwise-genuine
    # match; a finger whose own pass doesn't match any existing entry
    # for its group still gets its own independent feature exactly as
    # before (this is purely additive -- a design with no shared-axis
    # pair produces byte-identical output to pre-fix behavior). Mills
    # are deliberately NOT merged here -- not what was reported, and a
    # mill's own contact-cleaning role is a separate physical question
    # left for a future round if it turns out to need the same fix.
    idx_by_stable_id = {ff.stable_id: i for i, ff in enumerate(junction.fingers)}

    def _merge_key(deposited_second, theta, thickness):
        return (bool(deposited_second), round(theta, 6), round(thickness, 3) if thickness is not None else None)

    pass_feature_index: Dict[Tuple[bool, float, Optional[float]], SimulationFeature] = {}

    def _rebuild_deposit_label(feat, deposited_second, pass_num, total_passes):
        member_indices = sorted(
            {idx_by_stable_id[feat.source_id]} | {idx_by_stable_id[sid] for sid in (feat.group_source_ids or [])}
        )
        member_labels = [finger_display_label(i) for i in member_indices]
        suffix = " (1st)" if not deposited_second else " (2nd)"
        pass_suffix = f" (pass {pass_num} of {total_passes})" if total_passes > 1 else ""
        feat.label = "Deposit " + " + ".join(member_labels) + suffix + pass_suffix

    for f in fingers_in_order:
        # Generalized for series-chain
        # designs: exactly ONE oxidation step separates the whole
        # "first" axis-group's depositions from the whole "second" axis-
        # group's (apply_deposit_order's own generalization can now put
        # more than one finger in either group, e.g. Finger A and Finger
        # C sharing an axis) -- placed right before the FIRST second-
        # group finger is processed (fingers_in_order is sorted so every
        # first-group finger is already done by then), guarded by
        # oxidation_added so it's never appended more than once. For the
        # classic 2-finger design (exactly one finger per group) this
        # fires at exactly the same point as before.
        if f.deposited_second and any_first_group and not oxidation_added and len(junction.fingers) > 1:
            features.append(SimulationFeature(
                kind="oxidation", alpha=None, theta=None, thickness_nm=None,
                label="Oxidation",
                provenance="Forms the AlOx tunnel barrier between the first and second finger "
                           "depositions -- the standard SIS Josephson-junction sequence.",
            ))
            oxidation_added = True
        # Fixed after a report against a real
        # 4-finger series-chain design: this used to look up ONE shared
        # representative recommendation per deposit group ("second_finger"
        # -- always the SAME dict, for every finger in that group), so
        # every finger past the first two silently reused the first
        # second-group finger's own finger_index, angles, and warning
        # source -- exactly why a chain's 3rd/4th fingers were evaluated
        # against the wrong finger's own geometry. rec["fingers"] (see
        # compute_recommended_angles' own generalization) holds every
        # finger's own real recommendation, keyed by its own real index.
        finger_idx = junction.fingers.index(f)
        f_label = finger_display_label(finger_idx)
        finger_rec = rec.get("fingers", {}).get(finger_idx)
        if finger_rec is None:
            continue
        # Fixed to use the user's committed
        # layer list VERBATIM (in the given order) when one was passed in
        # for this finger, instead of always recomputing a single angle
        # fresh from the geometry recommendation -- see this function's
        # own docstring for the full story. Falls back to the original
        # single-recommended-angle behavior when no override applies to
        # this finger (an untouched finger, or no overrides passed at
        # all -- e.g. existing/test callers).
        override_thetas = finger_layer_overrides.get(finger_idx)
        if override_thetas:
            active_thetas = [t for t in override_thetas if t is not None]
            ambiguous = False
            committed = True
        else:
            active_theta, ambiguous = _pick_angle(finger_rec["angles"])
            active_thetas = [active_theta]
            committed = False
        if not active_thetas:
            continue
        finger_summaries.append(dict(stable_id=f.stable_id, warning_source=f.warning_source,
                                      deposited_second=bool(f.deposited_second), active_theta_deg=active_thetas[0]))

        # "Makes electrode contact" (needs a mill) and
        # "crosses an electrode boundary" (needs Warning #1 direction
        # checked) are two DIFFERENT physical questions that used to
        # share the same warning_source-based test. A shape fully buried
        # in electrode material (overlap fraction > CROSS_HI -- no
        # boundary in its own footprint, so warning_source lands on
        # "none"/"unknown" rather than "finger"/"both") still physically
        # sits ON the electrode and needs the same oxide-removal mill as
        # a shape that straddles the edge. Only a shape that's fully
        # CLEAR of the electrode (frac <= CROSS_LO, or no electrode
        # geometry at all) genuinely needs no mill.
        #
        # For a patch-integrated design,
        # milling a finger's OWN electrode contact is unnecessary --
        # and shouldn't be listed as a step -- when this finger's own
        # patch group ALSO reaches the electrode. In a patch-integrated
        # design, the finger stops in the substrate (or only partially
        # overlaps the electrode) and a patch, deposited over both the
        # finger and the electrode, is what carries the actual current
        # path: the patch mill (generated separately below) already
        # strips oxide off the finger's SIDE and the electrode surface
        # the patch lands on, giving a metal-on-metal patch-to-finger
        # and patch-to-electrode connection that carries current
        # instead of the finger's own (unmilled, oxidized) direct
        # end-contact. Milling the finger's own end-contact in addition
        # is physically harmless but not needed and not the point of a
        # patch-integrated design -- omit it whenever the finger's own
        # patch group already makes contact; keep it for a direct-
        # contact finger (no patches, or patches that don't themselves
        # reach the electrode), where the finger's own end-contact IS
        # the real electrical path and must be milled.
        patch_group_supersedes = (f.patch_outward_end is not None
                                   and (f.patch_group_overlap_frac or 0.0) > CROSS_LO)
        makes_electrode_contact = (f.finger_overlap_frac or 0.0) > CROSS_LO and not patch_group_supersedes
        if makes_electrode_contact:
            # Fixed after a real 2-finger direct-
            # contact design report: a finger's electrode contact happens
            # at ONE specific end -- its own patch_end, by construction
            # (for a direct-contact/no-patch design, the patch_end is
            # literally where the finger lands on/into the electrode --
            # the SAME axis Warning #1's own align_own_tip check in
            # main_gui.py is keyed to). A committed pass only ever
            # deposits metal anywhere near that contact if its OWN theta
            # points that direction (extends the patch end); a pass aimed
            # the other way (e.g. a second pass added purely to thicken
            # the free-standing stub end) never reaches the electrode at
            # all, so milling for it would mill a surface nothing is
            # about to touch -- exactly the reported spurious extra mill
            # (a second pass landing on the SAFE side, opposite the
            # electrode, still got a mill before this fix). One mill
            # sub-step per QUALIFYING pass -- mirrors how the patches
            # block below already handles needs_two_depositions (each
            # required angle gets its own contact-cleaning mill), just
            # filtered to the passes that actually reach the electrode.
            contact_passes = [
                (i, mt) for i, mt in enumerate(active_thetas)
                if math.cos(math.radians(mt - f.theta_toward_patch_deg)) > 0
            ]
            # Fixed for a naming bug's recipe-panel
            # manifestation: display strings use finger_display_label
            # (A/B/C/...) like every other consumer in this app --
            # source_id below stays the raw stable_id, the correct
            # internal identity key, unaffected.
            for k, (i, mt) in enumerate(contact_passes):
                label = f"Mill before {f_label}" if len(contact_passes) == 1 else \
                    f"Mill before {f_label} (pass {k + 1} of {len(contact_passes)})"
                electrode_metal = getattr(p, "electrode_metal_type", "Nb")
                skipped_note = (
                    f" ({len(active_thetas) - len(contact_passes)} other committed pass(es) for "
                    f"{f_label} land on the opposite/safe side and never reach the electrode at "
                    f"all, so they get no mill.)" if len(contact_passes) < len(active_thetas) else "")
                features.append(SimulationFeature(
                    kind="mill", alpha=alpha_mill, theta=mt, thickness_nm=None, label=label,
                    provenance=(f"{f_label} makes direct electrode contact "
                                f"({(f.finger_overlap_frac or 0.0)*100:.0f}% overlap) -- removes native oxide "
                                f"off the {electrode_metal} electrode for a clean contact. Milled at "
                                f"alpha={alpha_mill:g} deg. All mills happen before any deposition." + skipped_note),
                    warning_source=f.warning_source, source_id=f.stable_id,
                ))

        if committed:
            reason = "Angle(s) as committed in the Deposition Angles tree."
        elif f.warning_source in ("finger", "both"):
            reason = "Warning #1 (electrode-crossing): this direction reaches the electrode."
        elif f.deposited_second and f.self_shadow_safe_theta_deg is not None:
            reason = "Warning #2 (self-shadowing): deposited second, so this direction avoids thinning the JJ crossing."
        else:
            reason = "No warning applies -- deposited first, so there's no prior step to shadow against yet." \
                if not f.deposited_second else "No warning applies to this finger in either direction."
        finger_thickness_overrides_list = (finger_thickness_overrides or {}).get(finger_idx)
        for i, active_theta in enumerate(active_thetas):
            dep_label = f"Deposit {f_label}" + (" (1st)" if not f.deposited_second else " (2nd)")
            if len(active_thetas) > 1:
                dep_label += f" (pass {i + 1} of {len(active_thetas)})"
            pass_provenance = reason + (" Both directions were equally safe; the lower angle was chosen "
                                         "arbitrarily for a concrete recipe." if ambiguous else "")
            # Per-step deposition thickness
            # is something the user sets explicitly per pass, in New Tab
            # 2's Deposition Angles tree (right next to that pass's own
            # angle), rather than having it auto-placed into the
            # generated recipe. An
            # explicit override always wins when given. is_w2_risky_pass
            # is still computed either way, purely to explain WHY a
            # thicker second-finger pass matters in the provenance text --
            # not to silently override what the user actually typed.
            override_list = finger_thickness_overrides_list
            override_thickness = (override_list[i] if override_list is not None and i < len(override_list)
                                    else None)
            is_w2_risky_pass = (
                f.deposited_second and f.self_shadow_safe_theta_deg is not None
                and abs((active_theta - f.theta_toward_patch_deg + 180.0) % 360.0 - 180.0) < 0.5
            )
            if override_thickness is not None:
                thickness = override_thickness
                pass_provenance += " Thickness set explicitly for this pass in the Deposition Angles tree."
                if is_w2_risky_pass:
                    pass_provenance += (" (Also lands on the Warning #2-risky angle -- extra thickness "
                                         "helps avoid a self-shadowing bottleneck.)")
            elif not f.deposited_second:
                thickness = default_finger_pass_thickness(f, active_theta, p)
            elif is_w2_risky_pass:
                thickness = default_finger_pass_thickness(f, active_theta, p)
                pass_provenance += (" Lands on the Warning #2-risky angle, so it defaults to the "
                                     "thicker second-finger setting to avoid a self-shadowing bottleneck.")
            else:
                thickness = default_finger_pass_thickness(f, active_theta, p)
            # This pass is a duplicate
            # of an already-emitted pass in the SAME deposited_second
            # group at the exact same theta and thickness -- one real
            # wafer rotation covering both fingers, not two separate
            # recipe steps. Fold this finger into that existing feature
            # (group_source_ids) instead of appending a redundant one.
            merge_key = _merge_key(f.deposited_second, active_theta, thickness)
            existing_feat = pass_feature_index.get(merge_key)
            if existing_feat is not None:
                existing_feat.group_source_ids = (existing_feat.group_source_ids or []) + [f.stable_id]
                _rebuild_deposit_label(existing_feat, f.deposited_second, i + 1, len(active_thetas))
                if "Shared physical pass" not in existing_feat.provenance:
                    existing_feat.provenance += (
                        " Shared physical pass -- this parallel-axis group deposits together in one "
                        "wafer rotation, not as separate steps.")
                continue
            new_feat = SimulationFeature(
                kind="deposition", alpha=alpha_dep, theta=active_theta, thickness_nm=thickness,
                label=dep_label,
                provenance=pass_provenance,
                warning_source=f.warning_source, source_id=f.stable_id, ambiguous_choice=ambiguous,
            )
            features.append(new_feat)
            pass_feature_index[merge_key] = new_feat


    patches_rec = rec.get("patches")
    patches_needs_two = False
    patches_thetas: List[float] = []
    # Fixed to use the same override mechanism as
    # the finger loop above -- a committed patch_layer_overrides list
    # (self._sim_patch_layers) is used VERBATIM instead of always
    # recomputing from the geometry recommendation. Checked independent
    # of patches_rec (which is only populated when patches have their
    # own warning-driven angle conflict) so a user-added/edited patch
    # layer still takes effect even when patches_rec itself is None.
    override_patch_thetas = ([t for t in patch_layer_overrides if t is not None]
                              if patch_layer_overrides is not None else None)
    if patches_rec or override_patch_thetas:
        if override_patch_thetas:
            patches_thetas = override_patch_thetas
            patches_needs_two = len(patches_thetas) > 1
        else:
            patches_needs_two = bool(patches_rec.get("needs_two_depositions"))
            patches_thetas = list(patches_rec.get("angles", []))
        patch_thickness_default = default_patch_pass_thickness(p)
        # Same fix as makes_electrode_contact above: a patch group fully
        # buried in electrode material (an "OnElectrode"-style design) makes
        # real contact and needs a mill, even though it never registers
        # warning_source "patches"/"both" (no boundary crossing found).
        patches_make_contact = any((f.patch_group_overlap_frac or 0.0) > CROSS_LO for f in junction.fingers)
        if patches_make_contact:
            # Fixed after a report on a real design: this used to
            # always emit exactly ONE mill feature, at patches_thetas[0]
            # only, even when the patches genuinely need BOTH angles
            # (patches_needs_two True) to get metal-on-metal contact --
            # milling only one side left the other side's patch-to-
            # electrode contact un-milled (oxide-covered), which is
            # exactly the real failed-design scenario documented in
            # shadowing_design_rules.md section 2.1 (finger connection
            # milled, patch connection not, ultra-high resistance
            # result). Now mirrors the deposition loop below: one mill
            # sub-step per required patch theta.
            mill_thetas = patches_thetas if patches_needs_two else (patches_thetas[:1] if patches_thetas else [None])
            for i, mt in enumerate(mill_thetas):
                label = "Mill before patches" if len(mill_thetas) == 1 else f"Mill before patches (pass {i + 1} of {len(mill_thetas)})"
                electrode_metal = getattr(p, "electrode_metal_type", "Nb")
                features.append(SimulationFeature(
                    kind="mill", alpha=alpha_mill, theta=mt,
                    thickness_nm=None, label=label,
                    provenance="A finger's patch group makes direct electrode contact -- removes "
                               f"native oxide off the {electrode_metal} electrode before the patches "
                               f"are deposited. Milled at alpha={alpha_mill:g} deg." + (
                                   " Both patch deposition angles make electrode contact, so both get "
                                   "their own mill sub-step." if len(mill_thetas) > 1 else ""),
                    warning_source="patches", source_id="patches",
                ))
        patch_reason = ("Angle(s) as committed in the Deposition Angles tree."
                         if override_patch_thetas else
                         "The two fingers' patch groups disagree on which direction reaches their "
                         "own electrode -- all patches share one layer, so this requires two "
                         "separate depositions, not a compromise angle.")
        # Patch thickness is also settable
        # PER PASS now (an explicit override from the same Deposition
        # Angles tree row, index-matched to patches_thetas), falling back
        # to patch_thickness_default (DesignParameters.patch_thickness_nm)
        # for any pass with no override -- same override-wins-else-default
        # pattern as the finger loop above.
        def _patch_pass_thickness(idx):
            override_t = (patch_thickness_overrides[idx]
                          if patch_thickness_overrides is not None and idx < len(patch_thickness_overrides)
                          else None)
            return override_t if override_t is not None else patch_thickness_default, override_t is not None

        if patches_needs_two:
            for i, pt in enumerate(patches_thetas):
                pass_thickness, was_override = _patch_pass_thickness(i)
                pass_provenance = patch_reason + (
                    " Thickness set explicitly for this pass in the Deposition Angles tree." if was_override else "")
                features.append(SimulationFeature(
                    kind="deposition", alpha=alpha_dep, theta=pt, thickness_nm=pass_thickness,
                    label=f"Deposit patches (pass {i + 1} of {len(patches_thetas)})",
                    provenance=pass_provenance,
                    # Fixed after a report that adding a new deposition
                    # step could revert angles that had been manually
                    # changed on a previous deposition step back to their
                    # original values: every patch
                    # feature used to leave source_id unset (None), unlike
                    # a finger's own features (source_id=f.stable_id) --
                    # so main_gui.MainWindow._reapply_manual_angle_
                    # overrides' stable (kind, source_id, pass index) key
                    # fell back to matching by (kind, label) alone for
                    # patches specifically, which breaks the moment this
                    # label's own "N" changes (a new patch pass added).
                    # "patches" is a fixed, stable group id here (there is
                    # only ever one shared patch layer across both finger
                    # sides, unlike fingers which each have their own
                    # stable_id), giving every patch pass the same
                    # pass-count-change immunity a finger's passes already
                    # had.
                    warning_source="patches", source_id="patches",
                ))
        elif patches_thetas:
            active_theta, ambiguous = _pick_angle(patches_thetas)
            provenance = (patch_reason if override_patch_thetas else
                          "A single angle both patch groups agree is safe." + (
                              " Multiple angles were equally safe; the lower one was chosen "
                              "arbitrarily." if ambiguous else ""))
            pass_thickness, was_override = _patch_pass_thickness(0)
            if was_override:
                provenance += " Thickness set explicitly for this pass in the Deposition Angles tree."
            features.append(SimulationFeature(
                kind="deposition", alpha=alpha_dep, theta=active_theta, thickness_nm=pass_thickness,
                label="Deposit patches",
                provenance=provenance,
                # source_id="patches" -- see the two-pass branch above's
                # own comment for why this matters for manual-override
                # survival across a pass-count change.
                warning_source="patches", source_id="patches", ambiguous_choice=ambiguous,
            ))

    # For a design with NO patches or
    # adhesion plates where a finger contacts its electrode directly,
    # there's a THIRD mill the per-finger sidewall mills above never
    # cover -- a flat, untilted (alpha=0, theta=0) blanket top mill. It's
    # only safe here because nothing has been deposited yet at this point
    # in the recipe once mills are batched first: with no patches/plates,
    # every finger's own footprint is still bare resist-opening/exposed
    # electrode, so a top-down beam can blanket the ENTIRE exposed top
    # surface -- the substrate, every electrode's own top face, and every
    # finger's future footprint -- in one pass, cleaning the finger's
    # TOP-side overlap with its electrode too (not just the angled
    # sidewall contact the per-finger mills above target), for the
    # lowest-resistance metal-to-electrode connection.
    #
    # Fixed after a real patch-integrated design
    # report: the "batch every mill first, before any deposition"
    # reshuffle above was scoped too broadly -- it only reflects how a
    # NO-patch, direct-electrode-contact design actually runs (nothing is
    # deposited yet when the mills happen, so there's nothing to protect
    # by interleaving). A patch-integrated design is different: the
    # milling happens right before the patches are deposited, not before
    # any deposition happens -- the ion milling steps for
    # patch-integrated designs happen after the junction fingers have
    # been deposited. So for a patch design, `features` is left exactly
    # as generated above -- fingers (with their own tightly-sequenced
    # oxidation) deposit first, and any mill (a finger's own contact
    # mill, if it has one, or the patch contact mill) stays immediately
    # before the deposition it protects, right where it was generated.
    # Only a design with no PATCHES gets the mills-first reshuffle (plus
    # the blanket top mill inserted at the very front).
    #
    # A second fix, prompted by a report against a real no-patch
    # Manhattan JJ design: this used to also gate on `plate`
    # (any(s.kind in ("patch", "plate") ...)), so a direct-contact design
    # WITH adhesion plates but genuinely no patches -- exactly what a
    # "NoPatch" design is -- was wrongly treated as if it were patch-
    # integrated, silently keeping the old interleaved Mill A/Deposit A/
    # Oxidize/Mill B/Deposit B order and skipping the blanket mill
    # entirely. An adhesion plate is not a reason to interleave: per
    # _apply_junction_group_scale's own docstring elsewhere in this
    # codebase and _plate_thickness_bands' own logic, "a plate rides
    # along with every one of its owning finger's own real, committed
    # deposition passes -- it's never separately scheduled." Only a
    # genuine PATCH is deposited as its own later step, needing its own
    # contact mill placed immediately before it to protect the AlOx
    # tunnel barrier already laid down -- a plate has no such separate
    # timing to protect, so its presence alone must never block the
    # mills-first/blanket-mill treatment. Reproduced directly: has_
    # adhesion_pads=True, patch_integrated=False produced the exact
    # wrong "Mill A, Deposit A, Oxidize, Mill B, Deposit B" sequence
    # before this fix, and the correct batched-mills-plus-blanket
    # sequence after it -- while a genuinely patch_integrated=True
    # design (adhesion plates AND patches) still correctly keeps its
    # interleaved order, unaffected by this change.
    has_patches = any(s.kind == "patch" for s in junction.other_shapes)
    has_mills = any(ft.kind == "mill" for ft in features)
    if has_mills and not has_patches:
        mills = [ft for ft in features if ft.kind == "mill"]
        body = [ft for ft in features if ft.kind != "mill"]
        electrode_metal = getattr(p, "electrode_metal_type", "Nb")
        mills.insert(0, SimulationFeature(
            kind="mill", alpha=0.0, theta=0.0, thickness_nm=None,
            label="Top Mill (blanket)",
            provenance=(
                "No patches in this design (an adhesion plate, if present, rides along with its own "
                "finger's deposition pass and is never separately scheduled) -- nothing is deposited "
                "yet at this point in the recipe, so a flat, untilted (alpha=0) top-down mill is safe "
                "here. Blankets the substrate, every electrode's top face, and every finger's "
                f"footprint in one pass, so each finger's top-side overlap is oxide-free for a clean "
                f"metal-to-{electrode_metal} contact."
            ),
            warning_source=None,
        ))
        features = mills + body

    # Fixed after a report on a real test design: a final,
    # SHORT static oxidation is missing after the entire junction has
    # finished depositing, to protect the
    # junction once it leaves the chamber -- separate from the
    # AlOx-tunnel-barrier oxidation that (when present) sits between the
    # first and second finger depositions above. Always appended last,
    # for any design that deposits anything at all; uses its own
    # p.final_oxidation_time_min (default 5 min) rather than
    # p.oxidation_time_min, since this is a passivation step, not the
    # junction's own tunnel barrier.
    if any(ft.kind == "deposition" for ft in features):
        features = features + [SimulationFeature(
            kind="oxidation", alpha=None, theta=None, thickness_nm=None,
            label="Final protective oxidation",
            provenance=(f"Passivates the finished junction with a short "
                        f"{p.final_oxidation_time_min:g}-minute static oxidation once every "
                        "deposition pass is complete, before the wafer leaves the chamber -- "
                        "distinct from the AlOx tunnel-barrier oxidation (if any) between the "
                        "first and second finger depositions above."),
            warning_source="final_oxidation",
        )]

    return SimulationResult(
        fingerprint=fingerprint, finger_summaries=finger_summaries,
        patches_needs_two_depositions=patches_needs_two, patches_thetas=patches_thetas,
        alpha_deposition=alpha_dep, has_disconnections=bool(disconnection_notes),
        disconnection_notes=disconnection_notes, features=features, not_set=False,
    )
