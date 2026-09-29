"""
shape_editor.py

Generic, reusable shape-editing data model for the interactive canvas
editor -- a free-form list of shapes (rectangle for now; circle/oval/
polygon are straightforward additions once this core loop is proven),
each with an exact position/size/angle and a ROLE (electrode, finger,
patch, plate, other). This is deliberately NOT hardcoded to "exactly one
horizontal electrode and one vertical electrode" the way the old
DesignParameters fields are -- the whole point is to let you place as
many shapes as you want, of whatever role, freely.

Kept independent of Qt/matplotlib so it can be tested and reused
without a GUI -- the GUI layer (main_gui.py) owns mouse events and
calls into a ShapeScene instance.
"""

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import shapely.geometry as sg
import shapely.affinity as affinity


@dataclass
class EditableShape:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: str = "rect"          # "rect" now; "circle"/"oval"/"polygon" are natural extensions
    cx: float = 0.0             # center, world coords (um)
    cy: float = 0.0
    w: float = 2.0              # full width (um) -- for "circle", w==h and both mean diameter
    h: float = 2.0              # full height (um)
    angle_deg: float = 0.0      # "rect"/"circle"/"oval": rotation about (cx, cy), applied live by
                                 # to_world_polygon(). "polygon": NOT applied to anything (see
                                 # to_world_polygon's own docstring) -- purely a running,
                                 # always-continuous display/bookkeeping total of how far this
                                 # polygon has been rotated (by its rotation handle, or by typing a
                                 # value into the Angle field) since it was drawn. self.points always
                                 # already reflects the real rotation directly; this field exists only
                                 # so the Shape Properties Angle field has something exact and
                                 # jump-free to show, instead of re-deriving "the current angle" from
                                 # the polygon's own geometry on every redraw (a bounding-box-based
                                 # heuristic like _min_rotated_rect_angle_deg is only continuous modulo
                                 # the bounding rectangle's own symmetry, so it can jump unpredictably
                                 # as a shape rotates smoothly through certain angles).
    role: str = "electrode"     # "electrode" | "finger" | "patch" | "plate" | "other"
    points: Optional[List[Tuple[float, float]]] = None  # only used by kind="polygon"

    # Scale reference ("1.00") -- the w/h (or polygon points) this shape
    # had the last time its size was DELIBERATELY set, either at
    # creation or via a direct edit (typed W/H, a drag-resize, etc).
    # apply_scale() always measures against THIS, never against
    # whatever a previous apply_scale() left behind -- see apply_scale's
    # own docstring for why that distinction matters.
    base_w: Optional[float] = None
    base_h: Optional[float] = None
    base_points: Optional[List[Tuple[float, float]]] = None
    scale_factor: float = 1.0   # the factor currently applied relative to base_w/h/points

    def __post_init__(self):
        if self.base_w is None:
            self.capture_scale_base()

    def capture_scale_base(self):
        """Snapshots the CURRENT w/h (or polygon points) as the new
        "1.00" reference for future apply_scale() calls, and resets
        scale_factor back to 1.0 -- this size IS the original now. Call
        this after any DIRECT dimension edit (typed W/H fields, etc),
        anything that isn't itself an apply_scale() -- so 1.00 always
        means "the shape as it was last deliberately set," not
        "whatever a previous scale apply computed.\""""
        self.base_w = self.w
        self.base_h = self.h
        self.base_points = list(self.points) if self.points else None
        self.scale_factor = 1.0

    def apply_scale(self, factor: float):
        """Scales relative to the captured base (self.base_w/h/points),
        NOT the shape's current w/h/points -- so typing 0.5 then later
        2.00 both measure against the SAME original size (50% of
        original, then 200% of original), rather than compounding onto
        whatever the previous apply left behind (0.5x, then 2x-of-that
        == only 1.0x of the true original). This is what makes 1.00
        always mean "the original size," matching a real graphics
        editor's scale tool instead of a relative multiply-in-place."""
        if self.base_w is None:
            self.capture_scale_base()
        self.scale_factor = factor
        if self.kind == "polygon" and self.base_points:
            cx = sum(p[0] for p in self.base_points) / len(self.base_points)
            cy = sum(p[1] for p in self.base_points) / len(self.base_points)
            self.points = [(cx + (px - cx) * factor, cy + (py - cy) * factor)
                           for px, py in self.base_points]
        else:
            self.w = self.base_w * factor
            self.h = self.base_h * factor

    def to_world_polygon(self, n_ellipse_pts: int = 32) -> List[Tuple[float, float]]:
        """Exact rect corners, an ellipse approximation for circle/oval, or
        the raw points for a polygon -- always in true world coordinates,
        ready to feed straight into the same shapely/rendering pipeline as
        everything else in the app (ImportedJunction, electrode_polygons,
        etc.).

        A "polygon"'s own points are ALWAYS already-absolute,
        already-in-world coordinates (every tool that edits one --
        _apply_node_drag's per-vertex branch, the rotation-handle drag,
        typing a value into the Angle field -- writes directly into
        self.points, never into cx/cy/angle_deg), so unlike rect/circle/
        oval, a polygon is returned as-is with NO further rotate/translate
        applied on top. Previously the rotate-by-angle_deg-then-
        translate-by-(cx,cy) steps below ran unconditionally for every
        kind, polygon included -- harmless in practice ONLY because
        nothing ever gave a polygon shape a nonzero cx/cy/angle_deg (a
        fresh polygon is always created with all three at exactly 0, and
        every edit path deliberately left them untouched), so this was a
        latent, never-triggered double-transform bug: had anything ever
        set a nonzero angle_deg on a polygon, this would have rotated its
        already-absolute points a SECOND time around the origin. Fixed by
        skipping the transform for polygon kind entirely, which is a
        no-op for every pre-existing design (angle_deg/cx/cy
        were always 0 for a polygon) and makes it safe for angle_deg to
        now serve as a polygon's own plain, continuously-tracked display
        angle (the rotation-handle drag's docstring in main_gui.py
        explains why a continuous tracked value is needed instead of
        re-deriving "the current angle" from the polygon's own geometry
        on every redraw)."""
        if self.kind == "polygon" and self.points:
            return list(sg.Polygon(self.points).exterior.coords)
        elif self.kind in ("circle", "oval"):
            # unit circle scaled to (w/2, h/2), then rotated + translated --
            # shapely's affine ops keep this exact and cheap
            poly = sg.Point(0, 0).buffer(1.0, resolution=max(4, n_ellipse_pts // 4))
            poly = affinity.scale(poly, self.w / 2.0, self.h / 2.0, origin=(0, 0))
        else:  # "rect"
            hw, hh = self.w / 2.0, self.h / 2.0
            poly = sg.Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        poly = affinity.rotate(poly, self.angle_deg, origin=(0, 0), use_radians=False)
        poly = affinity.translate(poly, self.cx, self.cy)
        return list(poly.exterior.coords)

    def contains_point(self, x: float, y: float) -> bool:
        try:
            return sg.Point(x, y).within(sg.Polygon(self.to_world_polygon()))
        except Exception:
            return False


class ShapeScene:
    """A free-form collection of EditableShape, with selection state and
    hit-testing -- the model behind the interactive canvas editor.
    Deliberately simple (a flat list, linear hit-test) since a JJ design
    has at most a handful of shapes, not thousands."""

    def __init__(self):
        self.shapes: List[EditableShape] = []
        self.selected_id: Optional[str] = None

    def add(self, kind: str = "rect", cx: float = 0.0, cy: float = 0.0,
            w: float = 2.0, h: float = 2.0, angle_deg: float = 0.0,
            role: str = "electrode", points: Optional[List[Tuple[float, float]]] = None) -> str:
        shp = EditableShape(kind=kind, cx=cx, cy=cy, w=w, h=h, angle_deg=angle_deg, role=role, points=points)
        self.shapes.append(shp)
        self.selected_id = shp.id
        return shp.id

    def remove(self, shape_id: str):
        self.shapes = [s for s in self.shapes if s.id != shape_id]
        if self.selected_id == shape_id:
            self.selected_id = None

    def duplicate(self, shape_id: str, dx: Optional[float] = None, dy: Optional[float] = None) -> Optional[str]:
        """Clones the given shape (new id, offset by (dx, dy) so the
        copy doesn't land exactly on top of the original) and selects
        the new copy -- the real underlying-data addition, same as
        add(), so the clone feeds every downstream consumer (shadowing,
        recipe generation) exactly the way any other shape does, not
        just a visual duplicate. Defaults dx/dy to 20% of the shape's
        own width/height when not given, so a tiny finger stub and a
        large plate both get a visible, proportionate offset instead of
        a fixed physical distance that would be invisible on one and
        absurd on the other."""
        shp = self.get(shape_id)
        if shp is None:
            return None
        if dx is None:
            dx = max(shp.w, 0.2) * 0.2
        if dy is None:
            dy = max(shp.h, 0.2) * 0.2
        clone = copy.deepcopy(shp)
        clone.id = uuid.uuid4().hex[:8]
        clone.cx += dx
        clone.cy += dy
        if clone.points:
            clone.points = [(px + dx, py + dy) for px, py in clone.points]
        if clone.base_points:
            clone.base_points = [(px + dx, py + dy) for px, py in clone.base_points]
        self.shapes.append(clone)
        self.selected_id = clone.id
        return clone.id

    def get(self, shape_id: str) -> Optional[EditableShape]:
        return next((s for s in self.shapes if s.id == shape_id), None)

    def selected(self) -> Optional[EditableShape]:
        return self.get(self.selected_id) if self.selected_id else None

    def hit_test(self, x: float, y: float) -> Optional[str]:
        """Topmost (last-added) shape containing (x, y), or None. Linear
        scan in reverse insertion order so newer/foreground shapes win
        when overlapping, matching typical draw-order expectations."""
        for shp in reversed(self.shapes):
            if shp.contains_point(x, y):
                return shp.id
        return None

    def move_selected_to(self, x: float, y: float):
        shp = self.selected()
        if shp is not None:
            shp.cx, shp.cy = x, y

    def bounds(self):
        if not self.shapes:
            return (0.0, 0.0, 0.0, 0.0)
        xs, ys = [], []
        for shp in self.shapes:
            for x, y in shp.to_world_polygon():
                xs.append(x); ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))

    def to_electrode_polygons(self, roles: Tuple[str, ...] = ("electrode",)) -> List[List[Tuple[float, float]]]:
        """Convenience: world-coord polygon list for shapes of the given
        role(s) -- feeds directly into ImportedJunction.electrode_polygons/
        electrode_material, same format the GDS import path already uses."""
        return [s.to_world_polygon() for s in self.shapes if s.role in roles]


# ---------------------------------------------------------------------
# Polygon-tool auto-correction ("snapping")
# ---------------------------------------------------------------------
#
# The freehand polygon tool is flexible, but that flexibility is a
# liability for the common case: a user placing 4 clicks meaning "a
# rectangle" will, in practice, place a slightly skewed quadrilateral
# (unequal side lengths, corner angles a few degrees off 90) purely from
# hand imprecision, and without correction that imprecise shape is taken
# completely literally. Beyond visual sloppiness, an imprecise polygon is
# a real downstream-compatibility risk -- role classification, axis/end
# detection (junction_import._polygon_axis_and_ends), width measurement,
# and shadowing overlap checks all assume a well-defined long axis and a
# consistent width, which an irregular hand-drawn polygon degrades or
# breaks outright.
#
# regularize_drawn_polygon() is deliberately NOT a library of separate
# per-shape-type detectors ("is this a rectangle? is this a trapezoid?
# is this a triangle?"). A single unified algorithm handles all of
# those, plus many other shapes, without hardcoding a closed list of
# recognized names:
#
#   1. Find the polygon's dominant axis (the angle of its minimum-area
#      bounding rectangle -- a robust "which way is this shape mostly
#      pointing" answer for anything from a near-rectangle to a
#      lopsided trapezoid to an irregular hexagon).
#   2. Build a small set of "nice" candidate edge directions relative to
#      that axis (every 15 degrees: parallel, perpendicular, and the
#      45/30/60-degree diagonals a chamfered or trapezoidal patch
#      commonly wants).
#   3. For each edge of the ORIGINAL drawn polygon, if its actual angle
#      is already close to one of those candidates (within a tolerance),
#      snap it exactly onto that candidate direction; if it is not
#      close to any of them, leave that edge's direction exactly as
#      drawn. This is the crux of the approach: a shape that is
#      genuinely close to rectilinear gets cleaned up, while a
#      deliberately oblique or organic shape (no edge close to any nice
#      angle) is left alone rather than being forced into a shape that
#      was never actually drawn.
#   4. Walk the polygon rebuilding each edge at its (possibly snapped)
#      direction with its length rounded to a sensible round number,
#      and require the walk to close back up near where it started; if
#      it doesn't (the snapping was inconsistent for this shape, e.g. a
#      genuinely irregular pentagon), bail out and return the polygon
#      completely unchanged rather than emit something distorted --
#      "do nothing" is always a safe fallback here.
#
# A 4-vertex result whose four corner angles are ALL within tolerance of
# 90 degrees is returned as an exact axis-aligned-to-its-own-angle
# rectangle (kind="rect", not kind="polygon") -- this is the single
# highest-value case (most junction features -- fingers, patches,
# plates -- are rectangles) and it plugs directly into the EditableShape
# "rect" representation every other tool (Nudge & Scale, the rect-drawing
# tool, the Shape Properties card's Length/Width/Angle fields) already
# assumes, rather than leaving even a "perfect" rectangle as a 4-point
# polygon that downstream code has to re-derive w/h/angle from.
_SNAP_ANGLE_TOL_DEG = 6.0           # how close an edge must be to a "nice" angle to be snapped -- kept
                                     # BELOW half of _SNAP_CANDIDATE_STEP_DEG (7.5) deliberately, so a
                                     # genuine gap exists between candidates and a truly oblique/organic
                                     # edge angle can land in it and stay unsnapped, rather than every
                                     # possible angle being "close enough" to something
_SNAP_CANDIDATE_STEP_DEG = 15.0     # spacing of candidate edge directions, relative to the dominant axis
_LENGTH_ROUND_UM = 0.01             # round snapped/regularized edge lengths to the nearest 10 nm
_RECT_ANGLE_TOL_DEG = 6.0           # how close ALL 4 corners must be to 90 deg to call it a rectangle
_CLOSURE_TOL_FRACTION = 0.08        # max allowed re-closure gap, as a fraction of the shape's own span


def _angle_norm_180(deg: float) -> float:
    """Normalizes an UNDIRECTED line angle into (-90, 90] -- a polygon
    edge and its reverse point the same way as a line, so angles that
    differ by exactly 180 must compare equal."""
    a = deg % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def _round_to(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def _min_rotated_rect_angle_deg(points: List[Tuple[float, float]]) -> float:
    """Angle (degrees) of the polygon's minimum-area bounding rectangle
    -- a robust "dominant axis" for any roughly-rectilinear shape,
    including trapezoids and irregular quads/hexagons, not just true
    rectangles."""
    poly = sg.Polygon(points).convex_hull
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 3:
        return 0.0
    (x0, y0), (x1, y1) = coords[0], coords[1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _snap_edge_angles(points: List[Tuple[float, float]], angle_tol_deg: float,
                       candidate_step_deg: float) -> List[float]:
    """Returns, for each edge i -> i+1 (wrapping), either a snapped
    "nice" direction (degrees) if the edge's own angle is within
    tolerance of one, or None if it should keep its original angle."""
    n = len(points)
    dominant = _min_rotated_rect_angle_deg(points)
    n_candidates = int(round(180.0 / candidate_step_deg))
    candidates = [_angle_norm_180(dominant + k * candidate_step_deg) for k in range(n_candidates)]
    snapped = []
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        edge_angle = _angle_norm_180(math.degrees(math.atan2(y1 - y0, x1 - x0)))
        best = min(candidates, key=lambda c: min(abs(edge_angle - c), 180.0 - abs(edge_angle - c)))
        diff = min(abs(edge_angle - best), 180.0 - abs(edge_angle - best))
        snapped.append(best if diff <= angle_tol_deg else None)
    return snapped


def _rebuild_from_edge_angles(points: List[Tuple[float, float]], edge_angles: List[Optional[float]],
                               length_round_um: float, closure_tol_fraction: float) -> Optional[List[Tuple[float, float]]]:
    """Walks the polygon rebuilding each edge at its target angle
    (snapped, or the ORIGINAL edge's own angle where unsnapped) with its
    length rounded, starting from vertex 0. Returns the corrected vertex
    list, or None if the walk fails to close back up near vertex 0
    (a safety fallback -- an inconsistent set of snaps for this
    particular shape, so leave the original polygon untouched instead of
    emitting something distorted)."""
    n = len(points)
    x0, y0 = points[0]
    xs, ys = [x0], [y0]
    for i in range(n):
        px, py = points[i]
        qx, qy = points[(i + 1) % n]
        length = math.hypot(qx - px, qy - py)
        length = _round_to(max(length, 1e-9), length_round_um)
        target_angle = edge_angles[i]
        if target_angle is None:
            target_angle = math.degrees(math.atan2(qy - py, qx - px))
        rad = math.radians(target_angle)
        # preserve the ORIGINAL edge's direction sign (snapping is
        # undirected/mod-180, so recover which of the two directions
        # along that line this edge actually travels)
        orig_rad = math.atan2(qy - py, qx - px)
        if math.cos(orig_rad - rad) < 0:
            rad += math.pi
        nx, ny = xs[-1] + length * math.cos(rad), ys[-1] + length * math.sin(rad)
        if i < n - 1:
            xs.append(nx)
            ys.append(ny)
        else:
            closing_gap = math.hypot(nx - x0, ny - y0)
    span = max(max(p[0] for p in points) - min(p[0] for p in points),
               max(p[1] for p in points) - min(p[1] for p in points), 1e-9)
    if closing_gap > closure_tol_fraction * span:
        return None
    return list(zip(xs, ys))


def _corner_angles_deg(points: List[Tuple[float, float]]) -> List[float]:
    """Interior angle (degrees, 0-180) at each vertex of a simple
    polygon."""
    n = len(points)
    angles = []
    for i in range(n):
        px, py = points[i - 1]
        cx, cy = points[i]
        nx, ny = points[(i + 1) % n]
        v1 = (px - cx, py - cy)
        v2 = (nx - cx, ny - cy)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag = math.hypot(*v1) * math.hypot(*v2)
        if mag < 1e-12:
            angles.append(180.0)
            continue
        cos_a = max(-1.0, min(1.0, dot / mag))
        angles.append(math.degrees(math.acos(cos_a)))
    return angles


def regularize_drawn_polygon(points: List[Tuple[float, float]],
                              angle_tol_deg: float = _SNAP_ANGLE_TOL_DEG,
                              candidate_step_deg: float = _SNAP_CANDIDATE_STEP_DEG,
                              length_round_um: float = _LENGTH_ROUND_UM,
                              rect_angle_tol_deg: float = _RECT_ANGLE_TOL_DEG,
                              closure_tol_fraction: float = _CLOSURE_TOL_FRACTION) -> dict:
    """Auto-corrects a freshly hand-drawn polygon (the raw vertex list
    from the polygon tool) toward a clean, "even" version of whatever it
    is close to, WITHOUT hardcoding a fixed list of recognized shape
    names -- see the module-level comment above for the full algorithm
    and rationale.

    Returns a dict describing the corrected shape, one of:
      {"kind": "rect", "cx", "cy", "w", "h", "angle_deg", "note", "changed": True}
      {"kind": "polygon", "points": [...], "note", "changed": bool}

    "changed" is False when the drawn polygon was already left alone
    (either it was too irregular to confidently snap, or -- for a
    triangle, which has no useful universal angle-snap target -- only
    length rounding was applied and even that made no visible
    difference); callers can use it to skip a status message when
    nothing meaningfully happened.
    """
    pts = [(float(x), float(y)) for x, y in points]
    n = len(pts)
    if n < 3:
        return {"kind": "polygon", "points": pts, "note": "too few points to regularize", "changed": False}

    if n == 3:
        # No universal angle-snap target for a triangle (unlike a
        # quad/hexagon there's no natural "perpendicular" partner edge
        # to snap most triangles onto) -- but DO snap a near-right-angle
        # corner to exactly 90 degrees (a very common intent: an L-shaped
        # patch corner or a right-triangle chamfer), and round edge
        # lengths regardless.
        corner_angles = _corner_angles_deg(pts)
        right_idx = min(range(3), key=lambda i: abs(corner_angles[i] - 90.0))
        changed = False
        if abs(corner_angles[right_idx] - 90.0) <= angle_tol_deg:
            a = pts[right_idx]
            b = pts[(right_idx + 1) % 3]
            c = pts[(right_idx - 1) % 3]
            len_ab = _round_to(math.hypot(b[0] - a[0], b[1] - a[1]), length_round_um)
            len_ac = _round_to(math.hypot(c[0] - a[0], c[1] - a[1]), length_round_um)
            ab_hat = math.atan2(b[1] - a[1], b[0] - a[0])
            ac_hat_orig = math.atan2(c[1] - a[1], c[0] - a[0])
            # rotate ac to be exactly perpendicular to ab, keeping ab fixed
            perp = ab_hat + math.pi / 2.0
            if math.cos(ac_hat_orig - perp) < 0:
                perp += math.pi
            new_b = (a[0] + len_ab * math.cos(ab_hat), a[1] + len_ab * math.sin(ab_hat))
            new_c = (a[0] + len_ac * math.cos(perp), a[1] + len_ac * math.sin(perp))
            new_pts = [None, None, None]
            new_pts[right_idx] = a
            new_pts[(right_idx + 1) % 3] = new_b
            new_pts[(right_idx - 1) % 3] = new_c
            pts = new_pts
            changed = True
            note = "right-angle corner snapped to exactly 90 deg, edge lengths rounded"
        else:
            pts = [(_round_to(x, length_round_um), _round_to(y, length_round_um)) for x, y in pts]
            note = "no near-right-angle corner found; left angles as drawn (only rounded off minor imprecision)"
        return {"kind": "polygon", "points": pts, "note": note, "changed": changed}

    # n >= 4: unified axis-snap regularization
    edge_angles = _snap_edge_angles(pts, angle_tol_deg, candidate_step_deg)
    rebuilt = _rebuild_from_edge_angles(pts, edge_angles, length_round_um, closure_tol_fraction)
    if rebuilt is None:
        return {"kind": "polygon", "points": pts,
                "note": "shape was too irregular to confidently snap; left exactly as drawn", "changed": False}

    if n == 4:
        corner_angles = _corner_angles_deg(rebuilt)
        if all(abs(a - 90.0) <= rect_angle_tol_deg for a in corner_angles):
            poly = sg.Polygon(rebuilt)
            cx, cy = poly.centroid.x, poly.centroid.y
            side_lengths = [math.hypot(rebuilt[(i + 1) % 4][0] - rebuilt[i][0],
                                        rebuilt[(i + 1) % 4][1] - rebuilt[i][1]) for i in range(4)]
            w = _round_to((side_lengths[0] + side_lengths[2]) / 2.0, length_round_um)
            h = _round_to((side_lengths[1] + side_lengths[3]) / 2.0, length_round_um)
            angle_deg = _angle_norm_180(math.degrees(math.atan2(
                rebuilt[1][1] - rebuilt[0][1], rebuilt[1][0] - rebuilt[0][0])))
            angle_deg = _round_to(angle_deg, 1.0)
            # snap to a handful of especially "nice" round angles when close
            for nice in (0.0, 45.0, 90.0, -45.0, -90.0):
                if abs(_angle_norm_180(angle_deg - nice)) < 1.5:
                    angle_deg = nice
                    break
            return {"kind": "rect", "cx": cx, "cy": cy, "w": max(w, length_round_um), "h": max(h, length_round_um),
                    "angle_deg": angle_deg, "note": f"snapped to a clean {w:g} x {h:g} um rectangle at {angle_deg:g} deg",
                    "changed": True}

    changed = any(a is not None for a in edge_angles) or rebuilt != pts
    note = ("edges snapped toward the dominant axis where close, lengths rounded"
            if any(a is not None for a in edge_angles)
            else "no edge was close enough to a clean angle; only lengths rounded")
    return {"kind": "polygon", "points": rebuilt, "note": note, "changed": changed}
