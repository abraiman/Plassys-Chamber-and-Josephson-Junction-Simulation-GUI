"""
klayout_pya_shim.py

A minimal, gdstk-backed re-implementation of the small subset of KLayout's
`pya` API that typical junction-generator macros actually use:
DPoint/DVector, DBox, DPolygon, DCplxTrans/Trans, Layout/Cell/Shapes,
and harmless no-op stand-ins for the GUI-only calls (Application,
MainWindow, LayoutView) so scripts that drive the live KLayout GUI don't
crash when run headless.

This is NOT a general KLayout replacement -- it only implements what's
needed to execute a `create_jj_cell(...)`-style generator function and
capture the polygons it inserts. The goal is to run the user's REAL
script logic unmodified rather than pattern-match/guess at its geometry,
which is far more robust across script variations.

Usage:
    shim = install_pya_shim()   # patches sys.modules['pya']
    exec(script_source, {...})  # script's `import pya` now resolves to this
    ...
    uninstall_pya_shim(shim)    # restore whatever was there before
"""

import sys
import math
import types
import gdstk


# ============================================================================
# Geometry primitives
# ============================================================================

class DVector:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return f"DVector({self.x}, {self.y})"


class DPoint:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)

    def __repr__(self):
        return f"DPoint({self.x}, {self.y})"


class Point:
    """Integer-DBU point, used by the older `pya.Trans(pya.Point(...))` API."""
    def __init__(self, x=0, y=0):
        self.x, self.y = int(x), int(y)


class DBox:
    """Axis-aligned box in float (um) coordinates: DBox(left, bottom, right, top)."""
    def __init__(self, left, bottom, right, top):
        self.left, self.bottom, self.right, self.top = float(left), float(bottom), float(right), float(top)

    def to_points(self):
        return [
            (self.left, self.bottom), (self.right, self.bottom),
            (self.right, self.top), (self.left, self.top),
        ]


class DPolygon:
    """Wraps a list of (x, y) points (or a DBox). Tracks its own transform
    chain lazily -- `.transformed(t)` returns a NEW DPolygon with the
    transform applied to its points immediately (simpler and sufficient
    for the box+rotate+translate patterns these scripts use)."""
    def __init__(self, box_or_points):
        if isinstance(box_or_points, DBox):
            self.points = box_or_points.to_points()
        else:
            self.points = [(float(x), float(y)) for x, y in box_or_points]

    def transformed(self, t):
        return DPolygon([t.apply(x, y) for x, y in self.points])

    def __repr__(self):
        return f"DPolygon({self.points})"


class DCplxTrans:
    """DCplxTrans(mag, rotation_deg, mirrored, DVector(dx, dy))
    -- matches the signature every one of these macros actually uses."""
    def __init__(self, mag=1.0, rotation_deg=0.0, mirrored=False, disp=None):
        self.mag = float(mag)
        self.rotation_deg = float(rotation_deg)
        self.mirrored = bool(mirrored)
        self.disp = disp if disp is not None else DVector(0.0, 0.0)

    def apply(self, x, y):
        if self.mirrored:
            y = -y
        x, y = x * self.mag, y * self.mag
        a = math.radians(self.rotation_deg)
        c, s = math.cos(a), math.sin(a)
        rx, ry = x * c - y * s, x * s + y * c
        return (rx + self.disp.x, ry + self.disp.y)


class Trans:
    """Integer-DBU translation-only transform: Trans(Point(x, y))."""
    def __init__(self, point=None, x=0, y=0):
        if point is not None:
            self.x, self.y = point.x, point.y
        else:
            self.x, self.y = x, y

    def apply_um(self, x, y, dbu):
        return (x + self.x * dbu, y + self.y * dbu)


# ============================================================================
# Layout / Cell / Shapes
# ============================================================================

class _LayerRef:
    def __init__(self, layer, datatype):
        self.layer, self.datatype = layer, datatype


class Shapes:
    """Cell.shapes(layer_ref) returns one of these; .insert(shape) records
    the polygon (world coords already baked in by the caller's own
    .transformed() chain) against that cell/layer."""
    def __init__(self, cell, layer_ref):
        self._cell = cell
        self._layer_ref = layer_ref

    def insert(self, shape):
        if isinstance(shape, DBox):
            pts = shape.to_points()
        elif isinstance(shape, DPolygon):
            pts = shape.points
        else:
            raise TypeError(f"klayout_pya_shim: don't know how to insert {type(shape)}")
        self._cell.polygons.append((self._layer_ref.layer, self._layer_ref.datatype, list(pts)))


class Cell:
    def __init__(self, layout, name):
        self.layout = layout
        self.name = name
        self.polygons = []       # list of (layer, datatype, points)
        self.insts = []          # list of (child_cell, trans)
        self.cell_index_val = id(self)

    def shapes(self, layer_ref):
        return Shapes(self, layer_ref)

    def insert(self, inst_array):
        # CellInstArray / DCellInstArray -- we don't need real array
        # repetition (only one junction matters), just record the placement
        # so bounding boxes / hierarchy traversal still work if needed.
        self.insts.append(inst_array)

    def cell_index(self):
        return self.cell_index_val

    def copy_tree(self, other_cell):
        self.polygons.extend(other_cell.polygons)
        self.insts.extend(other_cell.insts)

    def bounding_box(self):
        xs, ys = [], []
        for _, _, pts in self.polygons:
            for x, y in pts:
                xs.append(x); ys.append(y)
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))


class _CellInstArray:
    def __init__(self, cell_index_or_cell, trans):
        self.cell_ref = cell_index_or_cell
        self.trans = trans


class Layout:
    def __init__(self):
        self.dbu = 0.001
        self.cells = {}          # name -> Cell
        self._by_index = {}      # cell_index -> Cell
        self._top_cell = None

    def create_cell(self, name):
        c = Cell(self, name)
        self.cells[name] = c
        self._by_index[c.cell_index_val] = c
        if self._top_cell is None:
            self._top_cell = c
        return c

    def cell(self, name):
        return self.cells.get(name)

    def delete_cell(self, cell_index):
        c = self._by_index.pop(cell_index, None)
        if c is not None:
            self.cells.pop(c.name, None)

    def layer(self, layer_num, datatype=0):
        return _LayerRef(layer_num, datatype)

    def top_cell(self):
        return self._top_cell

    def read(self, path):
        # Template GDS files referenced by these scripts are expected to
        # live on the local filesystem and may not always be present --
        # os.path is patched (see run_script_function) so scripts' own
        # `if not os.path.exists(...)` guards skip this gracefully. If a
        # real path IS reachable, do a best-effort real read via gdstk.
        try:
            lib = gdstk.read_gds(path)
        except Exception:
            return
        for gc in lib.cells:
            c = self.create_cell(gc.name)
            for p in gc.polygons:
                c.polygons.append((p.layer, p.datatype, [tuple(pt) for pt in p.points]))

    def write(self, path):
        pass  # not needed -- we read geometry directly from the Cell objects


# ============================================================================
# GUI no-ops (scripts that drive the live KLayout view call these; headless
# execution just needs them to not raise)
# ============================================================================

class _NoOpView:
    def __getattr__(self, name):
        def _noop(*a, **k):
            return None
        return _noop


class _NoOpMainWindow:
    def current_view(self):
        return None

    def create_layout(self, *a, **k):
        return _NoOpView()


class Application:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def main_window(self):
        return _NoOpMainWindow()


# ============================================================================
# Shim install/uninstall
# ============================================================================

def _build_module():
    mod = types.ModuleType("pya")
    mod.DVector = DVector
    mod.DPoint = DPoint
    mod.Point = Point
    mod.DBox = DBox
    mod.DPolygon = DPolygon
    mod.DCplxTrans = DCplxTrans
    mod.Trans = Trans
    mod.Layout = Layout
    mod.Cell = Cell
    mod.Application = Application
    mod.CellInstArray = _CellInstArray
    mod.DCellInstArray = _CellInstArray
    return mod


def install_pya_shim():
    """Patches sys.modules['pya'] with the shim. Returns the previous
    module (or None) so it can be restored later."""
    previous = sys.modules.get("pya")
    sys.modules["pya"] = _build_module()
    return previous


def uninstall_pya_shim(previous):
    if previous is None:
        sys.modules.pop("pya", None)
    else:
        sys.modules["pya"] = previous
