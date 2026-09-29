"""
scan_pattern.py -- pure, dependency-free scan-order generation for the
Room-Temperature Resistance Measurement tab.

Generates the ORDERED list of (row, col) cell coordinates a user walks
through while taking measurements, decoupled entirely from voice
capture and the GUI so it can be reasoned about and tested on its own,
with zero dependencies (no PySide6, no audio libraries). Coordinates
are 1-indexed (row, col), matching the row/column labels in the
reference measurement spreadsheets (row 1..N top to bottom, column
1..M left to right).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Coord = Tuple[int, int]  # (row, col), both 1-indexed


def generate_row_major(rows: int, cols: int) -> List[Coord]:
    """Row 1 left->right, then row 2 left->right, ... standard reading order."""
    return [(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1)]


def generate_column_major(rows: int, cols: int) -> List[Coord]:
    """Column 1 top->bottom, then column 2 top->bottom, ..."""
    return [(r, c) for c in range(1, cols + 1) for r in range(1, rows + 1)]


def generate_serpentine_row(rows: int, cols: int) -> List[Coord]:
    """Row 1 left->right, row 2 right->left, row 3 left->right, ...
    Minimizes probe travel between rows versus row-major, which has to
    jump all the way back to column 1 at the start of every row."""
    order: List[Coord] = []
    for r in range(1, rows + 1):
        col_range = range(1, cols + 1) if r % 2 == 1 else range(cols, 0, -1)
        order.extend((r, c) for c in col_range)
    return order


def generate_serpentine_column(rows: int, cols: int) -> List[Coord]:
    """Column 1 top->bottom, column 2 bottom->top, ... the column-major
    equivalent of serpentine_row."""
    order: List[Coord] = []
    for c in range(1, cols + 1):
        row_range = range(1, rows + 1) if c % 2 == 1 else range(rows, 0, -1)
        order.extend((r, c) for r in row_range)
    return order


def generate_spiral_edges_in(rows: int, cols: int) -> List[Coord]:
    """Edges first, spiraling inward -- a common manual-probing habit,
    sweeping each outer ring before moving inward: column 1 (top->bottom),
    then row `rows` (left->right), then column
    `cols` (bottom->top), then row 1 (right->left) -- that's the outer
    ring -- then shrink to the next ring inward (r0+1, r1-1, c0+1,
    c1-1) and repeat until the whole grid is consumed. Each edge after
    the first skips the corner cell the previous edge already visited,
    so no cell is ever repeated. Degenerates cleanly to a single
    remaining row or column at the center of an odd-sized grid."""
    order: List[Coord] = []
    r0, r1, c0, c1 = 1, rows, 1, cols
    while r0 <= r1 and c0 <= c1:
        if c0 == c1:
            order.extend((r, c0) for r in range(r0, r1 + 1))
            break
        if r0 == r1:
            order.extend((r0, c) for c in range(c0, c1 + 1))
            break
        order.extend((r, c0) for r in range(r0, r1 + 1))          # left edge (column c0), top->bottom
        order.extend((r1, c) for c in range(c0 + 1, c1 + 1))      # bottom edge (row r1), left->right
        order.extend((r, c1) for r in range(r1 - 1, r0 - 1, -1))  # right edge (column c1), bottom->top
        order.extend((r0, c) for c in range(c1 - 1, c0, -1))      # top edge (row r0), right->left
        r0 += 1
        r1 -= 1
        c0 += 1
        c1 -= 1
    return order


PATTERNS: Dict[str, Callable[[int, int], List[Coord]]] = {
    "row_major": generate_row_major,
    "column_major": generate_column_major,
    "serpentine_row": generate_serpentine_row,
    "serpentine_column": generate_serpentine_column,
    "spiral_edges_in": generate_spiral_edges_in,
}

PATTERN_LABELS: Dict[str, str] = {
    "row_major": "Row 1 -> Row N (left to right each row)",
    "column_major": "Column 1 -> Column M (top to bottom each column)",
    "serpentine_row": "Serpentine by row (alternates direction each row)",
    "serpentine_column": "Serpentine by column (alternates direction each column)",
    "spiral_edges_in": "Edges first, spiral inward (column 1, row N, column M, row 1, ...)",
}


def generate_scan_order(rows: int, cols: int, pattern: str) -> List[Coord]:
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from: {sorted(PATTERNS)}")
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must both be >= 1")
    return PATTERNS[pattern](rows, cols)


@dataclass
class ScanState:
    """Walks a precomputed scan order one cell at a time. Works with a
    fully CUSTOM order too (e.g. one recorded by the user clicking
    cells in their own preferred sequence, for a probe-card layout that
    doesn't match any named pattern) since it's constructed from any
    List[Coord], not just one of the presets above."""
    order: List[Coord]
    index: int = 0
    history: List[int] = field(default_factory=list)

    @classmethod
    def from_pattern(cls, rows: int, cols: int, pattern: str) -> "ScanState":
        return cls(order=generate_scan_order(rows, cols, pattern))

    @property
    def current(self) -> Optional[Coord]:
        return self.order[self.index] if 0 <= self.index < len(self.order) else None

    @property
    def done(self) -> bool:
        return self.index >= len(self.order)

    @property
    def progress(self) -> Tuple[int, int]:
        """(cells committed so far, total cells) for a progress readout."""
        return (min(self.index, len(self.order)), len(self.order))

    def advance(self) -> Optional[Coord]:
        """Commit the current cell and move to the next one. Returns
        the NEW current cell, or None once the scan is complete."""
        if not self.done:
            self.history.append(self.index)
            self.index += 1
        return self.current

    def undo(self) -> Optional[Coord]:
        """Step back to the previous position (e.g. after a misheard
        value was committed and needs to be re-taken) without
        discarding or reshuffling the rest of the order."""
        if self.history:
            self.index = self.history.pop()
        return self.current

    def skip(self) -> Optional[Coord]:
        """Move on WITHOUT recording a value for the current cell (a
        junction the user wants to measure later, or can't reach)."""
        return self.advance()

    def jump_to(self, row: int, col: int) -> bool:
        """Manually reposition the scan pointer to an arbitrary cell
        (e.g. the user clicks a cell directly in the table). Returns
        False, leaving the pointer unchanged, if that cell isn't part
        of this scan's own order."""
        try:
            idx = self.order.index((row, col))
        except ValueError:
            return False
        self.history.append(self.index)
        self.index = idx
        return True
