"""
electrode_convention.py

Holds ElectrodeConvention on its own, with zero dependencies on
junction_import.py, parametric_junction_view.py, or recipe_generator.py. This is what
lets the SAME convention object be attached to DesignParameters (Tab 1's
process parameters, in recipe_generator.py) and used directly by
junction_import.py's GDS/script import paths, without a circular import
(recipe_generator -> junction_import -> parametric_junction_view -> recipe_generator would
otherwise be a real cycle).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ElectrodeConvention:
    """Settings for how electrode geometry is identified in a given GDS
    file. The defaults match a common convention for these designs (look
    for a cell literally named "Pads_v4_Manhattan", or fall back to any
    cell whose name contains "pad"; treat the hollow absent-space inside
    a bracket-shaped piece of material as the electrode; extend the
    electrode past the array's outer edge). A design that uses a
    different naming convention, a "filled material is electrode"
    convention, or has no repeating array at all just needs a different
    ElectrodeConvention -- no code changes required.
    """
    # How to FIND the electrode-bearing content:
    cell_name_exact: Optional[str] = "Pads_v4_Manhattan"
    cell_name_fallback_substring: Optional[str] = "pad"
    layer_override: Optional[Tuple[int, int]] = None  # explicit (layer, datatype);
                                                        # set this to skip cell-name matching
                                                        # entirely -- for files that don't use
                                                        # any cell-name-based convention

    # What COUNTS as electrode, once that content is found:
    convention: str = "hollow"  # "hollow" = the absent-space (notch) inside a bracket-shaped
                                 # piece of material is the electrode -- the more common
                                 # convention for this style of design.
                                 # "filled" = the drawn material itself is the electrode (the
                                 # more common/intuitive convention elsewhere).

    # Array-edge extension (only meaningful for a repeating-array chip):
    array_edge_extension: bool = True
    standoff_pad_um: float = 15.0          # keeps a JJ crossing from being misclassified as
                                             # electrode just for sitting near the array's edge
    min_electrode_area_um2: float = 10000.0  # "big hole" threshold for finding the array's
                                               # real repeating pattern (excludes small
                                               # per-site connector notches from that specific
                                               # calculation)
    search_radius_um: float = 600.0


DEFAULT_ELECTRODE_CONVENTION = ElectrodeConvention()
