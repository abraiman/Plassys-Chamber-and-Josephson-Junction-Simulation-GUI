"""
recipe_model.py

Deterministic recipe generation for Manhattan-style and patch-integrated
Josephson junction fabrication on the Plassys PVD / Ion-Mill tool.

No machine learning, no LLM: every angle here comes from either (a) a
direct closed-form formula (the alpha/line-width relationship) or (b) a
fixed template rule inferred from your working recipe, documented inline.
Where a rule is a first-pass assumption rather than something you've
explicitly confirmed as general, it's flagged in ASSUMPTION comments and
surfaced in the generated recipe's warnings list so it's never silently
trusted.

============================================================================
RECIPE STRUCTURE (fixed template, derived from your working recipe sheet)
============================================================================
  1. Vacuum check / pump                          (prep, no angle)
  2. Ti gather (lower pressure)                    (prep, no angle)
  3. Argon Mill (ion milling step)  [STARRED]      (mill, multi sub-step)
       3a. Top mill:      alpha = 0,        theta = anything
       3b+ Sidewall mill(s): alpha, theta -- fully user-specified, as many
            as the design needs. No formula; this is a process choice.
  4. Ti gather (lower pressure)                    (prep, no angle)
  5. 1st Al deposition -- FIRST finger [STARRED]   (deposition, 1 sub-step)
       Whichever finger (horizontal or vertical) the user designates as
       deposited first. theta = that finger's drawn angle. This is a
       single sub-step: its job is a clean, shadow-free connection to
       the Nb electrode, plus its own adhesion plate (same layer, same
       sub-step, no separate angle -- the plate is just a wider region
       of the same evaporation).
  6. Oxidation step                                (misc, no angle)
  7. Ti gather (lower pressure)                    (prep, no angle)
  8. 2nd Al deposition -- SECOND finger [STARRED]  (deposition, 2 sub-steps)
       Whichever finger was NOT deposited first. This finger gets the
       connect/overlap pair because it's the one that defines the
       junction's crisp overlap length:
       8a. connect: theta = second_finger_angle
       8b. overlap: theta = second_finger_angle + 180 deg
            ASSUMPTION: the "+180" rule for defining the crisp overlap
            length is generalized from your one confirmed example
            (90 -> 270). You flagged that this sub-step does something
            geometry-specific at the finger intersection that isn't
            fully nailed down yet -- verify against your bridge/undercut
            geometry before committing a real wafer.
       alpha = the same fixed deposition alpha as step 5
  9. Oxidation step                                (misc, no angle; only if patch-integrated)
 10. 3rd Al deposition -- PATCHES [STARRED]        (deposition, 2 sub-steps,
       ONLY IF patch-integrated)
       Patches sit on their own layer-independent diagonal (45/225 deg by
       convention), overlapping the adhesion plates and Nb electrodes on
       each side, at an angle offset from both finger axes. This is
       ADDITIVE to the two finger deposition steps above -- it does not
       replace either one (per your clarification).
       10a. theta = 45 deg
       10b. theta = 45 + 180 = 225 deg
            ASSUMPTION (LITERATURE-INFORMED, STILL UNVERIFIED FOR YOUR
            PROCESS): published "patch-integrated cross-type" (PITC)
            junction work (e.g. Kim et al., Appl. Phys. Lett. 2021;
            "Fabrication of Al/AlOx/Al junctions..." arXiv:2305.10956)
            describes evaporating the patch material "from three
            optional angles" in a single lithography step, but does not
            spell out whether a two-sided patch pair needs one shared
            angle or two separate angles the way the finger overlap pair
            does. Since your two patches sit on opposite sides of the
            junction (one toward the horizontal finger's electrode, one
            toward the vertical finger's electrode), this module treats
            them the same way as the finger connect/overlap pair --
            two sub-steps, 180 deg apart -- on the reasoning that a
            single-angle deposition would fully coat the near-side patch
            while leaving the far-side patch partially shadowed. This is
            a modeling choice, not a confirmed rule -- verify against
            your own patch connectivity before trusting it on a real run.
 11. Oxidation step                                (misc, no angle)
 12. Pump                                          (misc, no angle)
 13. Wait (user-set duration, default 20 min)      (misc, no angle)

Exactly one horizontal finger and one vertical finger per junction --
hardware-constrained in this module (duplicate parallel fingers caused a
real short in a past design, so this is not offered as a configurable
list).
============================================================================
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

# Reuse the same hardware alpha range as the visualizer for consistency.
try:
    from rotation_core import ALPHA_MIN, ALPHA_MAX
except ImportError:
    ALPHA_MIN, ALPHA_MAX = -90.0, 90.0


# ----------------------------------------------------------------------
# Alpha-from-line-width calculator
# ----------------------------------------------------------------------

def compute_alpha_from_linewidth(line_width_nm: float, pmma_thickness_nm: float,
                                  safety_margin_nm: float = 100.0) -> float:
    """
    alpha = arctan( LW / (t_PMMA - safety_margin) )

    This is your confirmed formula, verified against your spreadsheet
    values (e.g. LW=145.5881 at alpha=20deg, t_PMMA=500nm, margin=100nm:
    (500-100)*tan(20deg) = 145.588, matches).
    """
    denom = pmma_thickness_nm - safety_margin_nm
    if denom <= 0:
        raise ValueError(
            f"PMMA thickness ({pmma_thickness_nm} nm) must exceed the safety "
            f"margin ({safety_margin_nm} nm) -- got a non-positive denominator."
        )
    return math.degrees(math.atan(line_width_nm / denom))


def compute_linewidth_from_alpha(alpha_deg: float, pmma_thickness_nm: float,
                                  safety_margin_nm: float = 100.0) -> float:
    """Inverse of compute_alpha_from_linewidth, for cross-checking a chosen alpha."""
    denom = pmma_thickness_nm - safety_margin_nm
    return denom * math.tan(math.radians(alpha_deg))


import math

def compute_overlap_shadow_bounds(pmma_thickness_nm: float, 
                                   pmgi_thickness_nm: float, 
                                   alpha_deg: float,
                                   chosen_overlap_nm: float = 1500.0):
    """
    Overlap shadow-evaporation bounds updated for tolerance analysis.
    
    h  = t_PMMA + t_PMGI      (total resist stack height)
    dx = h * tan(alpha)       (worst-case shift reducing the overlap)
    dy = t_PMGI * tan(alpha)  (shift increasing the overlap)
    
    A chosen junction overlap is safe if (chosen_overlap_nm - dx_nm) > 0,
    meaning even with the worst-case shadow shortening, a real physical 
    overlap still remains.
    
    Returns:
        dx_um (float): Shortening shadow error bound in micrometers.
        dy_um (float): Lengthening shadow error bound in micrometers.
        min_actual_overlap_um (float): The absolute worst-case minimum remaining overlap.
        max_actual_overlap_um (float): The maximum possible expanded overlap.
        is_safe (bool): True if the worst-case shift still guarantees an overlap.
    """
    h_nm = pmma_thickness_nm + pmgi_thickness_nm
    
    # Calculate shadow offsets from the resist profile (in nm)
    dx_nm = h_nm * math.tan(math.radians(alpha_deg))
    dy_nm = pmgi_thickness_nm * math.tan(math.radians(alpha_deg))
    
    # Convert error bounds to micrometers
    dx_um = dx_nm / 1000.0
    dy_um = dy_nm / 1000.0
    
    # Convert chosen target from nm to micrometers
    chosen_overlap_um = chosen_overlap_nm / 1000.0
    
    # Compute the physical layout boundaries based on your shift logic
    # dx shortens the overlap down, dy lengthens it out
    min_actual_overlap_um = round(chosen_overlap_um - dx_um, 3)
    max_actual_overlap_um = round(chosen_overlap_um + dy_um, 3)
    
    # Safe if the worst-case shortening still leaves an extra physical buffer
    is_safe = min_actual_overlap_um > 0.0
    
    return dx_um, dy_um, min_actual_overlap_um, max_actual_overlap_um, is_safe


# ----------------------------------------------------------------------
# Input data model
# ----------------------------------------------------------------------

@dataclass
class SidewallMill:
    alpha: float
    theta: float
    label: str = ""  # optional free-text description, e.g. "sidewall facing junction A"


@dataclass
class DesignParameters:
    # --- Deposition (fixed for the whole batch/wafer) ---
    alpha_deposition: float                     # deg; typed directly or via calculator

    # --- Finger geometry ---
    # These angles are now used DIRECTLY as both (a) the deposition theta
    # for that finger's connect sub-step, AND (b) the physical direction,
    # from the junction crossing, that the finger points toward its own
    # Nb electrode -- e.g. 0 deg = electrode to the right, 270 deg =
    # electrode at the bottom. Change these to match where YOUR electrodes
    # actually sit; the junction schematic draws each finger rotated to
    # point in its given direction, it no longer assumes horizontal=right
    # and vertical=top.
    horizontal_finger_angle: float = 0.0        # deg; direction to horizontal finger's electrode
    vertical_finger_angle: float = 270.0        # deg; direction to vertical finger's electrode
                                                 # (270 = bottom, matching your confirmed design;
                                                 # was defaulted to 90/top before this correction)
    deposit_first: str = "horizontal"           # 'horizontal' or 'vertical' -- which
                                                 # finger gets the single connect-only
                                                 # sub-step; the OTHER finger gets the
                                                 # connect+overlap pair.
    horizontal_lw_nm: Optional[float] = None    # line width, for JJ area + drawing
    vertical_lw_nm: Optional[float] = None
    horizontal_length_um: float = 3.0           # drawn length, proportional only
    vertical_length_um: float = 3.0
    overlap_target_um: float = 1.5              # documentation + pass/fail check

    # --- Adhesion plates (same for both fingers, per your confirmation) ---
    adhesion_plate_width_um: float = 1.0
    adhesion_plate_height_um: float = 1.0

    # --- Patch-integrated design ---
    patch_integrated: bool = False
    patch_count: int = 3
    patch_spacing_um: float = 1.0
    patch_width_um: float = 0.5
    patch_height_um: float = 1.5
    patch_angle: float = 45.0                   # deg; connect sub-step theta
                                                 # (overlap sub-step = +180)

    # --- Ion milling ---
    top_mill_theta: float = 0.0                 # arbitrary; alpha is always 0 for top mill
    sidewall_mills: List[SidewallMill] = field(default_factory=list)

    # --- Timing ---
    wait_minutes: float = 20.0

    # --- Resist stack (documentation + optional alpha-calculator inputs) ---
    pmma_thickness_nm: Optional[float] = None
    pmgi_thickness_nm: Optional[float] = None   # documentation only; not in the alpha formula
    line_width_nm: Optional[float] = None       # legacy alias, used by the alpha calculator UI
    safety_margin_nm: float = 100.0

    notes: str = ""

    def second_finger_angle(self) -> float:
        """The finger NOT deposited first -- the one that gets the connect/overlap pair."""
        return self.vertical_finger_angle if self.deposit_first == "horizontal" else self.horizontal_finger_angle

    def first_finger_angle(self) -> float:
        return self.horizontal_finger_angle if self.deposit_first == "horizontal" else self.vertical_finger_angle

    def jj_area_um2(self) -> Optional[float]:
        """JJ area = LW_horizontal x LW_vertical (nominal drawn overlap at the finger
        crossing), in um^2. Returns None if either line width is unset."""
        if self.horizontal_lw_nm is None or self.vertical_lw_nm is None:
            return None
        lw_h_um = self.horizontal_lw_nm / 1000.0
        lw_v_um = self.vertical_lw_nm / 1000.0
        return lw_h_um * lw_v_um


# ----------------------------------------------------------------------
# Output data model
# ----------------------------------------------------------------------

@dataclass
class RecipeStep:
    index: str                  # "1", "3a", "8b", etc.
    title: str                  # e.g. "1st Al deposition -- horizontal finger"
    category: str                # 'prep' | 'mill' | 'deposition' | 'misc'
    alpha: Optional[float]
    theta: Optional[float]
    starred: bool
    detail: str                  # human-readable traceability note
    feature: Optional[str] = None  # what this sub-step visually deposits, for the
                                     # junction-formation slideshow, e.g.
                                     # 'horizontal_finger', 'vertical_connect',
                                     # 'vertical_overlap', 'patch_a', 'patch_b'


def _wrap360(deg: float) -> float:
    return deg % 360.0


def generate_recipe_steps(p: DesignParameters):
    """
    Returns (steps: List[RecipeStep], warnings: List[str]).
    """
    steps: List[RecipeStep] = []
    warnings: List[str] = []

    if p.deposit_first not in ("horizontal", "vertical"):
        raise ValueError("deposit_first must be 'horizontal' or 'vertical'")

    def check_alpha_range(alpha, where):
        if alpha is not None and not (ALPHA_MIN <= alpha <= ALPHA_MAX):
            warnings.append(
                f"{where}: alpha={alpha:.2f} deg is outside the hardware range "
                f"[{ALPHA_MIN}, {ALPHA_MAX}]. Verify before running."
            )

    # 1. Vacuum check / pump
    steps.append(RecipeStep("1", "Vacuum check / pump", "prep", None, None, False,
                             "Fixed prep step, no angle."))

    # 2. Ti gather (lower pressure)
    steps.append(RecipeStep("2", "Ti gather (lower pressure)", "prep", None, None, False,
                             "Fixed prep step, no angle."))

    # 3. Argon Mill (ion milling step) -- starred, multi sub-step
    steps.append(RecipeStep("3", "Argon Mill (ion milling step)", "mill", 0.0, p.top_mill_theta, True,
                             "Top mill: alpha=0 by definition (full top-surface contact); "
                             "theta is not constraining here, any value is valid.",
                             feature="mill_top"))
    check_alpha_range(0.0, "Argon Mill top")

    if not p.sidewall_mills:
        warnings.append("No sidewall mill steps specified -- add at least one if your "
                         "design needs sidewall oxide removal.")
    for i, sw in enumerate(p.sidewall_mills):
        label = sw.label or f"sidewall {i + 1}"
        steps.append(RecipeStep(f"3.{i + 1}", f"Argon Mill -- {label}", "mill", sw.alpha, sw.theta, False,
                                 "User-specified sidewall mill angle; not derived from a formula, "
                                 "your process choice for sidewall coverage.",
                                 feature="mill_sidewall"))
        check_alpha_range(sw.alpha, f"Argon Mill {label}")

    # 4. Ti gather (lower pressure)
    steps.append(RecipeStep("4", "Ti gather (lower pressure)", "prep", None, None, False,
                             "Fixed prep step, no angle."))

    # 5. 1st Al deposition -- whichever finger is deposited first
    first_name = p.deposit_first                      # 'horizontal' or 'vertical'
    second_name = "vertical" if first_name == "horizontal" else "horizontal"
    first_angle = p.first_finger_angle()
    second_angle = p.second_finger_angle()

    steps.append(RecipeStep(
        "5", f"1st Al deposition -- {first_name} finger", "deposition",
        p.alpha_deposition, _wrap360(first_angle), True,
        f"theta = {first_name}_finger_angle ({first_angle:g} deg, as drawn in KLayout). "
        f"Single sub-step: deposits the {first_name} finger plus its adhesion plate "
        "(same layer, same evaporation) with a clean, shadow-free connection to the "
        "Nb electrode. Deposited first per your choice -- does not define the junction overlap.",
        feature=f"{first_name}_finger"
    ))
    check_alpha_range(p.alpha_deposition, "1st Al deposition")

    # 6. Oxidation step
    steps.append(RecipeStep("6", "Oxidation step", "misc", None, None, False,
                             "Fixed step, no angle."))

    # 7. Ti gather (lower pressure)
    steps.append(RecipeStep("7", "Ti gather (lower pressure)", "prep", None, None, False,
                             "Fixed prep step, no angle."))

    # 8. 2nd Al deposition -- the OTHER finger, connect + overlap pair
    overlap_angle = _wrap360(second_angle + 180.0)

    steps.append(RecipeStep(
        "8a", f"2nd Al deposition -- {second_name} finger (connect)", "deposition",
        p.alpha_deposition, _wrap360(second_angle), True,
        f"theta = {second_name}_finger_angle ({second_angle:g} deg). "
        f"Ensures full connection across the Nb step for the {second_name} finger and "
        "its adhesion plate, no shadow gap.",
        feature=f"{second_name}_connect"
    ))
    steps.append(RecipeStep(
        "8b", f"2nd Al deposition -- {second_name} finger (overlap)", "deposition",
        p.alpha_deposition, overlap_angle, True,
        f"theta = {second_name}_finger_angle + 180 = {overlap_angle:g} deg. "
        "ASSUMPTION carried over from your one confirmed example (90->270): "
        "this defines the crisp, self-aligned junction overlap "
        f"(target {p.overlap_target_um:g} um) via shadow evaporation at the "
        "finger intersection. You noted this sub-step does something "
        "geometry-specific at the junction that isn't fully pinned down yet "
        "-- verify against your bridge/undercut geometry before committing a real wafer.",
        feature=f"{second_name}_overlap"
    ))
    check_alpha_range(p.alpha_deposition, "2nd Al deposition")

    next_index = 9

    # 9/10. Patch-integrated: ADDITIVE third deposition (does not replace either finger step)
    if p.patch_integrated:
        steps.append(RecipeStep(str(next_index), "Oxidation step", "misc", None, None, False,
                                 "Fixed step, no angle."))
        next_index += 1
        steps.append(RecipeStep(str(next_index), "Ti gather (lower pressure)", "prep", None, None, False,
                                 "Fixed prep step, no angle."))
        next_index += 1

        patch_overlap_angle = _wrap360(p.patch_angle + 180.0)
        steps.append(RecipeStep(
            f"{next_index}a", "3rd Al deposition -- patches (side A)", "deposition",
            p.alpha_deposition, _wrap360(p.patch_angle), True,
            f"theta = patch_angle ({p.patch_angle:g} deg). Additive third deposition -- "
            f"does NOT replace either finger step. Lands {p.patch_count} patch(es) "
            "overlapping the adhesion plates/Nb electrodes at this diagonal. "
            "ASSUMPTION (literature-informed, unverified for your process): treated as a "
            "two-angle pair like the finger overlap step, since a single angle would "
            "leave the far-side patches partially shadowed -- see module docstring.",
            feature="patch_a"
        ))
        steps.append(RecipeStep(
            f"{next_index}b", "3rd Al deposition -- patches (side B)", "deposition",
            p.alpha_deposition, patch_overlap_angle, True,
            f"theta = patch_angle + 180 = {patch_overlap_angle:g} deg. Completes patch "
            "coverage on the opposite side. Same ASSUMPTION caveat as side A.",
            feature="patch_b"
        ))
        check_alpha_range(p.alpha_deposition, "3rd Al deposition (patches)")
        warnings.append(
            "Patch-integrated is on: added a 3rd Al deposition (2 sub-steps at "
            f"{p.patch_angle:g}/{patch_overlap_angle:g} deg) for {p.patch_count} patch(es), "
            "additive to both finger steps. The two-sub-step treatment of the patch pair "
            "is a literature-informed guess, not a confirmed rule for your process -- "
            "verify patch connectivity before running a real wafer."
        )
        next_index += 1

    # Oxidation / pump / wait
    steps.append(RecipeStep(str(next_index), "Oxidation step", "misc", None, None, False,
                             "Fixed step, no angle."))
    next_index += 1
    steps.append(RecipeStep(str(next_index), "Pump", "misc", None, None, False,
                             "Fixed step, no angle."))
    next_index += 1
    steps.append(RecipeStep(str(next_index), f"Wait ({p.wait_minutes:g} minutes)", "misc", None, None, False,
                             "Fixed step, no angle."))

    area = p.jj_area_um2()
    if area is not None:
        warnings.append(f"JJ area (LW_horizontal x LW_vertical) = {area:.4f} um^2 "
                         f"({p.horizontal_lw_nm:g} nm x {p.vertical_lw_nm:g} nm).")
    else:
        warnings.append("JJ area not computed -- set both horizontal and vertical line widths to get it.")

    return steps, warnings


def format_recipe_text(steps: List[RecipeStep], warnings: List[str], p: DesignParameters) -> str:
    """Render the recipe as plain text, echoing the style of the handwritten sheet."""
    lines = []
    lines.append("=" * 60)
    lines.append("PLASSYS RECIPE -- Manhattan-style Josephson Junction")
    if p.patch_integrated:
        lines.append(f"  (patch-integrated: +3rd Al deposition, patches at "
                      f"{p.patch_angle:g}/{_wrap360(p.patch_angle + 180):g} deg)")
    lines.append(f"  (deposit first: {p.deposit_first})")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Deposition alpha (fixed for whole batch): {p.alpha_deposition:.2f} deg")
    if p.pmma_thickness_nm:
        lines.append(f"PMMA thickness: {p.pmma_thickness_nm:g} nm   "
                      f"Safety margin: {p.safety_margin_nm:g} nm")
    if p.pmgi_thickness_nm:
        lines.append(f"PMGI thickness: {p.pmgi_thickness_nm:g} nm (documentation only)")
    area = p.jj_area_um2()
    if area is not None:
        lines.append(f"JJ area: {area:.4f} um^2  "
                      f"({p.horizontal_lw_nm:g} nm horiz x {p.vertical_lw_nm:g} nm vert)")
    lines.append("")

    for s in steps:
        star = "* " if s.starred else "  "
        if s.alpha is not None and s.theta is not None:
            angle_str = f"alpha={s.alpha:.2f} deg, theta={s.theta:.2f} deg"
        else:
            angle_str = ""
        header = f"{star}[{s.index}] {s.title}"
        if angle_str:
            header += f"   ({angle_str})"
        lines.append(header)
        lines.append(f"      {s.detail}")
        lines.append("")

    if warnings:
        lines.append("-" * 60)
        lines.append("WARNINGS / THINGS TO VERIFY:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("-" * 60)

    if p.notes:
        lines.append("")
        lines.append("Notes:")
        lines.append(f"  {p.notes}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check reproducing your known-good recipe (non-patch case).
    params = DesignParameters(
        alpha_deposition=30.0,
        horizontal_finger_angle=0.0,
        vertical_finger_angle=90.0,
        deposit_first="horizontal",
        horizontal_lw_nm=200.0,
        vertical_lw_nm=220.0,
        patch_integrated=False,
        sidewall_mills=[
            SidewallMill(alpha=60.0, theta=0.0, label="sidewall 1"),
            SidewallMill(alpha=60.0, theta=270.0, label="sidewall 2"),
        ],
        pmma_thickness_nm=875.0,
        pmgi_thickness_nm=600.0,
        safety_margin_nm=100.0,
    )
    steps, warnings = generate_recipe_steps(params)
    print(format_recipe_text(steps, warnings, params))

    print("\n\n--- patch-integrated case ---")
    params2 = DesignParameters(
        alpha_deposition=30.0,
        horizontal_finger_angle=0.0,
        vertical_finger_angle=90.0,
        deposit_first="horizontal",
        horizontal_lw_nm=200.0,
        vertical_lw_nm=220.0,
        patch_integrated=True,
        patch_count=3,
        patch_spacing_um=1.0,
        patch_width_um=0.5,
        patch_height_um=1.5,
        sidewall_mills=[
            SidewallMill(alpha=60.0, theta=0.0, label="sidewall 1"),
            SidewallMill(alpha=60.0, theta=270.0, label="sidewall 2"),
        ],
        pmma_thickness_nm=875.0,
        pmgi_thickness_nm=600.0,
    )
    steps2, warnings2 = generate_recipe_steps(params2)
    print(format_recipe_text(steps2, warnings2, params2))

    print("\n\n--- alpha calculator check ---")
    a = compute_alpha_from_linewidth(145.5881, 500.0, 100.0)
    print(f"expected ~20.00 deg, got {a:.4f} deg")
