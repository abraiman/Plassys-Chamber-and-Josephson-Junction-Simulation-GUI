"""
recipe_generator.py

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
       replace either one.
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

from electrode_convention import ElectrodeConvention  # shared with junction_import.py's GDS/script
                                                        # import paths -- no circular import risk since
                                                        # this module has zero dependencies of its own
from shape_editor import EditableShape  # same reasoning -- zero dependencies, safe to share

# Reuse the same hardware alpha range as the visualizer for consistency.
try:
    from wafer_orientation import ALPHA_MIN, ALPHA_MAX
except ImportError:
    ALPHA_MIN, ALPHA_MAX = -90.0, 90.0


# ----------------------------------------------------------------------
# Ramp profiles (e-gun emission-current ramps), keyed by metal + rate
# ----------------------------------------------------------------------
#
# Measured directly off a real Plassys MEB550S "Executing" step list,
# not estimated:
#
#   Ti gather, 0.2 nm/s:
#       1mA hold -> wait 0:15 -> ramp to 45mA over 0:20 -> Ramp Wait All
#       -> Alert -> Shutter Source Open -> ramp to 50mA over 0:10
#       -> Ramp Wait All -> Rate Control 0.20nm/s -> wait 0:5
#       -> (gather: substrate shutter stays closed) hold 4:00
#       -> ramp down to 5mA over 0:10 -> ramp to 0mA -> wait 5:00
#
#   Al deposition, 0.5 nm/s:
#       1mA hold -> wait 0:15 -> ramp to 150mA over 1:30 -> Ramp Wait All
#       -> Alert -> wait 0:5 -> Shutter Source Open -> ramp to 170mA over 1:00
#       -> Ramp Wait All -> Rate Control 0.50nm/s -> wait 0:10
#       -> Zero Thickness -> Substrate Shutter Open -> Wait for Termination
#       -> Substrate Shutter Close
#       -> ramp down to 5mA over 0:30 -> ramp to 0mA -> wait 0:30
#
# ONLY these two (metal, rate) points are confirmed. Any other metal or
# rate you pick is scaled from the nearest confirmed profile, roughly
# proportionally with target rate, and is flagged with a warning telling
# you to verify the ramp currents on the tool before running a real
# wafer -- do not trust an unconfirmed ramp current on real hardware.
@dataclass
class RampProfile:
    soak1_ma: float        # first ramp target current (source shutter still closed)
    soak1_time_s: float    # time to reach soak1_ma, in seconds
    pre_wait_s: float       # wait after reaching 1mA baseline, before soak1 ramp
    soak2_ma: float        # second ramp target current (source shutter opens for this one)
    soak2_time_s: float    # time to reach soak2_ma, in seconds
    settle_wait_s: float   # wait after Rate Control is enabled, before deposit/gather proper
    rampdown_ma: float      # intermediate ramp-down current
    rampdown_time_s: float
    cooldown_wait_s: float  # final wait after ramping to 0mA
    confirmed: bool = True   # False if scaled/estimated rather than read off the tool


_CONFIRMED_PROFILES = {
    # (metal, rate_nmps): RampProfile
    ("Ti", 0.2): RampProfile(
        soak1_ma=45.0, soak1_time_s=20, pre_wait_s=15,
        soak2_ma=50.0, soak2_time_s=10, settle_wait_s=5,
        rampdown_ma=5.0, rampdown_time_s=10, cooldown_wait_s=300,
        confirmed=True,
    ),
    ("Al", 0.5): RampProfile(
        soak1_ma=150.0, soak1_time_s=90, pre_wait_s=15,
        soak2_ma=170.0, soak2_time_s=60, settle_wait_s=10,
        rampdown_ma=5.0, rampdown_time_s=30, cooldown_wait_s=30,
        confirmed=True,
    ),
}

# Fallback reference profile (per metal family) used only to scale an
# unconfirmed rate. Aluminum-like metals (Al, Ta, Re, Nb, Custom) scale
# off the Al point; Ti gather always uses the Ti point since gather rate
# is fixed by convention (0.2 nm/s) rather than user-selected.
_METAL_REFERENCE = {
    "Ti": ("Ti", 0.2),
}


def get_ramp_profile(metal: str, rate_nmps: float, warnings: Optional[List[str]] = None) -> RampProfile:
    """
    Look up (or scale) the e-gun ramp profile for a given metal + target
    deposition rate. Exact (metal, rate) matches return the confirmed,
    tool-verified profile. Anything else is linearly scaled off the
    nearest confirmed reference for that metal family and flagged as
    unconfirmed -- ALWAYS verify ramp currents on the tool before running
    real hardware with an unconfirmed profile.
    """
    key = (metal, round(rate_nmps, 6))
    if key in _CONFIRMED_PROFILES:
        return _CONFIRMED_PROFILES[key]

    ref_key = _METAL_REFERENCE.get(metal, ("Al", 0.5))
    ref = _CONFIRMED_PROFILES[ref_key]
    ref_rate = ref_key[1]
    scale = rate_nmps / ref_rate if ref_rate else 1.0

    scaled = RampProfile(
        soak1_ma=round(ref.soak1_ma * scale, 1),
        soak1_time_s=ref.soak1_time_s,
        pre_wait_s=ref.pre_wait_s,
        soak2_ma=round(ref.soak2_ma * scale, 1),
        soak2_time_s=ref.soak2_time_s,
        settle_wait_s=ref.settle_wait_s,
        rampdown_ma=ref.rampdown_ma,
        rampdown_time_s=ref.rampdown_time_s,
        cooldown_wait_s=ref.cooldown_wait_s,
        confirmed=False,
    )
    if warnings is not None:
        warnings.append(
            f"Ramp profile for {metal} @ {rate_nmps:g} nm/s is estimated (scaled from "
            f"{ref_key[0]} @ {ref_key[1]:g} nm/s), not read off the tool. Verify soak "
            f"currents ({scaled.soak1_ma:g} mA / {scaled.soak2_ma:g} mA) before running."
        )
    return scaled


def _fmt_mmss(total_seconds: float) -> str:
    """Format seconds as Plassys-style m:ss, e.g. 90 -> '1:30'."""
    total_seconds = int(round(total_seconds))
    m, s = divmod(total_seconds, 60)
    return f"{m}:{s:02d}"


# ----------------------------------------------------------------------
# Alpha-from-line-width calculator
# ----------------------------------------------------------------------

def compute_alpha_from_linewidth(line_width_nm: float, pmma_thickness_nm: float,
                                  safety_margin_nm: float = 100.0) -> float:
    """
    alpha = arctan( LW / (t_PMMA - safety_margin) )

    This formula is verified against reference values (e.g. LW=145.5881
    at alpha=20deg, t_PMMA=500nm, margin=100nm:
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


def compute_electrode_shadow(electrode_height_nm: float, alpha_deg: float) -> float:
    """Electrode-topography shadow -- a genuinely different mechanism from
    the resist-stack overlap shadow above: the Nb electrode's OWN height
    can block the angled metal flux from reaching the substrate/sidewall
    right next to it, potentially leaving a sidewall disconnection even
    when the top-down (2D) overlap looks fine. Formula:
    tan(alpha) = shadow / electrode_height, i.e.
    shadow = electrode_height * tan(alpha) -- the horizontal distance,
    measured from the electrode's edge, over which no metal lands."""
    return electrode_height_nm * math.tan(math.radians(alpha_deg))


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

    # Compute the physical layout boundaries based on the shift logic above
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
    # --- GUI Compatibility Fields (read directly by main_gui.py's widgets) ---
    metal_type: str = "Al"
    # The electrode (the pre-existing pad each finger connects to) is a
    # SEPARATE metal choice from metal_type (what's actually being
    # evaporated for this deposition) -- distinct concepts that are easy
    # to conflate since electrode material is otherwise unmodeled
    # elsewhere in this codebase. This matters because the Ion Mill
    # step's whole purpose is cleaning NATIVE OXIDE off the ELECTRODE
    # surface before contact -- Nb oxidizes readily (long mill times),
    # Ta/Re oxidize far less, Ti barely oxidizes at all -- so
    # electrode_metal_type, not metal_type, is what should inform mill
    # duration/necessity guidance.
    electrode_metal_type: str = "Nb"
    project_name: str = ""  # e.g. "my_wafer_batch" -- your own naming for this
                             # project/recipe family; used to build the actual
                             # Plassys recipe-tree paths (Root\Evap\..., Root\Etching\...)
                             # below so they aren't hardcoded to any one lab's convention.
                             # Empty is fine -- generic fallback names are used instead.
    al_dep_rate_nmps: float = 0.5        # user-selectable deposition rate; drives the e-gun ramp profile
    # Per-step deposition thickness is one of the parameters most likely
    # to be adjusted per process run, since a process log typically tracks
    # exactly how thick each deposition step ran. These four are the
    # generalized defaults (junction_import.build_simulation_result applies
    # them per pass); any single step can still be hand-edited afterward via
    # the recipe's own "Target thickness" double-click field, independent of
    # these project-wide defaults. Typical values: a thin first
    # finger (~30nm) to establish the junction crossing, a thicker second
    # finger (40-60nm) so the JJ crossing keeps enough cross-sectional area
    # even as that pass self-thins, and a further bump on the second finger
    # specifically when Warning #2's risky angle is unavoidable (Warning #1
    # already forced the only safe direction) -- extra thickness there is
    # what keeps that self-thinning from ever bottlenecking the electron
    # path. Patches are their own thing entirely (adhesion onto the
    # electrode/finger/adhesion-plate, not junction transport), typically
    # 50-60nm PER deposition pass -- see patch_thickness_nm below.
    al_thickness_1_nm: float = 30.0      # first-deposited finger (thin -- establishes the crossing)
    al_thickness_2_nm: float = 50.0      # second-deposited finger, normal case
    al_thickness_2_warn2_nm: float = 60.0  # second-deposited finger, this pass forced onto the
                                            # Warning #2-risky angle -- thicker to avoid a self-shadowing bottleneck
    ti_getter_thickness_nm: float = 10.0
    oxidation_pressure_torr: float = 10.0   # kept for backward compatibility; NOT what the tool displays
    oxidation_pressure_mbar: float = 20.0   # confirmed unit off the tool ("Static Oxidation 20.00 mBar")
    oxidation_time_min: float = 25.0        # "Param wait" timer -- user-alterable
    # A SEPARATE, short static oxidation after the entire junction has
    # finished depositing -- intended to protect the junction surface
    # once it leaves the chamber -- distinct from oxidation_time_min
    # above, which grows the real AlOx tunnel barrier between the first
    # and second finger depositions. Same pressure (oxidation_pressure_mbar)
    # is reused since no separate pressure is typically specified; only
    # the duration is its own, shorter, user-alterable value.
    final_oxidation_time_min: float = 5.0
    # This used to default to 100.0, but junction_import.build_simulation_
    # result applies it PER deposition PASS, not divided across passes --
    # a patch region needing both passes (the common case) was silently
    # getting ~200nm total instead of the ~100nm a typical process target.
    # 55.0 is the midpoint of a typical 50-60nm-per-pass range, so two
    # passes land right around the intended ~100-110nm combined.
    patch_thickness_nm: float = 55.0

    # --- Ti gather (getter) -- fixed by convention, exposed for override ---
    ti_gather_rate_nmps: float = 0.2        # confirmed tool default
    ti_gather_duration_min: float = 4.0     # confirmed tool default ("Ti 0.2nm/s 4min")

    # --- Argon ion mill parameters (confirmed tool defaults, overridable) ---
    mill_ar_gas_sccm: float = 6.0
    mill_beam_voltage_v: float = 400.0
    mill_beam_current_ma: float = 15.0
    mill_accel_voltage_v: float = 80.0
    mill_duration_min: float = 1.0

    # --- Deposition (fixed for the whole batch/wafer) ---
    alpha_deposition: float = 30.0              # deg; typed directly or via calculator

    # --- Milling -- see also shadowing_design_rules.md section 2.1: mill
    # alpha is its OWN fixed, chosen convention -- steep enough to hit
    # the electrode's SIDEWALLS specifically (rather than its top
    # surface, which risks thinning the linewidth or the JJ itself) --
    # NOT derived from the deposition geometry/shadow formulas, and NOT
    # the same value as alpha_deposition. shadowing_design_rules.md's own
    # worked example cites 60 deg as a fixed constant; this defaults to a
    # typical real-process value (50 deg) instead -- override freely per
    # process. Previously, build_simulation_result silently reused
    # alpha_deposition for every mill step, which is the exact bug this
    # field fixes.
    alpha_mill: float = 50.0                    # deg; fixed sidewall-mill convention, independent of alpha_deposition

    # --- Finger geometry ---
    horizontal_finger_angle: float = 0.0        # deg; direction to horizontal finger's electrode
    vertical_finger_angle: float = 270.0        # deg; direction to vertical finger's electrode
    deposit_first: str = "horizontal"           # 'horizontal' or 'vertical'
    horizontal_lw_nm: Optional[float] = 200.0   # line width, for JJ area + drawing
    vertical_lw_nm: Optional[float] = 220.0
    horizontal_length_um: float = 5.0           # drawn length, proportional only
    vertical_length_um: float = 5.0
    overlap_target_um: float = 1.5              # documentation + pass/fail check

    # --- Adhesion plates (same convention for both fingers) ---
    has_adhesion_pads: bool = False
    adhesion_plate_width_um: float = 1.0
    adhesion_plate_height_um: float = 1.0

    # --- Electrode pads (INDEPENDENT layer -- position/size/rotation of
    # the wide contact pads. Decoupled from the finger's own draw angle:
    # a finger no longer drags its electrode around when you change
    # horizontal/vertical_finger_angle. Leave x/y/angle as None to keep
    # the legacy "attached right at the end of the finger" placement;
    # set explicit values once you add UI controls to reposition them
    # freely as their own layer. w/h always have concrete defaults since
    # those don't depend on where the pad ends up. ---
    horizontal_electrode_x_um: Optional[float] = None
    horizontal_electrode_y_um: Optional[float] = None
    horizontal_electrode_angle_deg: Optional[float] = None
    horizontal_electrode_w_um: float = 4.0
    horizontal_electrode_h_um: float = 1.5
    vertical_electrode_x_um: Optional[float] = None
    vertical_electrode_y_um: Optional[float] = None
    vertical_electrode_angle_deg: Optional[float] = None
    vertical_electrode_w_um: float = 4.0
    vertical_electrode_h_um: float = 1.5

    # Desired standoff (um) from a finger's outward tip to its own
    # electrode's near edge, used only for auto-placing that electrode
    # when its x/y aren't set explicitly (i.e. before you've "fixed" it
    # as its own independent layer). Can be 0 (touching) or negative
    # (intentional overlap) -- allow_electrode_overlap controls whether
    # that's flagged as a violation or treated as an intentional design.
    electrode_gap_um: float = -1.5              # NEGATIVE by default = intentional overlap:
                                                  # covers the adhesion plate (1.0um default)
                                                  # plus some of the bare finger beyond it;
                                                  # positive means a real gap.
    allow_electrode_overlap: bool = True          # matches the new default above being an
                                                    # intentional overlap, not a violation to flag

    # --- Patch-integrated design ---
    patch_integrated: bool = False
    patch_count: int = 3
    patch_spacing_um: float = 1.0
    patch_width_um: float = 0.5
    # patch_height_um used to define a symmetric w x h box centered at an
    # offset point independent of the finger's own geometry -- there was
    # no guarantee, and frequently no actual overlap, between the rotated
    # patch rectangle and the finger it was supposed to electrically
    # connect to. The reference KLayout fabrication script instead
    # defines patches by how far they extend BEYOND the finger's edge on
    # each side -- asymmetrically, a long extension one way and a short
    # extension the other -- which by construction always overlaps the
    # finger. These replace patch_height_um entirely.
    patch_long_extension_um: float = 6.0        # matches the real script's long_extension
    patch_short_extension_um: float = 1.8       # matches the real script's short_extension
    patch_angle: float = 45.0                   # deg; connect sub-step theta
    # Minimum finger/plate material that must remain PAST the outermost
    # patch, toward the very tip -- a separate shadow-margin concern from
    # the long/short extensions above. Deposition at a non-zero tilt can
    # land the finger's own metal shifted from its nominal (top-down)
    # position; if the outermost patch sits too close to the finger's
    # own tip, that shift alone could disconnect it even though the
    # design looks fine from directly above. Default: 2 um minimum,
    # adjustable per process.
    patch_tip_margin_um: float = 2.0

    # --- Ion milling ---
    top_mill_theta: float = 0.0                 # arbitrary; alpha is always 0 for top mill
    sidewall_mills: List[SidewallMill] = field(default_factory=list)

    # --- Timing ---
    wait_minutes: float = 20.0

    # --- Resist stack ---
    # Generalized bilayer, per shadowing_design_rules.md section 3: the
    # process is not fixed to PMMA A7 / PMGI SF9 specifically. The two
    # thickness fields keep their original names (pmma_thickness_nm /
    # pmgi_thickness_nm) so every existing formula call
    # (compute_overlap_shadow_bounds, compute_alpha_from_linewidth,
    # compute_linewidth_from_alpha -- all purely numeric, no name
    # dependency) and every other call site keeps working unchanged;
    # top_resist_name/bottom_resist_name are new, display-only fields
    # (used in generated recipe text / the Settings calculator) so a
    # user running a different bilayer sees their own resist names
    # instead of a hardcoded "PMMA"/"PMGI" label.
    top_resist_name: str = "PMMA A7"
    pmma_thickness_nm: Optional[float] = 875.0
    bottom_resist_name: str = "PMGI SF9"
    pmgi_thickness_nm: Optional[float] = 600.0
    line_width_nm: Optional[float] = None
    safety_margin_nm: float = 100.0

    # --- Electrode identification convention (shared with Tab 6's GDS/
    # script import) -- only meaningful when this design is built FROM an
    # import (see junction_import.from_design_parameters is the reverse
    # direction; this field matters when Tab 6 populates a
    # DesignParameters FROM an imported file, so process parameters and
    # the electrode convention that produced them travel together as one
    # object). None means "not import-derived" -- a purely hand-designed
    # Tab 1 junction has no file-derived electrode convention to carry.
    electrode_convention: Optional[ElectrodeConvention] = None

    # --- Free-form electrode shapes (interactive canvas editor) --
    # empty by default, which means "use the old horizontal_electrode_*/
    # vertical_electrode_* fields above exactly as before" -- fully
    # backward compatible. Once populated (by placing shapes on the
    # canvas), THIS list takes priority: it isn't limited to exactly one
    # horizontal + one vertical electrode the way the old fields are.
    electrode_shapes: List["EditableShape"] = field(default_factory=list)

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


# ============================================================================
# Electrode-layer geometry (pure math, no matplotlib -- parametric_junction_view.py
# imports these for drawing; generate_recipe_steps below uses them directly
# for Warning #1/#2 without needing to import parametric_junction_view, which would be
# circular since parametric_junction_view imports DesignParameters/RecipeStep from here).
# ============================================================================

def _dir_xy(angle_deg: float):
    a = math.radians(angle_deg)
    return math.cos(a), math.sin(a)


@dataclass
class ElectrodeGeom:
    cx: float           # near-edge midpoint (world) -- the anchor facing the junction
    cy: float
    angle_deg: float    # direction the electrode extends AWAY from the junction
    w_um: float
    h_um: float
    auto_placed: bool   # True if x/y weren't explicitly set (still following the finger)


def finger_tip_point(p: "DesignParameters", side: str):
    """World (x, y) of the outward-most point of this finger -- the far
    edge of its adhesion plate if present, else the finger's own tip.
    This is the point that can potentially touch/cross onto the electrode."""
    angle = getattr(p, f"{side}_finger_angle")
    length = getattr(p, f"{side}_length_um")
    overlap = p.overlap_target_um or 0.0
    end_x = length - overlap
    if p.has_adhesion_pads:
        end_x += p.adhesion_plate_height_um
    dx, dy = _dir_xy(angle)
    return end_x * dx, end_x * dy


def electrode_geometry(p: "DesignParameters", side: str) -> ElectrodeGeom:
    """Independent electrode-layer placement for one side. If x/y have been
    explicitly set (the electrode has been "fixed" as its own layer), that
    position is used verbatim and does NOT move when the finger angle
    changes. Otherwise it auto-follows the finger tip at electrode_gap_um
    standoff, same as the legacy "attached right at the finger" behavior.

    The auto-placed anchor is walked out from the finger tip along the
    ELECTRODE's OWN axis (its angle override if set, else the finger's own
    angle) -- not the finger's axis. That keeps the finger tip exactly
    `electrode_gap_um` deep into the electrode's near edge AND perfectly
    centered on it (zero sideways offset), for any override angle. Walking
    along the finger's axis instead (the old behavior) only gave that
    guarantee when the override matched the finger angle exactly: any
    rotation away from that shrank the along-axis overlap by cos(delta) and,
    worse, slid the tip sideways by sin(delta) of the (unshrunk) gap
    distance -- for a narrow electrode this sideways slide alone was enough
    to walk the tip clean off the electrode's near edge, which is what made
    a 45-degree override look like it had erased the standoff gap entirely."""
    x = getattr(p, f"{side}_electrode_x_um")
    y = getattr(p, f"{side}_electrode_y_um")
    ang = getattr(p, f"{side}_electrode_angle_deg")
    w = getattr(p, f"{side}_electrode_w_um")
    h = getattr(p, f"{side}_electrode_h_um")
    finger_angle = getattr(p, f"{side}_finger_angle")
    tip_x, tip_y = finger_tip_point(p, side)
    auto = x is None or y is None
    if ang is None:
        ang = finger_angle
    if auto:
        gap = p.electrode_gap_um if p.electrode_gap_um is not None else 1.0
        dx, dy = _dir_xy(ang)
        x = tip_x + dx * gap
        y = tip_y + dy * gap
    return ElectrodeGeom(x, y, ang, w, h, auto)


def finger_electrode_gap_um(p: "DesignParameters", side: str) -> float:
    """Signed standoff distance from the finger's outward tip to its
    electrode's near edge, measured along the electrode's own outward
    axis. Positive = a real gap (no overlap). Negative = the finger tip
    has crossed past the electrode's near edge -- the two features
    overlap by that many um."""
    tip_x, tip_y = finger_tip_point(p, side)
    e = electrode_geometry(p, side)
    dx, dy = _dir_xy(e.angle_deg)
    vx, vy = e.cx - tip_x, e.cy - tip_y
    return vx * dx + vy * dy


def electrode_safe_theta(p: "DesignParameters", side: str):
    """(safe_theta, unsafe_theta) in degrees for depositing a finger that
    crosses onto its own electrode (Warning #1). By convention
    {side}_finger_angle already points TOWARD that finger's electrode, so
    depositing AT that angle means material travels from the substrate
    side UPHILL onto the elevated electrode (unobstructed). Depositing at
    angle+180 means material arrives from beyond the electrode and travels
    DOWNHILL across the step edge, where the electrode's own raised edge
    shadows the substrate side right at the boundary -- breaking metal
    continuity between the electrode and the substrate for every junction
    in the run."""
    angle = getattr(p, f"{side}_finger_angle")
    return angle % 360.0, (angle + 180.0) % 360.0


def self_shadow_safe_theta(p: "DesignParameters"):
    """(safe_theta, unsafe_theta, side) for Warning #2 -- the SECOND
    finger's own self-shadowing direction, relevant only when the whole
    junction sits on the substrate (no electrode overlap on either side).
    The second finger crosses perpendicular over the first finger's own
    raised step edge; whichever way material travels, the resulting thin
    spot lands on the DOWNSTREAM side of that step. Depositing TOWARD the
    electrode (angle_deg, the Warning #1 "safe" direction) puts the thin
    spot on the electrical-connection side -- bad here, since there's no
    electrode step to justify it. Depositing AWAY from the electrode
    (angle+180) puts the thin spot on the harmless safety-overlap/JJ-tail
    side instead, which doesn't carry current."""
    side = "vertical" if p.deposit_first == "horizontal" else "horizontal"
    angle = getattr(p, f"{side}_finger_angle")
    return (angle + 180.0) % 360.0, angle % 360.0, side


def check_electrode_standoff_warnings(p: "DesignParameters"):
    """Tab-1-level design warnings about how each finger relates to its own
    (independently-placed) electrode -- doesn't require knowing which Tab 2
    angle was actually committed yet."""
    out = []
    for side in ("horizontal", "vertical"):
        gap = finger_electrode_gap_um(p, side)
        if gap >= 0:
            if not p.allow_electrode_overlap and gap < 0.05:
                out.append(
                    f"{side.capitalize()} finger sits only {gap:.3f} um from its electrode -- "
                    "very little margin for alignment error before it touches unintentionally."
                )
            continue
        safe_t, unsafe_t = electrode_safe_theta(p, side)
        if not p.allow_electrode_overlap:
            out.append(
                f"{side.capitalize()} finger overlaps its electrode by {-gap:.3f} um, but overlap "
                f"isn't allowed -- pull the finger/electrode back by {-gap:.3f} um, or enable "
                f"overlap and deposit at theta={safe_t:.0f} deg (not {unsafe_t:.0f} deg, which "
                f"disconnects the junction via electrode shadowing)."
            )
        else:
            out.append(
                f"{side.capitalize()} finger overlaps its electrode by {-gap:.3f} um (allowed). "
                f"Use theta={safe_t:.0f} deg, not theta={unsafe_t:.0f} deg (electrode shadowing "
                f"disconnects the junction)."
            )
    return out


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
    feature: Optional[str] = None  # what this sub-step visually deposits


@dataclass
class RecipeHandEdit:
    """One user hand-edit applied directly to an already-generated
    recipe step, after generation. Tracked as its own list
    (main_gui.py's self._recipe_hand_edits) rather than folded
    into RecipeStep itself, so the edit history is a
    single, easy-to-serialize source of truth for three independent
    consumers that each render it differently: the on-screen HTML
    (an inline highlight around the edited value), the exported PDF
    (the same highlight via reportlab's own markup), and the plain-text
    .txt export (a bracketed marker) -- plus the Design Library, which
    saves/restores this list verbatim so a reloaded design still shows
    which values were hand-touched. Applies equally to a deterministic-
    generated recipe and an adopted AI draft, since both end up as plain
    RecipeStep objects by the time an edit is possible."""
    step_index: str     # RecipeStep.index this edit applies to
    step_title: str      # snapshot of the title at edit time, for a readable tally
    field_label: str      # human label, e.g. "Deposition Rate"
    old_value: str          # formatted old value, for display
    new_value: str            # formatted new value, for display


def _wrap360(deg: float) -> float:
    return deg % 360.0


# ----------------------------------------------------------------------
# Literal machine-step detail builders (mirror the actual Plassys
# "Executing" step list, confirmed against real recipe screenshots)
# ----------------------------------------------------------------------

def _ti_gather_detail(p: DesignParameters, warnings: List[str]) -> str:
    prof = get_ramp_profile("Ti", p.ti_gather_rate_nmps, warnings)
    lines = [
        f"Root\\Evap\\Titanium\\Ti {p.ti_gather_rate_nmps:g}nm/s "
        f"{p.ti_gather_duration_min:g}min",
        "Move Table to Deposit + 0.00 deg",
        "Process Chamber / Process LL pressure checks",
        "Substrate Shutter EBGun Close",
        "Crucible - Ti",
        "Alert -- remain present while the beam is on",
        f"Egun emission ramp - 1mA, 0:0 -> wait {_fmt_mmss(prof.pre_wait_s)}",
        f"Egun emission ramp - {prof.soak1_ma:g}mA, {_fmt_mmss(prof.soak1_time_s)} -> Ramp Wait - All",
        "Alert -- check beam is centered on crucible",
        f"Shutter Source Open -> Egun emission ramp - {prof.soak2_ma:g}mA, "
        f"{_fmt_mmss(prof.soak2_time_s)} -> Ramp Wait - All",
        f"Rate Control {p.ti_gather_rate_nmps:g}nm/s -> wait {_fmt_mmss(prof.settle_wait_s)}",
        f"(gather: substrate shutter stays CLOSED) hold {p.ti_gather_duration_min:g}:00",
        f"Egun emission ramp - {prof.rampdown_ma:g}mA, {_fmt_mmss(prof.rampdown_time_s)} "
        f"-> Ramp Wait - All -> Egun emission ramp - 0mA, 0:0",
        f"wait {_fmt_mmss(prof.cooldown_wait_s)}",
    ]
    if not prof.confirmed:
        lines.append("[ESTIMATED ramp -- verify on tool before running]")
    return "\n      ".join(lines)


def _deposition_ramp_detail(p: DesignParameters, is_ramp_owner: bool,
                             warnings: List[str], thickness_nm: Optional[float] = None,
                             theta: Optional[float] = None) -> str:
    """
    is_ramp_owner=True: this sub-step actually ramps the e-gun up from
    cold (first angle of a deposition). is_ramp_owner=False: this
    sub-step reuses the SAME continuous ramp/soak as the previous
    sub-step (confirmed pattern: a second angle within one deposition
    only does "Move Planetary" + Zero Thickness + Shutter Open/Close,
    it does NOT re-ramp the e-gun from scratch).

    Verified against the real Plassys recipe editor (the actual
    Step Type / Parameter 1-4 grid, not just the Executing screen): the
    owning sub-step also has a substrate-position step ("Recipe Tilt
    Substrat -> Deposit"), its own planetary rotation to the FIRST
    angle, and a housekeeping pump/LL-pressure pair before the crucible
    is selected -- these were previously only implied, not written out.
    theta is now threaded through and actually substituted into the
    rotation line for every sub-step (a prior version of the non-owner
    branch left the literal word "theta" in place instead of the real
    angle).
    """
    prof = get_ramp_profile(p.metal_type, p.al_dep_rate_nmps, warnings)
    # A project name is entirely optional -- if the user hasn't set one,
    # don't leave a dangling "- " artifact in the recipe path (that was
    # the old hardcoded-default behavior); just use the metal name alone.
    name_part = f"{p.project_name}- " if p.project_name else ""
    thickness_note = f"{thickness_nm:g} nm" if thickness_nm is not None else "target thickness"
    theta_str = f"{theta:g}" if theta is not None else "(not set)"
    if is_ramp_owner:
        lines = [
            f"Root\\Evap\\{p.metal_type}\\{name_part}{p.metal_type} "
            f"{p.al_dep_rate_nmps:g}nm/s thick? tilt? Manhattan",
            "Move Table to Deposit + 0.00 deg",
            f"Move Planetary to Zero + {theta_str} deg",
            "Process Chamber / Process LL pressure checks",
            "Substrate Shutter EBGun Close",
            "Crucible select -> Alert -- remain present while the beam is on",
            f"Egun emission ramp - 1mA, 0:0 -> wait {_fmt_mmss(prof.pre_wait_s)}",
            f"Egun emission ramp - {prof.soak1_ma:g}mA, {_fmt_mmss(prof.soak1_time_s)} -> Ramp Wait - All",
            "Alert -- check beam is centered on crucible",
            f"Shutter Source Open -> Egun emission ramp - {prof.soak2_ma:g}mA, "
            f"{_fmt_mmss(prof.soak2_time_s)} -> Ramp Wait - All",
            f"Rate Control {p.al_dep_rate_nmps:g}nm/s -> wait {_fmt_mmss(prof.settle_wait_s)}",
            f"Zero Thickness -> Substrate Shutter EBGun Open -> Wait for Termination "
            f"(crystal monitor cuts shutter at {thickness_note}) -> Substrate Shutter EBGun Close",
        ]
        if not prof.confirmed:
            lines.append("[ESTIMATED ramp -- verify soak currents on tool before running]")
    else:
        lines = [
            "(same continuous ramp/soak as the previous sub-step -- e-gun is NOT re-ramped)",
            f"Move Planetary to Zero + {theta_str} deg",
            f"Zero Thickness -> Substrate Shutter EBGun Open -> Wait for Termination "
            f"(crystal monitor cuts shutter at {thickness_note}) -> Substrate Shutter EBGun Close",
        ]
    if thickness_nm is None:
        lines.append("[NO TARGET THICKNESS SET for this sub-step -- set it in Tab 2 before running]")
    return "\n      ".join(lines)


def _deposition_rampdown_detail(p: DesignParameters, warnings: List[str]) -> str:
    prof = get_ramp_profile(p.metal_type, p.al_dep_rate_nmps, warnings)
    return (f"Egun emission ramp - {prof.rampdown_ma:g}mA, {_fmt_mmss(prof.rampdown_time_s)} "
            f"-> Ramp Wait - All -> Egun emission ramp - 0mA, 0:0 -> "
            f"wait {_fmt_mmss(prof.cooldown_wait_s)}")


def _oxidation_detail(p: DesignParameters) -> str:
    return "\n      ".join([
        f"Root\\Static oxidation\\Static oxid {p.oxidation_pressure_mbar:g} mBar "
        f"time? {p.oxidation_time_min:g}:00",
        f"Static Oxidation {p.oxidation_pressure_mbar:.2f} mBar, Static O2",
        f"Param wait {p.oxidation_time_min:g} min  <- USER-ALTERABLE oxidation time",
        f"Static Oxidation 0.000 mBar, Static O2 (off) -> Pump LL -> wait 0:5",
    ])


def _mill_detail(p: DesignParameters, group_index: int, alpha: float, theta: Optional[float],
                  is_beam_owner: bool, is_beam_closer: bool) -> str:
    """Builds one sub-angle's slice of the mill's literal Plassys
    recipe-tree step text. group_index distinguishes multiple mill
    groups in the same recipe (e.g. one for each finger/patch sidewall)
    the same way separate named recipes would need to on the real tool,
    without assuming any particular lab's naming convention.

    Verified against the real Plassys recipe editor (the actual
    Step Type / Parameter 1-4 grid, not just an Executing-screen
    snapshot): a multi-angle mill group strikes the ion beam ONCE for
    the whole group -- Ar gas on, IBG discharge on, beam struck -- and
    every angle pass after the first is just a planetary rotation plus
    a substrate-shutter open/wait/close cycle; the beam, discharge, and
    gas are only switched off after the LAST angle pass. This mirrors
    the same "don't re-ramp for a second angle" pattern
    _deposition_ramp_detail already uses (is_ramp_owner there,
    is_beam_owner here) -- a prior version of this function had no such
    distinction and gave every sub-angle its own full, independent
    strike/mill/extinguish block, which doesn't match how the tool is
    actually run.
    is_beam_owner=True: this sub-step strikes the ion beam from cold
    (first angle of a mill group). is_beam_closer=True: this sub-step
    turns the ion beam back off (last angle of a mill group). A
    single-angle mill is both owner and closer.
    """
    if p.project_name:
        recipe_name = p.project_name if group_index == 1 else f"{p.project_name}_{group_index}"
    else:
        recipe_name = f"Mill{group_index}"
    theta_str = f"{theta:g}" if theta is not None else "(not set)"
    lines = []
    if is_beam_owner:
        lines += [
            f"Root\\Etching\\New Ion Gun Recipes\\{recipe_name}",
            f"Substrate Position -> Etch + {alpha:g} deg",
            f"Move Planetary to Zero + {theta_str} deg",
            "Substrate Shutter Ion Gun Close -> Pump Chamber / Process LL pressure checks",
            f"Ar Ion Gun Gas {p.mill_ar_gas_sccm:g}sccm -> wait 0:10",
            "IBG Discharge On -> wait 0:30",
            f"Ion Beam V={p.mill_beam_voltage_v:g}V, I={p.mill_beam_current_ma:.1f}mA, "
            f"Vacc={p.mill_accel_voltage_v:g}V -> wait 0:30",
        ]
    else:
        lines += [
            "(same continuous ion beam strike as the previous sub-step -- beam is NOT re-struck)",
            f"Move Planetary to Zero + {theta_str} deg",
        ]
    lines.append(f"Substrate Shutter Ion Gun Open -> mill for {p.mill_duration_min:g} min "
                  f"-> Substrate Shutter Ion Gun Close")
    if is_beam_closer:
        lines.append("Ion Beam Off -> IBG Discharge Off -> Ar Ion Gun Gas 0.0sccm -> Substrate Shutter Ion Gun Close")
    return "\n      ".join(lines)


def _classify_feature(theta: float, group_size: int, sub_index: int,
                       p: DesignParameters, patch_counter: List[int]) -> Optional[str]:
    """
    Best-effort feature tag for the Tab 4/5 junction-formation visualizer
    (parametric_junction_view.py), which expects legacy tags like "horizontal_finger",
    "vertical_connect"/"vertical_overlap", "patch_a"/"patch_b", "mill_top",
    "mill_sidewall". Since the sequence is now fully user-driven (not a
    fixed finger1/finger2/patch template), this matches each angle against
    the finger angles from Tab 1's geometry section within a small
    tolerance; anything that doesn't match becomes "patch_a"/"patch_b" in
    the order encountered (capped at two), else None (the visualizer just
    won't highlight that step specially -- it still renders, just without
    a feature-specific overlay).
    """
    TOL_DEG = 2.0

    def close(a, b):
        return abs(((a - b + 180.0) % 360.0) - 180.0) <= TOL_DEG

    if close(theta, p.horizontal_finger_angle):
        side = "horizontal"
    elif close(theta, p.vertical_finger_angle):
        side = "vertical"
    else:
        side = None

    if side is not None:
        if group_size == 1:
            return f"{side}_finger"
        return f"{side}_connect" if sub_index == 0 else f"{side}_overlap"

    # Not a finger angle -- treat as a patch-like feature, in encounter order.
    if patch_counter[0] < 2:
        tag = "patch_a" if patch_counter[0] == 0 else "patch_b"
        patch_counter[0] += 1
        return tag
    return None


def _mill_feature(alpha: float) -> str:
    """alpha ~ 0 reads as a full top-surface mill; anything else is a
    sidewall mill. This is just a visualization heuristic (parametric_junction_view.py
    draws these differently) -- it doesn't force a top mill to exist."""
    return "mill_top" if abs(alpha) < 1e-6 else "mill_sidewall"


def _theta_close(a: float, b: float, tol_deg: float = 2.0) -> bool:
    return abs(((a - b + 180.0) % 360.0) - 180.0) <= tol_deg


def _describe_mill_contact_side(theta: Optional[float], mill_feature_index: int, features: list,
                                 alpha: Optional[float] = None) -> str:
    """The governing rule: the part of the electrode being ion-milled must
    be the same side/part of the electrode that will make contact with
    the finger, adhesion plate, or patches -- the electrode side depends
    on theta and on the specific deposition steps that come ahead of it.
    This is text-only (no chamber-preview highlighting needed). This
    names the side this mill sub-step targets by its own theta (the
    parameter that actually determines which physical face of the
    electrode gets hit) and explicitly cross-checks it against the
    NEXT deposition feature ahead of it in the committed sequence -- the
    pass whose metal actually needs a clean, oxide-free contact -- rather
    than leaving that connection implicit across two separate, unlinked
    paragraphs of recipe text the way it was before. This applies
    uniformly whether the mill/deposition sequence came from New Tab 3's
    simulation (build_simulation_result, junction_import.py) or was
    hand-committed one step at a time in New Tab 2 (_flatten_committed_
    steps above) -- the latter has NO automatic angle linkage at all
    between a manually-added mill and a manually-added deposition, so
    nothing previously verified, or even stated, that they'd end up
    matching.

    build_simulation_result batches ALL mills before ANY deposition (a
    milled surface doesn't reoxidize under vacuum, so there's no reason
    to interleave), so a mill's own matching deposition is not
    necessarily the very NEXT feature -- it can be several mills further
    down the batched block. This searches every deposition feature that
    follows this mill (not just the first one encountered) for a theta
    match, rather than stopping at the first deposition group and
    reporting a mismatch if that one alone doesn't match.

    A blanket top mill (alpha=0, no tilt) is not a sidewall mill -- it
    has no "side" at all, since an untilted beam hits every exposed top
    surface uniformly regardless of theta. Its theta=0 is just a
    hardware placeholder, not a targeted direction, so it must never be
    compared against upcoming deposition thetas the way a real sidewall
    mill is -- that would falsely flag a "WARNING -- no deposition
    matches theta=0" on a step that was never supposed to match anything
    in the first place."""
    if alpha is not None and abs(alpha) < 1e-6:
        return ("This is the blanket top mill (alpha=0, no tilt) -- it mills the entire exposed top "
                "surface uniformly, not one particular electrode side, so theta here is just a hardware "
                "placeholder and there's nothing to cross-check it against.")
    if theta is None:
        return ("No theta was set for this mill sub-step, so the contact side it targets can't be "
                "named or cross-checked -- set an angle before running.")
    upcoming_dep_thetas = [
        a[1] for ftype, fangles in features[mill_feature_index + 1:] if ftype == "deposition"
        for a in fangles if a[1] is not None
    ]
    if any(_theta_close(theta, dt) for dt in upcoming_dep_thetas):
        return (f"Targets the electrode side facing theta={theta:g} deg -- a later deposition pass "
                 f"(also at theta={theta:g} deg) lands on the same side, leaving a clean contact.")
    if upcoming_dep_thetas:
        others = ", ".join(f"{dt:g}" for dt in sorted(set(upcoming_dep_thetas)))
        return (f"Warning: targets theta={theta:g} deg, but no later deposition pass lands at that "
                 f"angle (upcoming: {others} deg). Verify this mill actually cleans a surface a later "
                 f"pass will contact.")
    return (f"Targets the electrode side facing theta={theta:g} deg -- no deposition step follows "
             f"this mill, so there's nothing to cross-check against.")


def _flatten_committed_steps(custom_steps):
    """
    Turn Tab 2's committed-step list (list of dicts, possibly containing
    "Mill Group"/"Deposition Group" entries with an "items" sub-list) into
    an ordered list of features:
        ("mill", [(alpha, theta), ...])
        ("deposition", [(alpha, theta, thickness_nm), ...])
        ("oxidation", None)
    preserving your exact commit order. Unrecognized entries are skipped.
    thickness_nm is None for any deposition sub-step committed before the
    Tab 2 thickness field existed, or left blank -- generate_recipe_steps
    surfaces that as an explicit warning rather than silently guessing.
    """
    features = []
    for step in (custom_steps or []):
        t = step.get("type", "")
        if t == "Oxidation":
            features.append(("oxidation", None))
        elif t == "Mill Group":
            angles = [(it.get("alpha", 0.0), it.get("theta", 0.0)) for it in step.get("items", [])]
            if angles:
                features.append(("mill", angles))
        elif t == "Deposition Group":
            angles = [(it.get("alpha", 0.0), it.get("theta", 0.0), it.get("thickness_nm")) for it in step.get("items", [])]
            if angles:
                features.append(("deposition", angles))
        elif t == "Mill":
            features.append(("mill", [(step.get("alpha", 0.0), step.get("theta", 0.0))]))
        elif t == "Deposition":
            features.append(("deposition", [(step.get("alpha", 0.0), step.get("theta", 0.0), step.get("thickness_nm"))]))
        # Anything else (e.g. an "Oxidation Group" from an accidental grouping
        # of an oxidation entry with others) is skipped rather than guessed at.
    return features


def generate_recipe_steps(p: DesignParameters, custom_steps=None, skip_old_electrode_warnings: bool = False):
    """
    Returns (steps: List[RecipeStep], warnings: List[str]).

    skip_old_electrode_warnings: when True, skips the OLD, fixed-electrode-
    model warning checks below (check_electrode_standoff_warnings,
    finger_electrode_gap_um, electrode_safe_theta, self_shadow_safe_theta)
    -- these read p.horizontal_finger_angle/p.vertical_finger_angle and
    the old fixed 2-electrode geometry DIRECTLY, with no knowledge of the
    free-form p.electrode_shapes canvas or the unified junction_import.py
    warning_source classification. For a recipe built from a New Tab 2
    simulation (see generate_recipe_steps_from_simulation), these checks
    are comparing against whatever DEFAULT/unrelated fixed-angle fields
    happen to be sitting on `p` -- confirmed to produce actively
    misleading warnings (a fixed-model angle mismatch reported as a
    "PROJECT-BREAKING" disconnect that has nothing to do with the real,
    already-correctly-classified simulated geometry). Defaults to False
    so the existing Tab 2 manual-commit workflow is completely unchanged.

    The recipe is now driven ENTIRELY by the ordered list of committed
    steps from Tab 2 (`custom_steps`) -- whatever Mill/Deposition/
    Oxidation entries you commit, in whatever order, is exactly what gets
    built. There is no assumed structure (no forced top mill, no forced
    finger-1/finger-2/patch split, no forced oxidation placement): if you
    don't commit a mill step, no milling happens; if you don't commit an
    oxidation step, no oxidation happens; deposition order follows your
    commit order exactly.

    The generator's only job is filling in the boilerplate BETWEEN your
    committed features, using the same literal machine-step patterns
    confirmed against real Plassys MEB550S recipe screenshots:
      - Vacuum check/pump: once, fixed, at the very start.
      - Ti gather: always immediately before every deposition group.
      - Pump + final wait: once, fixed, at the very end.
      - Mill gas/discharge/beam boilerplate: wrapped around every mill
        group.
      - A deposition group with more than one angle is ONE continuous
        e-gun ramp/soak (the gun is not re-ramped between angles within
        the same group) -- only the FIRST angle "owns" the ramp-up; the
        rest are "Move Planetary" + Zero Thickness + Shutter cycles, and
        the ramp-down happens once, after the LAST angle in the group.
      - Each committed step keeps its OWN alpha (not a single global
        deposition alpha) -- different depositions in the same recipe can
        use different tilt angles.
      - Oxidation only happens where you've explicitly committed an
        Oxidation step -- it is never auto-inserted.
    """
    steps: List[RecipeStep] = []
    warnings: List[str] = []

    def check_alpha_range(alpha, where):
        if alpha is not None and not (ALPHA_MIN <= alpha <= ALPHA_MAX):
            warnings.append(
                f"{where}: alpha={alpha:.2f} deg is outside the hardware range "
                f"[{ALPHA_MIN}, {ALPHA_MAX}]. Verify before running."
            )

    features = _flatten_committed_steps(custom_steps)

    if not features:
        warnings.append(
            "No steps committed yet -- the recipe below is just the fixed vacuum/pump bookends."
        )

    # Vacuum check / pump (fixed, once, at the start)
    steps.append(RecipeStep("1", "Vacuum check / pump", "prep", None, None, False,
                             "Pump Chamber / Process LL to base pressure before any gather or mill."))

    patch_counter = [0]
    step_num = 2
    mill_count = 0
    deposition_count = 0
    oxidation_count = 0

    for feat_index, (feature_type, angles) in enumerate(features):

        if feature_type == "mill":
            mill_count += 1
            for j, (alpha, theta) in enumerate(angles):
                idx = str(step_num) if len(angles) == 1 else f"{step_num}{chr(ord('a') + j)}"
                fname = _mill_feature(alpha)
                title = f"Argon Mill #{mill_count}" + (f" ({j + 1}/{len(angles)})" if len(angles) > 1 else "")
                # theta CAN legitimately be None here (a patch-mill
                # sub-step built with no patches_thetas at all -- see
                # build_simulation_result) -- _wrap360(None) would raise
                # TypeError before this ever reached the RecipeStep
                # below. theta_wrapped/theta_str degrade gracefully
                # instead.
                theta_wrapped = _wrap360(theta) if theta is not None else None
                theta_str = f"{theta_wrapped:g}" if theta_wrapped is not None else "(not set)"
                # State explicitly which side of the electrode this mill
                # targets and cross-check it against the next deposition
                # pass ahead of it -- see _describe_mill_contact_side's
                # own docstring.
                contact_note = _describe_mill_contact_side(theta_wrapped, feat_index, features, alpha)
                is_beam_owner = (j == 0)
                is_beam_closer = (j == len(angles) - 1)
                steps.append(RecipeStep(
                    idx, title, "mill", alpha, theta_wrapped, True,
                    f"alpha={alpha:g} deg, theta={theta_str} deg (committed in Tab 2).\n      "
                    + contact_note + "\n      "
                    + _mill_detail(p, mill_count, alpha, theta_wrapped, is_beam_owner, is_beam_closer),
                    feature=fname
                ))
                check_alpha_range(alpha, title)
            step_num += 1

        elif feature_type == "deposition":
            deposition_count += 1
            # Ti gather -- always immediately before every deposition group.
            steps.append(RecipeStep(str(step_num), "Ti gather (lower pressure)", "prep", None, None, False,
                                     _ti_gather_detail(p, warnings)))
            step_num += 1

            n = len(angles)
            for j, angle_entry in enumerate(angles):
                # Backward-compatible: older committed steps (or mill-group
                # leftovers) may only carry (alpha, theta) with no thickness.
                padded = tuple(angle_entry) + (None, None, None)
                alpha, theta, thickness_nm = padded[0], padded[1], padded[2]
                idx = str(step_num) if n == 1 else f"{step_num}{chr(ord('a') + j)}"
                is_owner = (j == 0)
                is_last = (j == n - 1)
                feat = _classify_feature(theta, n, j, p, patch_counter)
                thickness_suffix = f" -- {thickness_nm:g}nm" if thickness_nm is not None else ""
                title = f"Deposition #{deposition_count}" + (f" ({j + 1}/{n})" if n > 1 else "") + thickness_suffix

                detail = f"alpha={alpha:g} deg, theta={theta:g} deg (from Tab 2)."
                if thickness_nm is not None:
                    detail += f" Target thickness: {thickness_nm:g} nm."
                detail += "\n      "
                detail += _deposition_ramp_detail(p, is_owner, warnings, thickness_nm=thickness_nm, theta=theta)
                if is_last:
                    detail += "\n      " + _deposition_rampdown_detail(p, warnings)

                if thickness_nm is None:
                    warnings.append(
                        f"Deposition #{deposition_count}" + (f" ({j + 1}/{n})" if n > 1 else "") +
                        ": no target thickness was set in Tab 2 for this sub-step before committing. "
                        "Set it before running -- the shutter's crystal-monitor cutoff needs a real number."
                    )

                steps.append(RecipeStep(
                    idx, title, "deposition", alpha, _wrap360(theta), True,
                    detail, feature=feat
                ))
                check_alpha_range(alpha, title)
            step_num += 1

        elif feature_type == "oxidation":
            oxidation_count += 1
            steps.append(RecipeStep(str(step_num), "Oxidation step", "misc", None, None, False,
                                     _oxidation_detail(p)))
            step_num += 1

    # Pump + final wait (fixed, once, at the very end)
    steps.append(RecipeStep(str(step_num), "Pump", "misc", None, None, False,
                             "Pump Chamber / Process LL back down."))
    step_num += 1
    steps.append(RecipeStep(str(step_num), f"Wait ({p.wait_minutes:g} minutes)", "misc", None, None, False,
                             "Final wait/cooldown before unload."))

    if deposition_count >= 2 and oxidation_count == 0:
        warnings.append(
            "Multiple deposition groups but no Oxidation step -- an SIS junction needs one "
            "between the base and counter electrode depositions to form the tunnel barrier."
        )

    # The old "JJ area (LW_horizontal x LW_vertical) = ..." warning that
    # used to be injected here was removed -- it was a crude nominal-
    # linewidth estimate with no shadowing/geometry awareness at all,
    # computed independently of (and often disagreeing with) the REAL
    # post-shadow JJ crossing area New Tab 2 already computes and
    # displays (report.jj_crossing_area_um2, from actual polygon
    # intersection). recipe_generator.py has no access to that real geometry
    # by design (avoids a circular import with junction_import.py), so
    # this was never going to be more than an unverified guess sitting
    # alongside a warnings list that's otherwise about real, geometry-
    # driven risk. The real area is exactly where it belongs: New Tab 2's
    # Warnings/Risk panel, computed from the real design.

    # --- Warning #1 (electrode shadowing disconnect, project-breaking) and
    # --- Warning #2 (self-shadowing overlap thinning, quality) -- cross-
    # check the ACTUAL committed Tab 2 deposition angles against this
    # design's electrode-overlap geometry (Tab 1). Independent of the
    # _classify_feature heuristic above, so a wrongly-committed angle still
    # gets flagged even if it doesn't happen to match either finger angle
    # closely enough to be tagged "*_finger"/"*_connect".
    TOL_DEG = 2.0

    def _theta_close(a, b):
        return abs(((a - b + 180.0) % 360.0) - 180.0) <= TOL_DEG

    if not skip_old_electrode_warnings:
        warnings.extend(check_electrode_standoff_warnings(p))

        committed_thetas = [a for (kind, angles) in features if kind == "deposition" for a in angles]
        both_clear_of_electrode = True
        for side in ("horizontal", "vertical"):
            gap = finger_electrode_gap_um(p, side)
            if gap < 0:
                both_clear_of_electrode = False
                safe_t, unsafe_t = electrode_safe_theta(p, side)
                for (_alpha, theta, _thick) in committed_thetas:
                    if _theta_close(theta, unsafe_t) and not _theta_close(theta, safe_t):
                        warnings.append(
                            f"A committed deposition step uses theta={theta:g} deg, which disconnects "
                            f"the {side} finger via electrode shadowing (it overlaps its electrode by "
                            f"{-gap:.3f} um). Use theta={safe_t:.0f} deg instead."
                        )

        if both_clear_of_electrode:
            safe_t2, unsafe_t2, second_side = self_shadow_safe_theta(p)
            for (_alpha, theta, _thick) in committed_thetas:
                if _theta_close(theta, unsafe_t2) and not _theta_close(theta, safe_t2):
                    warnings.append(
                        f"A committed deposition step uses theta={theta:g} deg for the {second_side} "
                        f"(second-deposited) finger, putting the self-shadowing thin spot on the "
                        f"electrode-connecting side -- expect inflated resistance or an open junction. "
                        f"Use theta={safe_t2:.0f} deg instead."
                    )

    if not (0 < p.al_dep_rate_nmps):
        raise ValueError("Deposition rate must be positive.")

    # De-duplicate identical warnings (e.g. the same "estimated ramp profile"
    # notice fires once per deposition sub-step that uses it).
    deduped_warnings = list(dict.fromkeys(warnings))

    return steps, deduped_warnings


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
    # project_name is real Settings state (it names the mill/deposition
    # recipe-tree entries every step below is actually built under, e.g.
    # Root\Etching\New Ion Gun Recipes\<project_name>, see _mill_detail)
    # so it's echoed in the header, giving an exported recipe a
    # standalone record of which project/naming convention it was
    # generated for.
    if getattr(p, "project_name", ""):
        lines.append(f"Project: {p.project_name}")
    lines.append(f"Deposition metal: {p.metal_type}   Electrode metal: {getattr(p, 'electrode_metal_type', 'Nb')}")
    lines.append(f"Deposition alpha (fixed for whole batch): {p.alpha_deposition:.2f} deg")
    lines.append(f"Mill alpha (fixed sidewall-mill convention): {p.alpha_mill:.2f} deg")
    if p.pmma_thickness_nm:
        lines.append(f"{p.top_resist_name} (top layer) thickness: {p.pmma_thickness_nm:g} nm   "
                      f"Safety margin: {p.safety_margin_nm:g} nm")
    if p.pmgi_thickness_nm:
        lines.append(f"{p.bottom_resist_name} (bottom layer) thickness: {p.pmgi_thickness_nm:g} nm "
                      f"(documentation only)")
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

# ============================================================================
# New Tab 3 (Recipe) bridge -- converts a junction_import.SimulationResult's
# high-level feature list into the literal custom_steps shape
# generate_recipe_steps already expects, then delegates ENTIRELY to
# generate_recipe_steps for the actual machine-step expansion (Ti gather,
# mill gas/discharge/beam boilerplate, shutter cycles, pump/wait). New Tab 3
# never reimplements any of that -- it only decides WHICH mill/deposition/
# oxidation events exist and in what order, the exact same division of
# responsibility generate_recipe_steps already has with Tab 2's manually-
# committed list. This is what keeps RecipeStep as the single union point
# regardless of whether a recipe was built by hand or from a simulation.
# ============================================================================

def generate_recipe_steps_from_simulation(sim_result, p: "DesignParameters"):
    """sim_result: a junction_import.SimulationResult (duck-typed here --
    only .not_set/.not_set_reason/.features/.has_disconnections/
    .disconnection_notes are read, so this file never needs to import
    junction_import.py and risk a circular dependency).

    Returns (steps, warnings, custom_steps). `custom_steps` is also
    returned (not just steps/warnings) so the caller can assign it to
    self.custom_committed_steps before triggering the existing AI review/
    draft methods -- those methods independently recompute
    generate_recipe_steps(params, self.custom_committed_steps) themselves,
    and returning the exact custom_steps used here lets that recomputation
    match this call exactly, with zero changes needed to the AI methods."""
    if getattr(sim_result, "not_set", False):
        return [], [getattr(sim_result, "not_set_reason", "Deposit order not set.")], []

    custom_steps = []
    for feat in sim_result.features:
        if feat.kind == "mill":
            custom_steps.append({"type": "Mill", "alpha": feat.alpha, "theta": feat.theta})
        elif feat.kind == "deposition":
            custom_steps.append({"type": "Deposition", "alpha": feat.alpha, "theta": feat.theta,
                                  "thickness_nm": feat.thickness_nm})
        elif feat.kind == "oxidation":
            custom_steps.append({"type": "Oxidation"})
        # unrecognized feature kinds are skipped rather than guessed at,
        # matching _flatten_committed_steps' own stated policy.

    steps, warnings = generate_recipe_steps(p, custom_steps, skip_old_electrode_warnings=True)
    if getattr(sim_result, "has_disconnections", False):
        # Post-shadow connectivity failures are more serious than an
        # ordinary recipe warning (Warning #1/#2 crossing risk) -- a
        # design that's confirmed to physically split apart or lose the
        # JJ crossing after shadowing should have that surfaced FIRST,
        # not buried at the end of an otherwise-normal warning list.
        warnings = list(sim_result.disconnection_notes) + warnings
    return steps, warnings, custom_steps
