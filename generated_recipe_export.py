"""
generated_recipe_export.py

Materializes the deterministic, design-driven recipe generator's output
(recipe_generator.py) into the REAL Plassys recipe-tree format
(recipe_builder_model.py) -- so a "Generate Recipe" click produces real,
directly-editable PlassysRecipeNode/PlassysStepRow entries in the SAME
shared recipe library the by-hand Recipe Builder already uses, rather
than a separate, read-only-feeling text rendering (the old "Traced
Recipe Timeline" + "Expanded Recipe" text blob).

The generated recipe format matches the format of a real Plassys/by-hand
recipe: angle parameters are automatically filled into the tilt and
planetary-rotation rows, along with recipe thickness and tilt. Every
placeholder that recipe_generator.py's own literal-text builders left
unfilled ("thick?"/"tilt?" in a recipe's NAME) is filled here with the
real, already-known number -- both in the generated recipe's NAME and in
its actual step rows (a RecipeThickness row's traceability comment, a
Position Planetary row's real theta, etc).

Three design decisions this module implements (see
shadowing_design_rules.md for the full discussion):

  1. SUB-RECIPE CONTENT is a literal snapshot -- real numbers baked
     directly into each generated recipe's own steps, never a
     `call_params` placeholder. Each project always gets its own
     separate, project-named node in the library, even when its content
     is byte-for-byte identical to another project's -- so a user
     working across many similar designs gets a separate, clearly
     project-named file for each one instead of silently sharing a node
     with an unrelated project. See find_identical_recipe's and
     resolve_and_write's own docstrings for the full rationale
     (regenerating a project's OWN unchanged recipe still reuses its own
     file -- only cross-project reuse is avoided).
  2. OXIDATION gets its own reusable leaf recipe (Root\\Static
     oxidation\\...), called via a "Recipe" step from the top-level
     "complete process" recipe -- not inlined -- matching how mill/
     Ti-gather/deposition are all real, separately-named, reusable
     sub-recipes too.
  3. REGENERATION is "duplicate-and-patch", not silent overwrite and
     not blind always-fresh dedup. This mirrors a common real lab
     workflow: a project often accumulates numerous evaporation recipes
     where only the angle differs (e.g. "Manhattan +45" vs.
     "Manhattan +90"), created by copying an old evap recipe and
     adjusting only the parameters that changed, while keeping whatever
     by-hand changes were made to it previously. So regenerating a
     design that was already generated once before:
       - if nothing changed since last time: reuse the same linked node,
         whatever hand-edits it carries, untouched.
       - if only specific parameter VALUES changed (same step shape):
         duplicate the CURRENT live node (hand-edits and all) under a
         new name reflecting what changed, patch just those cells, leave
         the OLD node sitting in the library untouched and still usable.
       - if the shape itself changed (rows added/removed/retyped, not
         just a value): can't cleanly patch -- fall back to fresh
         dedup-or-create, not derived from the old node.
     See resolve_and_write() for the implementation.

NAMING is fully verbose/labeled (real numbers, every field spelled out)
by design: the more detail a recipe's name carries, the easier it is for
a user to find their specific recipe later, especially when two recipes
differ by only one small change and that difference needs to be visible
at a glance.

This module has zero Qt dependency (plain dataclasses/strings only, like
recipe_builder_model.py and recipe_generator.py themselves) so it's independently
testable headless with no QApplication needed.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import recipe_builder_model as rtm
from recipe_builder_model import PlassysRecipeNode, PlassysStepRow
from recipe_generator import DesignParameters, get_ramp_profile, _fmt_mmss


# ----------------------------------------------------------------------
# Output data model
# ----------------------------------------------------------------------

@dataclass
class GeneratedSubRecipe:
    """One real recipe-tree leaf this generation wants to exist --
    either a reusable building block (a mill pass, a deposition pass,
    the shared Ti-gather, the shared oxidation) or the final
    "complete process" recipe that chains all of them together via
    "Recipe" steps. `tree_path` is the PROPOSED path -- resolve_and_write
    may redirect it (dedup onto an existing identical node, or a
    non-colliding rename) before anything is actually written."""
    kind_key: str
    tree_path: str
    steps: List[PlassysStepRow]
    label: str = ""


# ----------------------------------------------------------------------
# Small row-building helper
# ----------------------------------------------------------------------

def _row(step_type: str, p1: str = "", p2: str = "", p3: str = "", p4: str = "",
         comment: Optional[str] = None) -> PlassysStepRow:
    if comment is None:
        comment = rtm.STEP_TYPE_DEFAULT_COMMENT.get(step_type, "")
    return PlassysStepRow(step_type=step_type, param1=str(p1), param2=str(p2),
                           param3=str(p3), param4=str(p4), comment=comment)


# ----------------------------------------------------------------------
# Literal machine-step row builders -- near-mechanical ports of
# recipe_generator.py's own _ti_gather_detail / _deposition_ramp_detail /
# _deposition_rampdown_detail / _mill_detail / _oxidation_detail, one
# real PlassysStepRow per literal line those functions build as text.
# Every value (theta, thickness, alpha) is the real, already-known
# number -- nothing is left as a "?" placeholder in the step rows
# themselves (see module docstring, decision 1).
# ----------------------------------------------------------------------

def build_ti_gather_rows(p: DesignParameters, warnings: Optional[List[str]] = None) -> List[PlassysStepRow]:
    """Port of recipe_generator._ti_gather_detail. Content is fixed by
    convention (rate/duration are process-wide, not per-use), so this
    is always the SAME steps regardless of which design calls it --
    naturally dedups onto one shared node across every design/generation."""
    prof = get_ramp_profile("Ti", p.ti_gather_rate_nmps, warnings)
    rows = [
        _row("Recipe Tilt Substrate", "Deposit"),
        _row("Process Chamber"),
        _row("LL Process"),
        # Bug fix: this Close is genuinely never re-opened anywhere in
        # THIS sub-recipe -- by design, not by mistake: the substrate
        # shutter stays closed for the ENTIRE Ti-gather step (Ti is only
        # being conditioned/rate-checked on the source, not deposited
        # onto the substrate yet), and is only opened later by the
        # deposition sub-recipe this Ti-gather always precedes (see
        # build_generated_subrecipes' "Ti gather before this deposition"
        # comment). recipe_builder_rules.py's shutter_ebgun_pairing check only
        # ever sees ONE sub-recipe's own steps when a user opens Ti-
        # gather directly (it can't see the separately-called deposition
        # recipe that reopens it), so from that narrow view it would
        # otherwise look like a close with no matching open anywhere.
        # The "stays closed" phrase in this comment is the same
        # documented intent this module already states in its own
        # docstring ("Ti gather ... substrate shutter stays CLOSED") --
        # now also the literal marker _check_toggle_pairs looks for to
        # recognize a deliberate, intentionally-unpaired close and not
        # flag it as a mistake.
        _row("Substrate Shutter", "EGun", "Close",
             comment="Stays closed for this entire Ti-gather step -- opened again by the "
                     "deposition step this Ti-gather precedes."),
        _row("Material select", "Ti"),
        _row("Operator requested", "Remain present while the beam is on"),
        _row("E Gun Emission", "1", "0:0"),
        _row("Wait", _fmt_mmss(prof.pre_wait_s)),
        _row("E Gun Emission", f"{prof.soak1_ma:g}", _fmt_mmss(prof.soak1_time_s)),
        _row("Ramp wait", "All"),
        _row("Operator requested", "Check beam is centered on crucible"),
        _row("Source Shutter", "Open"),
        _row("E Gun Emission", f"{prof.soak2_ma:g}", _fmt_mmss(prof.soak2_time_s)),
        _row("Ramp wait", "All"),
        _row("Rate control", f"{p.ti_gather_rate_nmps:g}"),
        _row("Wait", _fmt_mmss(prof.settle_wait_s)),
        # Same class of bug fixed for mill duration above (build_mill_
        # group_rows) -- a hardcoded ":00" here would mis-render any
        # non-whole-minute ti_gather_duration_min (e.g. Settings' own
        # duration field set to 4.5 -> "4.5:00" instead of "4:30").
        # _fmt_mmss does the real minutes-to-M:SS conversion.
        _row("Wait", _fmt_mmss(p.ti_gather_duration_min * 60.0),
             comment="Gather -- substrate shutter stays CLOSED"),
        _row("E Gun Emission", f"{prof.rampdown_ma:g}", _fmt_mmss(prof.rampdown_time_s)),
        _row("Ramp wait", "All"),
        _row("E Gun Emission", "0", "0:0"),
        _row("Wait", _fmt_mmss(prof.cooldown_wait_s)),
    ]
    # By design, no "Comment only" narrative rows are added to describe
    # the table. An unconfirmed/estimated ramp profile is already
    # surfaced through the real warnings channel instead (see
    # get_ramp_profile's own `warnings.append(...)` above, which reaches
    # the Recipe Warnings panel) -- a narrative row would be redundant
    # with that, not the only place this information lives.
    return rows


def build_oxidation_rows(p: DesignParameters) -> List[PlassysStepRow]:
    """Port of recipe_generator._oxidation_detail. Content is fixed by
    convention (pressure/time are process-wide, not per-feature), so
    this is always the SAME steps -- one shared, reusable node
    (decision 2), regardless of how many times oxidation occurs across
    a design's feature sequence."""
    return [
        _row("Static Oxidation", f"{p.oxidation_pressure_mbar:g}", "Static O2"),
        _row("Param Wait", f"{p.oxidation_time_min:g}",
             comment="USER-ALTERABLE oxidation time"),
        _row("Static Oxidation", "0.000", "Static O2", comment="Oxidation off"),
        _row("Pump LL"),
        _row("Wait", "0:5"),
    ]


def build_final_oxidation_rows(p: DesignParameters) -> List[PlassysStepRow]:
    """Builds a separate, short static oxidation after the ENTIRE junction has
    finished depositing, to passivate/protect it before it leaves the
    chamber -- distinct from build_oxidation_rows above, which grows the
    real AlOx tunnel barrier between the first and second finger
    depositions. Same shape, but uses p.final_oxidation_time_min (default
    5 min) instead of p.oxidation_time_min -- its own, separately
    reusable leaf recipe (decision 2's "own reusable leaf" pattern
    applied here too), never confused with the tunnel-barrier oxidation
    even if a design happens to use both."""
    return [
        _row("Static Oxidation", f"{p.oxidation_pressure_mbar:g}", "Static O2"),
        _row("Param Wait", f"{p.final_oxidation_time_min:g}",
             comment="USER-ALTERABLE final protective oxidation time"),
        _row("Static Oxidation", "0.000", "Static O2", comment="Oxidation off"),
        _row("Pump LL"),
        _row("Wait", "0:5"),
    ]


def build_mill_group_rows(p: DesignParameters, angles: List[Tuple[Optional[float], Optional[float]]],
                           warnings: Optional[List[str]] = None,
                           duration_min: Optional[float] = None) -> List[PlassysStepRow]:
    """Port of recipe_generator._mill_detail across every sub-angle in
    `angles` (a mill group -- currently always length 1 from the
    Simulate Design feature pipeline, since each SimulationFeature
    already produces its own complete, self-contained mill pass; this
    stays general in case grouping is ever reintroduced). Only the
    FIRST angle strikes gas/discharge/beam; only the LAST shuts them
    off again -- recipe_generator.py's own "beam owner / beam closer"
    pattern, preserved exactly.

    `duration_min`, if given, overrides p.mill_duration_min for THIS
    mill pass specifically: different mill passes in a real recipe
    sometimes need different durations (e.g. a longer initial pass
    through native oxide vs. a shorter cleanup pass), so mill duration
    can no longer be assumed uniform across every mill step in the
    recipe the way the single Settings-wide default implied. Falls back
    to the Settings-wide default when not given (None), so nothing
    changes for a design that never sets a per-pass duration."""
    effective_duration = duration_min if duration_min is not None else p.mill_duration_min
    rows: List[PlassysStepRow] = []
    n = len(angles)
    for j, (alpha, theta) in enumerate(angles):
        is_owner = (j == 0)
        is_closer = (j == n - 1)
        theta_str = f"{theta:g}" if theta is not None else None
        alpha_str = f"{alpha:g}" if alpha is not None else "(not set)"
        if is_owner:
            rows.append(_row("Recipe Tilt Substrate", "Etch",
                              comment=f"Tilt to {alpha_str} deg for milling"))
            if theta_str is not None:
                rows.append(_row("Position Planetary", "Zero", theta_str))
            rows.append(_row("Substrate Shutter", "Ion Gun", "Close"))
            rows.append(_row("Pump Chamber"))
            rows.append(_row("LL Process"))
            rows.append(_row("Ar Ion Gun Gas", f"{p.mill_ar_gas_sccm:g}"))
            rows.append(_row("Wait", "0:10"))
            rows.append(_row("IBG Discharge", "On"))
            rows.append(_row("Wait", "0:30"))
            rows.append(_row("Ion Beam", f"V={p.mill_beam_voltage_v:g}V",
                              f"I={p.mill_beam_current_ma:.1f}mA",
                              f"Vacc={p.mill_accel_voltage_v:g}V"))
            rows.append(_row("Wait", "0:30"))
        else:
            # By design, no narrative "Comment Only" filler rows describe
            # the mechanics here -- the real machine's own table shows
            # this exact combined-recipe pattern with nothing more than
            # the Position Planetary/Substrate Shutter rows below; the
            # beam simply isn't re-struck because there's no IBG
            # Discharge/Ion Beam row here to strike it with.
            if theta_str is not None:
                rows.append(_row("Position Planetary", "Zero", theta_str))
        rows.append(_row("Substrate Shutter", "Ion Gun", "Open"))
        # Bug fix: a fractional-minute mill duration (e.g. an override of
        # 1.5 minutes, typed as "1:30") used to render wrong in two ways:
        # (1) the Recipe Parameters panel's mill-duration override field
        # only ever parsed a plain decimal number of minutes (see
        # main_gui.py's _minutes_opt), so a typed "1:30" failed to
        # parse and was silently discarded, leaving effective_duration at
        # the Settings-wide 1.0 min default instead; (2) even with a
        # correctly-parsed fractional-minute value, this line was
        # hardcoding a literal ":00" seconds suffix onto the raw minutes
        # number -- f"{1.5:g}:00" produces the nonsensical "1.5:00"
        # instead of the real M:SS the tool's own Wait step expects here
        # ("1:30"). _fmt_mmss (already used everywhere else a duration is
        # rendered in this file -- e.g. the e-gun ramp profile times)
        # does the real minutes-to-M:SS conversion properly, matching
        # exactly how this same 1.5-minute duration would read in the
        # Complete Process table's own Time column.
        rows.append(_row("Wait", _fmt_mmss(effective_duration * 60.0),
                          comment=f"Mill for {effective_duration:g} min"))
        rows.append(_row("Substrate Shutter", "Ion Gun", "Close"))
        if is_closer:
            rows.append(_row("Ion Beam", "Off"))
            rows.append(_row("IBG Discharge", "Off"))
            rows.append(_row("Ar Ion Gun Gas", "0.0"))
            rows.append(_row("Substrate Shutter", "Ion Gun", "Close"))
    return rows


def build_deposition_group_rows(p: DesignParameters,
                                  angles: List[Tuple[Optional[float], Optional[float], Optional[float]]],
                                  warnings: Optional[List[str]] = None) -> List[PlassysStepRow]:
    """Port of recipe_generator._deposition_ramp_detail +
    _deposition_rampdown_detail across every sub-angle in `angles`
    (currently always length 1 from the Simulate Design pipeline, for
    the same reason as build_mill_group_rows -- kept general). Only the
    FIRST angle ramps the e-gun up from cold; only the LAST ramps it
    back down -- recipe_generator.py's own "ramp owner" pattern, preserved
    exactly. thickness_nm is always the real target -- if a pass's
    thickness was never set, that's surfaced as a traceability comment
    on its RecipeThickness row rather than silently guessed."""
    prof = get_ramp_profile(p.metal_type, p.al_dep_rate_nmps, warnings)
    rows: List[PlassysStepRow] = []
    n = len(angles)
    for j, (alpha, theta, thickness_nm) in enumerate(angles):
        is_owner = (j == 0)
        is_last = (j == n - 1)
        theta_str = f"{theta:g}" if theta is not None else None
        alpha_str = f"{alpha:g}" if alpha is not None else "(not set)"
        thickness_comment = (f"Deposits until the crystal monitor reads {thickness_nm:g} nm."
                              if thickness_nm is not None else
                              "NO TARGET THICKNESS SET -- set it before running.")
        if is_owner:
            rows.append(_row("Recipe Tilt Substrate", "Deposit",
                              comment=f"Tilt to {alpha_str} deg for deposition"))
            if theta_str is not None:
                rows.append(_row("Position Planetary", "Zero", theta_str))
            rows.append(_row("Process Chamber"))
            rows.append(_row("LL Process"))
            rows.append(_row("Substrate Shutter", "EGun", "Close"))
            rows.append(_row("Material select", p.metal_type))
            rows.append(_row("Operator requested", "Remain present while the beam is on"))
            rows.append(_row("E Gun Emission", "1", "0:0"))
            rows.append(_row("Wait", _fmt_mmss(prof.pre_wait_s)))
            rows.append(_row("E Gun Emission", f"{prof.soak1_ma:g}", _fmt_mmss(prof.soak1_time_s)))
            rows.append(_row("Ramp wait", "All"))
            rows.append(_row("Operator requested", "Check beam is centered on crucible"))
            rows.append(_row("Source Shutter", "Open"))
            rows.append(_row("E Gun Emission", f"{prof.soak2_ma:g}", _fmt_mmss(prof.soak2_time_s)))
            rows.append(_row("Ramp wait", "All"))
            rows.append(_row("Rate control", f"{p.al_dep_rate_nmps:g}"))
            rows.append(_row("Wait", _fmt_mmss(prof.settle_wait_s)))
        else:
            # By design, no narrative "Comment Only" filler rows here --
            # same reasoning as build_mill_group_rows above.
            if theta_str is not None:
                rows.append(_row("Position Planetary", "Zero", theta_str))
        rows.append(_row("Zero Thickness"))
        rows.append(_row("Substrate Shutter", "EGun", "Open"))
        rows.append(_row("RecipeThickness", comment=thickness_comment))
        rows.append(_row("Substrate Shutter", "EGun", "Close"))
        if is_last:
            rows.append(_row("E Gun Emission", f"{prof.rampdown_ma:g}", _fmt_mmss(prof.rampdown_time_s)))
            rows.append(_row("Ramp wait", "All"))
            rows.append(_row("E Gun Emission", "0", "0:0"))
            rows.append(_row("Wait", _fmt_mmss(prof.cooldown_wait_s)))
    # By design, no narrative "Comment Only" filler rows are added here --
    # an unconfirmed/estimated ramp profile is already surfaced through
    # the real warnings channel instead (see get_ramp_profile's own
    # `warnings.append(...)`, threaded through to the Recipe Warnings
    # panel), so a duplicate descriptive row would be redundant with that.
    return rows


# ----------------------------------------------------------------------
# Naming + tree-path conventions -- fully verbose/labeled, real numbers
# baked in (decision 4/naming), so two recipes differing by only one
# small value are still easy to tell apart at a glance in the library
# tree.
# ----------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', ... -- used in a
    deposition recipe's name to mark which deposition step it is (1st,
    2nd, or 3rd). Handles the general case (11th/12th/13th don't take
    'st'/'nd'/'rd') even though a real design only ever reaches 2 or 3
    deposition steps."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def generated_name(kind: str, p: DesignParameters, *,
                    alpha: Optional[float] = None, theta: Optional[float] = None,
                    thetas: Optional[List[Optional[float]]] = None,
                    thickness_nm: Optional[float] = None, label: str = "",
                    project_name: str = "", design_name: str = "",
                    order_label: str = "", group_index: int = 1,
                    has_patches: bool = False) -> str:
    proj = project_name or "Project"
    design = design_name or "design"

    def _theta_str() -> str:
        # A double-mill or double-deposition at two planetary angles for
        # the SAME physical step is one real recipe, named with BOTH
        # angles spelled out in the real machine's own compact,
        # sign-prefixed form (e.g. "+270and+90") -- not two
        # separately-named recipes, and not a more verbose "0deg+180deg-
        # planetary" form. `thetas` (plural) carries every sub-angle in
        # the group; a single-angle call still just passes `theta` and
        # gets the same compact singular form (e.g. "+180").
        vals = thetas if thetas is not None else ([theta] if theta is not None else [])
        vals = [t for t in vals if t is not None]
        if not vals:
            return "planetary-not-set"
        if len(vals) == 1:
            return f"{vals[0]:+g}"
        return "and".join(f"{t:+g}" for t in vals)

    if kind == "deposition":
        # Final naming convention for a deposition recipe: project name,
        # metal type, deposition rate, junction type ("Manhattan" or
        # "Manhattan PatchInt"), which deposition step it is ((1st),
        # (2nd), or (3rd)), and the planetary angle(s). Target thickness
        # is deliberately NOT part of the name -- it's still a real
        # parameter, just not shown here, since the Complete Process
        # table's own "Thickness (nm)" column already shows it;
        # design_name is dropped entirely; label/order_label are no
        # longer used for the visible ordinal -- group_index (this
        # design's 1st/2nd/3rd deposition-kind feature overall, already
        # tracked by the caller) IS "which deposition step it is". Tilt
        # (alpha) is dropped from the name too, for the same reason as
        # thickness -- the Complete Process table's own Evap Tilt column
        # already shows it, so repeating it in the name would just add
        # clutter without adding information.
        jtype = "Manhattan PatchInt" if has_patches else "Manhattan"
        theta_str = _theta_str()
        return f"{proj} {p.metal_type} {p.al_dep_rate_nmps:g}nm-s {jtype} ({_ordinal(group_index)}) {theta_str}"
    if kind == "mill":
        # Same tilt-drop as deposition above, plus the redundant
        # standalone "Mill" word is dropped since mill_label already
        # contains "Mill" in virtually every real case (see
        # junction_import.py's labeling).
        theta_str = _theta_str()
        mill_label = label or f"pass {group_index}"
        return f"{proj} {design} {mill_label} {theta_str}"
    if kind == "ti_gather":
        return f"{proj} Ti-gather {p.ti_gather_rate_nmps:g}nm-s {p.ti_gather_duration_min:g}min"
    if kind == "oxidation":
        return f"{proj} Static-oxid {p.oxidation_pressure_mbar:g}mBar {p.oxidation_time_min:g}min"
    if kind == "final_oxidation":
        return f"{proj} Final protective static-oxid {p.oxidation_pressure_mbar:g}mBar {p.final_oxidation_time_min:g}min"
    if kind == "complete_process":
        return f"{proj} {design} Complete Process"
    return f"{proj} {kind} {group_index}"


def generated_tree_path(kind: str, name: str, project_name: str = "") -> str:
    proj = project_name or "Project"
    if kind == "deposition":
        # metal folder is threaded in by the caller (build_generated_subrecipes),
        # which already knows p.metal_type -- see there.
        raise ValueError("use generated_deposition_path (needs metal) instead")
    if kind == "mill":
        return rtm.join_path(["Root", "Etching", "New Ion Gun Recipes", proj, name])
    if kind == "ti_gather":
        return rtm.join_path(["Root", "Evap", "Titanium", name])
    if kind == "oxidation":
        return rtm.join_path(["Root", "Static oxidation", proj, name])
    if kind == "final_oxidation":
        return rtm.join_path(["Root", "Static oxidation", proj, name])
    if kind == "complete_process":
        return rtm.join_path(["Root", "Complete Process Recipes", proj, name])
    return rtm.join_path(["Root", proj, name])


def generated_deposition_path(metal: str, name: str, project_name: str = "") -> str:
    proj = project_name or "Project"
    return rtm.join_path(["Root", "Evap", metal, proj, name])


def _parent_path(path: str) -> str:
    parts = rtm.split_path(path)
    return rtm.join_path(parts[:-1])


def _leaf_name(path: str) -> str:
    parts = rtm.split_path(path)
    return parts[-1] if parts else path


# ----------------------------------------------------------------------
# "Complete Process" derived summary columns -- matching the real
# machine's own top-level overall recipe table, which shows, per row,
# which named sub-recipe it calls, how long that takes, and (for an
# evaporation/mill substep) the target thickness and substrate tilt
# actually configured for it. For every evaporation substep in the final
# recipe, these columns are filled out with that step's alpha tilt and
# target thickness; the time for oxidation and the final wait time are
# also shown under "Time". These values are
# never typed directly on the outer calling row itself (a "Recipe" step
# only ever carries the target path in param1) -- they're derived by
# resolving the call and reading the SAME comment-encoded values
# _steps_equal/_diff_cells above already treat as the authoritative
# source for "Recipe Tilt Substrate"/"RecipeThickness" (see
# _COMMENT_CARRIES_VALUE_STEP_TYPES), plus summing every real "Wait"/
# "Param Wait" duration inside the called recipe for its total time.
# ----------------------------------------------------------------------

_TILT_COMMENT_RE = re.compile(r"Tilt to ([+-]?\d+(?:\.\d+)?) deg")
_THICKNESS_COMMENT_RE = re.compile(r"reads ([+-]?\d+(?:\.\d+)?) nm")


def is_complete_process_path(path: str) -> bool:
    """True for the top-level 'Complete Process' recipe itself (or
    anything filed under that same library folder) -- the ONE level
    that gets the derived Subproc Name/Time/Thickness/Evap Tilt display
    instead of the generic Step Type/Parameter 1-4 grid. A nested
    sub-recipe (mill/deposition/oxidation/Ti-gather) a user drills into
    from there lives elsewhere in the tree and keeps the generic grid,
    since ITS OWN rows are the real, directly-editable Step Type/
    Parameter values."""
    parts = rtm.split_path(path or "")
    return len(parts) >= 2 and parts[0] == "Root" and parts[1] == "Complete Process Recipes"


def _mmss_to_seconds(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        pieces = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    total = 0.0
    for piece in pieces:
        total = total * 60.0 + piece
    return total


def _sum_duration_seconds(steps: List[PlassysStepRow]) -> float:
    """Total real elapsed time inside one sub-recipe's own steps -- the
    only two step types that actually consume clock time on the real
    tool: "Wait" (param1 is an "M:SS" duration) and "Param Wait" (used
    for oxidation; param1 is a plain number of MINUTES, no ':'), see
    build_oxidation_rows/build_final_oxidation_rows above."""
    total = 0.0
    for row in steps:
        if row.step_type == "Wait":
            secs = _mmss_to_seconds(row.param1)
            if secs is not None:
                total += secs
        elif row.step_type == "Param Wait":
            try:
                total += float(row.param1) * 60.0
            except (TypeError, ValueError):
                pass
    return total


def _oxidation_time_seconds(steps: List[PlassysStepRow]) -> float:
    """The Time column's goal is not to sum up every individual time
    step inside a sub-recipe -- for oxidation specifically, it should
    show the exact oxidation time a user configured, not a total elapsed
    time. So this is deliberately NOT _sum_duration_seconds (which would
    also pull in the trailing "Pump LL" / "Wait 0:5" housekeeping row
    build_oxidation_rows always appends) -- only the "Param Wait" row's
    own value, the literal user-typed oxidation duration, ever counts."""
    for row in steps:
        if row.step_type == "Param Wait":
            try:
                return float(row.param1) * 60.0
            except (TypeError, ValueError):
                pass
    return 0.0


def _extract_tilt_deg(steps: List[PlassysStepRow]) -> Optional[float]:
    for row in steps:
        if row.step_type == "Recipe Tilt Substrate":
            m = _TILT_COMMENT_RE.search(row.comment or "")
            if m:
                return float(m.group(1))
    return None


def _extract_thickness_str(steps: List[PlassysStepRow]) -> str:
    """Joined with '+' (matching the same convention generated_name
    already uses for a combined multi-angle recipe) when a single
    combined recipe carries more than one RecipeThickness row -- e.g. a
    two-angle patch deposition group."""
    vals = []
    for row in steps:
        if row.step_type == "RecipeThickness":
            m = _THICKNESS_COMMENT_RE.search(row.comment or "")
            if m:
                vals.append(m.group(1))
    return "+".join(vals)


def summarize_complete_process_step(root: PlassysRecipeNode, step: PlassysStepRow) -> dict:
    """Derives one row of the top-level Complete Process display: {name,
    time_str, thickness_nm, tilt_deg, comment}. For a "Recipe" call, all
    four derived fields come from resolving step.param1 and reading the
    values baked into ITS steps (see module docstring above); for a
    direct boilerplate row (Pump Chamber, LL Process, the final Wait),
    only whatever applies directly to that one row is filled in."""
    name = step.step_type
    tilt_deg: Optional[float] = None
    thickness_nm = ""
    duration_s = 0.0
    if step.step_type == rtm.RECIPE_STEP_TYPE and (step.param1 or "").strip():
        target = rtm.resolve_path(root, step.param1)
        if target is not None and not target.is_folder:
            name = _leaf_name(step.param1)
            tilt_deg = _extract_tilt_deg(target.steps)
            thickness_nm = _extract_thickness_str(target.steps)
            # Bug fix (extends an earlier deposition-only version of this
            # fix to every sub-recipe kind): the Time column must NEVER
            # show an accumulated/summed estimate for ANY sub-recipe --
            # Mill (was summing its own
            # fixed 0:10/0:30/0:30 strike sequence plus the mill-duration
            # Wait into a fabricated total like "4:10") and Ti-gather
            # (was summing its whole ramp/soak/gather/rampdown sequence
            # into "9:20") are just as wrong as the deposition case this
            # column already suppressed -- ONLY a real, user-set,
            # single-value duration belongs here, which today means
            # oxidation alone (its "Param Wait" row IS the literal
            # user-typed oxidation time, not a sum of anything). A
            # sub-recipe is oxidation-kind if and only if it contains a
            # "Param Wait" row -- mill/deposition/Ti-gather never emit
            # that step type (see build_oxidation_rows/
            # build_final_oxidation_rows, the only two builders that do).
            is_oxidation = any(r.step_type == "Param Wait" for r in target.steps)
            duration_s = _oxidation_time_seconds(target.steps) if is_oxidation else 0.0
        else:
            name = f"{_leaf_name(step.param1)} (missing!)"
    elif step.step_type == "Wait":
        duration_s = _mmss_to_seconds(step.param1) or 0.0
    elif step.step_type == "Param Wait":
        try:
            duration_s = float(step.param1) * 60.0
        except (TypeError, ValueError):
            duration_s = 0.0
    return {
        "name": name,
        "time_str": _fmt_mmss(duration_s) if duration_s else "",
        "thickness_nm": thickness_nm,
        "tilt_deg": f"{tilt_deg:g}" if tilt_deg is not None else "",
        "comment": step.comment,
    }


# ----------------------------------------------------------------------
# Full-recipe flattening for reports -- a PDF report should show the
# entire Plassys recipe, whether it was generated or created by hand, in
# the same list fashion the on-screen view uses, but with every level
# expanded. The on-screen Complete Process table only ever shows ONE
# level of derived summary per "Recipe"
# call row -- a user has to double-click a row to see the real Step Type/
# Parameter/Comment grid underneath. A report needs the WHOLE thing in
# one place, so this recursively walks every "Recipe"-type row, at ANY
# depth, and inlines the target's own steps right after it -- this works
# identically for an Auto-Generated recipe (always exactly one level of
# "Recipe" calls under Complete Process today) and for an arbitrarily
# hand-built recipe (which could nest "Recipe" calls as deep as a user
# likes).
# ----------------------------------------------------------------------

def flatten_recipe_for_report(root: PlassysRecipeNode, start_path: str,
                               *, max_depth: int = 8) -> Tuple[Optional[PlassysRecipeNode], List[dict]]:
    """Returns (start_node, rows). start_node is None if start_path
    doesn't resolve to a real (non-folder) recipe, in which case rows is
    always [].

    Each row is a dict: {"depth": int, "kind": str, "step": PlassysStepRow,
    "summary": Optional[dict]}.

      - "step": an ordinary literal row (Step Type/Param1-4/Comment as-is).
      - "call": a "Recipe"-type row that resolves to a real target --
        `summary` is the SAME derived {name, time_str, thickness_nm,
        tilt_deg, comment} dict summarize_complete_process_step already
        computes for the on-screen table ("the added details provided by
        this view"). The target's own rows immediately follow, at
        depth + 1, recursively -- so a fully-expanded story is just this
        list rendered top to bottom with each row indented by its depth.
      - "missing": a "Recipe" row whose target no longer resolves (e.g.
        the target node was since deleted) -- not expanded further.
      - "cycle": a "Recipe" row that would re-enter a recipe already open
        earlier in the SAME call chain -- guards a hand-built recipe that
        (accidentally or deliberately) calls itself, directly or through
        an intermediate, instead of recursing forever. Not expanded.

    max_depth is a second, blunter guard against absurd/runaway nesting
    (well beyond anything a real recipe would ever need) -- expansion
    simply stops there with one explanatory marker row.
    """
    start_node = rtm.resolve_path(root, start_path)
    if start_node is None or start_node.is_folder:
        return None, []

    rows: List[dict] = []

    def walk(node: PlassysRecipeNode, depth: int, chain: frozenset) -> None:
        for step in node.steps:
            if step.step_type == rtm.RECIPE_STEP_TYPE and (step.param1 or "").strip():
                target_path = (step.param1 or "").strip()
                if target_path in chain:
                    rows.append({"depth": depth, "kind": "cycle", "step": step, "summary": None})
                    continue
                target = rtm.resolve_path(root, target_path)
                if target is None or target.is_folder:
                    rows.append({"depth": depth, "kind": "missing", "step": step, "summary": None})
                    continue
                summary = summarize_complete_process_step(root, step)
                rows.append({"depth": depth, "kind": "call", "step": step, "summary": summary})
                if depth + 1 >= max_depth:
                    stub = _row("Comment Only", comment="(nesting too deep -- stopped expanding here)")
                    rows.append({"depth": depth + 1, "kind": "step", "step": stub, "summary": None})
                    continue
                walk(target, depth + 1, chain | {target_path})
            else:
                rows.append({"depth": depth, "kind": "step", "step": step, "summary": None})

    walk(start_node, 0, frozenset({start_path}))
    return start_node, rows


# ----------------------------------------------------------------------
# Multi-angle grouping -- matches how this is done on the real machine:
# a double-deposition (at 2 different angles) or milling twice in a row
# (at 2 different angles) can go in the same recipe table/step. Once the
# substrate shutter for the previous angle closes, the planetary angle
# is adjusted again, the shutter reopens, and the same next steps
# repeat. junction_import.build_simulation_result already labels exactly
# these multi-pass groups "<base label> (pass N of M)", consecutively,
# for a patch group or a single finger that needs two angles -- and
# build_mill_group_rows/build_deposition_group_rows already implement
# the real "beam owner/beam closer" pattern these combined recipes need
# (only the FIRST sub-angle re-strikes/re-ramps, only the LAST shuts
# down, same continuous beam/ramp in between). What was missing was
# actually grouping those consecutive features into ONE GeneratedSubRecipe
# instead of one leaf per feature -- fixed here.
# ----------------------------------------------------------------------

_PASS_LABEL_RE = re.compile(r"^(.*) \(pass (\d+) of (\d+)\)$")


def strip_pass_suffix(label: str) -> str:
    """Public wrapper around _PASS_LABEL_RE's stripping logic --
    main_gui's dynamic Recipe Parameters panel needs to group a
    mill pass's sub-angle features (e.g. "Mill before patches (pass 1
    of 2)" / "... (pass 2 of 2)") under ONE shared duration field, the
    same base-label grouping build_generated_subrecipes already uses
    below -- exposed here so the GUI doesn't need its own, possibly
    drifting, copy of this regex."""
    m = _PASS_LABEL_RE.match(label or "")
    return m.group(1) if m else (label or "")


def _group_multi_angle_passes(features: list) -> List[list]:
    """Returns `features` regrouped into runs -- most entries are
    singleton runs [feat], but a maximal consecutive run of "<base>
    (pass 1 of M)", "<base> (pass 2 of M)", ..., "<base> (pass M of M)"
    (same kind, same base label, strictly in order) collapses into one
    run [feat1, ..., featM]. A run that doesn't match cleanly (wrong
    kind, gap, out-of-order, or cut short by the list ending) is left
    ungrouped -- singleton runs -- rather than guessed at."""
    groups: List[list] = []
    i = 0
    n = len(features)
    while i < n:
        feat = features[i]
        kind = getattr(feat, "kind", None)
        m = _PASS_LABEL_RE.match(getattr(feat, "label", "") or "")
        if kind in ("mill", "deposition") and m and int(m.group(2)) == 1 and int(m.group(3)) > 1:
            base, total = m.group(1), int(m.group(3))
            run = [feat]
            j = i + 1
            while len(run) < total and j < n:
                nf = features[j]
                nm = _PASS_LABEL_RE.match(getattr(nf, "label", "") or "")
                if (getattr(nf, "kind", None) == kind and nm and nm.group(1) == base
                        and int(nm.group(3)) == total and int(nm.group(2)) == len(run) + 1):
                    run.append(nf)
                    j += 1
                else:
                    break
            if len(run) == total:
                groups.append(run)
                i = j
                continue
        groups.append([feat])
        i += 1
    return groups


# ----------------------------------------------------------------------
# Top-level entry point -- walks the SAME feature-list shape
# junction_import.SimulationResult.features / main_gui.py's
# self._recipe_features already use (each item: .kind in
# {"mill","deposition","oxidation"}, .alpha, .theta, .thickness_nm,
# .label -- duck-typed, no import of junction_import needed here).
# ----------------------------------------------------------------------

def build_generated_subrecipes(p: DesignParameters, features: list,
                                project_name: str = "", design_name: str = "",
                                ) -> Tuple[List[GeneratedSubRecipe], List[str]]:
    """Returns (subs, warnings). `subs` is in dependency order -- every
    leaf building block (mill passes, the shared Ti-gather, the shared
    oxidation, deposition passes) BEFORE the final "complete_process"
    entry that calls them, so a caller resolving/writing each in order
    (resolve_and_write, below) always knows the real path of anything
    the complete-process recipe needs to reference by the time it's
    built. One GeneratedSubRecipe per mill/deposition FEATURE (not
    grouped across features) -- matches today's actual behavior of the
    Simulate-Design-driven pipeline, where every feature already arrives
    as its own complete, self-contained pass (see build_mill_group_rows/
    build_deposition_group_rows docstrings)."""
    subs: List[GeneratedSubRecipe] = []
    warnings: List[str] = []
    # The "Manhattan" vs. "Manhattan PatchInt" tag used in a deposition
    # recipe's name is a whole-design property (every deposition
    # sub-recipe generated for THIS design uses the same tag), computed
    # once here from whether any feature in the sequence is a patches
    # deposition (warning_source == "patches" -- the same signal
    # junction_import.py already attaches to every patch-group
    # SimulationFeature), not re-derived per feature.
    has_patches = any(getattr(f, "warning_source", None) == "patches" for f in features)
    mill_index = 0
    dep_index = 0
    ti_gather_sub: Optional[GeneratedSubRecipe] = None
    oxidation_sub: Optional[GeneratedSubRecipe] = None
    final_oxidation_sub: Optional[GeneratedSubRecipe] = None
    complete_steps: List[PlassysStepRow] = [_row("Pump Chamber"), _row("LL Process")]

    for group in _group_multi_angle_passes(features):
        feat = group[0]
        kind = getattr(feat, "kind", None)
        alpha = getattr(feat, "alpha", None)
        theta = getattr(feat, "theta", None)
        thickness_nm = getattr(feat, "thickness_nm", None)
        label = getattr(feat, "label", "") or ""
        is_group = len(group) > 1
        # For a combined multi-angle recipe, strip the "(pass N of M)"
        # suffix back off for the base label/name -- the recipe's own
        # NAME shows every angle together (generated_name's `thetas`),
        # not a per-pass count.
        if is_group:
            m = _PASS_LABEL_RE.match(label)
            base_label = m.group(1) if m else label
        else:
            base_label = label
        thetas = [getattr(f, "theta", None) for f in group] if is_group else None

        if kind == "mill":
            mill_index += 1
            angles = [(getattr(f, "alpha", None), getattr(f, "theta", None)) for f in group]
            # A per-pass mill duration override (set via the
            # Recipe Parameters panel's dynamic duration fields, one
            # per mill pass) lives on the feature itself -- every
            # sub-angle in a multi-angle group is the SAME physical mill
            # pass, so group[0]'s value (any of them, since the panel
            # writes the same value onto all of a group's members) is
            # the one that applies to the whole group.
            group_duration = getattr(feat, "mill_duration_min", None)
            name = generated_name("mill", p, alpha=alpha, theta=theta, thetas=thetas, label=base_label,
                                   project_name=project_name, design_name=design_name,
                                   group_index=mill_index)
            path = generated_tree_path("mill", name, project_name)
            steps = build_mill_group_rows(p, angles, warnings, duration_min=group_duration)
            sub = GeneratedSubRecipe(kind_key=f"mill_{mill_index}", tree_path=path, steps=steps,
                                      label=f"Mill: {base_label}" if base_label else f"Mill #{mill_index}")
            subs.append(sub)
            complete_steps.append(_row(rtm.RECIPE_STEP_TYPE, path, comment=f"Calls {sub.label}"))

        elif kind == "deposition":
            dep_index += 1
            if ti_gather_sub is None:
                ti_name = generated_name("ti_gather", p, project_name=project_name)
                ti_path = generated_tree_path("ti_gather", ti_name, project_name)
                ti_gather_sub = GeneratedSubRecipe(kind_key="ti_gather", tree_path=ti_path,
                                                    steps=build_ti_gather_rows(p, warnings),
                                                    label="Ti gather (lower pressure)")
                subs.append(ti_gather_sub)
            complete_steps.append(_row(rtm.RECIPE_STEP_TYPE, ti_gather_sub.tree_path,
                                        comment="Ti gather before this deposition"))
            order_label = base_label or f"pass {dep_index}"
            angles = [(getattr(f, "alpha", None), getattr(f, "theta", None), getattr(f, "thickness_nm", None))
                      for f in group]
            name = generated_name("deposition", p, alpha=alpha, theta=theta, thetas=thetas,
                                   thickness_nm=thickness_nm, label=base_label,
                                   project_name=project_name, design_name=design_name,
                                   order_label=order_label, group_index=dep_index,
                                   has_patches=has_patches)
            path = generated_deposition_path(p.metal_type, name, project_name)
            steps = build_deposition_group_rows(p, angles, warnings)
            sub = GeneratedSubRecipe(kind_key=f"deposition_{dep_index}", tree_path=path, steps=steps,
                                      label=f"Deposition: {base_label}" if base_label else f"Deposition #{dep_index}")
            subs.append(sub)
            complete_steps.append(_row(rtm.RECIPE_STEP_TYPE, path, comment=f"Calls {sub.label}"))

        elif kind == "oxidation":
            # The final protective oxidation after the whole junction is
            # done depositing is a SEPARATE reusable leaf from the
            # between-deposition tunnel-barrier oxidation above -- own
            # kind_key, own steps (build_final_oxidation_rows, using
            # p.final_oxidation_time_min), even though both are
            # SimulationFeatures of kind "oxidation".
            is_final = (getattr(feat, "warning_source", "") or "") == "final_oxidation"
            if is_final:
                if final_oxidation_sub is None:
                    fox_name = generated_name("final_oxidation", p, project_name=project_name)
                    fox_path = generated_tree_path("final_oxidation", fox_name, project_name)
                    final_oxidation_sub = GeneratedSubRecipe(
                        kind_key="final_oxidation", tree_path=fox_path,
                        steps=build_final_oxidation_rows(p),
                        label="Final protective oxidation")
                    subs.append(final_oxidation_sub)
                complete_steps.append(_row(rtm.RECIPE_STEP_TYPE, final_oxidation_sub.tree_path,
                                            comment="Final protective oxidation before unload"))
            else:
                if oxidation_sub is None:
                    ox_name = generated_name("oxidation", p, project_name=project_name)
                    ox_path = generated_tree_path("oxidation", ox_name, project_name)
                    oxidation_sub = GeneratedSubRecipe(kind_key="oxidation", tree_path=ox_path,
                                                        steps=build_oxidation_rows(p),
                                                        label="Oxidation step")
                    subs.append(oxidation_sub)
                complete_steps.append(_row(rtm.RECIPE_STEP_TYPE, oxidation_sub.tree_path,
                                            comment="Oxidation step"))
        # unrecognized feature kinds are skipped, matching
        # recipe_generator._flatten_committed_steps' own stated policy.

    complete_steps.append(_row("Pump Chamber"))
    # Bug fix: the final wait/cooldown was showing minutes as if they
    # were seconds (e.g. a configured 20 minutes rendered as "0:20", 20
    # seconds). A "Wait" step's param1 is always M:SS everywhere else in this file
    # (see build_mill_group_rows/build_ti_gather_rows/build_oxidation_
    # rows) -- this was the one place still writing a bare minutes number
    # (f"{p.wait_minutes:g}", e.g. "20") straight into a "Wait" row, which
    # _mmss_to_seconds/summarize_complete_process_step then reads back as
    # 20 SECONDS ("0:20"), not 20 minutes. _fmt_mmss does the real
    # minutes-to-M:SS conversion, matching every other Wait row here.
    complete_steps.append(_row("Wait", _fmt_mmss(p.wait_minutes * 60.0),
                                comment="Final wait/cooldown before unload."))

    cp_name = generated_name("complete_process", p, project_name=project_name, design_name=design_name)
    cp_path = generated_tree_path("complete_process", cp_name, project_name)
    subs.append(GeneratedSubRecipe(kind_key="complete_process", tree_path=cp_path,
                                    steps=complete_steps, label="Complete process recipe"))
    return subs, list(dict.fromkeys(warnings))


# ----------------------------------------------------------------------
# Dedup + duplicate-and-patch (decision 3 -- see module docstring)
# ----------------------------------------------------------------------

# "Recipe Tilt Substrate" and "RecipeThickness" are the two real
# Plassys step types that have NO numeric parameter at all for the value
# that matters most about them (tilt angle / target thickness -- see
# recipe_builder_model.py's own STEP_TYPE_PARAM_HINTS notes: both are
# Recipe-LEVEL settings on the real tool, encoded only in the recipe's
# NAME/comment, never a row parameter). build_mill_group_rows and
# build_deposition_group_rows both bake that value into those two step
# types' own .comment specifically so it's recoverable from step content
# at all -- so for THESE two step types (and only these two -- every
# other .comment really is auto-filled boilerplate/traceability text,
# e.g. "Same continuous ion beam strike...", not a differing value),
# .comment must participate in equality, or two recipes that differ only
# by tilt angle or only by target thickness would incorrectly dedup onto
# the same node.
_COMMENT_CARRIES_VALUE_STEP_TYPES = {"Recipe Tilt Substrate", "RecipeThickness"}


def _steps_equal(a: List[PlassysStepRow], b: List[PlassysStepRow]) -> bool:
    """Ignores .comment for most step types -- auto-filled boilerplate,
    never meaningful for equality (two recipes with identical real
    machine behavior but a hand-edited comment should still count as
    identical) -- EXCEPT _COMMENT_CARRIES_VALUE_STEP_TYPES, where the
    comment IS the only place a real value (tilt angle / thickness)
    lives at all. See the comment above this function."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if (ra.step_type, ra.param1, ra.param2, ra.param3, ra.param4) != \
           (rb.step_type, rb.param1, rb.param2, rb.param3, rb.param4):
            return False
        if ra.step_type in _COMMENT_CARRIES_VALUE_STEP_TYPES and ra.comment != rb.comment:
            return False
    return True


def find_identical_recipe(root: PlassysRecipeNode, steps: List[PlassysStepRow]) -> Optional[str]:
    """DFS the WHOLE tree (small tree, same performance assumption
    recipe_builder_model.node_path already makes) for a non-folder node
    whose .steps passes _steps_equal against `steps`. Searches
    everywhere, not just the proposed destination folder. Returns its
    resolved path, or None.

    DESIGN NOTE: an earlier version of this generator wired this
    function into resolve_and_write (below) so that a recipe identical
    to one already in the library would be reused rather than
    duplicated. In practice, across many real designs, that meant two
    unrelated projects that happened to produce byte-for-byte identical
    steps would end up silently sharing one library node -- which made
    the library harder to navigate project-by-project, since a user
    looking for "their" recipe would instead find one filed under a
    different project's name. resolve_and_write no longer calls this
    function at all -- every leaf a generation proposes gets its own
    project-named node (see generated_name/generated_tree_path's own
    project_name threading), even when its content byte-for-byte matches
    some OTHER project's already-existing node. This function itself is
    kept, correct and independently tested (test_recipe_tree_export.py),
    as a standalone "does an identical recipe already exist anywhere"
    utility for any future manual lookup -- it just isn't invoked
    automatically as part of generation any more. Regenerating the SAME
    project's own already-generated recipe with nothing changed is a
    SEPARATE case (case 2 in resolve_and_write, keyed by that project's
    own stored link/baseline, not a content search) and is deliberately
    left unchanged by this design choice -- it still reuses that node,
    so re-clicking Generate Recipe on an unchanged design doesn't pile up
    redundant duplicate files for that same project."""
    def _walk(node: PlassysRecipeNode) -> Optional[PlassysRecipeNode]:
        if not node.is_folder and _steps_equal(node.steps, steps):
            return node
        for child in node.children.values():
            found = _walk(child)
            if found is not None:
                return found
        return None
    match = _walk(root)
    return rtm.node_path(root, match) if match is not None else None


def _diff_cells(old_steps: List[PlassysStepRow], new_steps: List[PlassysStepRow]
                 ) -> Optional[List[Tuple[int, int]]]:
    """Returns the list of (row_index, column) cells that differ between
    old_steps and new_steps, IF they have the same length and the same
    step_type row-for-row (a pure value-level change -- e.g. only an
    angle or a thickness differs). column is 1-4 for param1-param4, or
    0 as a sentinel meaning ".comment" -- needed for
    _COMMENT_CARRIES_VALUE_STEP_TYPES (see _steps_equal above), where a
    tilt-angle-only or thickness-only change shows up ONLY in the
    comment, never in any param. Returns None if the shapes differ at
    all (rows added/removed/reordered/retyped) -- signals "not a simple
    patch", per resolve_and_write's decision 3 fallback to a fresh,
    non-derived recipe."""
    if len(old_steps) != len(new_steps):
        return None
    diffs: List[Tuple[int, int]] = []
    for i, (ro, rn) in enumerate(zip(old_steps, new_steps)):
        if ro.step_type != rn.step_type:
            return None
        old_params = (ro.param1, ro.param2, ro.param3, ro.param4)
        new_params = (rn.param1, rn.param2, rn.param3, rn.param4)
        for col0, (vo, vn) in enumerate(zip(old_params, new_params)):
            if vo != vn:
                diffs.append((i, col0 + 1))
        if ro.step_type in _COMMENT_CARRIES_VALUE_STEP_TYPES and ro.comment != rn.comment:
            diffs.append((i, 0))
    return diffs


def _non_colliding_name(folder: PlassysRecipeNode, base_name: str) -> str:
    if base_name not in folder.children:
        return base_name
    i = 2
    while f"{base_name} ({i})" in folder.children:
        i += 1
    return f"{base_name} ({i})"


def _clone_steps(steps: List[PlassysStepRow]) -> List[PlassysStepRow]:
    return [PlassysStepRow(**dataclasses.asdict(r)) for r in steps]


def _set_param(row: PlassysStepRow, col: int, value: str) -> None:
    if col == 0:
        row.comment = value
    else:
        setattr(row, f"param{col}", value)


def _get_field(row: PlassysStepRow, col: int) -> str:
    return row.comment if col == 0 else getattr(row, f"param{col}")


def _create_leaf(root: PlassysRecipeNode, tree_path: str, steps: List[PlassysStepRow]) -> str:
    parent = rtm.ensure_folder_path(root, _parent_path(tree_path))
    leaf_name = _non_colliding_name(parent, _leaf_name(tree_path))
    new_node = rtm.add_child(parent, leaf_name, is_folder=False)
    new_node.steps = list(steps)
    return rtm.node_path(root, new_node)


def resolve_and_write(root: PlassysRecipeNode, sub: GeneratedSubRecipe, *,
                       prior_link_path: Optional[str] = None,
                       prior_baseline_steps: Optional[List[PlassysStepRow]] = None,
                       ) -> Tuple[str, List[PlassysStepRow]]:
    """Returns (final_path, new_baseline_steps_to_store) -- the caller
    saves new_baseline_steps as this design's stored baseline for
    `sub.kind_key`, to diff the NEXT generation against. See module
    docstring, decision 3, for the full "duplicate-and-patch" rationale.

    DESIGN NOTE: this function deliberately does not reach for an
    identical recipe belonging to a different project, so that every
    project keeps its own separate, clearly project-named file in the
    library, even when two projects happen to generate byte-for-byte
    identical steps. This means it does not use the two
    find_identical_recipe cross-tree content searches an earlier version
    of this function made (cases 1/4 and 3, below) -- a project whose
    generated steps happen to byte-for-byte match some OTHER
    already-existing project's node no longer gets silently redirected
    onto that other project's file; it always gets its own new node,
    named for its own project (generated_name/generated_tree_path
    already thread project_name through every kind's name/path). Case 2
    below (regenerating THIS SAME project's own already-generated,
    unchanged recipe) still reuses that project's own existing link --
    this is only about not reaching for a DIFFERENT project's node,
    never about re-duplicating a project's own unchanged file on every
    re-generation, which would just pile up redundant clutter for that
    one project instead of keeping the library organized.

    Order of operations:
      1. No prior_link_path, or it no longer resolves to a real node
         (deleted since): always create a brand-new node at sub.tree_path
         (non-colliding name if needed) -- never a cross-tree content
         search.
      2. prior_link_path resolves to a node, and prior_baseline_steps
         matches what THIS generation would produce (_steps_equal):
         nothing changed -- reuse the existing link untouched, whatever
         hand-edits it carries, with no write at all.
      3. prior_link_path resolves, but the fresh content differs from
         the stored baseline by VALUES ONLY (_diff_cells succeeds):
         deep-copy the node's CURRENT LIVE steps (hand-edits and all),
         patch just the diffed cells onto the copy, and file the copy as
         a new sibling node (a non-colliding verbose name) -- always, even
         if some OTHER already-existing node happens to now match it
         exactly. The OLD node is left completely untouched in the tree,
         still browsable/reusable.
      4. prior_link_path resolves, but the change is structural (rows
         added/removed/retyped, not just a value): can't cleanly patch
         -- falls through to the same fresh always-create as case 1, NOT
         derived from the old node.
    """
    prior_node = rtm.resolve_path(root, prior_link_path) if prior_link_path else None

    if prior_node is not None and not prior_node.is_folder and prior_baseline_steps is not None:
        if _steps_equal(prior_baseline_steps, sub.steps):
            return prior_link_path, _clone_steps(prior_baseline_steps)

        diffs = _diff_cells(prior_baseline_steps, sub.steps)
        if diffs is not None:
            patched_steps = _clone_steps(prior_node.steps)
            for row_i, col in diffs:
                if row_i < len(patched_steps):
                    new_val = _get_field(sub.steps[row_i], col)
                    _set_param(patched_steps[row_i], col, new_val)
            new_path = _create_leaf(root, sub.tree_path, patched_steps)
            return new_path, _clone_steps(sub.steps)
        # else: structural change -- fall through to fresh always-create.

    new_path = _create_leaf(root, sub.tree_path, sub.steps)
    return new_path, _clone_steps(sub.steps)
