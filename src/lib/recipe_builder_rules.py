"""
recipe_builder_rules.py

A validation/rules engine for the Recipe Builder (recipe_builder_model.py) --
walks a from-scratch, named Plassys recipe (a flat Step Type/Parameter
1-4/Comment table, possibly calling other named recipes as nested
sub-tables via a "Recipe" step) and raises a warning for every step or
sub-step that's missing something required, out of order, or left in an
unsafe state -- each warning carries a short, concrete description of
what to change so the warning goes away.

The goal is to look at how the Plassys tool is actually operated -- which
steps always go together, which are required so the machine doesn't get
damaged, and which are easy to get wrong -- and turn that into a checker
that raises a warning, with a concrete description of what to fix, for
any step or sub-step that's missing something required, out of order, or
left in an unsafe state.

VOCABULARY AND SEQUENCING PROVENANCE: the rules below are derived from
the step-type vocabulary and sequencing conventions established
elsewhere in this codebase (see recipe_builder_model.py's vocabulary
section and ai_assistant.py's APP_REFERENCE "REAL PLASSYS
STEP-TYPE VOCABULARY" section), which were themselves built from direct
observation of the tool's editor and Executing screen. Two behaviors in
particular come from a real, complete, working reference deposition
recipe: (1) it opens the source/crucible shutter and never explicitly
closes it again before the recipe ends -- so shutter_source_pairing
below does not require a matching close for that specific shutter; (2)
it also opens with a defensive "Substrate Shutter EGun Close" ahead of
that shutter's real Open/Close pair used later -- a normal, safe "reset
to a known state" step, not a missing-Open mistake -- so every
toggle-pair rule's "closed/off without ever being opened/on" half checks
whether the thing is EVER opened/turned on ANYWHERE in the whole recipe,
not just "not yet, at this specific point" (see _check_toggle_pairs' own
comment for the detail). The rest of the SEQUENCING/SAFETY LOGIC matches
the real tool's step-type vocabulary and conventions documented in:
  - recipe_builder_model.py's STEP_TYPE_VOCAB / STEP_TYPE_PARAM_HINTS and
    main_gui.py's _recipe_builder_template_step (the literal
    Step Type / Parameter 1-4 grid and its established value
    conventions -- e.g. an "Ion Beam" row with Parameter 1 = "Off"
    means the beam is off, one with "V=..." means it's struck).
  - ai_assistant.py's APP_REFERENCE "REAL PLASSYS STEP-TYPE
    VOCABULARY" section (a narrative transcription of the literal step
    lines and their sequencing, based on the tool's editor and
    Executing screen).
  - recipe_generator.py's own sequencing comments (e.g. "a mill covering
    more than one theta strikes the ion beam ONCE for the whole group",
    "Ti gather ... substrate shutter stays CLOSED", the ramp-up/
    ramp-down current profile shape).

Rules here are scoped to be generic to ANY Plassys recipe (mill and/or
deposition), not specific to this app's own JJ/bilayer conventions, since
the Recipe Builder is meant for "ANY process" (see recipe_builder_model.py's
own docstring). A convention that IS JJ/bilayer-specific (e.g. "Ti gather
always immediately before every deposition group") is deliberately left
OUT of this generic engine -- that convention already lives in, and is
enforced by, the deterministic generator in recipe_generator.py instead.
Nothing here should be read as claiming those JJ-specific conventions are
wrong for a JJ recipe; they're simply out of scope for a checker meant to
apply to any recipe a user builds from scratch.

Each rule inspects a FLATTENED step sequence -- "Recipe" steps are
expanded in place by resolving their target path against the whole tree
and splicing in that target's own steps recursively (with cycle
detection), so a rule sees the recipe exactly as it would actually run on
the tool, not just the one screen's worth of rows the user happens to be
looking at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import recipe_builder_model as rtm


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------

@dataclass
class RuleWarning:
    """One rule violation. `severity` is "critical" (machine-safety --
    could damage the tool, waste a wafer, or leave hardware energized)
    or "caution" (structural/logical -- likely a mistake, not a hazard).
    `recipe_path` + `step_index` locate the OFFENDING row (0-based index
    into that recipe's own .steps list) so the UI can point the user
    straight at it, even when the row lives inside a nested sub-recipe
    reached only through a "Recipe" call."""
    rule_id: str
    severity: str
    message: str
    fix: str
    recipe_path: str
    step_index: int = -1

    def display_text(self) -> str:
        loc = f'step {self.step_index + 1} of "{self.recipe_path}"' if self.step_index >= 0 else f'"{self.recipe_path}"'
        mark = "⚠" if self.severity == "critical" else "?"
        return f"{mark} [{self.severity.upper()}] {self.message} ({loc})\n   Fix: {self.fix}"


@dataclass
class FlatStep:
    """One row, in the order it would actually execute on the tool, with
    enough context to report exactly where it lives (its OWN recipe's
    path and index -- not the top-level recipe's -- so a warning about a
    row reached through a nested "Recipe" call still points at the right
    place to fix it)."""
    row: rtm.PlassysStepRow
    recipe_path: str
    step_index: int


# ----------------------------------------------------------------------
# Flattening -- expands nested "Recipe" calls in place, with cycle and
# depth guards so a bad call chain is reported as its own warning rather
# than recursing forever.
# ----------------------------------------------------------------------

_DEFAULT_MAX_DEPTH = 25


def flatten_recipe(root: rtm.PlassysRecipeNode, start_path: str,
                    max_depth: int = _DEFAULT_MAX_DEPTH) -> Tuple[List[FlatStep], List[RuleWarning]]:
    """Returns (flat_steps, structural_warnings) for the named recipe at
    start_path. structural_warnings covers the "Recipe" call mechanic
    itself -- an empty/unresolved target path, a call-parameter count
    mismatch against the target's declared call_params, or a call cycle
    -- rather than the machine-sequencing rules in this module's other
    functions."""
    flat: List[FlatStep] = []
    structural: List[RuleWarning] = []

    # Ancestors are tracked by NODE IDENTITY, not raw path string -- the
    # same recipe can legitimately be spelled two different ways (e.g.
    # "Evap\\Ti" vs "Root\\Evap\\Ti", both resolving to the same node),
    # and comparing strings would miss a real cycle written with mixed
    # spelling. `canon(node)` gives the one canonical display path for
    # messages/locations regardless of how a given "Recipe" step spelled
    # its call.
    def canon(node: rtm.PlassysRecipeNode) -> str:
        return rtm.node_path(root, node) or node.name

    def _expand(path: str, ancestor_nodes: List[rtm.PlassysRecipeNode], depth: int) -> None:
        node = rtm.resolve_path(root, path)
        if node is None or node.is_folder:
            return  # caller already recorded a warning about this -- see below
        if depth > max_depth:
            structural.append(RuleWarning(
                rule_id="call_depth_exceeded", severity="caution",
                message=f'Recipe calls nested under "{canon(node)}" go more than {max_depth} levels deep.',
                fix="Simplify the call chain, or check for a \"Recipe\" step that calls back into "
                    "one of its own callers (a cycle this checker could not otherwise detect).",
                recipe_path=canon(node),
            ))
            return
        if any(node is anc for anc in ancestor_nodes):
            chain = " -> ".join(canon(n) for n in ancestor_nodes + [node])
            structural.append(RuleWarning(
                rule_id="call_cycle", severity="critical",
                message=f'"{canon(node)}" is called again from within its own call chain ({chain}) -- '
                        f"this would call itself forever on the real tool.",
                fix=f'Remove or redirect one of the "Recipe" steps in this chain so "{canon(node)}" is '
                    f"never called from inside anything it itself (directly or indirectly) calls.",
                recipe_path=canon(node),
            ))
            return
        display_path = canon(node)
        for i, row in enumerate(node.steps):
            if row.step_type != rtm.RECIPE_STEP_TYPE:
                flat.append(FlatStep(row, display_path, i))
                continue
            flat.append(FlatStep(row, display_path, i))
            target_path = (row.param1 or "").strip()
            if not target_path:
                structural.append(RuleWarning(
                    rule_id="recipe_call_missing_path", severity="critical",
                    message='A "Recipe" step here has no path in Parameter 1, so it calls nothing.',
                    fix='Set Parameter 1 to the path of the recipe to call, e.g. "Evap\\\\Ti".',
                    recipe_path=display_path, step_index=i,
                ))
                continue
            target = rtm.resolve_path(root, target_path)
            if target is None or target.is_folder:
                structural.append(RuleWarning(
                    rule_id="recipe_call_unresolved", severity="critical",
                    message=f'This calls "{target_path}", which doesn\'t resolve to a real recipe '
                            f"(it doesn't exist, or it's a folder, not a recipe).",
                    fix=f'Fix Parameter 1 to point at an existing recipe, or create "{target_path}" '
                        f"as a recipe first.",
                    recipe_path=display_path, step_index=i,
                ))
                continue
            provided = [v for v in (row.param2, row.param3, row.param4) if (v or "").strip()]
            expected = list(target.call_params)
            if len(expected) != len(provided):
                structural.append(RuleWarning(
                    rule_id="recipe_call_param_mismatch", severity="caution",
                    message=(f'This calls "{target_path}", which expects {len(expected)} call value(s) '
                              f'({", ".join(expected) if expected else "none"}), but this call provides '
                              f'{len(provided)}.'),
                    fix=(f'Fill in Parameter 2 onward with exactly {len(expected)} value(s) matching '
                         f'{", ".join(expected) if expected else "(remove the extra ones -- none are needed)"}.'),
                    recipe_path=display_path, step_index=i,
                ))
            _expand(target_path, ancestor_nodes + [node], depth + 1)

    _expand(start_path, [], 0)
    return flat, structural


# ----------------------------------------------------------------------
# Small parsing helpers -- everything in a PlassysStepRow is plain text
# (matching the real tool's own editor), so state has to be read out of
# free text using the value conventions already established elsewhere in
# this codebase (see module docstring).
# ----------------------------------------------------------------------

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _first_number(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    m = _NUM_RE.search(s)
    return float(m.group()) if m else None


def _param(row: rtm.PlassysStepRow, col: int) -> str:
    """Reads row.param<col> (1-4) generically -- lets a toggle-pair spec
    say which parameter column actually carries its on/off state (e.g.
    the unified "Substrate Shutter" step keeps Open/Close in Parameter
    2, not Parameter 1, since Parameter 1 is which shutter)."""
    return getattr(row, f"param{col}", "") or ""


def _row_state(row: rtm.PlassysStepRow, kind: str, col: int = 1) -> Optional[bool]:
    """True = on/open/struck, False = off/close, None = can't tell (row
    doesn't carry a recognizable state for this `kind`). `col` selects
    which Parameter column (1-4) actually carries the state."""
    if kind == "open_close":
        p1 = _norm(_param(row, col))
        if "open" in p1 or p1 == "on":
            return True
        if "close" in p1 or p1 == "off":
            return False
        return None
    if kind == "on_off":
        p1 = _norm(_param(row, col))
        if p1 == "on" or "open" in p1:
            return True
        if p1 == "off" or "close" in p1:
            return False
        return None
    if kind == "numeric":
        n = _first_number(_param(row, col))
        if n is None:
            return None
        return n > 1e-9
    if kind == "ion_beam":
        # Convention established in _recipe_builder_template_step in
        # main_gui.py: an "Ion Beam" row with "Off" in a parameter
        # means the beam is off; one with "V=" means it's struck.
        blob = " ".join(_norm(x) for x in (row.param1, row.param2, row.param3, row.param4))
        if "off" in blob:
            return False
        if "v=" in blob:
            return True
        return None
    return None


# ----------------------------------------------------------------------
# Rule: paired on/off, open/close state must always resolve -- nothing
# left energized, flowing, or open when the recipe ends, and nothing
# turned off/closed that was never turned on/opened.
# ----------------------------------------------------------------------

_TOGGLE_RULES = [
    # Substrate-side shutters: ONE real step type, "Substrate Shutter",
    # with Parameter 1 = which shutter (EGun / Ion Gun) and Parameter 2 =
    # Open/Close -- see recipe_builder_model.py's STEP_TYPE_PARAM_CHOICES
    # note for why this is unified rather than two separate step types.
    # filter_param/filter_value pick out just the rows for ONE shutter
    # (they're independent pieces of hardware and must each pair up on
    # their own); state_param says Open/Close lives in Parameter 2 here.
    dict(step_type="Substrate Shutter", kind="open_close", rule_id="shutter_ebgun_pairing",
         label="the substrate EBGun shutter", filter_param=1, filter_value="egun", state_param=2,
         off_without_on_fix='Add a "Substrate Shutter" step (Parameter 1 = "EGun", Parameter 2 = "Open") '
                             "earlier in the recipe, before this Close, or remove this Close if it isn't needed.",
         on_without_off_fix='Add a "Substrate Shutter" step (Parameter 1 = "EGun", Parameter 2 = "Close") '
                             "after the last time this shutter is used, so it isn't left open when the "
                             "recipe ends."),
    dict(step_type="Substrate Shutter", kind="open_close", rule_id="shutter_iongun_pairing",
         label="the substrate ion-gun shutter", filter_param=1, filter_value="ion gun", state_param=2,
         off_without_on_fix='Add a "Substrate Shutter" step (Parameter 1 = "Ion Gun", Parameter 2 = "Open") '
                             "earlier in the recipe, before this Close, or remove this Close if it isn't needed.",
         on_without_off_fix='Add a "Substrate Shutter" step (Parameter 1 = "Ion Gun", Parameter 2 = "Close") '
                             "after the last time this shutter is used, so it isn't left open when the "
                             "recipe ends."),
    # Source (crucible) shutter: a real, complete, working reference
    # recipe opens this shutter and never explicitly closes it again
    # before the recipe ends (it's left open through the ramp-down/
    # cool-down at the end); the e-gun emission ramping back to 0mA is
    # what actually matters once the shutter is open, since no material
    # is being emitted regardless of shutter state. So unlike the
    # substrate-side shutters above (which really do protect the wafer
    # and must close), this one does NOT require a closing step --
    # require_close=False. The other half of the check (closing it
    # without ever having opened it, still a real mistake -- a Close
    # with nothing to close) is kept.
    dict(step_type="Source Shutter", kind="open_close", rule_id="shutter_source_pairing",
         label="the source (crucible) shutter", require_close=False,
         off_without_on_fix='This closes the source shutter, but it was never opened first -- add a '
                             '"Source Shutter" step with Parameter 1 = "Open" earlier in the recipe, or '
                             "remove this Close if it isn't needed.",
         on_without_off_fix=""),  # unused -- require_close=False means this never fires
    dict(step_type="IBG Discharge", kind="on_off", rule_id="ibg_discharge_pairing",
         label="the IBG discharge",
         off_without_on_fix='Add an "IBG Discharge" step with Parameter 1 = "On" earlier in the recipe, '
                             "before this Off, or remove this Off if it isn't needed.",
         on_without_off_fix='Add an "IBG Discharge" step with Parameter 1 = "Off" after the last mill pass, '
                             "so the discharge isn't left running when the recipe ends."),
    dict(step_type="Ar Ion Gun Gas", kind="numeric", rule_id="ar_gas_pairing",
         label="the Ar ion-gun gas flow",
         off_without_on_fix='This sets the gas to 0 sccm, but it was never turned on first -- add an '
                             '"Ar Ion Gun Gas" step with a nonzero sccm value earlier, or remove this step.',
         on_without_off_fix='Add an "Ar Ion Gun Gas" step set to 0 sccm after the last mill pass, so gas '
                             "isn't left flowing into the chamber when the recipe ends."),
    dict(step_type="E Gun Emission", kind="numeric", rule_id="egun_ramp_pairing",
         label="the e-gun emission ramp",
         off_without_on_fix='This ramps to 0mA, but the gun was never ramped up first -- add an "E Gun '
                             'Emission" step to a nonzero mA target earlier, or remove this step.',
         on_without_off_fix='Add an "E Gun Emission" step ramping back down to 0mA after the last '
                             "deposition/gather in this group, so the filament isn't left hot when the "
                             "recipe ends."),
    dict(step_type="Ion Beam", kind="ion_beam", rule_id="ion_beam_pairing",
         label="the ion beam",
         off_without_on_fix='This turns the beam off, but it was never struck first -- add an "Ion Beam" '
                             "step with V=/I=/Vacc= values earlier, or remove this Off row.",
         on_without_off_fix='Add an "Ion Beam" step with Parameter 1 = "Off" after the last mill pass, so '
                             "the beam isn't left striking when the recipe ends."),
]


def _check_toggle_pairs(flat: List[FlatStep]) -> List[RuleWarning]:
    warnings: List[RuleWarning] = []
    for spec in _TOGGLE_RULES:
        filter_param = spec.get("filter_param")
        filter_value = spec.get("filter_value")
        state_param = spec.get("state_param", 1)
        require_close = spec.get("require_close", True)

        # Collect every row for this spec first (rather than a single
        # left-to-right pass): a real, complete, working reference
        # recipe opens with "Substrate Shutter EGun Close" -- a
        # defensive reset to a known state, BEFORE the shutter's real
        # Open/Close pair used later for the actual deposition. A strict
        # left-to-right state machine flags that leading Close as
        # "closed but never opened" (a false positive on a genuinely
        # correct recipe), because it hadn't seen the later Open yet.
        # The fix: "off without on" is only a real mistake when this
        # thing is NEVER opened ANYWHERE in the whole recipe -- not just
        # "not yet, at this specific point" -- so a leading defensive
        # close ahead of a real later open is allowed, while a stray
        # close with truly no open anywhere (the actual mistake this
        # check exists to catch) still warns.
        matching = []
        for fs in flat:
            if fs.row.step_type != spec["step_type"]:
                continue
            if filter_param is not None and filter_value not in _norm(_param(fs.row, filter_param)):
                continue
            state = _row_state(fs.row, spec["kind"], col=state_param)
            if state is None:
                continue
            matching.append((fs, state))
        ever_on = any(state for _, state in matching)

        is_on = False
        on_loc: Optional[Tuple[str, int]] = None
        for fs, state in matching:
            if state:
                is_on = True
                on_loc = (fs.recipe_path, fs.step_index)
            else:
                # Bug fix: the Ti-gather sub-recipe closes the substrate
                # EGun shutter and NEVER reopens it -- by design, not by
                # mistake (see generated_recipe_export.
                # build_ti_gather_rows: it's meant to stay closed for the
                # whole gather, only reopened by the SEPARATE deposition
                # sub-recipe that's always called right after it). Since
                # this rule (like every rule here) only ever sees ONE
                # recipe's own flattened steps -- it has no visibility
                # into some OTHER, separately-called recipe that reopens
                # the same shutter later -- a deliberately-permanent close
                # like this one is indistinguishable, by state alone, from
                # the real mistake this rule exists to catch. So a close
                # step whose own comment documents that intent (the exact
                # phrase this module's own docstring already uses for Ti-
                # gather: "stays closed") is trusted and skipped here,
                # same as this file already trusts a real, confirmed
                # reference recipe's OTHER documented exceptions (e.g.
                # shutter_source_pairing's require_close=False above).
                intentionally_unpaired = "stays closed" in _norm(fs.row.comment or "")
                if not is_on and not ever_on and not intentionally_unpaired:
                    label = spec["label"]
                    warnings.append(RuleWarning(
                        rule_id=spec["rule_id"], severity="critical",
                        message=f"{label[0].upper()}{label[1:]} is turned off/closed here, but it was "
                                f"never turned on/opened anywhere in this recipe.",
                        fix=spec["off_without_on_fix"],
                        recipe_path=fs.recipe_path, step_index=fs.step_index,
                    ))
                is_on = False
                on_loc = None
        if is_on and on_loc is not None and require_close:
            label = spec["label"]
            warnings.append(RuleWarning(
                rule_id=spec["rule_id"], severity="critical",
                message=f"{label[0].upper()}{label[1:]} is turned on/opened here but never turned "
                        f"off/closed again before the recipe ends.",
                fix=spec["on_without_off_fix"],
                recipe_path=on_loc[0], step_index=on_loc[1],
            ))
    return warnings


# ----------------------------------------------------------------------
# Rule: a Crucible must be selected before the e-gun is ramped up --
# otherwise the gun is heating whatever material was last selected (or
# nothing at all), not the one this recipe actually means to evaporate.
# ----------------------------------------------------------------------

def _check_crucible_before_ramp(flat: List[FlatStep]) -> List[RuleWarning]:
    warnings: List[RuleWarning] = []
    seen_crucible = False
    already_warned = False
    for fs in flat:
        if fs.row.step_type == "Material select":
            seen_crucible = True
        elif fs.row.step_type == "E Gun Emission" and not seen_crucible and not already_warned:
            n = _first_number(fs.row.param1)
            if n is not None and n > 1e-9:
                warnings.append(RuleWarning(
                    rule_id="crucible_before_ramp", severity="critical",
                    message='This ramps the e-gun emission current up, but no "Material select" step '
                            "(selecting a material) appears earlier in the recipe.",
                    fix='Add a "Material select" step (Parameter 1 = the material, e.g. "Ti" or "Al") '
                        "before this ramp, so the e-gun is heating the crucible this recipe actually "
                        "intends to evaporate.",
                    recipe_path=fs.recipe_path, step_index=fs.step_index,
                ))
                already_warned = True
    return warnings


# ----------------------------------------------------------------------
# Rule: Rate Control should only be locked once the gun is actually
# ramped up and soaking -- locking a rate against a cold/off gun isn't
# meaningful.
# ----------------------------------------------------------------------

def _check_rate_control_after_ramp(flat: List[FlatStep]) -> List[RuleWarning]:
    warnings: List[RuleWarning] = []
    ramped_up = False
    for fs in flat:
        if fs.row.step_type == "E Gun Emission":
            n = _first_number(fs.row.param1)
            if n is not None:
                ramped_up = n > 1e-9
        elif fs.row.step_type == "Rate control" and not ramped_up:
            warnings.append(RuleWarning(
                rule_id="rate_control_after_ramp", severity="caution",
                message='This locks a deposition Rate control, but no "E Gun Emission" ramp to a nonzero '
                        "current appears earlier (or the gun was already ramped back down since).",
                fix='Add an "E Gun Emission" step (ramping up, then a "Ramp wait" step) before this '
                    "Rate control, so the rate is locked while the gun is actually up and soaking.",
                recipe_path=fs.recipe_path, step_index=fs.step_index,
            ))
    return warnings


# ----------------------------------------------------------------------
# Rule: every "RecipeThickness" (the real step that ends a deposition by
# depositing until the recipe's own configured thickness is reached --
# see recipe_builder_model.py's vocabulary note; this was previously,
# incorrectly, called "Wait for Termination") needs its own preceding
# "Zero Thickness" -- otherwise the crystal monitor isn't zeroed for
# this layer, and the thickness reading isn't measuring this layer from
# zero.
# ----------------------------------------------------------------------

def _check_zero_thickness_before_wait(flat: List[FlatStep]) -> List[RuleWarning]:
    warnings: List[RuleWarning] = []
    have_fresh_zero = False
    for fs in flat:
        if fs.row.step_type == "Zero Thickness":
            have_fresh_zero = True
        elif fs.row.step_type == "RecipeThickness":
            if not have_fresh_zero:
                warnings.append(RuleWarning(
                    rule_id="zero_thickness_before_wait", severity="critical",
                    message='This deposits to the Recipe\'s configured thickness, but no "Zero Thickness" '
                            "step (since the last one used) appears earlier -- the thickness monitor "
                            "reading may not start from zero for this layer.",
                    fix='Add a "Zero Thickness" step right before this "RecipeThickness" step, so the '
                        "crystal monitor is zeroed for this specific layer before it's timed.",
                    recipe_path=fs.recipe_path, step_index=fs.step_index,
                ))
            have_fresh_zero = False  # consumed -- the next RecipeThickness needs its own
    return warnings


# ----------------------------------------------------------------------
# Rule: chamber/load-lock pressure must be checked before any real
# process step runs (material selection, e-gun ramp, ion beam, gas,
# discharge) -- the vacuum/pump check happens once, at the very start,
# per the sequencing convention documented in recipe_generator.py.
# ----------------------------------------------------------------------

_PRESSURE_CHECK_TYPES = {"Process Chamber", "LL Process", "Pump Chamber", "Pump LL"}
_PROCESS_START_TYPES = {"Material select", "E Gun Emission", "Ion Beam", "Ar Ion Gun Gas", "IBG Discharge"}


def _check_vacuum_before_process(flat: List[FlatStep]) -> List[RuleWarning]:
    seen_pressure_check = False
    for fs in flat:
        st = fs.row.step_type
        if st in _PRESSURE_CHECK_TYPES:
            seen_pressure_check = True
        elif st in _PROCESS_START_TYPES and not seen_pressure_check:
            return [RuleWarning(
                rule_id="vacuum_check_before_process", severity="critical",
                message="This recipe starts real process steps (material selection / e-gun ramp / ion "
                        "beam / gas / discharge) without ever checking chamber or load-lock pressure first.",
                fix='Add a "Process Chamber" and/or "LL Process" (or "Pump Chamber"/"Pump LL") step before '
                    "the first Material select / E Gun Emission / Ion Beam / Ar Ion Gun Gas / IBG Discharge "
                    "step, confirming the chamber is actually at the right vacuum before running the process.",
                recipe_path=fs.recipe_path, step_index=fs.step_index,
            )]
    return []


# ----------------------------------------------------------------------
# Rule: the ion beam must never be struck without gas already flowing
# and the IBG discharge already on -- confirmed sequence order (APP_
# REFERENCE's "REAL PLASSYS STEP-TYPE VOCABULARY"): ... Ar Ion Gun Gas
# ... IBG Discharge On ... Ion Beam V=... -- striking the beam with no
# gas flow risks arcing/damage to the gun in vacuum.
# ----------------------------------------------------------------------

def _check_gas_and_discharge_before_beam(flat: List[FlatStep]) -> List[RuleWarning]:
    warnings: List[RuleWarning] = []
    gas_on = False
    disch_on = False
    for fs in flat:
        st = fs.row.step_type
        if st == "Ar Ion Gun Gas":
            state = _row_state(fs.row, "numeric")
            if state is not None:
                gas_on = state
        elif st == "IBG Discharge":
            state = _row_state(fs.row, "on_off")
            if state is not None:
                disch_on = state
        elif st == "Ion Beam" and _row_state(fs.row, "ion_beam") is True:
            if not gas_on:
                warnings.append(RuleWarning(
                    rule_id="gas_before_beam", severity="critical",
                    message="This strikes the ion beam, but the Ar ion-gun gas isn't flowing at this "
                            'point in the recipe (no earlier "Ar Ion Gun Gas" step with a nonzero sccm '
                            "value that hasn't since been turned off).",
                    fix='Add an "Ar Ion Gun Gas" step with a nonzero sccm value before this, and make '
                        "sure nothing turns it back to 0 sccm before the beam strikes -- striking the "
                        "beam with no gas flow risks arcing/damage in vacuum.",
                    recipe_path=fs.recipe_path, step_index=fs.step_index,
                ))
            if not disch_on:
                warnings.append(RuleWarning(
                    rule_id="discharge_before_beam", severity="critical",
                    message='This strikes the ion beam, but the IBG discharge isn\'t on at this point in '
                            'the recipe (no earlier "IBG Discharge" step set to "On" that hasn\'t since '
                            'been turned off).',
                    fix='Add an "IBG Discharge" step with Parameter 1 = "On" before this, and make sure '
                        "nothing turns it back off before the beam strikes.",
                    recipe_path=fs.recipe_path, step_index=fs.step_index,
                ))
    return warnings


# ----------------------------------------------------------------------
# Rule: the table must be tilted to the Etch position before the ion
# beam is struck.
# ----------------------------------------------------------------------

def _check_substrate_position_before_mill(flat: List[FlatStep]) -> List[RuleWarning]:
    seen_etch_position = False
    for fs in flat:
        if fs.row.step_type == "Recipe Tilt Substrate" and "etch" in _norm(fs.row.param1):
            seen_etch_position = True
        elif fs.row.step_type == "Ion Beam" and _row_state(fs.row, "ion_beam") is True and not seen_etch_position:
            return [RuleWarning(
                rule_id="substrate_position_before_mill", severity="critical",
                message='This strikes the ion beam, but no "Recipe Tilt Substrate" step with Parameter 1 = '
                        '"Etch" appears earlier in the recipe.',
                fix='Add a "Recipe Tilt Substrate" step (Parameter 1 = "Etch") before this, so the table '
                    "is actually tilted for milling before the beam strikes.",
                recipe_path=fs.recipe_path, step_index=fs.step_index,
            )]
    return []


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------

_ALL_STEP_RULES = [
    _check_toggle_pairs,
    _check_crucible_before_ramp,
    _check_rate_control_after_ramp,
    _check_zero_thickness_before_wait,
    _check_vacuum_before_process,
    _check_gas_and_discharge_before_beam,
    _check_substrate_position_before_mill,
]


def validate_recipe(root: rtm.PlassysRecipeNode, recipe_path: str,
                     max_depth: int = _DEFAULT_MAX_DEPTH) -> List[RuleWarning]:
    """Validates the named recipe at recipe_path -- expanding every
    nested "Recipe" call it makes -- against every rule in this module.
    Returns a flat list of RuleWarning, structural warnings first (an
    unresolved/missing call, a call-parameter mismatch, a call cycle),
    then every machine-sequencing warning, in the order the affected
    step would actually run on the tool. Returns [] for a path that
    doesn't resolve to a real recipe (nothing to validate)."""
    node = rtm.resolve_path(root, recipe_path)
    if node is None or node.is_folder:
        return []
    flat, structural = flatten_recipe(root, recipe_path, max_depth=max_depth)
    warnings: List[RuleWarning] = list(structural)
    for rule_fn in _ALL_STEP_RULES:
        warnings.extend(rule_fn(flat))
    return warnings
