"""
recipe_builder_model.py

Data model + persistence for the "Recipe Builder" -- a from-scratch
Plassys recipe creator that mirrors the REAL machine's own editor
mechanics: a tree of named recipes (e.g. "Root\\Evap\\Ti",
"Root\\Etching\\New Ion Gun Recipes\\Sidewall 60"), each one a flat
table of steps (Step Type / Parameter 1-4 / Comment -- the literal
grid the tool is authored in, confirmed against real photographs of
the tool's own recipe editor and Executing screen -- see
ai_assistant.py's APP_REFERENCE for the fuller narrative
version of the same vocabulary), where a step can itself be of type
"Recipe" and CALL another named recipe elsewhere in the tree as a
sub-routine -- passing call-time values for whatever placeholder
parameters that sub-recipe declares (e.g. a deposition recipe declares
"thick?"/"tilt?" placeholders since those vary per use even though its
own rate is fixed in its own name/definition, exactly as seen in real
recipe-tree text like "Root\\Evap\\Al\\Al 0.20nm/s thick? tilt?").

This is completely independent of the deterministic, design-driven
recipe generator in recipe_generator.py -- that engine computes a JJ
recipe automatically from simulated junction geometry; this module is
a manual, from-scratch, generic recipe author usable for ANY process,
matching the real tool's own tree-of-named-recipes authoring mechanic
one-to-one rather than the app's own RecipeStep abstraction. The two
are deliberately kept separate data models (see main_gui.py's
New Tab 3, which now offers both as two modes of the same tab): a
from-scratch Plassys recipe creator using the exact same mechanics the
Plassys uses to create a recipe -- essentially, nested tables.
"""

from __future__ import annotations

import os
import json
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

_ENV_OVERRIDE = "PLASSYS_RECIPE_TREE_DIR"

PATH_SEP = "\\"
ROOT_NAME = "Root"


# ----------------------------------------------------------------------
# Real Step Type vocabulary -- transcribed EXACTLY from a clear, direct
# photograph of the tool's own recipe editor (a real, complete Al
# deposition recipe on a Plassys MEB550S: "myproject- Al 0.5nm/s thick?
# tilt? Manhattan 2nd +45"), not an approximation, so the exact step
# type names match what appears on the machine rather than an
# approximate rendering of them.
# Earlier iterations of this vocabulary (from blurrier/indirect
# references) had several names slightly wrong -- corrected here against
# the clear photo:
#   WAS (approximate)              -> NOW (exact, from the real editor)
#   "Substrate Position"           -> "Recipe Tilt Substrate"
#   "Move Table" (a separate type) -> folded into "Recipe Tilt Substrate"
#                                      (Parameter 1 = the destination --
#                                      Deposit/Etch/Load -- there is no
#                                      separate literal tilt-angle
#                                      parameter on this step; the real
#                                      tilt magnitude is a Recipe-level
#                                      value, not typed per row, exactly
#                                      like RecipeThickness below)
#   "Move Planetary"               -> "Position Planetary"
#   "Process LL"                   -> "LL Process"
#   "Shutter Source"               -> "Source Shutter"
#   "Crucible"                     -> "Material select"
#   "Egun Emission Ramp"           -> "E Gun Emission"
#   "Ramp Wait"                    -> "Ramp wait"
#   "Rate Control"                 -> "Rate control"
#   "Wait for Termination"         -> "RecipeThickness" (this is the real
#                                      step that actually ends a
#                                      deposition -- it isn't a timed
#                                      "wait", it deposits until the
#                                      Recipe's own configured thickness
#                                      value is reached; "Wait for
#                                      Termination" never appeared on the
#                                      real screen at all)
#   "Substrate Shutter EBGun" /
#   "Substrate Shutter Ion Gun"    -> unified into one "Substrate
#                                      Shutter" step (Parameter 1 = which
#                                      shutter, EGun or Ion Gun --
#                                      exactly like Recipe Tilt
#                                      Substrate's destination
#                                      parameter -- Parameter 2 =
#                                      Open/Close)
# Two real step types were missing from the vocabulary entirely and are
# added here: "Material select" mentioned above (was miscalled
# "Crucible"), "Operator requested" (a message-alert step -- e.g. "About
# to turn on the..."/"Check the beam position"), and "RecipeThickness"
# (see above). Everything the mill/ion-gun side used ("Ar Ion Gun Gas",
# "IBG Discharge", "Ion Beam") is unchanged -- this photo only covers a
# deposition recipe, and nothing in it contradicts those names.
#
# PARAM_CHOICES gives, for a handful of step types, the small fixed set
# of values Parameter 1/2 can actually take on the real tool (a
# secondary dropdown rather than free text): for some step types the
# real editor offers a small dropdown of two or three options rather
# than a free-text field for "Open"/"Close" and similar values. Every
# entry here is either read directly off the photo (Deposit; EGun; Open/
# Close; Zero; Al) or a well-confirmed sibling value already established
# elsewhere in this codebase (Etch/Load for the tilt destination; Ion
# Gun for the other shutter; the rest of METAL_OPTIONS for material
# select) -- still an editable combo box, not a locked-in choice, so an
# unusual real value can always be typed in directly.
# DEFAULT_COMMENT gives the auto-filled Comment text for a step type,
# matching how the real editor automatically populates a Comment for
# each step type -- transcribed verbatim from the photo where a row of
# that exact type appears there, and written in the same descriptive
# style for the few real-but-unphotographed types. "Comment Only" is the one
# deliberate exception: it exists purely so the user can write their own
# free-form note, so it has no auto-comment and stays user-editable.
# DEFAULT_PARAMS gives the values a brand-new (or freshly retyped) row
# of that step type starts with -- always the safe/off/neutral state
# (Close, Off, 0mA, 0sccm) rather than blank, so a fresh row is never
# accidentally left in an energized state until edited.
# ----------------------------------------------------------------------

RECIPE_STEP_TYPE = "Recipe"  # the special type that calls a sub-recipe

# Same metal list as main_gui.py's own METAL_OPTIONS (kept as an
# independent copy here, not an import, to avoid a GUI->model import
# cycle -- recipe_builder_model.py has no other dependency on Qt at all).
MATERIAL_OPTIONS = ["Nb", "Ta", "Re", "Al", "Ti"]

STEP_TYPE_VOCAB = [
    RECIPE_STEP_TYPE,   # call another named recipe in the tree
    "Recipe Tilt Substrate",
    "Position Planetary",
    "Process Chamber",
    "LL Process",
    "Pump Chamber",
    "Pump LL",
    "Substrate Shutter",
    "Source Shutter",
    "Material select",
    "Operator requested",
    "E Gun Emission",
    "Ramp wait",
    "Rate control",
    "Zero Thickness",
    "RecipeThickness",
    "Ar Ion Gun Gas",
    "IBG Discharge",
    "Ion Beam",
    "Static Oxidation",
    "Param Wait",
    "Wait",
    "Comment Only",
]

# step_type -> (param1 hint, param2 hint, param3 hint, param4 hint) --
# shown only for columns that AREN'T a fixed dropdown (see PARAM_CHOICES).
# Each hint gives a subtle indication of what belongs in that field --
# not a predetermined value, but a greyed-out placeholder word naming
# the parameter (e.g. theta for the planetary rotation) and its units.
# These strings are rendered as real, greyed-out QLineEdit
# placeholder text (see _recipe_builder_param_cell in main_gui.py)
# on every free-text Parameter column -- they disappear the instant the
# user types a real value and reappear if the field is ever cleared, so
# they can never be mistaken for an actual saved value the way a fake
# prefilled number could be. theta (rotation) is the only per-row angle
# that exists in this table at all -- the real machine's tilt/alpha
# magnitude is a Recipe-level setting, not typed on any row here (see
# "Recipe Tilt Substrate" below), so there is deliberately no "alpha"
# placeholder anywhere in this dict.
STEP_TYPE_PARAM_HINTS: Dict[str, Tuple[str, str, str, str]] = {
    RECIPE_STEP_TYPE: ("recipe path, e.g. Evap\\Ti", "call value 1 (e.g. thick?)",
                        "call value 2 (e.g. tilt?)", ""),
    "Recipe Tilt Substrate": ("", "", "", ""),
    "Position Planetary": ("", "θ -- rotation, deg (e.g. 45.0°)", "", ""),
    "Process Chamber": ("", "timeout, m:s", "", ""),
    "LL Process": ("", "timeout, m:s", "", ""),
    "Pump Chamber": ("", "", "", ""),
    "Pump LL": ("", "", "", ""),
    "Substrate Shutter": ("", "", "", ""),
    "Source Shutter": ("", "", "", ""),
    "Material select": ("", "", "", ""),
    "Operator requested": ("message to display", "", "", ""),
    "E Gun Emission": ("target current, mA", "ramp time, m:s", "", ""),
    "Ramp wait": ("", "", "", ""),
    "Rate control": ("deposition rate, nm/s", "", "", ""),
    "Zero Thickness": ("", "", "", ""),
    "RecipeThickness": ("", "", "", ""),
    "Ar Ion Gun Gas": ("gas flow, sccm (0 to turn off)", "", "", ""),
    "IBG Discharge": ("", "", "", ""),
    "Ion Beam": ("V= volts, or Off", "I= mA", "Vacc= volts", ""),
    "Static Oxidation": ("pressure, mBar", "gas, e.g. Static O2", "", ""),
    "Param Wait": ("duration, m:s or minutes", "", "", ""),
    "Wait": ("duration, m:s or minutes", "", "", ""),
    "Comment Only": ("", "", "", ""),
}

# step_type -> {param column (1-4): [real, confirmed choices]}. Rendered
# as an editable QComboBox in the GUI -- the listed values are a
# convenience, not an enforced schema.
STEP_TYPE_PARAM_CHOICES: Dict[str, Dict[int, List[str]]] = {
    "Recipe Tilt Substrate": {1: ["Deposit", "Etch", "Load"]},
    "Position Planetary": {1: ["Zero"]},
    "Process Chamber": {1: ["1.0e-07mbar"]},
    "LL Process": {1: ["5.0e-06mbar", "5.0e-07mbar"]},
    "Substrate Shutter": {1: ["EGun", "Ion Gun"], 2: ["Open", "Close"]},
    "Source Shutter": {1: ["Open", "Close"]},
    "Material select": {1: list(MATERIAL_OPTIONS)},
    "Ramp wait": {1: ["All"]},
    "IBG Discharge": {1: ["On", "Off"]},
}

# step_type -> the Comment column's auto-filled text (see note above).
STEP_TYPE_DEFAULT_COMMENT: Dict[str, str] = {
    RECIPE_STEP_TYPE: "Call another recipe",
    "Recipe Tilt Substrate": "Substrate position in the Recipe",
    "Position Planetary": "Position the planetary",
    "Process Chamber": "Process chamber",
    "LL Process": "LL process",
    "Pump Chamber": "Pump the chamber",
    "Pump LL": "Pump the load lock",
    "Substrate Shutter": "Move Substrate Shutter",
    "Source Shutter": "Move Source Shutter",
    "Material select": "E gun Crucible selection",
    "Operator requested": "Display a message alert",
    "E Gun Emission": "Set the emission current",
    "Ramp wait": "Wait for ramp to be finished before next step",
    "Rate control": "Set the evaporation rate",
    "Zero Thickness": "Set the Thickness at zero",
    "RecipeThickness": "Thickness value in the Recipe",
    "Ar Ion Gun Gas": "Set the Ar ion-gun gas flow",
    "IBG Discharge": "Turn the IBG discharge on/off",
    "Ion Beam": "Strike or stop the ion beam",
    "Static Oxidation": "Set the static oxidation pressure",
    "Param Wait": "Wait this time before the next step",
    "Wait": "Wait this time before the next step",
    "Comment Only": "",  # the one exception -- free-form, user-authored
}

# step_type -> the safe/neutral (param1, param2, param3, param4) a
# brand-new row of that type starts with. A DROPDOWN column (see
# PARAM_CHOICES above) still gets a real, meaningful default --
# Close/Off/a confirmed pressure/"Zero" are genuine safe machine states,
# not guesses, so prefilling them is correct. A FREE-TEXT numeric/message
# column with no real "safe" value of its own (a target current, a ramp
# time, a rotation angle, a rate, a gas flow, a duration, a message)
# previously got a made-up placeholder-ish number instead (e.g. "0mA",
# "0:10", "0.0°") that looked exactly like a real, deliberately-chosen
# value -- indistinguishable from something actually typed in by hand.
# Those are blanked to "" here so the greyed-out STEP_TYPE_PARAM_HINTS placeholder
# text shows through instead (see _recipe_builder_param_cell in
# main_gui.py); "Ion Beam"'s Parameter 1 = "Off" is the one
# exception kept as a real default rather than blanked, since "Off" is
# itself the literal, confirmed real machine value for a struck-down
# beam (see APP_REFERENCE's "Ion Beam Off -- extinguish the beam"), not
# a guessed number.
STEP_TYPE_DEFAULT_PARAMS: Dict[str, Tuple[str, str, str, str]] = {
    RECIPE_STEP_TYPE: ("", "", "", ""),
    "Recipe Tilt Substrate": ("Deposit", "", "", ""),
    "Position Planetary": ("Zero", "", "", ""),
    "Process Chamber": ("1.0e-07mbar", "", "", ""),
    "LL Process": ("5.0e-06mbar", "", "", ""),
    "Pump Chamber": ("", "", "", ""),
    "Pump LL": ("", "", "", ""),
    "Substrate Shutter": ("EGun", "Close", "", ""),
    "Source Shutter": ("Close", "", "", ""),
    "Material select": ("Al", "", "", ""),
    "Operator requested": ("", "", "", ""),
    "E Gun Emission": ("", "", "", ""),
    "Ramp wait": ("All", "", "", ""),
    "Rate control": ("", "", "", ""),
    "Zero Thickness": ("", "", "", ""),
    "RecipeThickness": ("", "", "", ""),
    "Ar Ion Gun Gas": ("", "", "", ""),
    "IBG Discharge": ("Off", "", "", ""),
    "Ion Beam": ("Off", "", "", ""),
    "Static Oxidation": ("", "", "", ""),
    "Param Wait": ("", "", "", ""),
    "Wait": ("", "", "", ""),
    "Comment Only": ("", "", "", ""),
}


@dataclass
class PlassysStepRow:
    """One row of the real recipe editor's Step Type / Parameter 1-4 /
    Comment grid. All fields are plain strings -- exactly like the
    real tool's own text-entry grid -- rather than typed numerics,
    since a Step Type's parameters mean completely different things
    depending on which Step Type it is (see STEP_TYPE_PARAM_HINTS),
    and forcing one typed schema across all of them would either
    reject valid real-world text (angles with a leading '+', 'On'/
    'Off', material names) or need a parallel per-type schema the
    real tool doesn't have either."""
    step_type: str = "Wait"
    param1: str = ""
    param2: str = ""
    param3: str = ""
    param4: str = ""
    comment: str = ""

    def line_text(self) -> str:
        """One literal grid line, close to how the tool's own editor
        reads left to right -- used for the plain-text export."""
        params = [p for p in (self.param1, self.param2, self.param3, self.param4) if p]
        out = self.step_type
        if params:
            out += "  " + ", ".join(params)
        if self.comment:
            out += f"   # {self.comment}"
        return out


# ----------------------------------------------------------------------
# One-time migration of a step saved under an earlier session's
# approximate step-type vocabulary (before it was corrected against a
# clear, direct photo of the real tool's editor -- see the STEP_TYPE_
# VOCAB rename table above) to the exact real names/structure. Applied
# automatically in PlassysRecipeNode.from_dict (so both load_tree() and
# duplicate_child pick it up) -- an already-in-progress recipe built
# before this correction keeps working with the Recipe Warnings engine
# and the exact-name dropdown with zero data loss and no action needed
# from the user. Idempotent: a step already using the new names simply
# matches nothing here and passes through unchanged.
# ----------------------------------------------------------------------

_SIMPLE_STEP_RENAME: Dict[str, str] = {
    "Substrate Position": "Recipe Tilt Substrate",
    "Move Planetary": "Position Planetary",
    "Process LL": "LL Process",
    "Shutter Source": "Source Shutter",
    "Crucible": "Material select",
    "Egun Emission Ramp": "E Gun Emission",
    "Ramp Wait": "Ramp wait",
    "Rate Control": "Rate control",
    "Wait for Termination": "RecipeThickness",
}


def _migrate_step_row(row: PlassysStepRow) -> PlassysStepRow:
    st = row.step_type
    if st == "Move Table":
        # Absorbed into "Recipe Tilt Substrate" -- the old destination
        # (param1, e.g. "Deposit") is kept; the old literal tilt-angle
        # param2 is dropped, since the real tool has no such per-row
        # parameter here (the tilt magnitude is a Recipe-level value).
        row.step_type = "Recipe Tilt Substrate"
        row.param2 = ""
    elif st == "Substrate Shutter EBGun":
        row.step_type = "Substrate Shutter"
        row.param1, row.param2 = "EGun", row.param1
    elif st == "Substrate Shutter Ion Gun":
        row.step_type = "Substrate Shutter"
        row.param1, row.param2 = "Ion Gun", row.param1
    elif st in _SIMPLE_STEP_RENAME:
        row.step_type = _SIMPLE_STEP_RENAME[st]
    return row


# ----------------------------------------------------------------------
# Node-name validation -- a folder or recipe name typed with a literal
# backslash in it (e.g. a recipe named
# "Aluminum\\myproject - 0.5 nm/s Al Manhattan 1st + 0") silently corrupts
# every path-based operation on that node, forever, because PATH_SEP
# ("\\") is ALSO this tree's own path separator: resolve_path (below)
# splits a path string back into segments on that exact character, so a
# name containing it gets misread as several nested segments instead of
# one real node -- resolve_path then looks for a child matching just the
# first segment, doesn't find one, and returns None. This is exactly why
# clicking that one recipe in the tree silently did nothing (the click
# handler calls resolve_path, gets None, and bails out) while every
# other recipe -- with no backslash in its name -- opened normally: it
# had nothing to do with which folder the recipe lived in or the order
# things were clicked in, only with that one recipe's own corrupted name.
# ----------------------------------------------------------------------

def _check_name_has_no_separator(name: str) -> None:
    if PATH_SEP in name:
        raise ValueError(
            f'A name can\'t contain "{PATH_SEP}" -- that character is this recipe tree\'s own '
            "path separator, so it would silently break opening this item, renaming it, and any "
            '"Recipe" step elsewhere that calls it by path. Use something else instead (a space '
            "or a dash both work fine).")


def _sanitize_node_name(name: str) -> str:
    """Heals a name that was saved before add_child/rename_child/
    duplicate_child validated against an embedded PATH_SEP (see above) --
    runs on every load (from_dict below), the same idempotent auto-heal
    pattern _migrate_step_row already uses for old step-type vocabulary:
    an already-clean name passes through completely unchanged, while a
    name saved with the bug already baked in gets its separator character
    replaced so the node becomes openable/resolvable again without the
    user having to notice and manually rename it."""
    return name.replace(PATH_SEP, "-") if PATH_SEP in name else name


@dataclass
class PlassysRecipeNode:
    """One node in the recipe tree -- either a pure organizational
    FOLDER (is_folder=True, no steps of its own, e.g. "Evap" or "New
    Ion Gun Recipes") or an actual named RECIPE (is_folder=False, a
    real ordered list of PlassysStepRow, e.g. "Ti" under "Evap").
    call_params names the placeholder values THIS recipe expects to be
    filled in by whoever calls it as a sub-recipe (e.g. ["thick?",
    "tilt?"] for a deposition recipe whose own rate is fixed by its
    name/definition but whose thickness/tilt vary per use) -- purely
    documentation for whoever authors a "Recipe" step that calls this
    one; nothing here substitutes the values automatically, since that
    substitution happens for real only on the tool itself."""
    name: str
    is_folder: bool = False
    steps: List[PlassysStepRow] = field(default_factory=list)
    call_params: List[str] = field(default_factory=list)
    children: Dict[str, "PlassysRecipeNode"] = field(default_factory=dict)

    def child_names(self) -> List[str]:
        return sorted(self.children.keys(), key=str.lower)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "is_folder": self.is_folder,
            "steps": [asdict(s) for s in self.steps],
            "call_params": list(self.call_params),
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "PlassysRecipeNode":
        known_step_fields = {f.name for f in dataclasses.fields(PlassysStepRow)}
        steps = [_migrate_step_row(PlassysStepRow(**{k: v for k, v in s.items() if k in known_step_fields}))
                 for s in (d.get("steps") or [])]
        # Sanitize the dict KEY the same way as the child's own .name
        # (see _sanitize_node_name above) -- these must always match,
        # since every add_child/rename_child call keeps a node's dict
        # key and its own .name in lockstep; sanitizing only one of the
        # two here would leave them silently mismatched.
        children = {_sanitize_node_name(k): PlassysRecipeNode.from_dict(v)
                    for k, v in (d.get("children") or {}).items()}
        return PlassysRecipeNode(
            name=_sanitize_node_name(d.get("name", "Untitled")),
            is_folder=bool(d.get("is_folder", False)),
            steps=steps,
            call_params=list(d.get("call_params") or []),
            children=children,
        )


def default_tree() -> PlassysRecipeNode:
    """A fresh tree with just the standard top-level folders seen on
    the real tool (Evap under Root, Etching\\New Ion Gun Recipes) --
    matches the paths already referenced literally by the deterministic
    generator's own recipe-tree text (recipe_generator.py's
    _mill_detail/_deposition_ramp_detail: "Root\\Evap\\...",
    "Root\\Etching\\New Ion Gun Recipes\\..."), so a from-scratch
    recipe naturally lands in the same places a generated one's own
    text already refers to. Purely a starting convenience -- every
    folder/recipe can be renamed, moved, or deleted freely afterward.

    Also seeds a top-level "Complete Process Recipes" folder, giving a
    place to save a recipe table for reuse across multiple projects,
    alongside the Evap and Etching folders. A "complete process"
    recipe is just an ordinary recipe leaf whose own steps are mostly
    "Recipe" calls chaining an Evap sub-recipe and an Etching sub-recipe
    together (the nested-call mechanic already supports this) -- this
    folder exists purely to keep those whole, reusable, known-good
    process recipes organized separately from their Evap/Etching
    building blocks, matching a typical workflow: pull up an
    old complete process recipe that's known to work, duplicate it (see
    duplicate_child below), and adjust angles/thickness/times for the
    new project."""
    root = PlassysRecipeNode(name=ROOT_NAME, is_folder=True)
    evap = PlassysRecipeNode(name="Evap", is_folder=True)
    etching = PlassysRecipeNode(name="Etching", is_folder=True)
    new_ion_gun = PlassysRecipeNode(name="New Ion Gun Recipes", is_folder=True)
    etching.children["New Ion Gun Recipes"] = new_ion_gun
    complete_process = PlassysRecipeNode(name="Complete Process Recipes", is_folder=True)
    root.children["Evap"] = evap
    root.children["Etching"] = etching
    root.children["Complete Process Recipes"] = complete_process
    return root


# ----------------------------------------------------------------------
# Path helpers -- "Root\\Evap\\Ti" style paths, the same separator/
# backslash convention the deterministic generator's own literal
# recipe text already uses (recipe_generator.py:
# f"Root\\Evap\\{p.metal_type}\\...").
# ----------------------------------------------------------------------

def split_path(path: str) -> List[str]:
    return [p for p in path.split(PATH_SEP) if p]


def join_path(parts: List[str]) -> str:
    return PATH_SEP.join(parts)


def resolve_path(root: PlassysRecipeNode, path: str) -> Optional[PlassysRecipeNode]:
    """Walks `path` (e.g. "Root\\Evap\\Ti" or "Evap\\Ti", either with or
    without a leading "Root") from `root`. Returns None if any segment
    doesn't exist."""
    parts = split_path(path)
    if parts and parts[0] == root.name:
        parts = parts[1:]
    node = root
    for part in parts:
        node = node.children.get(part)
        if node is None:
            return None
    return node


def ensure_folder_path(root: PlassysRecipeNode, path: str) -> PlassysRecipeNode:
    """Like resolve_path, but creates any missing folder segments along
    the way. Never creates the final segment as a real recipe -- add
    that leaf explicitly via add_child(..., is_folder=False)."""
    parts = split_path(path)
    if parts and parts[0] == root.name:
        parts = parts[1:]
    node = root
    for part in parts:
        child = node.children.get(part)
        if child is None:
            child = PlassysRecipeNode(name=part, is_folder=True)
            node.children[part] = child
        node = child
    return node


def node_path(root: PlassysRecipeNode, target: PlassysRecipeNode) -> Optional[str]:
    """Finds `target`'s full path from `root` by identity search (the
    tree is small -- a handful of named recipes per lab -- so a plain
    DFS is plenty fast and avoids every node needing its own parent
    pointer)."""
    if target is root:
        return root.name

    def _walk(node, trail):
        for name, child in node.children.items():
            new_trail = trail + [name]
            if child is target:
                return new_trail
            found = _walk(child, new_trail)
            if found is not None:
                return found
        return None

    found = _walk(root, [root.name])
    return join_path(found) if found is not None else None


def add_child(parent: PlassysRecipeNode, name: str, *, is_folder: bool) -> PlassysRecipeNode:
    name = name.strip()
    if not name:
        raise ValueError("Name can't be empty.")
    _check_name_has_no_separator(name)
    if name in parent.children:
        raise ValueError(f'"{name}" already exists here.')
    node = PlassysRecipeNode(name=name, is_folder=is_folder)
    parent.children[name] = node
    return node


def remove_child(parent: PlassysRecipeNode, name: str) -> None:
    parent.children.pop(name, None)


def duplicate_child(parent: PlassysRecipeNode, name: str, new_name: str) -> PlassysRecipeNode:
    """Deep-copies parent.children[name] (steps, call_params, and every
    descendant if it's a folder) under new_name in the same parent, so a
    recipe table that's known to work can be reused across multiple
    projects: pull up an old known-good recipe, duplicate it, then tweak
    angles/thickness/times on the COPY for a new project -- leaving the
    original reusable template untouched. Raises ValueError if `name`
    doesn't exist or `new_name` is already taken (same failure mode as
    add_child/rename_child)."""
    if name not in parent.children:
        raise ValueError(f'"{name}" doesn\'t exist here.')
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("Name can't be empty.")
    _check_name_has_no_separator(new_name)
    if new_name in parent.children:
        raise ValueError(f'"{new_name}" already exists here.')
    original = parent.children[name]
    copy = PlassysRecipeNode.from_dict(original.to_dict())
    copy.name = new_name
    parent.children[new_name] = copy
    return copy


def rename_child(parent: PlassysRecipeNode, old_name: str, new_name: str) -> None:
    new_name = new_name.strip()
    if not new_name or new_name == old_name:
        return
    _check_name_has_no_separator(new_name)
    if new_name in parent.children:
        raise ValueError(f'"{new_name}" already exists here.')
    node = parent.children.pop(old_name)
    node.name = new_name
    parent.children[new_name] = node


def _is_node_or_descendant(ancestor: PlassysRecipeNode, node: PlassysRecipeNode) -> bool:
    """True if `node` is `ancestor` itself, or somewhere inside
    ancestor's own subtree -- used by move_child below to reject a move
    that would disconnect a folder from the tree by nesting it under
    one of its own children (or under itself)."""
    if node is ancestor:
        return True
    return any(_is_node_or_descendant(child, node) for child in ancestor.children.values())


def move_child(source_parent: PlassysRecipeNode, name: str, dest_folder: PlassysRecipeNode) -> PlassysRecipeNode:
    """Moves source_parent.children[name] -- a folder OR a recipe, with
    every step/sub-recipe underneath it completely intact -- to become
    a child of dest_folder instead. Lets a recipe or folder created in
    the wrong place (e.g. an Evap recipe accidentally created under
    Etching) be relocated to the correct folder without losing its
    steps. Used by both the tree toolbar's "Move..." dialog and dragging an item
    onto a folder in the tree (see _RecipeTreeWidget in
    main_gui.py). A plain reparent -- nothing about the moved
    node itself (its steps, call_params, or, if it's a folder, anything
    underneath it) is touched, unlike duplicate_child, which deep-copies.

    Raises ValueError (same failure style as add_child/rename_child/
    duplicate_child) if: `name` doesn't exist in source_parent;
    dest_folder isn't actually a folder (a recipe leaf has no children
    to move anything into); dest_folder is the node being moved, or is
    somewhere inside that node's own subtree (either would disconnect
    it from the tree -- a cycle); dest_folder is already the node's
    current parent (a no-op, surfaced rather than silently doing
    nothing, so a caller doesn't rebuild/reopen anything for no reason);
    or a node with the same name already exists at the destination."""
    if name not in source_parent.children:
        raise ValueError(f'"{name}" doesn\'t exist here.')
    if not dest_folder.is_folder:
        raise ValueError(f'"{dest_folder.name}" isn\'t a folder -- pick a folder to move "{name}" into.')
    node = source_parent.children[name]
    if dest_folder is source_parent:
        raise ValueError(f'"{name}" is already in "{dest_folder.name}".')
    if _is_node_or_descendant(node, dest_folder):
        raise ValueError(f'Can\'t move "{name}" into itself or one of its own subfolders.')
    if name in dest_folder.children:
        raise ValueError(f'"{name}" already exists in "{dest_folder.name}".')
    del source_parent.children[name]
    dest_folder.children[name] = node
    return node


def move_step(node: PlassysRecipeNode, index: int, delta: int) -> int:
    """Moves node.steps[index] up (delta=-1) or down (delta=+1) by one
    position. Returns the step's new index (unchanged if the move
    would go out of bounds)."""
    if index < 0 or index >= len(node.steps):
        return index
    new_index = index + delta
    if new_index < 0 or new_index >= len(node.steps):
        return index
    node.steps[index], node.steps[new_index] = node.steps[new_index], node.steps[index]
    return new_index


# ----------------------------------------------------------------------
# Storage location -- same portable, no-hardcoded-path convention as
# design_library.py, but a SINGLE file for the whole tree (not one
# file per recipe): a "Recipe" step refers to another recipe BY PATH,
# so paths must resolve directly against one shared tree rather than
# through a separate name/id lookup layer.
# ----------------------------------------------------------------------

def _app_data_dir() -> str:
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return override
    try:
        from PySide6.QtCore import QCoreApplication, QStandardPaths
        if not QCoreApplication.organizationName():
            QCoreApplication.setOrganizationName("PlassysShadowSim")
        if not QCoreApplication.applicationName():
            QCoreApplication.setApplicationName("PlassysShadowSim")
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if base:
            return base
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), ".plassys_shadow_sim")


def _tree_path() -> str:
    return os.path.join(_app_data_dir(), "recipe_tree.json")


def save_tree(root: PlassysRecipeNode, path: Optional[str] = None) -> str:
    path = path or _tree_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "root": root.to_dict()}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)  # atomic-ish: never leaves a half-written .json behind
    return path


def load_tree(path: Optional[str] = None) -> PlassysRecipeNode:
    """Loads the saved tree, or returns a fresh default_tree() if
    nothing's been saved yet (or the file is missing/corrupt -- a
    corrupt tree file shouldn't ever crash the app, just start fresh,
    the same defensive stance design_library.py's list_designs takes
    against a corrupt/foreign file)."""
    path = path or _tree_path()
    try:
        with open(path) as f:
            payload = json.load(f)
        return PlassysRecipeNode.from_dict(payload["root"])
    except Exception:
        return default_tree()
