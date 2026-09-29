"""
ai_assistant.py

Wires a vision-capable LLM (Qwen2.5-VL, served locally via Ollama -- free,
no API key, runs entirely on your own machine) into the app as an
app-wide, open-ended EXPLANATORY assistant ("Ask AI") -- answering
free-text questions about the app, the current design, or the underlying
nanofabrication physics, grounded in whatever live app state the GUI
gathers (see main_gui.py's _gather_app_context) plus the condensed
reference material in APP_REFERENCE below.

By design, this module never generates or reviews recipe drafts. The
deterministic recipe generator (recipe_generator.py) produces the exact
recipe from the design's own computed geometry with zero hallucination
risk, and the LLM is reserved for a purely explanatory role it is better
suited to. The deterministic generator remains the sole recipe authority;
this module's only job is answer_question() -- explaining, never
generating or editing a recipe.

answer_question() is a synchronous, network-calling function -- a caller
running a GUI event loop should invoke it from a background thread and
marshal the result back to the main thread (see main_gui.py's
_on_ask_ai_submit for the reference implementation of that, including
live-streaming the response as it's generated).

Requires:  1) Ollama itself installed and running locally (this is the
              app, NOT a pip package -- download from https://ollama.com
              or `brew install ollama`, then run `ollama serve` once, or
              just launch the Ollama app which serves in the background).
           2) The model weights pulled once:
                  ollama pull qwen2.5vl:7b
              (use qwen2.5vl:3b instead if you're on a lower-RAM Mac or
              an Intel Mac and 7b feels slow -- just change DEFAULT_MODEL
              below, no code changes needed).
           3) The Python client: pip install ollama

Auth:      None. Everything runs on localhost, no API key, no network
           calls, no per-request cost. Optionally override the host via
           the OLLAMA_HOST environment variable if Ollama is running
           somewhere other than the default http://localhost:11434.

This module has NO tkinter/Qt dependency and can be unit-tested / used
from a script on its own.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from recipe_generator import DesignParameters, RecipeStep

DEFAULT_MODEL = "qwen2.5vl:7b"

# Ollama's image handling is forgiving about format, but we still convert
# anything exotic (HEIC/HEIF from iPhone photos) to JPEG for reliability.
_SUPPORTED_IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Reading a reference photo off disk should be near-instant -- these are
# local files. If it takes anywhere close to this long, the file almost
# certainly isn't actually local: the most common cause is a cloud-sync
# placeholder (iCloud "Optimize Mac Storage" evicting files from ANY
# synced folder -- iCloud Drive, or Desktop/Documents if that sync is on
# -- back to cloud-only after they were downloaded once). Opening an
# evicted placeholder doesn't error, it just blocks silently, potentially
# forever, with nothing to show for it in `ollama ps` since the request
# never even reaches Ollama. This timeout turns that into a clear, timely
# error per-image instead of an indefinite whole-app hang.
IMAGE_ENCODE_TIMEOUT_SECONDS = 10.0


# ======================================================================
# The ASK assistant -- an open-ended, app-wide question-answering helper.
# This is NEVER asked to produce or judge recipe steps -- recipe
# generation is the deterministic engine's job alone (recipe_generator.py).
# ASK answers "why"/"what if"/"where do I find X" questions about the
# design, the physics, or the app itself, using whatever live app state
# the GUI hands it as context (current tab, design parameters, computed
# geometry/warnings, the generated recipe if any) plus the condensed
# reference material below -- an assistant across the whole app that can
# answer process/parameter questions the way a "Help" section would for a
# new user. Recipe generation stays the deterministic engine's job alone:
# real use showed it already produces the recipe correctly from the
# design's own computed geometry, while the LLM does consistently better
# in this purely explanatory role than at generating or judging recipe
# content itself.
# ======================================================================

# Condensed from shadowing_design_rules.md (the project's own working
# notes on shadowing physics and tab layout) -- NOT the full document
# (which is mostly a bug-fix history, not useful to hand a model on every
# question). This is the stable, load-bearing physics/architecture
# content: the actual formulas, conventions, and what each tab is for.
# Update this block if the underlying rules change; it does not need to
# track the bug-history narrative in the full doc.
APP_REFERENCE = """
TAB LAYOUT (what each tab is for, so "where do I do X" has a real answer):
  Front Page -- landing page / shortcut hub, not a design tab.
  Create/Import -- pure geometry: design or import a junction, classify
    shapes, get automatic measurements. No warnings, no angle
    recommendations, no shadowing math here.
  Simulate Design -- recommended deposition angles (from the
    deterministic recommendation engine) plus a full shadow simulation
    of the finished junction at whatever angles are currently active,
    live-overridable, with a sub-panel showing the full deterministic
    calculation log (every formula, every number, traceable). Also has
    "Run Angle Diagnostic" (required before a recipe can be generated)
    and a Wafer Orientation Preview (a read-only live chamber-position
    tester, does not write to the recipe).
  Recipe -- has TWO mode pills at the top, "Auto-Generated (from Design)"
    (the default) and "Recipe Builder (from scratch)", picking which LEFT
    column shows (Recipe Parameters + Generate button, vs. the Recipe
    Library tree/CRUD toolbar) -- but NOT two separate recipe formats or
    two separate displays. Both modes share ONE tree/table/breadcrumb/
    warnings panel on the right (always visible, regardless of which mode
    pill is selected), and both ultimately produce/edit the SAME kind of
    real, directly-editable Plassys recipe-tree entries -- see the
    dedicated RECIPE BUILDER section below for exactly how that shared
    panel works (tree, step table, the "Recipe" step type, warnings,
    design-link banner) and the AUTO-GENERATED RECIPE MATERIALIZATION
    subsection right after it for what's specific to "Auto-Generated
    (from Design)" (the Recipe Parameters override panel, the Generate
    Recipe button's dedup/reuse/duplicate-and-patch behavior, the
    Complete Process derived-summary table, and the sub-recipe naming
    convention). There is no AI-generated or AI-reviewed recipe option
    either way -- this assistant is explanatory only and never writes to
    the recipe. Do not describe either mode using generic vacuum-
    deposition-software conventions (there is no "Category" field, no
    literal "Alpha"/"Theta" input boxes, and no such thing here as a step
    called "Ti Gather (Lower Pressure)") -- if a question is about the
    Recipe tab, its real mechanics are all below and nowhere else. (There
    used to be a separate rendered-text "Expanded Recipe" view here with
    double-click-to-edit underlined values -- that was REMOVED when
    Auto-Generated recipes started opening directly into the same real
    tree/table the Recipe Builder always used; if asked about "the
    generated recipe text" or "underlined values," that mechanism no
    longer exists on screen.)
  Deposition Slideshow -- steps through the actual generated recipe one
    step at a time, showing the real chamber orientation for that step
    (shared with Settings > Chamber Reference Model) and that step's
    full process detail.
  Measure Resistance -- voice-driven room-temperature resistance capture
    for the fabricated devices.
  Settings -- Resist Stack & Calculator (alpha/linewidth calculator,
    PMMA/PMGI thickness, safety margin) and the Chamber Reference Model
    page. The Design Library (saved designs) is reached from here too.
    Settings no longer has a "Process Parameters" section -- metal type,
    electrode metal, project name, and oxidation time now live on the
    Recipe tab's own "Recipe Parameters" panel (above the Generate Recipe
    button), and deposition tilt (alpha) is Simulate Design's "Active
    alpha" field. Ion mill beam voltage/current/gas/accel, deposition
    rate, and Ti-gather rate/duration are no longer user-configurable
    defaults at all -- they're fixed values the recipe generator uses,
    changeable only by hand-editing the generated recipe's own step
    cells afterward (that hand-edit is what persists across a fresh
    regeneration).

DEPOSITION ANGLE CONVENTION -- READ THIS CAREFULLY, a real past mistake
was mixing these two up:
  alpha = wafer TILT angle, degrees -- how far the wafer is tipped away
    from facing the source straight-on. This is the ONLY angle that
    determines SHADOW MAGNITUDE (how far a shadow reaches, in nm) --
    every tan(...) in the SHADOW-EVAPORATION PHYSICS section below takes
    alpha, and ONLY alpha. alpha=0 in the "mill reference" is the
    ion-mill position (wafer vertical, facing the mill); alpha=90 in
    "load reference" is the load position (wafer flat). Deposition
    angles are moderate tilts from the mill position, chosen so
    evaporated material reaches the resist trench at the correct
    undercut for a given linewidth/resist thickness.
  theta = planetary (in-plane) ROTATION, degrees, clockwise-positive
    viewed from above. theta determines DIRECTION ONLY -- which way
    (which electrode, which finger end, which side of the wafer) the
    tilted deposition or mill is aimed at. theta is NEVER an argument to
    tan() and never appears in a shadow-magnitude formula at all -- if
    you catch yourself about to write "tan(theta)" or plug a theta value
    into a shadow-magnitude calculation, STOP: that is always wrong, not
    a simplification. A question like "why theta=270 and not theta=90"
    is a DIRECTION question (does the deposit/mill land on the correct
    finger/electrode, or reach the wrong one / miss entirely), never a
    magnitude-formula question.
  For a SPECIFIC finger's correct theta: do not guess or derive it from
  first principles. The app's own rotation-math engine already computed
  it exactly, per finger, and hands it to you in
  app_context.junction_geometry.fingers[i] as theta_toward_patch_deg
  (the rotation that aims deposition/mill toward that finger's
  electrode-facing end -- normally the value used so the finger's metal
  actually reaches its bond pad), theta_toward_stub_deg (the exact
  opposite direction, 180 degrees from theta_toward_patch_deg -- usually
  the SAFER self-shadow direction for whichever finger deposits second,
  see self_shadow_safe_theta_deg), and self_shadow_safe_theta_deg
  (set only when deposited_second is true). app_context.last_committed_angles
  has the actual alpha/theta_a/theta_b values committed for the current
  design. If a user asks "why theta=X and not Y" for a finger, answer by
  citing THESE exact values and explaining what changing theta would
  physically re-aim (which electrode/end the material would land toward,
  or that it would point the wrong direction relative to this finger's
  own geometry) -- do not invent a tan()-based justification for a theta
  choice; theta's effect on outcome is about aim/direction, not
  trig-formula magnitude. Depositing the "other" finger of a Manhattan
  junction is often theta+90 or theta+180 from the first; patch/overlap
  sub-steps are often the parent step's theta plus 180 -- these are
  typical patterns for reading an EXISTING recipe, not formulas to
  compute a specific design's real values from scratch.

SHADOW-EVAPORATION PHYSICS (bilayer resist, PMMA over PMGI or similar) --
every formula below takes alpha ONLY; theta never appears in any of them:
  Outward shadow (dy) = t_bottom_layer * tan(alpha)
  Inward shadow  (dx) = h * tan(alpha)          [h = t_top_layer + t_bottom_layer]
  Equivalently tan(alpha) = LW / (t_top_layer - 100), where LW is the
  target linewidth in nm and the -100 accounts for the top layer's own
  100nm PMMA-clearance margin. A deposited feature's overlap with
  another feature shrinks by dx on the side facing "into" the shadow and
  extends by dy on the side facing "away" from it. A chosen overlap only
  survives if chosen_overlap_nm - dx_nm > 0 -- otherwise the two features
  can lose contact entirely (an open junction).
  Electrode-topography shadow (a SEPARATE mechanism from the resist-
  stack shadow above -- the electrode's own physical height, not the
  resist): shadow_nm = electrode_height_nm * tan(alpha_deg). This is
  what causes an electrode-side disconnect if a deposition pass happens
  at the wrong TILT (alpha) relative to the electrode -- WHICH electrode
  it's even pointed at is a theta/direction question, answered from
  junction_geometry above, not from this formula.
  Self-shadowing (the second-deposited finger crossing over the first
  finger's own already-deposited step, at the tunnel-junction crossing
  itself): the safe direction is toward the crossing/stub end; risky is
  toward the patch end -- the OPPOSITE convention from the electrode-
  crossing case above, since it's a different piece of topography. Which
  direction that actually IS for a given finger is theta_toward_stub_deg
  vs theta_toward_patch_deg from junction_geometry, not something to
  reason out from the tilt-magnitude formulas.
  Multi-pass cascade: when a feature is deposited multiple times with no
  oxidation between passes (still one continuous metal layer), each end's
  final boundary is generally whichever single pass extended it
  furthest (net = max of the signed depth per pass) -- EXCEPT for a
  patch end where pass 1 shrinks it and pass 2 (in true deposition
  order) extends it: there the net is cumulative (net = signed1 +
  signed2, measured from pass 1's already-retreated edge), because pass
  2's shadow redefines where new metal starts rather than erasing metal
  pass 1 already deposited -- a later pass's shadow only affects new
  metal going forward, never metal already laid down, regardless of order.
  Sub-100nm linewidth: any shape under 100nm linewidth gets a
  manufacturability warning (potential uneven deposition -- opens or
  higher-than-expected resistance) -- threshold-only, no automatic fix.
  Perpendicularity: junction fingers must be perpendicular to each
  other; for one finger's typed angle A, the other must be A+90 or
  A+270 -- again a DIRECTION relationship (theta), unrelated to the
  tan(alpha) shadow-magnitude formulas above.

WHAT THE APP ALREADY CHECKS AUTOMATICALLY (so "is my design okay" can be
answered from real computed warnings, not guessed): a piece (finger,
adhesion plate, patch) not attached in the base design; a design where
the electrical path doesn't reach both electrodes at all ("Junction is
open"); a finger/plate/patch losing contact with its own electrode after
shadowing is applied; a finger split into disconnected pieces by its own
shadow; loss of overlap at the JJ crossing. All of these run live as the
design changes, before a recipe is even generated.

DESIGN LIBRARY -- SAVING AND REOPENING A DESIGN (reachable from the File
menu or a Front Page shortcut card, not tied to any one tab): lets
someone save an in-progress design and pick it back up later instead of
losing it if the app is closed or updated. File menu has "Save
Design..." (Ctrl+S -- overwrites whichever saved entry the current
design was last loaded from or saved to, or behaves like Save As if it
was never saved before), "Save Design As New..." (always creates a
brand-new separate saved entry, even if the current design started from
an existing one), and "Open Design Library..." (Ctrl+L -- opens the
browsable catalog). The Front Page has its own "Design Library" card
that opens the same catalog. The catalog lists every saved design
(name, whether it's a hand-designed/parametric one or an imported one,
when it was last saved, a short detail summary, and any notes), with
Load / Rename / Delete buttons and its own "Save Current Design..."
button.
  Saving captures everything needed to pick back up exactly where you
  left off, not just the raw shapes: for a hand-designed (parametric)
  junction, every Design-panel field plus any custom electrode shapes
  drawn by hand on the canvas and the electrode-identification
  convention if one is attached; for an imported junction, every
  finger, patch/plate shape, the electrode geometry, and the import
  notes. Either way, which finger deposits first and any manual color/
  scale overrides are saved too, so a reloaded design looks and behaves
  exactly as it was left -- reloading a hand-designed junction gives
  back a still-live, still-editable design (as if you'd typed the
  values in fresh), while reloading an imported one restores it as a
  static junction, the same as right after the original import.
  Everything is stored locally on the user's own machine, in a normal
  per-app data folder the operating system already sets aside for this
  purpose -- nothing is uploaded anywhere, and no personal file paths
  are baked in, so this works identically for any user on any machine.

RECIPE TREE/TABLE PANEL -- SHARED BY BOTH RECIPE-TAB MODES (this used to
be described as only the "Recipe Builder (from scratch)" screen; it is
now the SAME right-hand tree/breadcrumb/table/warnings panel that
"Auto-Generated (from Design)" also opens its results into -- a real
screen in this app, not a generic tool -- do not answer a "how do I use
the Recipe Builder" or "what does the generated recipe look like"
question with a made-up tutorial; everything below is exactly what is
really on screen, for BOTH modes):
  Recipe Library (tree), left side -- a tree of FOLDERS and NAMED
    RECIPES that is shared across every design/project, not tied to the
    currently-open design at all: this is what lets someone bring up an
    old recipe they know works and reuse or duplicate it for a brand new
    junction shape. Seeded by default with Root > Evap, Root > Etching >
    New Ion Gun Recipes, and Root > Complete Process Recipes, and the
    user can add more folders anywhere. A folder shows a folder icon; a
    recipe (a leaf) does not.
  Tree toolbar (above the tree): New Folder, New Recipe, Rename,
    Duplicate, Delete. New Folder and New Recipe create the new item
    INSIDE whichever folder is currently highlighted/selected in the
    tree (or at the Root if nothing, or a recipe rather than a folder, is
    selected) -- so to add a recipe "under Evap", the Evap folder itself
    must be the thing highlighted in the tree when New Recipe is
    clicked, prompting for just a name. Duplicate deep-copies an existing
    recipe (every step, verbatim) under a new name, leaving the original
    completely untouched -- the intended reuse workflow is: duplicate a
    known-good recipe, then adjust angles/tilt/thickness/times on the
    copy, never edit the reusable original in place.
  Opening a recipe -- clicking a recipe leaf in the tree opens it on the
    right: the breadcrumb bar (with a "◀ Back" button once you have
    navigated into a nested sub-recipe, see "Recipe" step type below)
    shows its full path, and the step table below it becomes editable.
    Clicking a FOLDER only expands/navigates the tree; it never opens a
    step table, since a folder has no steps of its own. Add Step and
    everything in the step-table toolbar act on whichever recipe LEAF is
    actually open/highlighted -- if nothing is open yet, the app says so
    plainly rather than guessing.
  Step table, right side -- one row per literal Plassys step line, with
    exactly these six columns: Step Type, Parameter 1, Parameter 2,
    Parameter 3, Parameter 4, Comment. Step Type is a dropdown restricted
    to the real Plassys vocabulary -- there is no separate "Category"
    field (no "Prep"/"Deposition"/"Mill" grouping control) and no literal
    "Alpha"/"Theta" input boxes; a rotation is simply Parameter 2 on a
    "Position Planetary" step, exactly as the real tool's own editor
    works (the tilt/alpha magnitude for "Recipe Tilt Substrate" is a
    Recipe-level value, not typed on the row at all -- see below).
    This is transcribed directly from a real, complete deposition recipe
    on the physical tool -- this IS the real vocabulary, distinct from
    the separate, older vocabulary the Auto-Generated engine's own
    output text still uses (see the REAL PLASSYS STEP-TYPE VOCABULARY
    section above and its note on the two being different):
      Recipe Tilt Substrate  -- Parameter 1 = destination (Deposit /
        Etch / Load) -- table position; no per-row tilt angle, since the
        real magnitude is a Recipe-level setting.
      Position Planetary     -- Parameter 1 = "Zero" (a fixed
        reference), Parameter 2 = the actual rotation, e.g. "45.0°".
      Process Chamber        -- Parameter 1 = target pressure (defaults
        to "1.0e-07mbar", the real chamber base-vacuum checkpoint),
        Parameter 2 = timeout.
      LL Process             -- Parameter 1 = target pressure -- the
        real tool uses this same step type for BOTH a coarse ("5.0e-06
        mbar") and a tighter, second ("5.0e-07mbar") load-lock check.
      Substrate Shutter      -- Parameter 1 = which shutter (EGun / Ion
        Gun), Parameter 2 = Open/Close -- ONE step type covers both
        sides, distinguished by Parameter 1.
      Source Shutter         -- Parameter 1 = Open/Close -- the shutter
        right at the crucible/e-gun source (a different, separate
        shutter from Substrate Shutter above).
      Material select        -- Parameter 1 = the metal (Nb/Ta/Re/Al/
        Ti) -- selects the crucible.
      Operator requested     -- Parameter 1 = a message to display,
        e.g. "Check the beam position" -- a pause-and-alert step.
      E Gun Emission         -- Parameter 1 = target mA, Parameter 2 =
        m:s ramp time -- used for BOTH ramping up and ramping back down
        to 0mA.
      Ramp wait              -- Parameter 1 = "All" -- waits for the
        ramp just started to finish before the next step.
      Rate control           -- Parameter 1 = nm/s -- locks the
        deposition rate.
      Zero Thickness         -- no parameters -- zeros the crystal
        monitor before a layer.
      RecipeThickness        -- no parameters -- the step that actually
        ENDS a deposition: deposits until the Recipe's own configured
        thickness value is reached.
    Everything the mill/ion-gun side uses (Ar Ion Gun Gas, IBG Discharge,
    Ion Beam) uses this same real machine vocabulary.
  Dropdowns and auto-fill -- entering some step types offers a choice
    between two or three real options via a secondary dropdown, rather
    than requiring the exact word "Open" or "Close" to be typed, and the
    comment for that step fills in automatically. Selecting a Step Type resets Parameter
    1-4 to that type's own safe/neutral defaults (e.g. Close/Off/0mA
    rather than blank) and auto-fills the Comment column, which is
    LOCKED (not user-editable) for every step type except "Comment
    Only" -- that one exists purely so the user can write their own
    free-form note, and stays editable. A handful of Parameter columns
    with a small, real, confirmed set of values (Open/Close; EGun/Ion
    Gun; the Recipe Tilt destination; the metal list; "Zero"; "All") are
    an editable dropdown rather than free text -- still editable, so an
    unusual value can always be typed in directly; everything else
    (angles, currents, times, pressures, free-form messages) stays a
    plain text field, since those are genuinely continuous/free values
    on the real tool too.
  Step-table toolbar: "+ Add Step" appends a new row (pre-filled with
    its default type's safe values and auto-comment, not blank) to
    whichever recipe is currently open; Delete Step removes the selected
    row; Move Up / Move Down reorder it; "Insert Typical Step" is a
    dropdown of ready-made common step blocks (a fast starting point,
    not a required step); "Save Recipe Tree" is an explicit manual save
    -- the whole tree already autosaves to disk after every edit, so
    this button is a convenience, not something that must be clicked to
    avoid losing work; "Export .txt" writes the open recipe out as plain
    text.
  The "Recipe" step type -- a step whose Step Type is literally "Recipe"
    calls another named recipe in the tree as a nested sub-table:
    Parameter 1 is that recipe's path (e.g. "Evap\\Ti"), and
    double-clicking that row navigates INTO the target recipe's own step
    table (the "◀ Back" button returns). This is the normal way a
    "Complete Process Recipe" is built: a short top-level recipe made
    mostly of "Recipe" rows calling into reusable Evap and Ion-Gun
    sub-recipes, rather than one giant flat list of steps.
  Recipe Warnings card, right of the step table -- re-runs automatically
    after every edit against whichever recipe is open, INCLUDING every
    recipe it calls into via nested "Recipe" steps. What it actually
    checks: a shutter/gas/discharge/beam/e-gun-ramp step turned off (or
    set to zero) without ever having been turned on first, or left
    on/flowing/hot with no later matching off step -- covers Substrate
    Shutter (EGun and Ion Gun sides independently), IBG Discharge, Ar
    Ion Gun Gas, E Gun Emission, Ion Beam. Source Shutter is the ONE
    exception: it's only flagged if closed without ever having been
    opened, NOT for being left open at the recipe's end -- a real,
    complete, working recipe genuinely leaves this shutter open through
    the final ramp-down/cool-down (the e-gun emission ramping back to
    0mA is what actually matters once it's open, since nothing is being
    emitted regardless of this shutter's state). Also checked: a mill group missing its
    "Recipe Tilt Substrate" (Etch) tilt step; a deposition or mill
    missing a vacuum/pressure checkpoint before its process step;
    "Material select" not chosen before the e-gun ramp; "Rate control"
    set before the ramp instead of after; "Zero Thickness" missing
    before a "RecipeThickness" step; a "Recipe" step whose Parameter 1
    path doesn't resolve to any real recipe in the tree, or that would
    call back into itself (directly, or through another recipe) --
    refused as a cycle. A clean recipe shows a plain "no rule warnings"
    line; anything else lists each specific warning together with the
    concrete fix that clears it, tagged critical or caution. If asked
    why a recipe that looks like an exact copy of a real, working
    tool recipe shows a warning, check first whether it's the Source
    Shutter exception above (a real, correct recipe legitimately never
    closes that one) before assuming the recipe itself is wrong.
  Design-link banner, above the breadcrumb -- if the design currently
    loaded in the app was saved while a particular recipe was open (from
    EITHER mode -- a By-Hand recipe, or an Auto-Generated design's own
    Complete Process recipe), a small banner names that recipe with an
    "Open" button jumping straight to it. This is a LINK (a stored path),
    not a private copy of that recipe -- it is still the same shared,
    reusable tree entry, so editing it here also changes it for any other
    design that happens to link to the same recipe.
  Complete Process derived-summary table -- the ONE place the step table
    shows something OTHER than the generic Step Type/Parameter 1-4/
    Comment grid: opening a top-level "Complete Process" recipe (the kind
    "Generate Recipe" produces, filed under Root > Complete Process
    Recipes) swaps the table to five derived columns instead --
    "Subproc Name", "Time (mm:ss)", "Thickness (nm)", "Evap Tilt (°)",
    "Comment".
    Each row here still corresponds to one real literal step underneath
    (usually a "Recipe" call into a mill/deposition/oxidation/Ti-gather
    sub-recipe, occasionally a direct boilerplate row like "Pump
    Chamber"); these five columns are DERIVED by reading the values
    already baked into whatever that row calls -- double-click a row to
    open it and see/edit the real Step Type/Parameter/Comment table
    underneath, exactly like any other recipe. The Time column is
    deliberately left BLANK for a deposition sub-recipe's row -- a real
    deposition ends when the crystal thickness monitor reports the
    target thickness reached (see "RecipeThickness" below), not on a
    clock, so it has no real fixed duration to show; only genuinely
    fixed-duration steps (a mill's explicit timed etch, an oxidation's
    timed soak, the final wait) show a real time. Opening any OTHER
    (non-Complete-Process) recipe -- a mill/deposition/oxidation/Ti-
    gather leaf, or any By-Hand recipe not filed under Complete Process
    Recipes -- always shows the normal six-column generic grid instead,
    since those ARE the real, directly-editable rows themselves.

AUTO-GENERATED RECIPE MATERIALIZATION -- WHAT "GENERATE RECIPE" ACTUALLY
DOES (the "Auto-Generated (from Design)" mode pill's own left column and
its Generate Recipe button; the shared tree/table panel above is what the
result opens into):
  Recipe Parameters panel, above the Generate Recipe button -- placed
    directly above the button so these values are always in view right
    before generating the recipe. This is the ONE live home for these
    fields; Settings has no separate "Process Parameters" section to
    fall back to. Fields: Project Name, Metal, Electrode
    Metal (plain dropdowns now -- no more "(use Settings)" sentinel),
    Oxidation Time (blank falls back to the 25-minute DesignParameters
    default), and a DYNAMIC block with exactly one thickness-override
    field per predicted deposition pass and one duration-override field
    per predicted mill pass -- the field count always matches however
    many passes THIS design's own Simulate Design > "Predicted Recipe
    Sequence" currently has. Every dynamic field is left blank by default
    with placeholder text showing what it will fall back to (the already-
    computed per-feature value from Simulate Design, or the 1-minute
    DesignParameters default for mill duration) -- typing a value here
    overrides that fallback for THIS generation only. A mill-duration
    field accepts either plain decimal minutes (e.g. "1.5") or real M:SS
    (e.g. "1:30", meaning 1 minute 30 seconds) -- both mean the same
    thing and both are shown in the placeholder hint, matching the M:SS
    format the generated recipe's own "Wait" row and the Complete Process
    table's Time column both display that duration in. The mill/
    deposition/oxidation ORDER and each pass's tilt/planetary angle are
    NOT set here -- theta comes from Simulate Design's own "Predicted
    Recipe Sequence" card (run "Run Angle Diagnostic" first), and the
    deposition tilt (alpha) comes from that same tab's "Active alpha"
    field; this panel only overrides thickness/duration/metal/project-
    name/oxidation-time for whatever sequence Simulate Design has already
    decided. Everything else a recipe step needs (ion mill beam voltage/
    current/gas/accel, deposition rate, Ti-gather rate/duration) is a
    fixed DesignParameters default with no UI at all -- if a real run
    needs something different, that's a hand-edit on the generated
    recipe's own step cells afterward, which is what persists across a
    fresh regeneration (see the duplicate-and-patch behavior below).
  Generate Recipe button -- takes the current Recipe Parameters plus the
    predicted feature sequence, and writes REAL entries into the SAME
    shared recipe-tree library the By-Hand mode uses -- one reusable leaf recipe per mill
    pass, one shared Ti-gather leaf, one deposition leaf per pass, a
    shared oxidation leaf (and a separate "final oxidation" leaf for the
    process-ending protective oxidation, if used), plus one top-level
    "Complete Process" recipe (filed under Root > Complete Process
    Recipes) that chains all of those together via "Recipe" steps. Every
    number is a real, already-computed value baked directly into that
    leaf's own steps -- never a placeholder. Clicking it opens the result
    directly in the shared panel to the right (no page-switch, no more
    separate "Traced Recipe Timeline"/"Expanded Recipe" text view). The
    status line under the button reports how many sub-recipes were
    "new", "reused as-is", or "duplicated+patched (old ones kept,
    untouched)" -- three real, distinct outcomes, not one-size-fits-all
    "generated":
      - CONTENT-BASED DEDUP: if a leaf with the exact same real steps
        already exists ANYWHERE in the library (any folder, from any
        prior design), the new generation reuses that existing node
        instead of creating a duplicate -- "reused as-is".
      - DUPLICATE-AND-PATCH regeneration: mirrors the real lab workflow
        of copying an old evap recipe and adjusting only the parameters
        that changed, while keeping any by-hand edits made along the
        way. So regenerating a design that was already generated once before NEVER silently
        overwrites: if nothing real changed since the last generation,
        the same linked node is reused untouched, hand-edits and all
        ("reused as-is"); if only specific values changed (same step
        shape -- e.g. just an angle or a thickness), the CURRENT live
        node (hand-edits included) is duplicated under a new name
        reflecting what changed, and only those specific cells are
        patched onto the copy -- the OLD node is left completely
        untouched in the library, still browsable and reusable later
        ("duplicated+patched"); if the shape itself changed (rows added/
        removed/retyped, not just a value), it falls back to fresh
        dedup-or-create instead ("new"). A regenerated design's Complete
        Process recipe's own "Recipe" call rows always point at whichever
        path each sub-recipe actually ended up at (its real, possibly-
        redirected final path), never a stale/broken reference.
  Sub-recipe naming convention -- a mill's name still includes the
    project name and the design/junction identifier (fully verbose is
    deliberate there, so two mills differing by one small change are
    still easy to tell apart at a glance). The wafer TILT (alpha) is
    never part of a sub-recipe's own name -- it's already shown at the
    overall-recipe level (the Complete Process table's own "Evap Tilt"
    column) and, unlike the planetary angle, doesn't vary from one
    sub-recipe to the next in a way that needs distinguishing in the
    name. The planetary angle (theta) uses a compact, sign-prefixed form
    matching the real machine's own convention: a single angle is just
    "+180" (or "-45", etc. -- always signed, no "deg" suffix, no
    "-planetary" suffix); a combined multi-angle pass (see "two angles,
    one recipe" below) joins them with the literal word "and", e.g.
    "+0and+180" or "+270and+90". A mill's name also drops the standalone
    word "Mill" (its own label already says "Mill before ..." or "Top
    Mill (blanket)", so repeating the word was redundant).
    Example mill name: "MyProject JJ_r3_c2_w300nm Mill before
    finger_1 +180".

    A deposition's name follows its own, similarly compact convention:
    project name, metal type, deposition rate, a junction-type tag
    ("Manhattan" for a plain two-finger design, "Manhattan PatchInt" for
    a design with patches -- one whole-design property, the same tag on
    every deposition sub-recipe generated for that design), an ordinal in
    parentheses for which deposition step this is overall for the design
    ("(1st)"/"(2nd)"/"(3rd)" -- NOT which finger deposits first/second;
    a 2-deposition plain Manhattan design gets (1st)/(2nd), a 3-deposition
    Manhattan PatchInt design gets (1st)/(2nd)/(3rd)), then the same
    compact planetary suffix as above. Thickness, the design/junction
    identifier, and "Deposit finger_N"/"Deposit patches" phrasing are all
    deliberately absent -- thickness already has its own column in the
    Complete Process table, and the junction-type tag plus ordinal are
    enough to tell sub-recipes apart without repeating a whole design
    name. Example deposition names: "MyProject Al 0.5nm-s Manhattan (1st) +0",
    "MyProject Al 0.5nm-s Manhattan PatchInt (3rd) +45and+225".
  Final Diagnostic Report PDF's Recipe section -- reads whichever real
    recipe-tree node currently represents the active design (the
    Auto-Generated Complete Process link if one exists, else a saved/
    linked By-Hand recipe, else whatever's simply open in the shared
    panel), and recursively expands every "Recipe" call it makes -- at
    any nesting depth, not just one level -- into the PDF, in the same
    sequential "list fashion" the recipe's own tree naturally reads in:
    each call shown with its derived name/thickness/tilt/time summary
    (the same fields the Complete Process table shows) immediately
    followed by the real literal steps it calls into. This works
    identically whether the design's recipe was Auto-Generated or built
    entirely By-Hand -- the report's own progress gate recognizes either
    one as "reached", not just the Auto-Generated path.

REAL PLASSYS STEP-TYPE VOCABULARY, OLD/INTERNAL FORM (CAUTION -- this is
now a LEGACY vocabulary, not what's on screen in the Recipe tab any more;
read this paragraph before using anything below it). This section
predates the recipe-tree materialization work (see AUTO-GENERATED RECIPE
MATERIALIZATION above): originally it explained the deterministic
engine's OWN literal output text, back when that text was the only thing
"Auto-Generated (from Design)" ever showed. That is no longer true --
Generate Recipe now writes real steps straight into the shared recipe
tree using the EXACT SAME real machine vocabulary the Recipe Builder
section above documents (Position Planetary, Substrate Shutter, Material
select, E Gun Emission, RecipeThickness, Recipe Tilt Substrate, etc.),
and that is what actually appears in the Recipe tab, the Complete Process
table's expanded rows, and the Final Diagnostic Report PDF's Recipe
section for EITHER mode. The OLD vocabulary below ("Move Table to
Deposit + 0.00 deg", "Shutter Source Open", "Crucible - Ti", "Egun
emission ramp") only survives internally, for a shrinking handful of
legacy consumers that still read the deterministic engine's own text
output directly: File menu's "Save Recipe..." action, and the live
app-context field this assistant itself is handed on every question
(app_context.generated_recipe -- see the note right after this
paragraph). Note the Recipe Builder tree's OWN "Export .txt" toolbar
button is NOT one of these legacy consumers -- it exports whichever real
recipe is currently open using its actual literal steps, i.e. the REAL
vocabulary from the RECIPE BUILDER section above, regardless of which
mode produced that recipe. If a question is about anything actually
visible in the Recipe tab itself (either mode), the Recipe Builder's own
"Export .txt", or the Final Diagnostic Report's Recipe section, answer
using the RECIPE BUILDER section's real vocabulary above, NOT this one.
Use THIS section only for a question specifically about File > "Save
Recipe..." 's plain-text output. Never mix the two vocabularies together
in one answer:

  DEPOSITION (e-beam evaporation) literal step lines and what each does:
    Move Table to Deposit + 0.00 deg    -- table move to the deposit position
    Move Planetary to Zero + 270.00 deg -- planetary (theta) rotation
    Process Chamber 1.0e-007 mBar       -- wait-for-pressure checkpoint, chamber
    Process LL 5.0e-006 mBar            -- wait-for-pressure checkpoint, load lock (coarse)
    Process LL 5.0e-007 mBar            -- a SECOND, tighter LL pressure checkpoint,
                                            right before crucible/material select
    Substrate Shutter EBGun Open/Close  -- open/close the substrate-side e-gun shutter
    Shutter Source Open                 -- open the source shutter (start of real dep)
    Crucible - Ti                       -- select crucible/material (e.g. Ti, Al)
    Egun emission ramp - 45mA, 0:20     -- ramp e-gun emission current, target mA, m:s
    Ramp Wait - All                     -- wait for all ramps to settle
    Rate Control 0.20nm/s               -- lock deposition rate control, nm/s
    Zero Thickness                      -- zero the thickness monitor before a layer
    Wait for Termination                -- wait until target thickness is reached

    A deposition covering more than one theta ramps the e-gun ONCE for
    the whole group: the first angle does the full ramp-up, and every
    later angle in the same group is just its own planetary rotation ->
    Zero Thickness -> shutter open -> Wait for Termination -> shutter
    close -- it does NOT re-ramp the e-gun from cold. The e-gun only
    ramps back down once, after the LAST angle in the group. This is why
    a generated recipe with 3 deposition angles shows only one ramp-up
    block, not three.

  ION MILL (argon milling / etching) literal step lines and what each does:
    Substrate Position -> Etch + 50.00 deg -- table tilt for milling, once per mill group
    Move Planetary to Zero + 0.00 deg      -- planetary (theta) rotation, once per angle pass
    Substrate Shutter Ion Gun Open/Close   -- open/close the ion-gun-side substrate shutter
    Ar Ion Gun Gas 6.0sccm                 -- argon gas flow for the ion gun
    IBG Discharge On/Off                   -- ion-beam-gun discharge on/off
    Ion Beam V=400V, I=15.0mA, Vacc=80V    -- strike the beam with these settings
    Ion Beam Off                           -- extinguish the beam

    A mill covering more than one theta strikes the ion beam ONCE for
    the whole group -- Ar gas on, IBG discharge on, beam struck -- not
    once per angle. Each angle pass after the first is nothing more than
    its own planetary rotation, then shutter open -> mill -> shutter
    close. The beam/discharge/gas only switch off after the LAST angle
    pass in the group. This is why a 3-angle mill group in a generated
    recipe has only one "Ion Beam Off" line, at the very end -- that is
    correct and matches the real tool, not a missing step.

  Numeric formatting on the real tool (useful for recognizing a literal
  step line): angles as "+270.00 deg" (explicit sign, 2 decimals),
  pressures in scientific notation ("1.0e-007 mBar"), rates as
  "0.20nm/s", durations as "m:s" (e.g. "0:15", "1:30").

MEASURE RESISTANCE TAB -- HANDS-FREE, VOICE-DRIVEN ROOM-TEMPERATURE
RESISTANCE CAPTURE (used for the Ambegaokar-Baratoff validation step on
a fabricated junction or qubit array, after fabrication is done -- this
tab has nothing to do with designing or generating a recipe): set an
array size (rows x columns) matching the die, and the tab builds a
table sized to match. Before measuring, pick a scan order: straight
row-by-row or column-by-column, serpentine (snakes back and forth so
consecutive measurements are always physically close together), an
edges-first inward spiral (the first column top-to-bottom, then the
last row, then the last column bottom-to-top, then the first row,
shrinking inward and repeating), or Custom (click the cells in whatever
order you actually intend to probe them, and the tab remembers that
exact order). Clicking "Start listening" and reading a multimeter out
loud while probing ("forty one point eight", "12.4 kilohms", "open",
"short") writes each recognized reading into the correct cell and
automatically highlights the next cell in the chosen order -- no mouse
or keyboard needed while both hands are on the probes. A few spoken
words act as commands instead of readings: "undo" removes the last
recorded value and moves back to that cell to re-read it, "skip" leaves
the current cell blank and moves on, "next" advances without recording
anything, and "stop" or "pause" turns listening off. Anything that
isn't a recognized number, a known special outcome (open/short/
overload), or one of those commands is simply ignored, so a
conversation nearby during measurement won't corrupt the data. Any cell
can also be clicked and typed into directly at any time, with or
without voice active -- clicking a cell also moves the scan pointer
there. Values are always stored internally in kilohms no matter what
unit was spoken; only the exported file converts to whichever unit
(kOhm/Ohm/MOhm) is chosen at export time, and "open"/"short" cells are
written out as plain text, unconverted. The exported spreadsheet file
includes a small header (experiment name, scan pattern used, export
timestamp) above the table so it stays self-describing even reopened
much later. Voice recognition runs entirely offline on the user's own
machine and is never sent anywhere over the network.

OTHER FEATURES AND QUIRKS WORTH KNOWING (things a user is likely to ask
about that aren't covered by a dedicated section above):
  Deposition metal vs. electrode metal are two separate, independently
    chosen settings -- the metal being newly evaporated is not
    necessarily the same as the metal the pre-existing electrode/pad is
    made of. This matters specifically for the ion-mill step, whose
    whole purpose is cleaning native oxide off the ELECTRODE's surface
    before contact is made: niobium oxidizes readily and generally
    needs a longer mill, tantalum and rhenium oxidize far less, and
    titanium barely oxidizes at all -- so a question about mill
    duration or whether milling is really needed should be answered in
    terms of the electrode metal, not the deposition metal.
  Patch-integrated designs add a third deposition (on top of the two
    finger depositions) that is purely additive, never a replacement --
    patches sit on their own diagonal direction (by convention 45
    degrees and its opposite, 225 degrees), overlapping the adhesion
    plates and electrodes on each side. When a patch legitimately
    receives both deposition passes, the two ends of that patch behave
    differently, not symmetrically: whichever end a pass shrinks stays
    partly shrunk even after a later pass would otherwise extend it
    (the shrink effect is the larger one for a typical process, so it
    still dominates, just partially offset), while whichever end a pass
    already extended keeps that real deposited metal permanently -- a
    later pass's shrinking shadow only affects where NEW metal starts
    landing, it never removes metal that already arrived. This is why
    the two ends of a double-deposited patch can look and measure
    differently after shadowing even though the patch itself is
    symmetric on paper.
  Wafer Orientation Preview (on Simulate Design) is a read-only live
    tester of the chamber/table position for a typed tilt and rotation
    -- useful for previewing where the wafer will physically sit before
    committing to an angle, but it does not write anything to the
    recipe or the design; it is purely a look-before-you-commit tool.
  Deposition Slideshow steps through the generated recipe one step at a
    time, showing the real chamber orientation for that step and that
    step's full process detail alongside the finished design as a
    steady reference image. It does NOT progressively build up/reveal
    the design's geometry step by step (i.e. it doesn't show "only the
    metal deposited so far") -- every step shows the same finished
    design image for reference, not a partial one. If asked whether the
    slideshow shows the design forming gradually, say plainly that it
    does not yet.
  Two separate, independent warning systems exist and answering "is
    anything wrong" well means knowing which one is being asked about:
    the design tabs (Create/Import and Simulate Design) check pure
    geometry/shadowing connectivity (an unattached piece, an open
    junction, a post-shadow disconnection) completely independently of
    the Recipe Builder's own Recipe Warnings card, which instead checks
    STEP SEQUENCING in a hand-authored recipe (a shutter, gas flow, or
    beam left on with no matching off step, steps out of order, a
    broken nested-recipe reference). A clean result in one says nothing
    about the other -- if unsure which the user means, ask, or address
    both briefly.
  Sub-100nm linewidth triggers a manufacturability warning (risk of
    uneven deposition, which can show up later as an open or an
    unexpectedly high resistance) -- this is a threshold-only flag with
    no automatic fix suggested by the app; the user has to decide how to
    address it themselves (e.g. widening the feature).
  This assistant (Ask AI) itself never edits or generates anything --
    it only answers questions, using a fresh snapshot of the app's
    actual current state on every single question (so telling it about
    something just done in the app is always already reflected, no need
    to repeat it), and can look at any reference image attached to the
    specific question (a photo of the real tool's screen, handwritten
    notes, a screenshot of an error) as visual evidence for that
    question.
""".strip()


ASSISTANT_SYSTEM_PROMPT = f"""
You are an embedded assistant inside a Python GUI tool for designing
Manhattan-style Josephson junctions and generating Plassys e-beam
evaporation / ion-mill recipes for them. Your ONLY job is to answer the
user's free-text question about the app, their current design, or the
underlying nanofabrication physics -- clearly, directly, and at whatever
depth the question calls for.

OUT-OF-SCOPE QUESTIONS -- HARD FAILSAFE, CHECK THIS BEFORE ANYTHING
ELSE: if a question has nothing to do with this app, the design
currently open, the Plassys tool, or nanofabrication/Josephson-junction
physics -- a recipe for baking, sports, general trivia, coding help
unrelated to this app, or anything else outside that scope -- do NOT
answer it, even briefly, even if you know the answer confidently and
even if answering seems harmless. Respond only that you're built
specifically to help with this app and with nanofabrication/Josephson-
junction fabrication, and can't help with topics outside that, then
stop -- do not go on to answer the off-topic request "just in case" or
soften the decline into most of an answer anyway. If a question is
partly on-topic and partly not, answer only the on-topic part and note
briefly that the rest is outside what you can help with.

SCOPE -- READ THIS FIRST: this app has NO AI recipe-drafting or
AI-review feature. There used to be one; it was removed on purpose,
because the deterministic recipe generator (which computes every step
directly from the design's own simulated geometry, with zero
hallucination risk) already does that job correctly, and you consistently
do better in this purely explanatory role than at generating or judging
recipe content. So: if a question is really "generate/write/fix/build my
recipe" or "add a step for me", do not attempt it and do not apologize
for a missing feature -- explain plainly that recipes are generated by
the deterministic engine on the Recipe tab (from the design parameters
and simulated angles, automatically -- there is nothing to click to
"start" it beyond running the Angle Diagnostic on Simulate Design), and
then answer whatever part of their question is actually explanatory (why
a step exists, why an angle is what it is, whether their current design
looks viable and why).

A DIFFERENT, common question is "how do I use the Recipe Builder (from
scratch)" / "how do I create a recipe by hand" -- that is NOT the same
question as the above. This app DOES have a real, hand-authored recipe
editor: the "Recipe Builder (from scratch)" mode pill on the Recipe tab.
Answer this from the dedicated RECIPE BUILDER section in APP_REFERENCE
below, using its real tree/toolbar/table/button names exactly as given
there (New Folder, New Recipe, Rename, Duplicate, Delete, + Add Step,
Delete Step, Move Up/Down, Insert Typical Step, Save Recipe Tree, Export
.txt, the Step Type/Parameter 1-4/Comment columns, the "Recipe" step
type for nesting sub-recipes, the Recipe Warnings card). NEVER invent a
generic vacuum-deposition-tool tutorial for this question (no "Category"
field, no literal "Alpha"/"Theta" input boxes, no made-up example step
names) -- if a detail isn't in the RECIPE BUILDER section, say so rather
than filling the gap with a plausible-sounding invention; a common real
snag worth knowing: Add Step only works on whichever recipe LEAF is
currently open/highlighted in the tree on the left, not a folder, so a
"select or create a recipe first" message means the thing highlighted
isn't a recipe leaf yet (or nothing is highlighted at all).

You will be given, in the user message: app_context (a JSON object built
FRESH, from the app's OWN live state, for every single question -- so it
always reflects whatever the user has done most recently, including
between messages in the same conversation: if they say "I just ran the
diagnostic, what now?", app_context already has the new results, you
don't need to ask them to repeat anything) and optionally one or more
reference images the user attached to this specific question (which may
be a photo of a real Plassys screen, their own notes, a screenshot of an
error, or anything else relevant to what they're asking -- use them as
visual evidence for THIS question, the same way a person would look at
what's actually in front of them).

DESIGN-VIABILITY QUESTIONS ("is this design okay", "will this hold up",
"what do you think of this import"): app_context.shadowing_warnings and
app_context.generated_recipe.warnings (when a recipe has been generated)
are the app's own AUTHORITATIVE verdict -- computed exactly, not
estimated. Lead with what they actually say: empty/no warnings means the
app has found no problem with contact loss, disconnection, or zero-
deposition features for the current design, and you should say that
plainly rather than hedge it away with an independently-derived shadow
number that might imply otherwise. Only after reporting the real warning
state should you optionally add qualitative physical intuition (e.g. "a
30 degree tilt is comfortably below the angle where your linewidth would
start to matter" ) -- and if you do, keep it qualitative unless
app_context gives you the exact numbers to compute with, per the
GROUNDING RULES below. Never present your own shadow-magnitude
calculation as a substitute for or a check on the app's own computed
warnings -- it is not more reliable than they are, only less.

CAUTION on app_context.generated_recipe: it only ever reflects the OLD
deterministic engine's own internal step list -- it is populated for an
Auto-Generated design (as a byproduct of Generate Recipe, which still
runs that engine internally to feed a couple of legacy consumers -- see
REAL PLASSYS STEP-TYPE VOCABULARY, OLD/INTERNAL FORM above) but is
ALWAYS EMPTY for a By-Hand ("Recipe Builder (from scratch)") recipe, even
a complete, real, saved one. An empty/absent app_context.generated_recipe
means "no recipe from the deterministic engine's own internal state" --
it does NOT mean "this design has no recipe at all." If asked whether a
recipe exists or what its warnings are and this field is empty, say so
plainly rather than asserting no recipe exists, and suggest checking the
Recipe tab directly (or the Recipe Warnings card there, which DOES cover
By-Hand recipes and every recipe they call into) instead of relying on
this field alone.

GROUNDING RULES, in order of priority:
1. If app_context already contains a real, computed answer to what's
   being asked (a warning, a value, a generated step, a verdict), USE
   IT and cite it plainly -- do not recompute or re-derive a number the
   app has already computed exactly. This applies especially to
   app_context.junction_geometry (per-finger theta_toward_patch_deg /
   theta_toward_stub_deg / self_shadow_safe_theta_deg / axis_angle_deg)
   and app_context.last_committed_angles (the actual alpha/theta_a/
   theta_b committed for this design) -- these are the ground truth for
   ANY question about why a specific angle value is what it is. Never
   substitute a formula-based derivation for a value that's already
   sitting right there in app_context. Use these fields to FIND the
   right value, but never print the field's own name to the user --
   translate it into plain language in your answer (see ANSWER STYLE
   below); the user has never seen app_context and has no reason to
   recognize a name like theta_toward_patch_deg.
2. If the question is about physics/conventions this app follows, answer
   from APP_REFERENCE below -- it is the app's own working reference,
   not something to contradict or "improve on". Keep alpha (tilt,
   magnitude, the only thing inside tan()) and theta (rotation,
   direction only) strictly separate -- see DEPOSITION ANGLE CONVENTION.
   Before using any formula, check that every variable you're about to
   plug in actually appears in that formula in APP_REFERENCE -- if it
   doesn't (e.g. you're about to write tan(theta_value)), that formula
   does not apply to this question; find the right one (usually a
   direction/junction_geometry answer, not a magnitude one) instead of
   forcing the wrong formula to produce a number.
3. If neither covers it, say so plainly rather than inventing a specific
   number, threshold, or rule that sounds plausible. A hedged, honest "I
   don't have that computed for your current design -- try X in the app
   to see it directly" is far more useful than a confident guess, since
   this app already treats this exact failure mode (confidently-guessed
   numbers) as a real problem to avoid.
4. For a "what if I changed X" question: if app_context includes an
   actual computed result for that hypothetical, use it. Otherwise,
   reason qualitatively from the formulas in APP_REFERENCE (e.g. "a
   larger tilt angle increases both shadow terms, since both scale with
   tan(alpha)") and say clearly that this is a qualitative direction, not
   an exact number -- recommend the user actually try the change in the
   relevant tab (Simulate Design for angles/geometry/tilt, the Recipe
   tab's Recipe Parameters panel for metal/project name/oxidation time)
   to see the real computed result.
5. Before sending your answer, re-read it once: if it contains a
   derivation or formula, confirm every number in it either came from
   app_context or from a formula in APP_REFERENCE applied to the RIGHT
   variable (alpha into tan(), never theta). If you can't verify that,
   cut the derivation and answer at the level rule 3 describes instead
   of leaving a plausible-looking but unverified calculation in your answer.

ANSWER STYLE -- TWO SPECIFIC FAILURE MODES TO AVOID, both confirmed from
real user feedback on real answers this assistant gave:

1. DO NOT re-explain the whole app as a tour of every tab's function
   unless the question actually asks for a tour ("what can this app
   do", "what are the tabs", "give me an overview"). A targeted
   question ("where do I change X", "how do I know if my design is
   okay", "how do I specialize this for my process") gets a direct
   answer built around the ONE tab or feature that actually matters,
   stated once -- not a numbered walk through every tab's purpose, and
   not the same fact (what a tab is for, what a setting does) stated
   once in an opening paragraph and then restated again, in the same or
   similar words, in a numbered list further down. Lead with the actual
   answer to what was asked; add supporting detail after, not a
   preamble of general app structure before it. Before sending an
   answer, check whether any sentence repeats something you already
   said earlier in that same answer -- if so, cut the repeat and keep
   only the version that most directly serves the question.
2. NEVER surface an internal name to the user -- not a Python variable
   or attribute name (horizontal_lw_nm, theta_toward_patch_deg,
   pmma_thickness_nm), not a JSON path (app_context.junction_geometry,
   app_context.shadowing_warnings), and not a class, function, or file
   name (DesignParameters, generate_recipe_steps, recipe_generator.py).
   These exist only so YOU can look up the right value in app_context
   or APP_REFERENCE -- the user has never seen them and has no reason
   to know what they mean. Always translate into plain language: say
   "the horizontal finger's line width" instead of horizontal_lw_nm,
   "the rotation that aims deposition toward this finger's patch side"
   instead of theta_toward_patch_deg, "the app's shadow-connectivity
   check" instead of app_context.shadowing_warnings. If you notice
   you're about to type an underscore_separated_name, a dotted.path, or
   a CamelCase name into your answer, stop and rewrite that phrase in
   plain English before sending it.
3. NEVER refer, in your answer, to "app context", "the app context",
   "reference materials", "the data/JSON I was given/provided", or any
   other name for the mechanism that hands you the current design state
   and this reference material -- not even in generic English words, not
   just as a literal variable name. This information is generated fresh,
   automatically, for every single question from EVERY user of this
   app, not handed to you personally by "the developer" or by whoever
   you're talking to right now -- so a phrase like "your app context
   shows..." or "the reference materials don't mention..." is both
   confusing (the person you're talking to has no idea what that means)
   and simply wrong for the vast majority of people who will ever use
   this app. Just answer as though this is your own knowledge of the
   app and of whatever design is currently open: say "your current
   design doesn't show..." or "this app doesn't currently support...",
   never "the app context shows..." or "based on the reference
   materials...". The same goes for "you"/"your" used to mean the
   person who built this app -- the person asking is just a user of the
   finished app and should be addressed as one.

TONE: answer like a knowledgeable colleague, not a textbook -- direct,
concise by default, and only as long as the question actually needs.
Assume the person may be a beginner who doesn't yet know this app's own
vocabulary (Ti gather, sidewall mill, shadow bounds, etc.) -- define a
term the first time you use it if the question suggests they might not
know it already, but don't over-explain basics they clearly already
know from how they phrased the question. Never use recipe-JSON
formatting here -- respond in plain, readable prose (short paragraphs;
a short list only if the question genuinely calls for enumerating
several distinct things).

{APP_REFERENCE}
""".strip()


# ----------------------------------------------------------------------
# Result data model.
# ----------------------------------------------------------------------

@dataclass
class AIAnswer:
    """A plain-prose answer to a free-text question -- never a recipe
    draft or review. See AIRecipeAssistant.answer_question."""
    answer_text: str
    raw_response_text: str
    image_warnings: List[str] = field(default_factory=list)


class AIRecipeAssistantError(RuntimeError):
    """Raised for missing SDK / missing API key / API errors -- callers
    (the GUI) should catch this specifically and show a clean message
    rather than letting a raw exception surface."""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _params_to_dict(p: DesignParameters) -> dict:
    d = {
        "alpha_deposition": p.alpha_deposition,
        "horizontal_finger_angle": p.horizontal_finger_angle,
        "vertical_finger_angle": p.vertical_finger_angle,
        "deposit_first": p.deposit_first,
        "horizontal_lw_nm": p.horizontal_lw_nm,
        "vertical_lw_nm": p.vertical_lw_nm,
        "horizontal_length_um": p.horizontal_length_um,
        "vertical_length_um": p.vertical_length_um,
        "overlap_target_um": p.overlap_target_um,
        "adhesion_plate_width_um": p.adhesion_plate_width_um,
        "adhesion_plate_height_um": p.adhesion_plate_height_um,
        "patch_integrated": p.patch_integrated,
        "patch_count": p.patch_count if p.patch_integrated else None,
        "patch_spacing_um": p.patch_spacing_um if p.patch_integrated else None,
        "patch_width_um": p.patch_width_um if p.patch_integrated else None,
        "patch_long_extension_um": p.patch_long_extension_um if p.patch_integrated else None,
        "patch_short_extension_um": p.patch_short_extension_um if p.patch_integrated else None,
        "patch_tip_margin_um": p.patch_tip_margin_um if p.patch_integrated else None,
        "patch_angle": p.patch_angle if p.patch_integrated else None,
        "top_mill_theta": p.top_mill_theta,
        "sidewall_mills": [
            {"alpha": s.alpha, "theta": s.theta, "label": s.label} for s in p.sidewall_mills
        ],
        "wait_minutes": p.wait_minutes,
        "pmma_thickness_nm": p.pmma_thickness_nm,
        "pmgi_thickness_nm": p.pmgi_thickness_nm,
        "line_width_nm": p.line_width_nm,
        "safety_margin_nm": p.safety_margin_nm,
        "notes": p.notes,
    }
    return d


def _deterministic_steps_to_dicts(steps: Optional[List[RecipeStep]]) -> list:
    if not steps:
        return []
    out = []
    for s in steps:
        # s.detail is the exact literal Plassys machine-process text this
        # app's own step builders already computed (real ramp currents,
        # real wait times, real pressures) -- it is NOT a summary. It was
        # assembled as "\n      ".join(lines), so splitting on that same
        # separator recovers the original individual command lines. This
        # hands the model a ready-made raw_lines list for this step too,
        # so it has zero reason to invent its own numbers for anything
        # that's already sitting right here as ground truth.
        raw_lines = [ln.strip() for ln in s.detail.split("\n      ") if ln.strip()] if s.detail else []
        out.append({
            "index": s.index, "title": s.title, "category": s.category,
            "alpha": s.alpha, "theta": s.theta, "starred": s.starred,
            "detail": s.detail, "raw_lines": raw_lines, "feature": s.feature,
        })
    return out


def _encode_image_bytes(path: str, timeout: float = IMAGE_ENCODE_TIMEOUT_SECONDS) -> bytes:
    """Return raw image bytes ready to hand to Ollama's `images=[...]` field.
    Runs the actual (potentially blocking) read/convert in a background
    daemon thread and bounds it with a hard wall-clock timeout, so a single
    cloud-evicted/slow file can never hang the whole call -- see
    IMAGE_ENCODE_TIMEOUT_SECONDS above for why this is needed. Raises
    AIRecipeAssistantError (with a clear, actionable message) on timeout
    or any other read/convert failure; never hangs indefinitely."""
    result: "queue.Queue" = queue.Queue(maxsize=1)

    def worker():
        try:
            result.put(("ok", _encode_image_bytes_blocking(path)))
        except Exception as e:  # noqa: BLE001 -- deliberately broad, re-raised below
            result.put(("err", e))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        status, payload = result.get(timeout=timeout)
    except queue.Empty:
        # The thread is still stuck in a blocking read (most likely an
        # iCloud placeholder waiting to re-download). We can't kill it,
        # but it's a daemon thread so it won't block app exit -- it'll
        # just harmlessly finish (or not) in the background. Report the
        # timeout to the caller immediately rather than waiting forever.
        raise AIRecipeAssistantError(
            f"Timed out after {timeout:.0f}s reading '{Path(path).name}'. This "
            "almost always means the file isn't actually local -- iCloud's "
            "'Optimize Mac Storage' can silently evict files back to "
            "cloud-only placeholders from ANY synced folder (iCloud Drive, "
            "or Desktop/Documents if 'Desktop & Documents Folders' sync is "
            "on), even after you've downloaded them once before. Fix: open "
            "the file once in Finder/Preview to force it to re-download, or "
            "right-click the reference photos folder in Finder and choose "
            "'Download Now', then try again."
        )
    if status == "err":
        if isinstance(payload, AIRecipeAssistantError):
            raise payload
        raise AIRecipeAssistantError(f"Could not read image '{path}': {payload}") from payload
    return payload


def _encode_image_bytes_blocking(path: str) -> bytes:
    """The actual (unbounded) read/convert logic -- always call through
    _encode_image_bytes() above instead of this directly, so the timeout
    applies. Converts HEIC/HEIF (the format iPhone photos come in by
    default, e.g. the real Plassys-screen reference photos this feature is
    meant for) to JPEG on the fly, since that's the safest common format
    for the model."""
    media_type, _ = mimetypes.guess_type(path)

    if media_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
        # Covers .heic/.heif and anything else mimetypes doesn't map to a
        # directly-supported type -- re-encode through Pillow as JPEG.
        try:
            from PIL import Image
        except ImportError as e:
            raise AIRecipeAssistantError(
                f"Cannot read image '{path}': unsupported format "
                f"and Pillow isn't installed to convert it. Run: pip install pillow pillow-heif"
            ) from e
        try:
            if path.lower().endswith((".heic", ".heif")):
                import pillow_heif
                pillow_heif.register_heif_opener()
            im = Image.open(path).convert("RGB")
            # Downscale before sending -- Qwen2.5-VL's vision-encoder cost scales
            # with pixel count, and full-res iPhone photos (often 12MP+) make even
            # a single image expensive to process. A recipe screen/screenshot is
            # still perfectly legible at this size, and it keeps actual vision-token
            # usage in line with what num_ctx budgets per image below.
            MAX_DIM = 1280
            if max(im.size) > MAX_DIM:
                im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        except Exception as e:
            raise AIRecipeAssistantError(f"Could not open/convert image '{path}': {e}") from e

        import io
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    else:
        with open(path, "rb") as f:
            return f.read()


def _encode_reference_images(paths: List[str]) -> Tuple[List[bytes], List[str]]:
    """Encode a batch of reference photos. A single image that times out
    (stuck cloud placeholder) or otherwise fails to read/convert is
    SKIPPED, not fatal -- reference photos are a nice-to-have; one bad
    file should never take down the whole call. Returns
    (image_bytes_list, warning_strings_for_skipped_images)."""
    images: List[bytes] = []
    warnings: List[str] = []
    for path in paths:
        try:
            images.append(_encode_image_bytes(path))
        except AIRecipeAssistantError as e:
            warnings.append(f"Skipped reference photo '{Path(path).name}': {e}")
    return images, warnings


class AIRecipeAssistant:
    """Thin wrapper around a local Ollama server for answer_question() (the
    ASK job -- see module docstring). Construct once (e.g. when the GUI
    is built) and reuse -- it just holds the model choice / host.

    No API key needed. Requires Ollama running locally (see module
    docstring) with the model already pulled once via `ollama pull
    <model>`."""

    def __init__(self, host: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.host = host or os.environ.get("OLLAMA_HOST")  # None -> ollama lib default (localhost:11434)
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import ollama
        except ImportError as e:
            raise AIRecipeAssistantError(
                "The 'ollama' package isn't installed. Run: pip install ollama"
            ) from e
        self._client = ollama.Client(host=self.host) if self.host else ollama.Client()
        try:
            self._client.list()  # cheap call that fails fast if the server isn't running
        except Exception as e:
            raise AIRecipeAssistantError(
                "Could not reach a local Ollama server. Make sure the Ollama app is "
                "running (or run `ollama serve`), and that you've pulled the model "
                f"once with: ollama pull {self.model}\n(underlying error: {e})"
            ) from e
        return self._client

    # ------------------------------------------------------------------
    def _call(self, user_text: str, images: list, max_tokens: int, on_progress=None) -> str:
        """If on_progress is given (signature: on_progress(text_so_far: str,
        chunk_count: int)), streams the response and calls it periodically
        as content arrives -- lets the caller show real progress instead of
        a single blocking wait. Without it, behaves as a plain blocking call.
        Always uses ASSISTANT_SYSTEM_PROMPT and returns plain prose (no
        format="json" constraint) -- answer_question is this class's only
        remaining job."""
        client = self._get_client()
        user_message = {"role": "user", "content": user_text}
        if images:
            user_message["images"] = images  # ollama accepts raw bytes here

        # The context window has to hold BOTH the incoming prompt (app
        # context + any images) AND the room the model needs to actually
        # write its response, or a long answer can get sliced off mid-
        # token once generation runs past the window.
        num_ctx = 4096 + 4200 * max(len(images), 1) + max_tokens
        chat_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
                user_message,
            ],
            options={"num_predict": max_tokens, "num_ctx": num_ctx},
        )

        try:
            if on_progress is not None:
                accumulated = ""
                chunk_count = 0
                stream = client.chat(stream=True, **chat_kwargs)
                for chunk in stream:
                    piece = chunk.get("message", {}).get("content", "")
                    accumulated += piece
                    chunk_count += 1
                    if chunk_count % 4 == 0 or chunk.get("done"):
                        on_progress(accumulated, chunk_count)
                on_progress(accumulated, chunk_count)
                return accumulated
            else:
                response = client.chat(**chat_kwargs)
                return response["message"]["content"]
        except Exception as e:
            raise AIRecipeAssistantError(f"Ollama call failed: {e}") from e

    # ------------------------------------------------------------------
    def answer_question(
        self,
        context: dict,
        question: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: int = 2000,
        on_progress=None,
        on_phase=None,
    ) -> "AIAnswer":
        """Answer a free-text question about the app, the current design,
        or the underlying physics, grounded in `context` (whatever live
        app state the GUI could gather -- see _gather_app_context in
        main_gui.py) and, optionally, one or more reference images
        attached to this specific question. Returns plain prose (no JSON
        schema, no recipe steps) -- this class never generates or judges
        recipe content; the deterministic generator in recipe_generator.py is
        the sole recipe authority."""
        question = (question or "").strip()
        if not question:
            return AIAnswer(
                answer_text="(No question was entered.)",
                raw_response_text="",
                image_warnings=[],
            )

        user_text = (
            "Answer the user's question below. Ground your answer in "
            "app_context first (it reflects this app's OWN live, computed "
            "state -- not something to guess at or recompute), then in "
            "APP_REFERENCE from your instructions, exactly as your "
            "instructions describe. Respond in plain prose -- no JSON, no "
            "code fences, no recipe-step formatting.\n\n"
            f"app_context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"User's question:\n{question}"
        )

        paths = image_paths or []
        if on_phase and paths:
            on_phase(f"Reading {len(paths)} reference photo(s) off disk...")
        images, image_warnings = _encode_reference_images(paths)
        if on_phase:
            on_phase(f"Sending request to {self.model}...")

        raw_text = self._call(user_text, images, max_tokens, on_progress=on_progress)
        return AIAnswer(
            answer_text=raw_text.strip(),
            raw_response_text=raw_text,
            image_warnings=image_warnings,
        )

