# Plassys Shadow Simulator

A desktop application for designing Josephson-junction fabrication layouts, simulating the shadow-evaporation deposition process, generating a real machine recipe for a Plassys e-beam / ion-mill deposition tool, and analyzing the resulting room-temperature resistance measurements — one continuous pipeline, from a blank design to a validated result.

**To launch the app, run:**

```
python3 src/lib/main_gui.py
```

See `requirements.txt` for the full setup process (required packages, optional voice/AI features, and exact tested versions).

---

## Version History

This section works like a changelog/archive: each release gets its own dated entry below, oldest at the bottom. Development continues past this point, so future updates will be added above this one as new dated entries.

### v1.0.0 — September 29, 2026 — First Full Release

The first complete, public-ready release of the application. Every tab described in this README is implemented and working: parametric design creation and real GDS/KLayout-script import, automatic shadow-evaporation simulation with a full warnings/verdict system, deterministic Plassys recipe generation alongside a from-scratch manual Recipe Builder, a step-by-step deposition slideshow, hands-free voice-driven resistance measurement, and full Ambegaokar-Baratoff resistance-vs-area analysis with experiment comparison. This release also generalized the codebase so it works with any Josephson-junction design rather than one specific lab's own layouts, and polished the code and documentation (including this README) for outside use.

---

## How the pieces fit together

The app is built around one pipeline, reflected directly in its tab order (left to right in the Contents panel):

1. **Create / Import** — draw a junction design from scratch, or import a real one from a `.gds` file or KLayout script.
2. **Simulate Design** — compute the recommended deposition tilt/rotation angles for that design and simulate exactly what the shadow-evaporation process will produce, including any warnings.
3. **Recipe** — turn that simulation into a real, step-by-step Plassys machine recipe (or build one by hand from scratch).
4. **Deposition Slideshow** — watch the recipe build the design up, one deposition/mill step at a time.
5. **Measure Resistance** — after the chip is actually fabricated, capture room-temperature resistance readings for every junction on it, hands-free if you like.
6. **Interpret Results** — plot those measurements against each junction's area and validate the fabrication run against theory.

A **Front Page** ties all of this together with one-click shortcuts, and a hidden **Settings** page (reached from the gear icon) holds configuration that doesn't need to be touched often.

---

## A Tour of the App

### Front Page

The landing screen shown on startup. Purely navigational — no data entry. It shows the app's name, tagline, and credits, a one-line description of the overall pipeline, a shortcut card for each of the six tabs below (click a card to jump straight there), and a shortcut to the Design Library for saving/loading designs.

### Create / Import

*"Design or import a junction, place and classify shapes, and see automatic measurements."*

This is the pipeline's starting point, with two interchangeable sources for a design — a toggle at the top of the tab switches between them:

- **Created** — build a design parametrically by typing numeric geometry: electrode standoff/size, each finger's angle/length/linewidth, adhesion pads, and optional shadow-evaporation patches (number, spacing, thickness, extension). A live suggested-linewidth hint is available from the Settings resist-stack calculator.
- **Imported** — import a real `.gds` layout file or the actual KLayout Python script that generated one. For a GDS file, pick the layer and candidate junction cell; for a script, pick which generator function to run and fill in its own parameters (pre-filled from the script). The app then automatically classifies the geometry — which end of each finger is the electrode end vs. the crossing end, where patches and plates are, and so on.

Both sources share the same interactive canvas and tools:

- Select, drag, resize, and rotate any shape; marquee-select a batch of shapes; nudge by an exact step size or scale by an exact factor.
- Draw custom rectangle, circle/oval, or polygon shapes by hand, and assign or correct each shape's role (Finger, Patch, Adhesion Plate, Electrode, Substrate) with one click.
- A ruler tool for one-off distance measurements, an on/off toggle for live dimension callouts drawn directly on the geometry, and standard zoom/pan/zoom-to-fit controls.
- Undo/redo covering essentially every edit.
- A **Measurements** panel automatically computes and lists every relevant geometric quantity (finger length/linewidth, junction overlap, patch contact area, spacings) and can export the full set to a text file.
- A **Design Warnings** banner flags geometry problems as you work (fingers that don't cross, fingers that aren't perpendicular, a patch not actually attached to anything, insufficient patch coverage).
- **Save/load designs** to a personal Design Library (File menu, or the Front Page shortcut) — every saved design, its geometry, roles, and settings survive closing and reopening the app.

### Simulate Design

*"Recommended deposition angles, warning explanations, and the shadowing visualization for the active design."*

Takes whichever design is active and turns it into a physically grounded deposition plan, in two steps:

1. **Run Angle Diagnostic** — computes recommended tilt (alpha) and rotation (theta) angles for every finger and patch, with a plain-English explanation of each choice, and an editable tree for overriding any angle by hand (including adding or removing individual deposition passes per feature).
2. **Run Shadow Simulation** — commits those angles and computes the real, physically simulated result: where the metal actually lands once shadowing is accounted for, including any broken connections or stray "ghost" slivers. After the first run, further edits recompute live.

A three-way view toggle switches the shared canvas between the as-drawn **Optimal Design**, the angle-annotated **Simulated Design**, and the fully shadow-computed **Post-Shadow Result**. A **Predicted Recipe Sequence** panel lists every mill/deposition/oxidation pass in order (with per-pass angle overrides, reordering, and manual insertion) — this list is exactly what the Recipe tab later reads to auto-generate the machine recipe. A **Wafer Orientation Preview** shows the chamber's physical geometry for the current tilt/rotation. On the right, a running **warnings and verdict system** reports per-finger angle safety, any zero-deposition or disconnection problems, and an overall verdict (OK / OPEN / SHORTED / AREA REDUCED / CAUTION, which can combine).

### Recipe

*"Generates a recipe from the last Simulate Design run — each step traces back to the finger/patch and warning that produced it."*

Two modes, selected by a toggle at the top, sharing one recipe view/table and one warnings panel on the right:

- **Auto-Generated (from Design)** — fill in a handful of process parameters (metal, electrode metal, oxidation time, mill angle, per-pass thickness/duration) and click **Generate Recipe**. The deterministic generator turns the design's simulated sequence into a complete, real Plassys recipe — written directly into the shared recipe library in the exact format the real tool uses, so it's immediately hand-editable. Regenerating an already-generated recipe preserves any hand-edits rather than overwriting them.
- **Recipe Builder (from scratch)** — a general-purpose recipe author, independent of any design, that mirrors the real Plassys tool's own editing mechanics exactly: a tree of named, reusable recipes (folders and recipes, drag-and-drop reorderable), where a step can itself call another named recipe as a sub-routine. Full create/rename/duplicate/move/delete support, plus a library of common step templates to insert.

Either way, the shared **step table** shows Step Type / Parameter 1–4 / Comment (or, for the top-level generated recipe, a friendlier derived summary view), with add/delete/move-up/move-down controls and a plain-text export. The **Recipe Warnings** panel continuously re-checks whichever recipe is open — and everything it calls — against sequencing and machine-safety rules, flagging anything missing, out of order, or unsafe.

### Deposition Slideshow

*"Steps through the generated recipe, showing chamber orientation and detail text for each step alongside the finished design."*

A step-by-step, read-only walkthrough of the generated recipe. Previous/Next buttons (or clicking any step in the sidebar list) move through the recipe one step at a time, showing: the step's full descriptive text and angle values in a detail panel; the wafer's real chamber orientation for that step; and — the main feature — a progressive design view that shows only the material actually deposited so far, so the junction visibly "grows" one pass at a time rather than appearing all at once. A **Post Deposition** / **Shadow Sim** toggle switches between the true accumulated post-shadow geometry and a per-step idealized shadow-band view; mill steps additionally highlight whatever surface is being milled.

### Measure Resistance

*"Room-temperature resistance capture for a junction array. Set the array size and scan pattern, read values aloud (Whisper or Vosk) hands-free, and export to CSV."*

Set the array's row/column size and a scan pattern (row-major, column-major, either serpentine direction, edges-in spiral, or fully custom by clicking cells in order), then build the table. With probes in hand, speak each reading aloud — a background speech-to-text engine (Whisper or Vosk, selectable) transcribes it, parses the number and its units, and drops it straight into the current cell, then automatically advances. Spoken commands ("undo," "skip," "repeat") work hands-free too, and a reading can be marked "open" or "short" instead of a number. Cells are color-coded by status (next up / done / open-or-short), a transcript log records every recognized utterance, and the finished table exports to CSV with experiment name, units, and timestamp.

### Interpret Results

*"Resistance vs. inverse junction area (Ambegaokar-Baratoff validation). Load data, adjust geometry or apply an offset, plot, and save/compare experiments."*

The final analysis step, in three sub-tabs:

- **Main Plot** — load data live from the Measure Resistance tab or import a CSV, set filtering thresholds (open/short cutoffs, an outlier Z-score limit) and the superconducting critical temperature, then plot resistance against inverse junction area. Reports a linear fit (R², slope, intercept) and an estimated critical current density, with a log of exactly which readings were filtered out and why.
- **Area Adjustment** — correct each junction's effective area before fitting (a global or per-row lithography offset in nanometers, pasted linewidth ladders, or a directly-typed area override), useful for correcting over/under-exposure.
- **Compare Experiments** — save a processed run to a comparison list (or load previously saved ones from disk) and overlay multiple experiments on one plot, each with its own fit line and stats.

### Settings *(reached via the gear icon, not a main tab)*

A one-time/rarely-revisited configuration page with three sections:

- **Resist Stack & Calculator** — define the lithography resist bilayer (top/bottom layer names and thicknesses, safety margin) and convert between a target line width and the deposition tilt angle (alpha) that would produce it; push either value straight into the Simulate Design tab.
- **Chamber Reference Model** — a standalone, nothing-is-saved reference view of the deposition chamber: type any alpha/theta/reference point and see the resulting wafer orientation from the top and the side.
- **Hotkeys** — a full table of every keyboard shortcut in the app (grouped by category), each one reassignable by clicking its key and pressing a new combination, with per-row clear/reset and an automatic conflict warning if two actions ever share a key.

An always-available **"Ask AI"** button (visible from every tab) opens a chat panel that answers free-text questions about the app, your current design, or the underlying physics, using a local AI model — it is purely explanatory and never generates or edits a recipe itself (recipe generation stays fully deterministic).

---

## Project Files

**`main_gui.py` — start here.** The application itself; run this file to launch the app. Contains every tab, the Front Page, and all window/menu/toolbar chrome described above, and imports and coordinates every module below. None of the other modules depend on Qt themselves, so each can be understood or reused independently of the GUI.

| File | What it does |
|---|---|
| `junction_import.py` | Reads a real fabricated design — a `.gds` file or the actual KLayout script that generated one — and turns it into a measured, analyzable junction: which finger is which, which end is the electrode end vs. the crossing end, overlaps, and contact areas. |
| `parametric_junction_view.py` | Draws the parametric ("Created") design: the top-down and side (cross-section) views, and the shared color palette used across the app's drawings. |
| `imported_junction_view.py` | Draws an imported design, cropped and styled to show clearly whether/where it overlaps the electrode, with hit-testing so clicking a shape can explain why it's flagged. |
| `shape_editor.py` | The generic, framework-independent data model behind the interactive canvas — a free-form list of shapes, each with a position, size, angle, and role. Powers the Create/Import tab's editing behavior. |
| `design_library.py` | Saves and loads named designs to a per-user data folder, so a design survives closing and reopening the app. Storage location is computed automatically for whichever OS it runs on. |
| `electrode_convention.py` | A small, dependency-free settings object describing how a GDS file's electrode geometry is identified — letting the app work with more than one lab's own GDS conventions. |
| `hotkey_registry.py` | The master list of every keyboard shortcut in the app, plus the logic for saving and restoring custom key bindings. Powers Settings > Hotkeys. |
| `recipe_generator.py` | The deterministic engine that turns a simulated design into a complete, ordered Plassys recipe — documented formulas and template rules, never a guess, with its own warnings wherever a rule is a first-pass assumption. |
| `recipe_builder_model.py` | Data model and disk persistence for the manual Recipe Builder's tree of named, reusable recipes, matching the real Plassys tool's own editor mechanics. |
| `recipe_builder_rules.py` | Validates a Recipe Builder recipe (and everything it calls) against sequencing and machine-safety conventions. |
| `generated_recipe_export.py` | Bridges the automatic generator's output into the Recipe Builder's own format, so a generated recipe becomes real, directly-editable entries in the shared recipe library. |
| `ai_assistant.py` | Wires a local AI model (via Ollama, free and running entirely on your own machine) into the "Ask AI" assistant. Purely explanatory — never generates or edits a recipe. |
| `wafer_orientation.py` | Core rotation math for the deposition chamber: converts a tilt (alpha) and rotation (theta) into the wafer's real 3D orientation. |
| `chamber_plotting.py` | Drawing routines for the top-down and side views of the physical Plassys chamber, used throughout Simulate Design, Settings, and the Slideshow. |
| `scan_pattern.py` | Generates the ordered list of array coordinates for the Measure Resistance tab's scan patterns (row-major, column-major, serpentine, spiral, or custom). |
| `resistance_parser.py` | Turns a spoken transcript into a resistance value, a special "open"/"short" marker, or a voice command, handling both digit-style and spelled-out numbers. |
| `voice_engine.py` | Microphone capture and speech-to-text for Measure Resistance — a shared interface over the Whisper and Vosk engines, plus automatic utterance segmentation. |
| `resistance_interpretation.py` | The analysis behind Interpret Results: open/short classification, outlier rejection, and the Ambegaokar-Baratoff linear regression. |
| `klayout_pya_shim.py` | A minimal, self-contained stand-in for the part of KLayout's own scripting API a typical junction-generator script uses, so that script runs unmodified without KLayout itself installed. |

---

## Building a Desktop App

Running `python3 main_gui.py` works from any terminal, but the app can also be packaged into a real, double-clickable desktop app with its own icon (the same comb-cross mark used on the Front Page and window icon). See `packaging/BUILD_INSTRUCTIONS.md` for the full walkthrough. This currently covers macOS (build it into a real `.app` in a couple of minutes with one command); the icon has also already been generated in Windows (`.ico`) and Linux (`.png`) formats under `icon_assets/` for a future Windows/Linux build.

---

## Credits

Made by **Alexander Braiman** — SQMS Center, Fermilab.

With design and shadow-evaporation guidance from **Francesco Crisa**, and support from **Akshay Murthy**.
