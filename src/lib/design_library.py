"""
design_library.py

Local, persistent storage for junction designs -- both "Created"
(parametric, Tab 1 Design panel -- a DesignParameters snapshot,
including any custom electrode_shapes drawn on the canvas) and
"Imported" (GDS/KLayout-script -- a full ImportedJunction snapshot) --
so a design survives closing the app (e.g. to pick up a new code
iteration) instead of being lost.

GENERALIZABILITY: nothing in this module points at any specific
user's files or file paths. Every location used is computed at
runtime from the operating system's own "where does this app's
per-user data go" convention:

  - Qt's QStandardPaths.AppDataLocation when PySide6/a Qt app context
    is available (the normal case -- this module is imported by the
    Qt GUI): e.g. ~/.local/share/<Org>/<App>/designs on Linux,
    ~/Library/Application Support/<Org>/<App>/designs on macOS,
    %APPDATA%/<Org>/<App>/designs on Windows.
  - A plain ~/.plassys_shadow_sim/designs fallback if Qt isn't
    importable in whatever context this module is used from (e.g. a
    pure-CLI/test script with no Qt event loop at all).
  - The PLASSYS_DESIGN_LIBRARY_DIR environment variable overrides
    both of the above when set -- lets any user relocate storage (a
    synced folder, a shared drive, ...) without touching code, and
    lets tests point at an isolated temp directory.

None of these is a hardcoded absolute path tied to any one person's
account or machine -- every user of the app gets their own library,
in the right place for their own OS, automatically.

One JSON file per saved design, named by a random id (never by the
user-given display name -- names can collide, get renamed, or contain
characters that aren't safe in a filename). The catalog is built by
scanning the directory rather than trusting a separate index file, so
it can never drift out of sync with what's actually on disk.

Round-trips through plain dataclasses.asdict()/reconstruction, the
same pattern resistance_interpretation.py's InterpretationDataset uses
for its own save/load (to_json_dict/from_json_dict) -- proven elsewhere
in this codebase.
"""

import os
import json
import uuid
import datetime
import dataclasses
from dataclasses import asdict
from typing import Optional, List, Dict, Any

from recipe_generator import DesignParameters, SidewallMill, RecipeStep, RecipeHandEdit
from shape_editor import EditableShape
from electrode_convention import ElectrodeConvention
from junction_import import ImportedJunction, ImportedFinger, ImportedShape, SimulationFeature
from recipe_builder_model import PlassysStepRow

SCHEMA_VERSION = 1

_ENV_OVERRIDE = "PLASSYS_DESIGN_LIBRARY_DIR"


# ----------------------------------------------------------------------
# Storage location (see module docstring -- fully portable, no
# hardcoded per-user paths)
# ----------------------------------------------------------------------

def _app_data_dir() -> str:
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return override
    try:
        from PySide6.QtCore import QCoreApplication, QStandardPaths
        # Only set these if nothing (e.g. the running app itself) has
        # already claimed them -- QStandardPaths.AppDataLocation folds
        # organizationName/applicationName into the path it returns, and
        # without them set it falls back to the interpreter's own name
        # (e.g. "python3"), which isn't a stable or sensible location.
        if not QCoreApplication.organizationName():
            QCoreApplication.setOrganizationName("PlassysShadowSim")
        if not QCoreApplication.applicationName():
            QCoreApplication.setApplicationName("PlassysShadowSim")
        base = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if base:
            return os.path.join(base, "designs")
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), ".plassys_shadow_sim", "designs")


def _design_path(design_id: str) -> str:
    return os.path.join(_app_data_dir(), f"{design_id}.json")


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# Reconstruction helpers -- filter unknown keys against each
# dataclass's OWN current fields (same defensive pattern
# InterpretationDataset.from_json_dict uses) so a design saved by an
# older/newer version of the app still loads instead of raising on an
# added/removed field.
# ----------------------------------------------------------------------

def _filtered(cls, d: dict) -> dict:
    known = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in known}


def _shape_from_dict(d: dict) -> EditableShape:
    kwargs = dict(_filtered(EditableShape, d))
    if kwargs.get("points") is not None:
        kwargs["points"] = [tuple(p) for p in kwargs["points"]]
    if kwargs.get("base_points") is not None:
        kwargs["base_points"] = [tuple(p) for p in kwargs["base_points"]]
    return EditableShape(**kwargs)


def _convention_from_dict(d: Optional[dict]) -> Optional[ElectrodeConvention]:
    if d is None:
        return None
    kwargs = dict(_filtered(ElectrodeConvention, d))
    if kwargs.get("layer_override") is not None:
        kwargs["layer_override"] = tuple(kwargs["layer_override"])
    return ElectrodeConvention(**kwargs)


def _mill_from_dict(d: dict) -> SidewallMill:
    return SidewallMill(**_filtered(SidewallMill, d))


def _params_from_dict(d: dict) -> DesignParameters:
    d = dict(d)
    shapes = [_shape_from_dict(s) for s in (d.pop("electrode_shapes", None) or [])]
    conv = _convention_from_dict(d.pop("electrode_convention", None))
    mills = [_mill_from_dict(m) for m in (d.pop("sidewall_mills", None) or [])]
    kwargs = _filtered(DesignParameters, d)
    return DesignParameters(electrode_shapes=shapes, electrode_convention=conv,
                             sidewall_mills=mills, **kwargs)


def _finger_from_dict(d: dict) -> ImportedFinger:
    kwargs = dict(_filtered(ImportedFinger, d))
    for key in ("points",):
        if kwargs.get(key) is not None:
            kwargs[key] = [tuple(p) for p in kwargs[key]]
    for key in ("stub_end", "patch_end", "patch_outward_end", "patch_inward_end"):
        if kwargs.get(key) is not None:
            kwargs[key] = tuple(kwargs[key])
    return ImportedFinger(**kwargs)


def _other_shape_from_dict(d: dict) -> ImportedShape:
    kwargs = dict(_filtered(ImportedShape, d))
    if kwargs.get("points") is not None:
        kwargs["points"] = [tuple(p) for p in kwargs["points"]]
    return ImportedShape(**kwargs)


def _step_row_from_dict(d: dict) -> PlassysStepRow:
    # The auto-generated recipe materializes into real
    # recipe_builder_model.PlassysStepRow entries (see generated_recipe_export.py)
    # instead of a text blob -- a design's per-link "baseline" snapshot
    # (the fresh-generated, pre-hand-edit steps from the generation that
    # produced its CURRENT recipe_builder_links entry) is a list of
    # these, saved/restored the same defensive way every other
    # sub-object here is.
    return PlassysStepRow(**_filtered(PlassysStepRow, d))


def _recipe_step_from_dict(d: dict) -> RecipeStep:
    # A hand-edited recipe (its steps AND its RecipeHandEdit tally --
    # see recipe_generator.RecipeHandEdit) is saved/restored just like the
    # design geometry itself, so a reloaded design shows the same
    # recipe with the same highlighted edits, rather than needing to be
    # regenerated (and losing the hand-edits if it were). RecipeStep is
    # a flat dataclass -- no nested sub-objects -- so a straight
    # _filtered() round-trip is enough.
    return RecipeStep(**_filtered(RecipeStep, d))


def _hand_edit_from_dict(d: dict) -> RecipeHandEdit:
    return RecipeHandEdit(**_filtered(RecipeHandEdit, d))


def _simulation_feature_from_dict(d: dict) -> SimulationFeature:
    # Reloading a design from the library should bring back all of the
    # predicted recipe steps from when the simulation was last run,
    # including any manual edits, without requiring a rerun. A plain,
    # flat dataclass (no nested sub-objects) -- see
    # junction_import.SimulationFeature's own docstring -- so the usual
    # defensive _filtered() round-trip (forward/backward compatible with
    # a field added/removed by a later app version) is all this needs.
    kwargs = dict(_filtered(SimulationFeature, d))
    if kwargs.get("group_source_ids") is not None:
        kwargs["group_source_ids"] = list(kwargs["group_source_ids"])
    return SimulationFeature(**kwargs)


def _junction_from_dict(d: dict) -> ImportedJunction:
    d = dict(d)
    fingers = [_finger_from_dict(f) for f in (d.pop("fingers", None) or [])]
    others = [_other_shape_from_dict(s) for s in (d.pop("other_shapes", None) or [])]
    if d.get("electrode_polygons") is not None:
        d["electrode_polygons"] = [[tuple(pt) for pt in poly] for poly in d["electrode_polygons"]]
    if d.get("electrode_material") is not None:
        d["electrode_material"] = [[tuple(pt) for pt in poly] for poly in d["electrode_material"]]
    kwargs = _filtered(ImportedJunction, d)
    return ImportedJunction(fingers=fingers, other_shapes=others, **kwargs)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def save_design(*, name: str, source: str, notes: str = "",
                 params: Optional[DesignParameters] = None,
                 junction: Optional[ImportedJunction] = None,
                 deposit_first_index: int = 0,
                 role_overrides: Optional[Dict[str, str]] = None,
                 junction_scale_base: Optional[Dict[str, Any]] = None,
                 param_shape_offsets: Optional[Dict[str, Any]] = None,
                 param_shape_edits: Optional[Dict[str, Any]] = None,
                 param_shape_deletions: Optional[Any] = None,
                 param_shape_duplicates: Optional[Dict[str, Any]] = None,
                 design_id: Optional[str] = None,
                 recipe_steps: Optional[List[RecipeStep]] = None,
                 recipe_warnings: Optional[List[str]] = None,
                 recipe_hand_edits: Optional[List[RecipeHandEdit]] = None,
                 recipe_source_label: Optional[str] = None,
                 recipe_params: Optional[DesignParameters] = None,
                 recipe_builder_link: Optional[str] = None,
                 recipe_builder_links: Optional[Dict[str, str]] = None,
                 recipe_builder_baselines: Optional[Dict[str, List[PlassysStepRow]]] = None,
                 sim_recipe_features: Optional[List[SimulationFeature]] = None,
                 sim_has_run: bool = False,
                 sim_angle_diagnostic_run: bool = False,
                 sim_view_mode: Optional[str] = None,
                 recipe_panel_thickness_overrides: Optional[List[Dict[str, str]]] = None,
                 recipe_panel_mill_duration_overrides: Optional[List[Dict[str, str]]] = None) -> str:
    """Writes one design to the library. source is 'created' (a
    DesignParameters snapshot -- pass params) or 'imported' (a static
    ImportedJunction snapshot -- pass junction). Pass design_id to
    overwrite an already-saved design in place (keeps its id and
    original created_at, e.g. a plain "Save" after "Save As..." once);
    omit it to create a new entry. Returns the design's id.

    params may ALSO be passed alongside junction for source='imported'
    -- an Imported design has no use for params' geometry fields
    (electrode_shapes, finger angles/lengths -- its real geometry lives
    entirely in junction), but its SETTINGS (metal types, mill/
    deposition alpha, resist stack, Process Parameters) are just as
    real and just as worth saving with it, so the whole
    design_parameters payload is written either way; the caller
    (main_gui.py's _load_design_from_library) is what applies
    only the settings subset back for an 'imported' load.

    The recipe_* args are all optional and independent of source/params/
    junction. If this design currently has a generated (or AI-adopted)
    recipe on screen, including any hand-edits made to it (see
    recipe_generator.RecipeHandEdit), that whole state is saved too so a
    reloaded design shows the same recipe with the same highlighted
    edits rather than needing to be regenerated. recipe_params is a
    SEPARATE snapshot from `params`/`junction` above (the live design)
    since the recipe may have been generated from a design state that's
    since been tweaked further before this save.

    recipe_builder_link is a THIRD, independent kind of recipe from the
    ones above -- a PATH (e.g. "Root\\Evap\\MyJunction_run3") into the
    separate, shared Recipe Builder tree (recipe_builder_model.py), not a
    snapshot of steps at all. Saving a design also saves the link to
    any by-hand recipe the user created for it. A by-hand recipe is
    deliberately kept as a LINK rather than an embedded copy: the
    Recipe Builder's whole point is a shared library of reusable, named
    recipes that can be brought up and reused across many designs, so
    embedding a private copy per design would fork it away from that
    shared, editable original. This link just remembers which recipe
    (if any) this design was using at save time, so reloading the
    design can jump straight back to it.

    A generated recipe is not just one link -- it's a whole family of
    real recipe-tree nodes this design's generation produced/reused
    (one per mill pass, one per deposition pass, the shared Ti-gather,
    the shared oxidation, plus the final "complete process" recipe
    chaining them). recipe_builder_links replaces the single
    recipe_builder_link with a dict keyed by a stable per-role
    "kind_key" (see generated_recipe_export.GeneratedSubRecipe.kind_key) ->
    current tree path; recipe_builder_baselines is the parallel dict of
    each link's fresh-generated (pre-hand-edit) PlassysStepRow list from
    the generation that produced it -- what the NEXT generation diffs
    against to decide whether to reuse/patch/fork that node (see
    generated_recipe_export.resolve_and_write's "duplicate-and-patch" logic).
    recipe_builder_link (singular) is kept as a DEPRECATED, still-
    written/still-read convenience for any older caller: when
    recipe_builder_links is given, recipe_builder_link is auto-derived
    from its "complete_process" entry unless explicitly overridden.

    Reloading a design from the library should bring back all of the
    predicted recipe steps from when the simulation (angle diagnostic +
    simulation) was last run, without requiring a rerun, with any
    manual edits carefully preserved -- including the parameters
    hand-typed in the recipe tab. Five more independent, all-
    optional fields for this: sim_recipe_features is Simulate Design's
    live, editable self._recipe_features list (the Predicted Recipe
    Sequence card -- including any manual_override angle/theta edit and
    any hand-inserted "__manual_insert__" freeform step); sim_has_run/
    sim_angle_diagnostic_run/sim_view_mode are the two-step-workflow
    gating flags (self._sim_has_run/_sim_angle_diagnostic_run/
    _sim_view_mode) that decide whether a reload can show the same
    Simulated/Post-Shadow views immediately, with no re-click needed.
    self._sim_finger_layers/_sim_patch_layers/thicknesses/alphas -- the
    lower-level state the actual shadow math reads -- are deliberately
    NOT saved separately: they're a documented lossless, order-
    preserving derivation of sim_recipe_features alone (see
    main_gui._sync_sim_layers_from_recipe_features's own
    docstring), so restoring sim_recipe_features and re-running that
    same derivation is enough, and avoids two copies of the same state
    silently drifting apart. recipe_panel_thickness_overrides/
    recipe_panel_mill_duration_overrides are the Recipe Parameters
    ("this generation only") panel's per-pass hand-typed text fields --
    saved as plain {"kind","label","text"}/{"base","text"} dict lists
    (rather than a dict keyed by a tuple, which JSON can't represent)
    since those widgets are rebuilt fresh, keyed by pass identity, every
    time the predicted sequence itself changes."""
    if source not in ("created", "imported"):
        raise ValueError(f"source must be 'created' or 'imported', got {source!r}")
    if source == "created" and params is None:
        raise ValueError("source='created' requires params")
    if source == "imported" and junction is None:
        raise ValueError("source='imported' requires junction")

    now = _now()
    created_at = now
    if design_id:
        try:
            with open(_design_path(design_id)) as f:
                created_at = json.load(f).get("created_at", now)
        except Exception:
            pass
    else:
        design_id = uuid.uuid4().hex

    payload = {
        "schema_version": SCHEMA_VERSION,
        "id": design_id,
        "name": name.strip() or "Untitled design",
        "notes": notes,
        "source": source,
        "created_at": created_at,
        "updated_at": now,
        "deposit_first_index": deposit_first_index,
        "role_overrides": dict(role_overrides or {}),
        "junction_scale_base": dict(junction_scale_base or {}),
        # Bug fix: a PARAMETRIC design's duplicated/edited/deleted
        # fingers/patches/plates (Ctrl+D, the Shape Properties card,
        # Delete Selected -- see _param_shape_duplicates/_edits/
        # _deletions/_offsets' own comments in main_gui.py) live
        # ONLY as MainWindow session state, re-applied to the
        # freshly-rebuilt junction every _update_preview() call -- they
        # were never part of this payload at all, so a duplicated
        # finger (or an edited/deleted one) previously vanished
        # silently on reload, reverting to whatever plain
        # DesignParameters alone describes. Persisted here the same
        # session-state-alongside-design_parameters way role_overrides/
        # junction_scale_base already are (not folded into
        # DesignParameters itself, since they're keyed by transient
        # stable_ids, not real recipe/geometry parameters).
        "param_shape_offsets": dict(param_shape_offsets or {}),
        "param_shape_edits": dict(param_shape_edits or {}),
        "param_shape_deletions": list(param_shape_deletions or []),
        "param_shape_duplicates": dict(param_shape_duplicates or {}),
        "design_parameters": asdict(params) if params is not None else None,
        "imported_junction": asdict(junction) if junction is not None else None,
        "recipe_steps": [asdict(s) for s in recipe_steps] if recipe_steps else None,
        "recipe_warnings": list(recipe_warnings) if recipe_warnings else None,
        "recipe_hand_edits": [asdict(h) for h in recipe_hand_edits] if recipe_hand_edits else None,
        "recipe_source_label": recipe_source_label,
        "recipe_params": asdict(recipe_params) if recipe_params is not None else None,
        "recipe_builder_link": recipe_builder_link if recipe_builder_link is not None
                                 else (dict(recipe_builder_links or {}).get("complete_process")),
        "recipe_builder_links": dict(recipe_builder_links) if recipe_builder_links else None,
        "recipe_builder_baselines": (
            {k: [asdict(s) for s in v] for k, v in recipe_builder_baselines.items()}
            if recipe_builder_baselines else None
        ),
        "sim_recipe_features": [asdict(f) for f in sim_recipe_features] if sim_recipe_features else None,
        "sim_has_run": bool(sim_has_run),
        "sim_angle_diagnostic_run": bool(sim_angle_diagnostic_run),
        "sim_view_mode": sim_view_mode,
        "recipe_panel_thickness_overrides": list(recipe_panel_thickness_overrides or []),
        "recipe_panel_mill_duration_overrides": list(recipe_panel_mill_duration_overrides or []),
    }

    directory = _app_data_dir()
    os.makedirs(directory, exist_ok=True)
    path = _design_path(design_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)  # atomic-ish: never leaves a half-written .json behind
    return design_id


def _links_with_legacy_fallback(payload: dict) -> Dict[str, str]:
    """A design saved before recipe_builder_links existed only has the
    old singular recipe_builder_link -- map it onto {"complete_process":
    link} so every caller can read recipe_builder_links uniformly
    regardless of which version of the app saved this design."""
    links = payload.get("recipe_builder_links")
    if links:
        return dict(links)
    legacy = payload.get("recipe_builder_link")
    return {"complete_process": legacy} if legacy else {}


def _summarize(payload: dict) -> dict:
    summary = {
        "id": payload.get("id"),
        "name": payload.get("name", "Untitled design"),
        "notes": payload.get("notes", ""),
        "source": payload.get("source", "created"),
        "created_at": payload.get("created_at", ""),
        "updated_at": payload.get("updated_at", payload.get("created_at", "")),
        "finger_count": 0,
        "shape_count": 0,
        "detail": "",
        "has_recipe": bool(payload.get("recipe_steps")),
        "hand_edit_count": len(payload.get("recipe_hand_edits") or []),
        "recipe_builder_link": payload.get("recipe_builder_link"),
        "recipe_builder_links": _links_with_legacy_fallback(payload),
    }
    if payload.get("source") == "imported" and payload.get("imported_junction"):
        j = payload["imported_junction"]
        summary["finger_count"] = len(j.get("fingers") or [])
        summary["shape_count"] = len(j.get("other_shapes") or [])
        summary["detail"] = j.get("source_label") or "Imported junction"
    elif payload.get("design_parameters"):
        p = payload["design_parameters"]
        summary["finger_count"] = 2
        summary["shape_count"] = len(p.get("electrode_shapes") or [])
        bits = [p.get("metal_type", "") or "Al"]
        if p.get("patch_integrated"):
            bits.append("patch-integrated")
        if p.get("has_adhesion_pads"):
            bits.append("adhesion pads")
        summary["detail"] = ", ".join(bits)
    return summary


def list_designs() -> List[dict]:
    """Catalog listing (metadata only -- name/notes/source/timestamps/
    a short detail string) for every saved design, newest-updated
    first. Built by scanning the storage directory directly rather
    than a separate index, so it can never go stale relative to what's
    actually saved."""
    directory = _app_data_dir()
    if not os.path.isdir(directory):
        return []
    out = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json") or fn.endswith(".tmp"):
            continue
        try:
            with open(os.path.join(directory, fn)) as f:
                payload = json.load(f)
            if "id" not in payload:
                continue
            out.append(_summarize(payload))
        except Exception:
            continue  # a corrupt/foreign file in the folder shouldn't break the whole catalog
    out.sort(key=lambda s: s["updated_at"], reverse=True)
    return out


def load_design(design_id: str) -> dict:
    """Full round-trip load: reconstructs real DesignParameters (with
    its EditableShape/ElectrodeConvention/SidewallMill sub-objects) or
    a real ImportedJunction (with its ImportedFinger/ImportedShape
    sub-objects) -- ready to hand straight back to the app's own
    display-state fields."""
    with open(_design_path(design_id)) as f:
        payload = json.load(f)
    return {
        "id": payload["id"],
        "name": payload.get("name", "Untitled design"),
        "notes": payload.get("notes", ""),
        "source": payload.get("source", "created"),
        "created_at": payload.get("created_at", ""),
        "updated_at": payload.get("updated_at", ""),
        "deposit_first_index": payload.get("deposit_first_index", 0),
        "role_overrides": dict(payload.get("role_overrides") or {}),
        # Bug fix: a group's scale base used to be a flat
        # (length_um, width_um) 2-tuple per shape -- JSON round-trips a
        # tuple as a plain array, so `tuple(v)` faithfully rebuilt it.
        # A later group-scale rework (see
        # main_gui._apply_junction_group_scale) replaced that with
        # a richer {"origin": [...], "fingers": {...}, "shapes": {...}}
        # DICT per group -- and this line's blind `tuple(v)` was still
        # firing on every entry regardless of shape, silently turning
        # that dict into a tuple of its own KEYS (`('fingers', 'shapes',
        # 'origin')`), destroying the origin/fingers/shapes payload on
        # every single load, new saves included -- caught by an actual
        # save/load round-trip check, not just by reading the code.
        # Passed through untouched here for a dict (every list/tuple
        # inside it -- origin's (x,y), a finger's (stub, patch_end,
        # width) -- is consumed purely by indexing/unpacking downstream,
        # which works identically on a JSON list or a real tuple, so no
        # further reconstruction is needed); old-format plain arrays
        # still get the original tuple(v) treatment so a design saved
        # before this fix keeps loading exactly as it always did.
        "junction_scale_base": {
            k: (v if isinstance(v, dict) else tuple(v))
            for k, v in (payload.get("junction_scale_base") or {}).items()
        },
        # See save_design's own comment: absent on any design saved
        # before this fix, in which case the {}/{}/set()/{} defaults
        # below are exactly the old (broken but harmless) "nothing to
        # restore" behavior, not a new failure mode.
        "param_shape_offsets": {k: tuple(v) for k, v in (payload.get("param_shape_offsets") or {}).items()},
        "param_shape_edits": dict(payload.get("param_shape_edits") or {}),
        "param_shape_deletions": set(payload.get("param_shape_deletions") or []),
        "param_shape_duplicates": dict(payload.get("param_shape_duplicates") or {}),
        "params": _params_from_dict(payload["design_parameters"]) if payload.get("design_parameters") else None,
        "junction": _junction_from_dict(payload["imported_junction"]) if payload.get("imported_junction") else None,
        "recipe_steps": [_recipe_step_from_dict(s) for s in (payload.get("recipe_steps") or [])],
        "recipe_warnings": list(payload.get("recipe_warnings") or []),
        "recipe_hand_edits": [_hand_edit_from_dict(h) for h in (payload.get("recipe_hand_edits") or [])],
        "recipe_source_label": payload.get("recipe_source_label"),
        "recipe_params": _params_from_dict(payload["recipe_params"]) if payload.get("recipe_params") else None,
        "recipe_builder_link": payload.get("recipe_builder_link"),
        "recipe_builder_links": _links_with_legacy_fallback(payload),
        "recipe_builder_baselines": {
            k: [_step_row_from_dict(s) for s in v]
            for k, v in (payload.get("recipe_builder_baselines") or {}).items()
        },
        # See save_design's own comment: absent on any design saved
        # before this fix, in which case the []/False/False/None/[]/[]
        # defaults below mean "nothing to restore" -- the caller
        # (main_gui._load_design_from_library) treats an empty
        # sim_recipe_features list as "fall back to today's behavior",
        # so an older save loads exactly as it always has.
        "sim_recipe_features": [_simulation_feature_from_dict(f) for f in (payload.get("sim_recipe_features") or [])],
        "sim_has_run": bool(payload.get("sim_has_run", False)),
        "sim_angle_diagnostic_run": bool(payload.get("sim_angle_diagnostic_run", False)),
        "sim_view_mode": payload.get("sim_view_mode"),
        "recipe_panel_thickness_overrides": list(payload.get("recipe_panel_thickness_overrides") or []),
        "recipe_panel_mill_duration_overrides": list(payload.get("recipe_panel_mill_duration_overrides") or []),
    }


def delete_design(design_id: str) -> None:
    path = _design_path(design_id)
    if os.path.exists(path):
        os.remove(path)


def rename_design(design_id: str, new_name: str, new_notes: Optional[str] = None) -> None:
    path = _design_path(design_id)
    with open(path) as f:
        payload = json.load(f)
    payload["name"] = new_name.strip() or "Untitled design"
    if new_notes is not None:
        payload["notes"] = new_notes
    payload["updated_at"] = _now()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
