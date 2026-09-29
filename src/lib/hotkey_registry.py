"""
hotkey_registry.py

Central catalog of every keyboard shortcut in the Plassys app -- both the
handful that already existed (File/Edit/View menu items, arrow-key tab
and slideshow navigation) and the new ones added alongside the new
Settings > Hotkeys database (Delete Selected Shape, Duplicate Selected
Shape, Deselect, Zoom to Fit, Reset View). The goal is a hotkey system
that is easy to inspect and customize -- add, remove, or rebind any
shortcut by clicking a key -- with a single, clearly organized database
that also covers every hotkey that already existed before this database
was introduced (menu-item shortcuts and the arrow-key navigation).

Kept as its own small, GUI-independent module -- same reasoning
electrode_convention.py gives for staying dependency-free -- so the
canonical hotkey list and its persistence logic can be imported/tested
without booting a QApplication, and so main_gui.py doesn't carry
this bookkeeping inline.

Every entry is one of two "kind"s, and this distinction matters:

  - kind="action": a real QAction.setShortcut(...) -- Qt's own shortcut
    map dispatches it, application-wide, regardless of which widget has
    focus. This is safe to reassign to any key for exactly the same
    reason the existing Ctrl+Z/Ctrl+S shortcuts already coexist fine
    with normal text-field editing: Qt's standard editing widgets
    (QLineEdit etc.) send a ShortcutOverride event to claim the small
    set of QKeySequence::StandardKey bindings (Undo, Redo, Copy, Paste,
    Delete, ...) for themselves FIRST, before the application-level
    QAction ever sees them -- confirmed by this app already shipping
    Ctrl+Z as a QAction shortcut without breaking any QLineEdit's own
    undo. Delete/Backspace fall under that same protection.

  - kind="keyevent": main_gui.py's own MainWindow.keyPressEvent
    reads the resolved (key, modifiers) pair directly, via plain Qt
    key-event bubbling -- NOT a QAction shortcut. This is deliberate,
    not an oversight: Escape is NOT one of the StandardKey bindings a
    focused widget auto-protects, so a global QAction shortcut on
    Escape would silently steal it away from _InlineHandEditPopup's own
    Escape-cancels-popup handling. Plain keyPressEvent bubbling means
    this hotkey only ever fires once no more-specific focused widget
    has already claimed the key for its own purpose -- see
    MainWindow.keyPressEvent's own docstring for the full reasoning.
    Arrow-key tab/slideshow navigation stays on this same mechanism for
    the identical reason (so a focused QLineEdit/QListWidget/spin box
    keeps its own arrow-key behavior).

On macOS, the default bindings for deleting and duplicating a selected
shape were effectively unreachable. Root-caused to two SEPARATE,
unrelated gaps -- neither is a Cmd-vs-Ctrl problem (Qt already remaps
that transparently, see build_key_combo_text's own docstring below):

  1. delete_selected's only bound key was "Del", which Qt maps to
     Qt.Key_Delete (forward-delete). A MacBook's own built-in keyboard
     has no dedicated forward-delete key -- its single "delete" key
     sends Qt.Key_Backspace; genuine forward-delete needs Fn+Delete,
     which most users never press or realize exists. So "Del" alone is
     effectively unreachable on a Mac laptop keyboard.
  2. duplicate_selected's only bound key was "Ctrl+D" (Cmd+D on real
     macOS hardware, via Qt's remap) -- never "Ctrl+C", the far more
     common "duplicate" convention in other design tools (copy-in-place,
     no separate paste step needed), which this app's own binding
     didn't match.

Rather than just swapping the default key (which would only relocate
the same kind of surprise for whichever key was dropped), ALIAS_KEYS
below adds a small, fixed set of EXTRA keys that a "action"-kind entry
always also answers to, unconditionally, alongside its own real
(possibly user-customized) current_key -- see _apply_hotkey_entry in
main_gui.py, which now calls QAction.setShortcuts (plural) with
current_key plus every one of that entry's own aliases. Aliases are
deliberately NOT user-editable, NOT persisted to hotkeys.json, and NOT
considered by find_conflicts below -- they are a fixed compatibility
layer, not a customizable binding a user could reassign or collide.
"""
import os
import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from PySide6.QtCore import QStandardPaths, QCoreApplication
    from PySide6.QtGui import QKeySequence
except Exception:  # pragma: no cover - only during non-Qt unit testing
    QStandardPaths = None
    QCoreApplication = None
    QKeySequence = None

try:
    # Qt6 split a key and its modifiers into two separate flag/enum types
    # (Qt.Key, Qt.KeyboardModifier) that can no longer be bitwise-OR'd
    # together into one plain int the way Qt5 allowed -- QKeyCombination is
    # Qt6's replacement for that combination step. It lives under QtCore in
    # every PySide6 build observed so far, but the import is wrapped
    # defensively (see build_key_combo_text's fallback) in case a future/
    # older PySide6 build organizes it differently or omits it.
    from PySide6.QtCore import QKeyCombination
except Exception:  # pragma: no cover - only if QKeyCombination is unavailable
    QKeyCombination = None


@dataclass
class HotkeyEntry:
    id: str
    label: str
    category: str
    default_key: str
    kind: str  # "action" | "keyevent"
    description: str = ""
    current_key: Optional[str] = None  # None (not "") means "use default_key" --
                                         # see merge_defaults_with_overrides/__post_init__.
                                         # An explicit "" means "no hotkey assigned."

    def __post_init__(self):
        if self.current_key is None:
            self.current_key = self.default_key

    @property
    def is_customized(self) -> bool:
        return self.current_key != self.default_key


# The canonical, ordered list -- order here is the order the Settings >
# Hotkeys table displays within each category.
DEFAULT_HOTKEYS: List[HotkeyEntry] = [
    # -- File --
    HotkeyEntry("new_junction", "New Junction", "File", "Ctrl+N", "action",
                "Clears the current design and starts a fresh parametric junction."),
    HotkeyEntry("save_design", "Save Design", "File", "Ctrl+S", "action",
                "Saves the active design back to the Design Library."),
    HotkeyEntry("open_design_library", "Open Design Library", "File", "Ctrl+L", "action",
                "Opens the Design Library browser."),
    # -- Edit --
    HotkeyEntry("undo", "Undo", "Edit", "Ctrl+Z", "action", "Undoes the last design edit."),
    HotkeyEntry("redo", "Redo", "Edit", "Ctrl+Shift+Z", "action", "Redoes the last undone edit."),
    HotkeyEntry("delete_selected", "Delete Selected Shape", "Edit", "Del", "action",
                "Deletes the shape selected in Create/Import -- a hand-drawn custom shape, or a "
                "real finger/patch/plate belonging to either a created or imported junction (or "
                "clears a role override on one that has no other properties to delete). Undoable. "
                "Always also answers to Backspace (see ALIAS_KEYS below), since a Mac laptop's own "
                "delete key sends Backspace, not Del."),
    HotkeyEntry("duplicate_selected", "Duplicate Selected Shape", "Edit", "Ctrl+D", "action",
                "Clones the selected shape a short offset away -- a hand-drawn custom shape, or a "
                "real finger/patch/plate belonging to either a created or imported junction. "
                "Undoable. Always also answers to Ctrl+C/Cmd+C (see ALIAS_KEYS below), the more "
                "common convention in most design tools."),
    HotkeyEntry("deselect", "Deselect", "Edit", "Esc", "keyevent",
                "Clears whichever shape or junction finger/patch/plate is currently selected."),
    # -- View --
    HotkeyEntry("zoom_to_fit", "Zoom to Fit", "View", "Ctrl+0", "action",
                "Fits the Create/Import design to the view."),
    HotkeyEntry("reset_view", "Reset View", "View", "Ctrl+9", "action",
                "Fits the view and clears rulers/selection highlight."),
    # -- Navigation --
    HotkeyEntry("nav_prev_tab", "Previous Tab", "Navigation", "Left", "keyevent",
                "Moves to the previous tab in the left nav rail. Same key everywhere in the "
                "window except where a focused field already uses it (text cursor, spin box, "
                "list row selection)."),
    HotkeyEntry("nav_next_tab", "Next Tab", "Navigation", "Right", "keyevent",
                "Moves to the next tab in the left nav rail."),
    # -- Deposition Slideshow --
    HotkeyEntry("slideshow_prev", "Previous Step", "Deposition Slideshow", "Up", "keyevent",
                "Steps back one recipe step. Only active while the Deposition Slideshow tab "
                "is open."),
    HotkeyEntry("slideshow_next", "Next Step", "Deposition Slideshow", "Down", "keyevent",
                "Steps forward one recipe step. Only active while the Deposition Slideshow "
                "tab is open."),
]


ALIAS_KEYS: Dict[str, List[str]] = {
    # See this module's own top-of-file docstring for the exact bug and
    # root cause each of these fixes. Only kind="action" entries are ever
    # looked up here (see
    # _apply_hotkey_entry in main_gui.py); a kind="keyevent" entry
    # has no alias mechanism since MainWindow.keyPressEvent compares
    # against exactly one parsed combo per entry.
    "delete_selected": ["Backspace"],
    "duplicate_selected": ["Ctrl+C"],
}


_ENV_OVERRIDE = "PLASSYS_HOTKEYS_DIR"  # same override convention design_library.py
                                          # already established (PLASSYS_DESIGN_LIBRARY_DIR)
                                          # -- lets tests point this at a throwaway directory
                                          # instead of touching the real per-user app-data folder


def _app_data_dir() -> str:
    """Same location convention design_library.py already established
    for saved designs (QStandardPaths.AppDataLocation, falling back to
    ~/.plassys_shadow_sim) -- hotkeys.json is a sibling of designs/, not
    nested inside it, since it isn't a design."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        return override
    try:
        if QStandardPaths is not None:
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


def _hotkeys_path() -> str:
    return os.path.join(_app_data_dir(), "hotkeys.json")


def load_overrides() -> Dict[str, str]:
    """Returns {hotkey_id: current_key} for every entry the user has
    ever touched (including one deliberately cleared to "", meaning "no
    hotkey assigned") -- entries left at their default are never
    written, keeping the file a sparse diff against DEFAULT_HOTKEYS. A
    missing, unreadable, or corrupt file quietly returns {} (all
    defaults) rather than raising -- a bad hotkeys.json should never
    stop the app from launching."""
    try:
        with open(_hotkeys_path()) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except Exception:
        pass
    return {}


def save_overrides(overrides: Dict[str, str]) -> None:
    """Atomic write (tmp file + os.replace), matching design_library.py's
    own save pattern, so a crash mid-write never leaves a half-written
    hotkeys.json behind that load_overrides() would then choke on."""
    directory = _app_data_dir()
    os.makedirs(directory, exist_ok=True)
    path = _hotkeys_path()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(overrides, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def merge_defaults_with_overrides(overrides: Optional[Dict[str, str]] = None) -> List[HotkeyEntry]:
    """Fresh HotkeyEntry list (safe for one MainWindow instance to hold
    and mutate without touching DEFAULT_HOTKEYS itself) with current_key
    set from `overrides` where present, else left at each entry's own
    default_key. Pass overrides explicitly to merge an in-memory dict
    (e.g. right after a Settings > Hotkeys edit, before it's saved);
    omit it to load from disk."""
    if overrides is None:
        overrides = load_overrides()
    out = []
    for entry in DEFAULT_HOTKEYS:
        current = overrides.get(entry.id, entry.default_key)
        out.append(dataclasses.replace(entry, current_key=current))
    return out


def entries_to_overrides(entries: List[HotkeyEntry]) -> Dict[str, str]:
    """The sparse dict save_overrides() expects: only entries whose
    current_key differs from their own default_key."""
    return {e.id: e.current_key for e in entries if e.is_customized}


def find_conflicts(entries: List[HotkeyEntry]) -> Dict[str, List[str]]:
    """Maps a key-sequence string -> list of entry ids sharing it, for
    every key currently used by 2+ entries -- surfaced as a warning in
    the Settings > Hotkeys table rather than letting two actions
    silently fight over the same key. An entry with no hotkey assigned
    (current_key == "") is never treated as a conflict.

    Only checks each entry's own PRIMARY current_key -- never
    ALIAS_KEYS. Aliases are a fixed, always-on compatibility layer (see
    ALIAS_KEYS' own comment above), not a user-assignable binding, so
    they're deliberately excluded from conflict detection the same way
    they're excluded from persistence and from the Settings > Hotkeys
    editor."""
    by_key: Dict[str, List[str]] = {}
    for entry in entries:
        key = (entry.current_key or "").strip()
        if not key:
            continue
        by_key.setdefault(key, []).append(entry.id)
    return {key: ids for key, ids in by_key.items() if len(ids) > 1}


def parse_key_combo(key_text: str) -> Optional[Tuple[int, object]]:
    """Parses a stored key-sequence string (Qt PortableText, e.g.
    "Ctrl+D", "Del", "Esc", "Left") into the (key, modifiers) pair a
    real QKeyEvent reports -- what a kind="keyevent" entry needs
    MainWindow.keyPressEvent to compare against directly (see this
    module's docstring for why those stay on plain key-event bubbling
    instead of a QAction shortcut). Returns None for an empty/
    unassigned string or one Qt can't parse as a single key press."""
    if not key_text or QKeySequence is None:
        return None
    seq = QKeySequence(key_text)
    if seq.count() == 0:
        return None
    combo = seq[0]
    return (combo.key(), combo.keyboardModifiers())


def build_key_combo_text(key: int, modifiers) -> str:
    """The other direction of parse_key_combo: turns a live (Qt.Key,
    Qt.KeyboardModifier) pair -- exactly what a real QKeyEvent reports
    while _HotkeyCaptureButton is listening -- into the same Qt
    PortableText string every hotkey is stored/compared as (e.g.
    "Ctrl+Shift+D").

    This is the fix for a real bug: the original capture code built the
    QKeySequence as `key | int(modifiers)`, copying Qt5's convention where
    Key and KeyboardModifier were both plain ints. On PySide6 6.x
    (Qt6), KeyboardModifier is a flag type with no __int__ any more --
    `int(event.modifiers())` raises TypeError -- and because this ran
    inside an overridden Qt virtual method (keyPressEvent), the exception
    couldn't propagate normally: depending on the platform/PySide6 build
    this either silently aborted the event (nothing happened -- reported
    as "pressing any key doesn't set the bind") or crashed outright.
    QKeyCombination is Qt6's supported way to combine the two again.

    Any number of simultaneously-held real modifier keys (Ctrl/Shift/
    Alt/Meta, in any combination -- e.g. Ctrl+Shift+Alt+A) round-trip
    correctly through this. Cmd-vs-Ctrl on macOS needs no special-casing
    here: Qt itself remaps which physical key reports as ControlModifier
    vs MetaModifier per-platform, transparently, before this function
    ever sees it. Callers are expected to have already stripped
    KeypadModifier/GroupSwitchModifier if they don't want a numpad key or
    an AltGr-typed character baking that physical detail into the saved
    binding (_HotkeyCaptureButton.keyPressEvent does this)."""
    seq = None
    if QKeyCombination is not None:
        try:
            seq = QKeySequence(QKeyCombination(modifiers, key))
        except Exception:
            seq = None
    if seq is None:
        # Fallback if QKeyCombination isn't available: reproduce the old
        # Qt5-style int combination, but tolerate a modifiers value that
        # doesn't support int() directly (Qt6 flag types expose .value).
        try:
            mods_int = int(modifiers)
        except TypeError:
            mods_int = int(getattr(modifiers, "value", 0))
        seq = QKeySequence(int(key) | mods_int)
    return seq.toString(QKeySequence.PortableText)
