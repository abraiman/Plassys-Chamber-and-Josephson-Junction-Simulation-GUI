# Turning Plassys Shadow Simulator into a real Mac app

This turns the app into a normal double-clickable `.app` you can put in your
Applications folder or Dock, with its own real icon (the same comb-cross mark
from the Front Page and window icon).

This has to be done **on your own Mac** — not in this chat session. The
packaging tool (PyInstaller) bundles the actual, compiled Qt libraries for
whichever computer it runs on, so a Mac app can only be built on a Mac. What's
provided here is everything needed to do that build in a few minutes.

## What's in this folder

- `plassys_app.spec` — the build recipe (tells PyInstaller what to bundle,
  what to name it, and which icon to use). You won't need to edit this.
- `build_macos_app.sh` — runs the build for you with one command.
- `BUILD_INSTRUCTIONS.md` — this file.

The icon itself lives one folder up, in `../icon_assets/app_icon.icns`.

## Step 1 — Install the app's own dependencies

If you haven't already, follow `requirements.txt` (one folder up) to install
Python 3.11+, PySide6, and the rest of the required packages. You do not need
the optional voice/AI extras just to build the app — only if you want those
features to work once it's running.

## Step 2 — Run the build

Open Terminal, navigate into this `packaging` folder, and run:

```
bash build_macos_app.sh
```

This will:
1. Install PyInstaller itself (the packaging tool — a one-time install, not
   something the app needs at runtime).
2. Build the app, which takes a few minutes the first time.
3. Tell you where the finished app landed:
   `packaging/dist/Plassys Shadow Simulator.app`

## Step 3 — Use it

Drag `Plassys Shadow Simulator.app` into your `/Applications` folder (or
anywhere else you like — the Dock, a project folder, your Desktop). From then
on, double-clicking it launches the app directly, with the real icon showing
in Finder, the Dock, and Cmd+Tab.

## First launch on macOS

The very first time you open an app that wasn't downloaded from the App
Store or signed with a paid Apple Developer certificate, macOS's Gatekeeper
blocks it and shows a warning — this is expected for an app you just built
locally, not a sign anything is wrong. Two ways to clear it, one time only:

- **Right-click (or Control-click) the app → Open → Open.** This is the
  simplest option and only needs doing once; every launch after that works
  normally with a plain double-click.
- If macOS instead says the app **"is damaged and can't be opened,"** that's
  the same underlying Gatekeeper quarantine flag, just a stricter-sounding
  message for a locally-built app. Clear it from Terminal with:
  ```
  xattr -cr "/Applications/Plassys Shadow Simulator.app"
  ```
  (adjust the path if you put it somewhere other than `/Applications`), then
  double-click normally.

Neither of these is a workaround for a bug — it's standard macOS behavior for
any app that isn't distributed through the App Store or notarized by Apple
(which requires a paid $99/year Apple Developer account and is a separate,
optional step well beyond what's needed to just run this yourself or share it
directly with your advisors).

## Rebuilding after a code change

If you edit any of the app's `.py` files later and want the app bundle to
reflect the change, just re-run `bash build_macos_app.sh` again — it cleans
up the previous build automatically before rebuilding.

## Troubleshooting

- **The build fails partway through with a `ModuleNotFoundError` for some
  package.** Make sure you've installed everything in `requirements.txt`
  first (Step 1) — PyInstaller can only bundle packages that are actually
  installed in the Python environment you run it from.
- **The app builds fine but crashes immediately on launch, with no visible
  error.** Try running the built executable directly from Terminal instead
  of double-clicking, so you can see the real error message:
  ```
  "./dist/Plassys Shadow Simulator.app/Contents/MacOS/Plassys Shadow Simulator"
  ```
- **That error is `ImportError: Failed to import any of the following Qt
  binding modules: PyQt6, PySide6, PyQt5, PySide2`.** This is a
  well-documented PyInstaller + matplotlib packaging quirk, not a sign
  PySide6 didn't get bundled: matplotlib tries to auto-detect a Qt binding
  by test-importing each one in turn, and that detection can fail inside a
  frozen app even when PySide6 itself is present and working. `main_gui.py`
  already works around this by setting `QT_API=PySide6` before matplotlib
  is ever imported, so a fresh build shouldn't hit this. If you're seeing it
  anyway, you likely have an older copy of `main_gui.py` from before this
  fix — grab the current release copy and rebuild.
- **That error is `ModuleNotFoundError: No module named 'PySide6'`** (note:
  no module, not the "Failed to import any of..." message above — these
  look similar but mean different things). This means the `python3` you
  ran `bash build_macos_app.sh` with does not have PySide6 installed, even
  if some other Python on your Mac does (common with pyenv, Homebrew
  Python, and the system Python all coexisting). Confirm which one you're
  actually using before rebuilding:
  ```
  python3 -c "import PySide6; print(PySide6.__file__)"
  ```
  If that also fails, install everything into that exact interpreter —
  from the `plassys_release` folder (one level up from `packaging`):
  ```
  python3 -m pip install -r ../requirements.txt
  ```
  then re-run `bash build_macos_app.sh`.
- **The app builds and launches, but immediately shows a traceback ending
  in `AttributeError: '_SixMetaPathImporter' object has no attribute
  '_path'`.** This is a separate, unrelated-looking symptom of the exact
  same PySide6/shiboken6 quirk described above, triggered by a different,
  unavoidable part of matplotlib (its date-axis support, which pulls in
  `dateutil` and `six`) rather than the Qt-binding detection. `main_gui.py`
  permanently disables the one function inside PySide6's import hook that
  causes this, for the whole life of the app, so a current copy shouldn't
  hit this at all. If you're seeing it anyway, you have an older copy of
  `main_gui.py` from before this fix — grab the current release copy and
  rebuild.
- **The app builds and launches, but immediately shows a traceback ending
  in `NameError: name 'obj' is not defined`, inside
  `scipy/stats/_distn_infrastructure.py`.** This one is *not* caused by
  this app, scipy, or PyInstaller's design — it's a confirmed bug in
  Python **3.12.0 itself** (specifically in `code.replace()`, which
  PyInstaller's build step relies on internally), already fixed upstream
  for Python 3.12.1 and later. Check your exact version:
  ```
  python3 --version
  ```
  If it prints exactly `3.12.0`, install a newer 3.12.x (or 3.11.x) —
  with pyenv:
  ```
  pyenv install --list | grep " 3.12"
  pyenv install 3.12.<latest from that list>
  cd /path/to/plassys_release
  pyenv local 3.12.<that version>
  python3 --version                        # confirm it changed
  python3 -m pip install -r requirements.txt
  ```
  Then rebuild from `packaging/` as usual. No code fix can work around
  this — it's baked into how the exact Python interpreter you build with
  compiles the app, not into anything this app's own `main_gui.py` does.
- **Icons/toolbar buttons are missing or blank in the built app, even though
  they show up fine when you run `python3 main_gui.py` directly.** This
  almost always means qtawesome's font files didn't get bundled — the spec
  file already collects these explicitly, but if you've upgraded qtawesome
  since this spec file was written, re-run the build; if it persists, add
  `--collect-all qtawesome` to the PyInstaller command as a stronger
  fallback (edit `build_macos_app.sh`'s `python3 -m PyInstaller` line).
- **Plots/graphs are missing or matplotlib raises a font-related error in
  the built app.** Same idea as above, but for `matplotlib` — try
  `--collect-all matplotlib` as a fallback.
- **Something else entirely.** PyInstaller's own error output (scroll up in
  Terminal after a failed build) almost always names the exact missing
  package or file — searching that exact error message alongside
  "pyinstaller" usually turns up the fix quickly, since these are common,
  well-documented packaging issues rather than anything specific to this
  app's own code.
