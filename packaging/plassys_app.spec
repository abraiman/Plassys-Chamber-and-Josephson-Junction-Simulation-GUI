# -----------------------------------------------------------------------------
# PyInstaller spec file -- builds Plassys Shadow Simulator into a real,
# double-clickable macOS .app, with its own icon (the same comb-cross mark
# used on the Front Page and window icon).
#
# You do not run this file directly with `python3`. Run it with PyInstaller,
# from the `packaging/` folder:
#
#     pyinstaller plassys_app.spec
#
# See BUILD_INSTRUCTIONS.md in this same folder for the full step-by-step
# process (installing PyInstaller, running this, and what to do with the
# result). The short version: this must be run ON A MAC, because PyInstaller
# bundles the real, compiled Qt/PySide6 libraries for whichever OS it's run
# on -- a macOS .app cannot be built from Linux or Windows.
# -----------------------------------------------------------------------------

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# This spec file lives in plassys_release/packaging/ -- the actual app
# source (main_gui.py and everything it imports) is one level up.
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), ".."))
ICON_PATH = os.path.join(SOURCE_DIR, "icon_assets", "app_icon.icns")

a = Analysis(
    [os.path.join(SOURCE_DIR, "main_gui.py")],
    pathex=[SOURCE_DIR],
    binaries=[],
    datas=[
        # qtawesome ships its own icon-font files (.ttf) and charmap JSON
        # as package data -- PyInstaller's static import analysis can't
        # see those (they're loaded by qtawesome internally, not
        # `import`ed), so they're collected explicitly here. Without
        # this, every toolbar/button icon in the packaged app would
        # silently fail to draw.
        *collect_data_files("qtawesome"),
        # matplotlib ships default fonts, styles, and other runtime data
        # under its own package directory (mpl-data) the same way --
        # collected explicitly so plots/canvases render correctly in the
        # packaged app even on a machine with no other matplotlib
        # install to fall back on.
        *collect_data_files("matplotlib"),
    ],
    hiddenimports=[
        # PyInstaller's static analysis sometimes misses a
        # dynamically-loaded submodule inside these packages' own C
        # extensions. Listed explicitly so the build fails loudly here
        # (a missing package at BUILD time) rather than as a mysterious
        # crash the first time a user opens the relevant tab.
        #
        # ("scipy.special._cdflib" used to be listed here too, but scipy
        # 1.17.1 has no such submodule -- confirmed directly, and
        # confirmed it was already harmless, just a "hidden import not
        # found" warning in the build log with no effect on the app.
        # Removed since it no longer means anything for this scipy
        # version, and a nonexistent entry here is just noise to read
        # past in a future build log.)
        "scipy._lib.array_api_compat.numpy.fft",
        "shapely.geometry",
        "shapely.affinity",
        "shapely.ops",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Plassys Shadow Simulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-compressed Qt binaries are a common source of
                         # mysterious startup crashes on macOS; not worth the
                         # smaller file size here.
    console=False,       # windowed app -- no terminal window behind it
    icon=ICON_PATH,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Plassys Shadow Simulator",
)

app = BUNDLE(
    coll,
    name="Plassys Shadow Simulator.app",
    icon=ICON_PATH,
    bundle_identifier="com.alexanderbraiman.plassysshadowsimulator",
    info_plist={
        "CFBundleName": "Plassys Shadow Simulator",
        "CFBundleDisplayName": "Plassys Shadow Simulator",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "NSHumanReadableCopyright": "Alexander Braiman -- SQMS Center, Fermilab",
        "NSHighResolutionCapable": True,
        # Belt-and-suspenders: macOS occasionally shows a "app is damaged"
        # dialog for an unsigned, freshly-built .app it hasn't seen before,
        # purely a Gatekeeper quarantine-flag quirk unrelated to the app
        # itself -- see BUILD_INSTRUCTIONS.md for the one-line fix if you
        # ever hit it.
    },
)
