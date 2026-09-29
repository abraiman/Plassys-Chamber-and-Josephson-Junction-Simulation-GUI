#!/bin/bash
# -----------------------------------------------------------------------------
# Builds Plassys Shadow Simulator into a real, double-clickable macOS app,
# with its own icon (the same comb-cross mark from the Front Page).
#
# Run this ON YOUR MAC, from a Terminal, with this command:
#
#     bash build_macos_app.sh
#
# See BUILD_INSTRUCTIONS.md in this same folder for the full walkthrough
# (what this script does, what to do with the result, and what to do if
# something goes wrong). This script assumes you've already followed
# requirements.txt (one level up) to install the app's own dependencies.
# -----------------------------------------------------------------------------

set -e  # stop immediately if any step below fails, instead of plowing ahead

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== Checking for PyInstaller =="
if ! python3 -m pip show pyinstaller > /dev/null 2>&1; then
    echo "PyInstaller not found -- installing it now (this is the packaging"
    echo "tool itself; it is not one of the app's own runtime dependencies,"
    echo "so it isn't in requirements.txt)."
    python3 -m pip install pyinstaller
else
    echo "PyInstaller is already installed."
fi

echo ""
echo "== Cleaning up any previous build =="
rm -rf build dist

echo ""
echo "== Building the app (this can take a few minutes) =="
python3 -m PyInstaller plassys_app.spec --noconfirm

echo ""
if [ -d "dist/Plassys Shadow Simulator.app" ]; then
    echo "== Done =="
    echo "Your app is ready at:"
    echo "    $SCRIPT_DIR/dist/Plassys Shadow Simulator.app"
    echo ""
    echo "Drag it into /Applications (or anywhere you like), then double-click"
    echo "it to launch. If macOS refuses to open it the first time with an"
    echo "'unidentified developer' or 'is damaged' warning, see the"
    echo "'First launch on macOS' section of BUILD_INSTRUCTIONS.md -- this is"
    echo "expected for an app built locally like this one, and takes one click"
    echo "to clear permanently."
else
    echo "Something went wrong -- 'dist/Plassys Shadow Simulator.app' was not"
    echo "created. Scroll up for the actual PyInstaller error, or see the"
    echo "Troubleshooting section of BUILD_INSTRUCTIONS.md."
    exit 1
fi
