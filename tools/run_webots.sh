#!/bin/bash
# Starts Webots R2025a with the macOS 27 port-check workaround (see webots_macos27_fix.c).
# Usage: ./run_webots.sh            or   ./run_webots.sh path/to/world.wbt
DIR="$(cd "$(dirname "$0")" && pwd)"
LIB="$DIR/webots_macos27_fix.dylib"
if [ ! -f "$LIB" ]; then
  echo "Building $LIB ..."
  clang -dynamiclib -O2 -arch arm64 -arch x86_64 -o "$LIB" "$DIR/webots_macos27_fix.c" || exit 1
fi
export DYLD_INSERT_LIBRARIES="$LIB"
# Use the project Python environment (numpy, opencv) for all Python controllers:
# Webots runs "python3" from PATH when no Python command is set in its preferences.
if [ -x "$HOME/.venvs/webots/bin/python3" ]; then
  export PATH="$HOME/.venvs/webots/bin:$PATH"
else
  echo "Warning: $HOME/.venvs/webots not found - controllers will use the default python3"
fi
echo "Controllers will use: $(command -v python3)"
exec /Applications/Webots.app/Contents/MacOS/webots "$@"
# Reference: script above generated with Claude (Anthropic LLM), 2026-10-18
