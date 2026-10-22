#!/bin/bash
# Double-click in Finder: starts Webots with Maze 1 and records all console output
# to tools/last_run_output.txt (so problems can be diagnosed afterwards).
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/last_run_output.txt"
{
  echo "=== $(date)"
  echo "which python3 (Terminal PATH): $(command -v python3)"
  [ -x "$HOME/.venvs/webots/bin/python3" ] && "$HOME/.venvs/webots/bin/python3" -c "import sys, numpy, cv2, scipy, skimage; print('venv python', sys.version.split()[0], 'numpy', numpy.__version__, 'cv2', cv2.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__)" 2>&1
} > "$OUT"
"$DIR/run_webots.sh" --stdout --stderr "$DIR/../Mazes/Maze1/worlds/Maze1.wbt" 2>&1 | tee -a "$OUT"
# Reference: generated with Claude (Anthropic LLM, claude.ai), Oct 2026.
