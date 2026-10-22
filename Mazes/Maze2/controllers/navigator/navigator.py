"""Launcher used by Webots (Webots needs a controller file inside each maze project).
All navigation code lives in ONE shared folder: <Autonomous robots>/navigator/ - this file only starts it.

It uses the standard library only, so it always runs, whichever Python Webots picked:
  * writes navigator/logs/launcher.log (which Python, which packages, any error)
  * if this Python lacks the required packages, it restarts itself with the project
    environment ~/.venvs/webots (the Webots controller connection is inherited)."""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "navigator"))
LOGS = os.path.join(SHARED, "logs")
VENV_DIR = os.path.expanduser("~/.venvs/webots")
VENV_PY = os.path.join(VENV_DIR, "bin", "python3")
REQUIRED = ("numpy", "cv2", "scipy", "skimage")


def log(msg):
    try:
        os.makedirs(LOGS, exist_ok=True)
        with open(os.path.join(LOGS, "launcher.log"), "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except Exception:
        pass
    if msg.startswith(("ERROR", "missing")):     # console: problems only (everything is in launcher.log)
        print("[launcher] " + msg, flush=True)


log(f"start: python {sys.version.split()[0]} at {sys.executable} (prefix {sys.prefix}); script {os.path.abspath(__file__)}")
missing = []
for mod in REQUIRED:
    try:
        __import__(mod)
    except Exception as ex:
        missing.append(f"{mod} ({ex.__class__.__name__}: {ex})")

if missing:
    log("missing packages in this Python: " + "; ".join(missing))
    in_venv = os.path.realpath(sys.prefix) == os.path.realpath(VENV_DIR)
    if os.path.exists(VENV_PY) and not in_venv and not os.environ.get("NAV_REEXEC"):
        log(f"restarting with the project environment {VENV_PY}")
        os.environ["NAV_REEXEC"] = "1"
        os.execv(VENV_PY, [VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])
    log("ERROR: no Python with numpy, opencv-python, scipy and scikit-image found. Run:\n"
        "  source ~/.venvs/webots/bin/activate && pip install numpy opencv-python scipy scikit-image")
    sys.exit(1)

log("all packages found, starting navigator")
sys.path.insert(0, SHARED)
try:
    import runpy
    runpy.run_path(os.path.join(SHARED, "navigator.py"), run_name="__main__")
except SystemExit:
    raise
except Exception:
    log("ERROR while running the navigator:\n" + traceback.format_exc())
    raise

# Reference: generated with Claude (Anthropic LLM, claude.ai), Oct 2026.
