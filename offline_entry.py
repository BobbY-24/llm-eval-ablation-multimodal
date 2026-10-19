"""Bounded dispatch used only inside run_offline.sh's OS sandbox."""
import runpy
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11 or later is required")
signal.alarm(180)
if sys.argv[1:2] == ["test"]:
    sys.argv = [str(ROOT / "test_runner.py"), *sys.argv[2:]]
else:
    sys.argv = [str(ROOT / "run.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
