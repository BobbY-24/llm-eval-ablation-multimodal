"""Source-checkout entry point; no installation or environment variables required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from multimodal_game_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
