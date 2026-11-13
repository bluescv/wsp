#!/usr/bin/env python3
"""Run the local control CLI directly from this checkout; no installation needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wsp_project.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
