"""
Cross-platform task runner mirroring the Makefile, for machines without `make`
(e.g. a stock Windows install -- `make` is not on PATH by default, unlike
every Unix system and macOS). Targets are identical in meaning:

    python run.py setup
    python run.py test
    python run.py audit
    python run.py replay
    python run.py precompute
    python run.py ui
    python run.py demo     # test + audit + replay + precompute, then launch the UI

This file is a thin wrapper, not a second source of truth -- it shells out to
the exact same commands the Makefile uses, so the two can never drift apart
on *what* gets run, only on *how* you invoke it.
"""
from __future__ import annotations

import subprocess
import sys

TARGETS = {
    "setup": [[sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]],
    "test": [[sys.executable, "-m", "pytest", "tests/", "-q"]],
    "audit": [[sys.executable, "scripts/audit.py"]],
    "synthetic": [[sys.executable, "scripts/generate_synthetic.py"]],
    "replay": [[sys.executable, "scripts/replay.py"]],
    "precompute": [[sys.executable, "scripts/precompute.py"]],
    "ui": [[sys.executable, "-m", "streamlit", "run", "ui/app.py"]],
    "app": [[sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"]],
}
TARGETS["demo"] = TARGETS["test"] + TARGETS["audit"] + TARGETS["replay"] + TARGETS["precompute"] + TARGETS["app"]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        print(f"Usage: python run.py <{'|'.join(TARGETS)}>")
        sys.exit(1)
    for cmd in TARGETS[sys.argv[1]]:
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
