"""Compatibility launcher for the packaged scheduler."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from job_search.scheduler import *  # noqa: F401,F403,E402
from job_search.scheduler import main


if __name__ == "__main__":
    main()