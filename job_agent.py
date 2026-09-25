"""Compatibility launcher for the packaged job-search agent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from job_search.agent import *  # noqa: F401,F403,E402
from job_search.agent import main


if __name__ == "__main__":
    main()
