"""Compatibility imports for the packaged resume tailoring module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from job_search.resume_tailor import *  # noqa: F401,F403,E402