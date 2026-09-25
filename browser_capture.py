"""Compatibility imports for the packaged browser capture module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from job_search.browser_capture import *  # noqa: F401,F403,E402