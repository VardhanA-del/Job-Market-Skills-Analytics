"""Command-line entry point for rebuilding processed analytics tables.

This file can be executed directly from the project root or another working
directory without requiring the project to be installed as a package.
"""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_processed_data


if __name__ == "__main__":
    print(json.dumps(build_processed_data(), indent=2))
