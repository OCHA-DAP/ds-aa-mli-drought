"""Project constants and the data location.

Data never lives in the repo. It is kept on the team's dev blob under PROJECT_PREFIX; pull it into a
temporary folder for a run and point MLI_DATA_DIR at that folder (see scripts/blob_data.py).
"""

import os
from pathlib import Path

PROJECT_PREFIX = "ds-aa-mli-drought"


def data_dir() -> Path:
    d = os.environ.get("MLI_DATA_DIR")
    if not d:
        raise SystemExit(
            "MLI_DATA_DIR is not set. Pull the data first:\n"
            "  export MLI_DATA_DIR=$(uv run --with ocha-stratus python scripts/blob_data.py pull)\n"
            "and push outputs back with: uv run --with ocha-stratus python scripts/blob_data.py push"
        )
    return Path(d)
