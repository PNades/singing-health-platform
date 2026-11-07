"""Held-out test singers (spec Section 7.1).

data/splits.json is {"test": [singer_id, ...], ...}. Test singers are off
limits: code that reads clips drops them unless it was run with --final.

    test = test_singers()          # empty set if splits.json doesn't exist yet
    allowed = [c for c in clips if c["singer_id"] not in test]
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "splits.json"


def test_singers(path: Path = SPLITS_PATH) -> set[str]:
    """Singer ids held out for the final test. Empty until the split is made."""
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")).get("test", []))
