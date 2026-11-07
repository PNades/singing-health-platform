"""Run the quality checks (api/quality.py) over every clip in data/raw.

Prints how many clips each check and each flag code catches. Each clip is
data/raw/<clip_id>.wav with its metadata in <clip_id>.json (spec 2.6), which
gives voice_type, track_settings and singer_id. Clips from test singers
(data/splits.json) are skipped unless run with --final.

    uv run python scripts/quality_report.py
    uv run python scripts/quality_report.py --final
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.quality import CHECK_OF, FLAG_NOTES, FLAG_ORDER, check  # noqa: E402
from ml.features.segment import load_mono  # noqa: E402
from ml.splits import test_singers  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--final", action="store_true", help="include held-out test singers")
    ap.add_argument("--raw", type=Path, default=RAW_DIR, help="clip folder (default data/raw)")
    args = ap.parse_args()

    wavs = sorted(args.raw.glob("*.wav")) if args.raw.is_dir() else []
    if not wavs:
        print(f"No clips in {args.raw} yet; nothing to report.")
        return 0

    held_out = set() if args.final else test_singers()
    codes: Counter[str] = Counter()
    checks: Counter[str] = Counter()
    n_ok = n_run = 0
    skipped: Counter[str] = Counter()
    for wav in wavs:
        meta_path = wav.with_suffix(".json")
        if not meta_path.exists():
            skipped["no metadata"] += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("singer_id") in held_out:
            skipped["test singer"] += 1
            continue
        try:
            x, sr = load_mono(wav)
            result = check(x, sr, meta["voice_type"], meta.get("track_settings"))
        except Exception as e:  # report and carry on: one bad file shouldn't stop the report
            print(f"  {wav.name}: error: {e}")
            skipped["error"] += 1
            continue
        n_run += 1
        n_ok += result["ok"]
        codes.update(result["flags"])
        checks.update({CHECK_OF[f] for f in result["flags"]})

    def pct(k: int) -> str:
        return f"{k:4d}  {100 * k / n_run:5.1f}%" if n_run else f"{k:4d}"

    print(f"Clips checked: {n_run}" + (f"  (skipped: {dict(skipped)})" if skipped else ""))
    if not n_run:
        return 0
    print(f"Passed:        {pct(n_ok)}")
    print(f"Rejected:      {pct(n_run - n_ok)}")
    print("\nClips caught, by check:")
    for name in dict.fromkeys(CHECK_OF.values()):
        print(f"  {name:14s} {pct(checks[name])}")
    print("\nBy code:")
    for code in FLAG_ORDER:
        kind = "flag" if code in FLAG_NOTES else "reject"
        print(f"  {code:20s} {kind:6s} {pct(codes[code])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
