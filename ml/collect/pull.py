"""Download submitted sessions' takes into data/raw/ (spec 2.8 "Tools").

    uv run python -m ml.collect.pull             # bucket from COLLECT_BUCKET
    uv run python -m ml.collect.pull --local     # dev store

Only sessions with status "submitted" are pulled, never open ones, and only
the clips their accepted slots point at (a half-written take left behind by
a crash is never pulled). Files land at data/raw/<clip_id>.wav and .json
(spec 2.6); a file already there with the same content is skipped.

This is ingest, so held-out test singers are pulled too: data/splits.json is
made from data/raw/ afterwards, and the code that reads clips for training
is what keeps test singers out (ml/splits.py).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from ml.collect import session
from ml.collect.storage import ROOT, Store, get_json, store_from_env

RAW = ROOT / "data" / "raw"


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def pull(store: Store, out_dir: Path = RAW) -> dict[str, int]:
    """Returns counts: sessions pulled, files written, files unchanged, files missing in the bucket."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"sessions": 0, "written": 0, "unchanged": 0, "missing": 0, "skipped_open": 0}
    for path in store.list("sessions/"):
        got = get_json(store, path)
        if not got:
            continue
        s = got[0]
        if s["status"] != "submitted":
            counts["skipped_open"] += 1
            continue
        counts["sessions"] += 1
        sid = s["singer_id"]
        for clip_id in session.accepted_clips(s).values():
            for ext in ("wav", "json"):
                obj = store.get_bytes(f"takes/{sid}/{clip_id}.{ext}")
                if obj is None:
                    counts["missing"] += 1
                    print(f"missing in bucket: takes/{sid}/{clip_id}.{ext}", file=sys.stderr)
                    continue
                dest = out_dir / f"{clip_id}.{ext}"
                if dest.exists() and _md5(dest.read_bytes()) == _md5(obj[0]):
                    counts["unchanged"] += 1
                    continue
                dest.write_bytes(obj[0])
                counts["written"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local", action="store_true", help="use the local dev store (data/collect_dev)")
    ap.add_argument("--out", type=Path, default=RAW)
    a = ap.parse_args(argv)
    c = pull(store_from_env(local=a.local), a.out)
    print(
        f"{c['sessions']} submitted session(s): {c['written']} file(s) written, {c['unchanged']} unchanged, "
        f"{c['missing']} missing; {c['skipped_open']} open session(s) not pulled -> {a.out}"
    )
    return 1 if c["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
