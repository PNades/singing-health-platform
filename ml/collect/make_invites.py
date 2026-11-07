"""Create invite links for the collection site (spec 2.8 "Invites"). Prem's machine only.

    uv run python -m ml.collect.make_invites 5 --url https://strainless-collect-xxxx.run.app
    uv run python -m ml.collect.make_invites 1 --local --url http://localhost:5000   # dev store
    uv run python -m ml.collect.make_invites --revoke s007

Each new singer gets the next sequential id (s001, s002, ...) and a random
URL-safe token (24 random bytes). Only the token's SHA-256 is stored, at
invites/<hash>.json = {singer_id, created_at, revoked}. The links go to
data/invites.csv (gitignored, never committed): singer_id,link. Email each
recruit their link and keep the email -> singer_id mapping in the private
recruiting sheet, never here.

The bucket comes from COLLECT_BUCKET in .env; Prem's gcloud application-default
credentials are used (gcloud auth application-default login).
"""

from __future__ import annotations

import argparse
import csv
import re
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

from ml.collect.session import token_hash
from ml.collect.storage import ROOT, Conflict, Store, create_json, get_json, put_json, store_from_env

INVITES_CSV = ROOT / "data" / "invites.csv"
TOKEN_BYTES = 24  # spec: at least 16
_SID = re.compile(r"^s(\d+)$")


def _invites(store: Store) -> list[tuple[str, dict]]:
    out = []
    for path in store.list("invites/"):
        got = get_json(store, path)
        if got:
            out.append((path, got[0]))
    return out


def next_number(store: Store) -> int:
    nums = [int(m.group(1)) for _, inv in _invites(store) if (m := _SID.match(inv.get("singer_id", "")))]
    return max(nums, default=0) + 1


def make_invites(store: Store, n: int, base_url: str, out_csv: Path = INVITES_CSV,
                 now: datetime | None = None) -> list[tuple[str, str]]:
    """Create n invites; append (singer_id, link) rows to out_csv and return them."""
    now = now or datetime.now(UTC)
    base = base_url.rstrip("/")
    num = next_number(store)
    rows = []
    for _ in range(n):
        while True:
            sid = f"s{num:03d}"
            token = secrets.token_urlsafe(TOKEN_BYTES)
            try:
                create_json(store, f"invites/{token_hash(token)}.json",
                            {"singer_id": sid, "created_at": now.isoformat(timespec="seconds"), "revoked": False})
                break
            except Conflict:  # same hash: practically impossible, just draw again
                continue
        rows.append((sid, f"{base}/s/{token}"))
        num += 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["singer_id", "link"])
        w.writerows(rows)
    return rows


def revoke(store: Store, singer_id: str) -> int:
    """Mark every invite of singer_id revoked. Returns how many were changed."""
    count = 0
    for path, inv in _invites(store):
        if inv.get("singer_id") == singer_id and not inv.get("revoked"):
            inv["revoked"] = True
            put_json(store, path, inv)
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("n", type=int, nargs="?", help="number of new invites")
    ap.add_argument("--url", help="base URL of the collection site")
    ap.add_argument("--local", action="store_true", help="use the local dev store (data/collect_dev)")
    ap.add_argument("--revoke", metavar="SINGER_ID", help="revoke this singer's invite instead")
    ap.add_argument("--out", type=Path, default=INVITES_CSV)
    a = ap.parse_args(argv)
    store = store_from_env(local=a.local)
    if a.revoke:
        k = revoke(store, a.revoke)
        print(f"revoked {k} invite(s) for {a.revoke}")
        return 0 if k else 1
    if not a.n or a.n < 1 or not a.url:
        ap.error("give N >= 1 and --url (or --revoke SINGER_ID)")
    rows = make_invites(store, a.n, a.url, a.out)
    print(f"created {len(rows)} invite(s): {rows[0][0]}..{rows[-1][0]}; links appended to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
