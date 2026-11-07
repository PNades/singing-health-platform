"""One line per singer: progress and completion codes (spec 2.8 "Payment").

    uv run python -m ml.collect.report            # bucket from COLLECT_BUCKET
    uv run python -m ml.collect.report --local    # dev store

Match each completion code in the payment form against this list, pay, then
delete that form response. Nothing here identifies a singer beyond singer_id.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from typing import Any

from ml.collect import session
from ml.collect.storage import Store, get_json, store_from_env


def rows(store: Store, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    invites: dict[str, dict] = {}
    for path in store.list("invites/"):
        got = get_json(store, path)
        if got:
            inv = got[0]
            # a singer may have a revoked and a fresh invite: keep the valid one if any
            if inv["singer_id"] not in invites or session.invite_valid(inv, now):
                invites[inv["singer_id"]] = inv
    sessions = {}
    for path in store.list("sessions/"):
        got = get_json(store, path)
        if got:
            sessions[got[0]["singer_id"]] = got[0]
    out = []
    for sid in sorted(set(invites) | set(sessions)):
        s, inv = sessions.get(sid), invites.get(sid)
        if inv is None:
            link = "no invite"
        elif inv.get("revoked"):
            link = "revoked"
        elif not session.invite_valid(inv, now):
            link = "expired"
        else:
            link = "ok"
        out.append({
            "singer_id": sid,
            "accepted": session.accepted_count(s) if s else 0,
            "status": s["status"] if s else "not started",
            "code": (s or {}).get("completion_code") or "",
            "rejected": sum(v["rejected_count"] for v in s["slots"].values()) if s else 0,
            "voice_type": ((s or {}).get("profile") or {}).get("voice_type", ""),
            "last_activity": (s or {}).get("updated_at", ""),
            "link": link,
        })
    return out


def format_rows(rs: list[dict[str, Any]]) -> str:
    head = f"{'singer':<8} {'takes':>6}  {'status':<12} {'code':<9} {'rejected':>8}  {'voice':<9} {'link':<8} last activity"
    lines = [head, "-" * len(head)]
    for r in rs:
        lines.append(
            f"{r['singer_id']:<8} {r['accepted']:>3}/16  {r['status']:<12} {r['code']:<9} {r['rejected']:>8}  "
            f"{r['voice_type']:<9} {r['link']:<8} {r['last_activity']}"
        )
    n_sub = sum(r["status"] == "submitted" for r in rs)
    lines.append(f"\n{len(rs)} singer(s), {n_sub} submitted")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local", action="store_true", help="use the local dev store (data/collect_dev)")
    a = ap.parse_args(argv)
    print(format_rows(rows(store_from_env(local=a.local))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
