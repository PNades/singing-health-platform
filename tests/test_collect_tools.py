"""Prem's collection tools: make_invites -> report -> pull, against a LocalStore."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta

from ml.collect import make_invites, pull, report
from ml.collect import session as S
from ml.collect.session import token_hash
from ml.collect.storage import LocalStore, get_json, put_json

T0 = datetime(2026, 11, 20, 14, 0, tzinfo=UTC)
PROFILE = {"voice_type": "tenor", "sex": "M", "device": "phone", "mic": "built-in"}


def test_make_invites(tmp_path):
    store = LocalStore(tmp_path / "b")
    out = tmp_path / "invites.csv"
    rows = make_invites.make_invites(store, 3, "https://collect.example/", out, now=T0)
    assert [r[0] for r in rows] == ["s001", "s002", "s003"]
    rows2 = make_invites.make_invites(store, 2, "https://collect.example", out, now=T0)
    assert [r[0] for r in rows2] == ["s004", "s005"]

    with out.open(encoding="utf-8") as f:
        lines = list(csv.reader(f))
    assert lines[0] == ["singer_id", "link"] and len(lines) == 6
    for sid, link in lines[1:]:
        assert link.startswith("https://collect.example/s/")
        token = link.rsplit("/", 1)[1]
        assert len(token) >= 22  # >= 16 random bytes, url-safe base64
        inv = get_json(store, f"invites/{token_hash(token)}.json")[0]
        assert inv == {"singer_id": sid, "created_at": T0.isoformat(timespec="seconds"), "revoked": False}
        # The token itself is never stored.
        assert all(token.encode() not in store.get_bytes(p)[0] for p in store.list(""))

    assert make_invites.revoke(store, "s002") == 1
    assert make_invites.revoke(store, "s999") == 0
    revoked = [get_json(store, p)[0] for p in store.list("invites/")]
    assert [i["singer_id"] for i in revoked if i["revoked"]] == ["s002"]


def _session(store, sid, n_accepted, submit=False):
    s = S.new_session(sid, T0)
    S.set_consent(s, "consent-v1", T0)
    S.set_profile(s, PROFILE, T0)
    S.set_mic_check(s, True, [], T0)
    for n in range(1, n_accepted + 1):
        clip = f"c{sid}{n:02d}xxx"
        S.accept_take(s, n, clip, [], T0)
        store.put_bytes(f"takes/{sid}/{clip}.wav", b"RIFF" + bytes([n]), "audio/wav")
        put_json(store, f"takes/{sid}/{clip}.json", {"clip_id": clip, "singer_id": sid, "slot": n})
    if submit:
        S.submit(s, "ABCDEFGH" if sid == "s001" else "HGFEDCBA", T0)
    put_json(store, f"sessions/{sid}.json", s)
    return s


def test_report_and_pull(tmp_path):
    store = LocalStore(tmp_path / "b")
    make_invites.make_invites(store, 4, "https://c.example", tmp_path / "i.csv", now=T0)
    _session(store, "s001", 16, submit=True)
    _session(store, "s002", 7)
    # an orphan take (crash between upload and session write) and a submitted session
    store.put_bytes("takes/s001/corphan.wav", b"x", "audio/wav")
    make_invites.revoke(store, "s004")

    rs = {r["singer_id"]: r for r in report.rows(store, now=T0 + timedelta(days=1))}
    assert rs["s001"]["status"] == "submitted" and rs["s001"]["code"] == "ABCDEFGH" and rs["s001"]["accepted"] == 16
    assert rs["s002"]["status"] == "open" and rs["s002"]["accepted"] == 7 and rs["s002"]["code"] == ""
    assert rs["s003"]["status"] == "not started" and rs["s003"]["link"] == "ok"
    assert rs["s004"]["link"] == "revoked"
    assert report.rows(store, now=T0 + timedelta(days=31))[2]["link"] == "expired"
    text = report.format_rows(list(rs.values()))
    assert "ABCDEFGH" in text and "1 submitted" in text

    raw = tmp_path / "raw"
    c = pull.pull(store, raw)
    assert c == {"sessions": 1, "written": 32, "unchanged": 0, "missing": 0, "skipped_open": 1}
    names = sorted(p.name for p in raw.iterdir())
    assert len(names) == 32 and all(n.startswith("cs001") for n in names)  # nothing from the open session
    assert "corphan.wav" not in names
    c = pull.pull(store, raw)
    assert c["written"] == 0 and c["unchanged"] == 32
