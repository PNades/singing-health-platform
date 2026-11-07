"""Collection-site API (ml/collect/app.py) against a LocalStore in tmp_path.

Most tests swap the take check for a stub so they run fast; one test sends a
real generated arpeggio through api/quality.py end to end, and the mic check
always runs the real measurements.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from scipy.io import wavfile

from ml.collect.app import create_app
from ml.collect.session import token_hash
from ml.collect.storage import Conflict, LocalStore, create_json, get_json, put_json
from tests.test_quality import sing

T0 = datetime(2026, 11, 20, 14, 0, tzinfo=UTC)
TOKEN = "tok_" + "a" * 28
FORM = "https://docs.google.com/forms/d/e/example/viewform"
H = {"X-Invite-Token": TOKEN}
NAME = "Jordan Example Singer"
PROFILE = {"voice_type": "alto", "sex": "", "device": "laptop", "mic": "USB mic", "tradition": ""}


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    store = LocalStore(tmp_path / "bucket")
    clock = Clock()
    verdict = {"ok": True, "flags": [], "message": None}
    app = create_app(store=store, now=clock, payment_form_url=FORM, check=lambda *a, **k: dict(verdict))
    create_json(store, f"invites/{token_hash(TOKEN)}.json",
                {"singer_id": "s007", "created_at": T0.isoformat(), "revoked": False})
    return app.test_client(), store, clock, verdict


def wav_bytes(x: np.ndarray | None = None, sr: int = 48_000) -> bytes:
    if x is None:
        x = (np.random.default_rng(0).normal(size=2 * sr) * 1000).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, x)
    return buf.getvalue()


def upload(client, path, data: bytes | None = None, ts=None):
    form = {"audio": (io.BytesIO(data if data is not None else wav_bytes()), "take.wav"),
            "track_settings": json.dumps(ts or {"sampleRate": 48000, "echoCancellation": False})}
    return client.post(path, data=form, headers=H, content_type="multipart/form-data")


def setup_session(client, store):
    """Consent + profile + a passed mic check (set directly: the real check is tested separately)."""
    assert client.get("/api/state", headers=H).status_code == 200
    r = client.post("/api/consent", json={"name": NAME, "age_ok": True, "version": "consent-v1"}, headers=H)
    assert r.status_code == 200, r.json
    assert client.post("/api/profile", json=PROFILE, headers=H).status_code == 200
    s, gen = get_json(store, "sessions/s007.json")
    s["mic_check"] = {"passed_at": T0.isoformat(), "flags": []}
    put_json(store, "sessions/s007.json", s, if_generation=gen)


def objects(store, prefix=""):
    return store.list(prefix)


# ---------------------------------------------------------------- auth and pages


def test_pages_and_headers(env):
    client, *_ = env
    r = client.get("/")
    assert r.status_code == 200 and b"invite-only" in r.data
    r = client.get(f"/s/{TOKEN}")
    assert r.status_code == 200 and b"StrainlessAI recordings" in r.data
    assert r.headers["Referrer-Policy"] == "no-referrer"
    assert r.headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    assert client.get("/audio/alto.wav").status_code == 200
    assert client.get("/audio/bass.wav").status_code == 404
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-Invite-Token": "short"}, {"X-Invite-Token": "x" * 32}])
def test_bad_token_is_refused(env, headers):
    client, store, *_ = env
    assert client.get("/api/state", headers=headers).status_code == 403
    assert objects(store, "sessions/") == []


def test_revoked_and_expired_links(env):
    client, store, clock, _ = env
    clock.t = T0 + timedelta(days=30, seconds=1)
    r = client.get("/api/state", headers=H)
    assert r.status_code == 403 and "expired" in r.json["message"]
    clock.t = T0
    inv, gen = get_json(store, f"invites/{token_hash(TOKEN)}.json")
    put_json(store, f"invites/{token_hash(TOKEN)}.json", {**inv, "revoked": True}, if_generation=gen)
    assert client.get("/api/state", headers=H).status_code == 403


def test_state_creates_session_once(env):
    client, store, *_ = env
    r = client.get("/api/state", headers=H)
    body = r.json
    assert body["session"]["singer_id"] == "s007" and body["session"]["status"] == "open"
    assert len(body["slots"]) == 16 and body["consent"]["version"] == "consent-v1"
    assert body["payment_form_url"] is None  # only after submit
    assert body["arpeggios"]["alto"]["notes"][0] == "F4"
    assert "upload_times" not in body["session"]
    client.get("/api/state", headers=H)
    assert objects(store, "sessions/") == ["sessions/s007.json"]


# ---------------------------------------------------------------- consent and profile


def test_consent_is_stored_apart_from_the_session(env):
    client, store, *_ = env
    for bad in ({"name": NAME, "age_ok": False, "version": "consent-v1"},
                {"name": " ", "age_ok": True, "version": "consent-v1"},
                {"name": NAME, "age_ok": True, "version": "consent-v0"}):
        assert client.post("/api/consent", json=bad, headers=H).status_code == 400
    assert objects(store, "consent/") == []
    setup_session(client, store)
    rec = get_json(store, "consent/s007.json")[0]
    assert rec["name"] == NAME and rec["consent_version"] == "consent-v1" and rec["singer_id"] == "s007"
    s = get_json(store, "sessions/s007.json")[0]
    assert s["consent_version"] == "consent-v1" and NAME not in json.dumps(s)


def test_profile_requires_consent(env):
    client, *_ = env
    r = client.post("/api/profile", json=PROFILE, headers=H)
    assert r.status_code == 400 and r.json["error"] == "no_consent"


# ---------------------------------------------------------------- takes


def test_take_needs_mic_check(env):
    client, store, *_ = env
    client.get("/api/state", headers=H)
    client.post("/api/consent", json={"name": NAME, "age_ok": True, "version": "consent-v1"}, headers=H)
    client.post("/api/profile", json=PROFILE, headers=H)
    r = upload(client, "/api/takes/1")
    assert r.status_code == 400 and r.json["error"] == "no_mic_check"


def test_accepted_take_is_stored_with_metadata(env):
    client, store, _, verdict = env
    setup_session(client, store)
    verdict.update(ok=True, flags=["noise_moderate"])
    r = upload(client, "/api/takes/5", ts={"sampleRate": 48000})
    assert r.status_code == 200 and r.json["accepted"] is True
    st = r.json["session"]["slots"]["5"]
    assert st["status"] == "accepted" and st["flags"] == ["noise_moderate"]
    clip = st["clip_id"]
    assert sorted(objects(store, "takes/")) == [f"takes/s007/{clip}.json", f"takes/s007/{clip}.wav"]
    meta = get_json(store, f"takes/s007/{clip}.json")[0]
    assert meta["slot"] == 5 and meta["loud"] is True and meta["condition"] == "comfortable"
    assert meta["arpeggio_id"] == "alto_F4" and meta["sex"] == "unknown"
    assert meta["device"] == "laptop / USB mic" and meta["sample_rate"] == 48000
    assert meta["track_settings"] == {"sampleRate": 48000}
    assert meta["quality_flags"] == ["noise_moderate"] and "tradition" not in meta
    sr, x = wavfile.read(io.BytesIO(store.get_bytes(f"takes/s007/{clip}.wav")[0]))
    assert sr == 48000 and x.dtype == np.int16
    a = client.get("/api/takes/5/audio", headers=H)
    assert a.status_code == 200 and a.mimetype == "audio/wav"


def test_rejected_take_is_not_stored(env):
    client, store, _, verdict = env
    setup_session(client, store)
    verdict.update(ok=False, flags=["clipping"], message="Too loud for the mic.")
    r = upload(client, "/api/takes/2")
    assert r.status_code == 200 and r.json["accepted"] is False
    assert r.json["message"] == "Too loud for the mic."
    st = r.json["session"]["slots"]["2"]
    assert st["status"] == "rejected" and st["rejected_count"] == 1
    assert objects(store, "takes/") == []
    assert client.get("/api/takes/2/audio", headers=H).status_code == 404


def test_rerecord_replaces_and_delete_empties(env):
    client, store, _, verdict = env
    setup_session(client, store)
    first = upload(client, "/api/takes/1").json["session"]["slots"]["1"]["clip_id"]
    verdict.update(ok=False, flags=["noise_high"], message="Too much background noise.")
    r = upload(client, "/api/takes/1")
    assert r.json["session"]["slots"]["1"]["clip_id"] == first  # rejected re-record keeps the old take
    verdict.update(ok=True, flags=[], message=None)
    second = upload(client, "/api/takes/1").json["session"]["slots"]["1"]["clip_id"]
    assert second != first
    assert all(first not in p for p in objects(store, "takes/"))
    assert len(objects(store, "takes/")) == 2
    r = client.delete("/api/takes/1", headers=H)
    assert r.status_code == 200 and r.json["session"]["slots"]["1"]["status"] == "empty"
    assert objects(store, "takes/") == []


def test_bad_uploads(env):
    client, store, *_ = env
    setup_session(client, store)
    assert upload(client, "/api/takes/1", data=b"not a wav").status_code == 400
    stereo = (np.zeros((48_000, 2)) + 10).astype(np.int16)
    assert upload(client, "/api/takes/1", data=wav_bytes(stereo)).status_code == 400
    floats = np.zeros(48_000, dtype=np.float32)
    assert upload(client, "/api/takes/1", data=wav_bytes(floats)).status_code == 400
    big = upload(client, "/api/takes/1", data=b"\0" * (17 * 1024 * 1024))
    assert big.status_code == 413
    assert upload(client, "/api/takes/17").status_code == 404
    assert objects(store, "takes/") == []


def test_rate_limit_30_uploads_per_hour(env):
    client, store, clock, verdict = env
    setup_session(client, store)
    verdict.update(ok=False, flags=["incomplete"], message="x")
    for k in range(30):
        clock.t = T0 + timedelta(seconds=10 * k)
        assert upload(client, "/api/takes/1").status_code == 200
    r = upload(client, "/api/takes/1")
    assert r.status_code == 429 and r.json["error"] == "rate_limited"
    clock.t = T0 + timedelta(hours=1, seconds=1)
    assert upload(client, "/api/takes/1").status_code == 200


# ---------------------------------------------------------------- submit


def test_submit_flow(env):
    client, store, clock, _ = env
    setup_session(client, store)
    for n in range(1, 16):
        upload(client, f"/api/takes/{n}")
    r = client.post("/api/submit", headers=H)
    assert r.status_code == 400 and r.json["error"] == "incomplete"
    assert objects(store, "codes/") == []
    upload(client, "/api/takes/16")

    r = client.post("/api/submit", headers=H)
    assert r.status_code == 200
    code = r.json["completion_code"]
    assert len(code) == 8 and r.json["payment_form_url"] == FORM
    assert objects(store, "codes/") == [f"codes/{code}"]
    assert client.post("/api/submit", headers=H).json["completion_code"] == code  # idempotent
    assert objects(store, "codes/") == [f"codes/{code}"]

    # Locked: no uploads, deletes or profile edits.
    assert upload(client, "/api/takes/1").status_code == 409
    assert client.delete("/api/takes/1", headers=H).status_code == 409
    assert client.post("/api/profile", json=PROFILE, headers=H).status_code == 409
    assert len(objects(store, "takes/")) == 32

    # The code stays visible after the link expires; recording doesn't.
    clock.t = T0 + timedelta(days=45)
    st = client.get("/api/state", headers=H).json
    assert st["session"]["completion_code"] == code and st["payment_form_url"] == FORM


def test_nothing_personal_outside_consent(env):
    client, store, *_ = env
    setup_session(client, store)
    for n in range(1, 17):
        upload(client, f"/api/takes/{n}")
    client.post("/api/submit", headers=H)
    for path in objects(store):
        data = store.get_bytes(path)[0]
        if path.startswith("consent/"):
            continue
        assert NAME.encode() not in data, path
        if path.endswith(".json"):
            assert "email" not in data.decode().lower(), path
    assert {p.split("/")[0] for p in objects(store)} == {"invites", "consent", "sessions", "takes", "codes"}


# ---------------------------------------------------------------- real checks


def test_mic_check_uses_real_measurements(tmp_path):
    store = LocalStore(tmp_path / "b")
    app = create_app(store=store, now=lambda: T0, payment_form_url=FORM)
    client = app.test_client()
    create_json(store, f"invites/{token_hash(TOKEN)}.json",
                {"singer_id": "s007", "created_at": T0.isoformat(), "revoked": False})
    client.get("/api/state", headers=H)
    client.post("/api/consent", json={"name": NAME, "age_ok": True, "version": "consent-v1"}, headers=H)
    client.post("/api/profile", json=PROFILE, headers=H)

    silent = np.zeros(4 * 44_100, dtype=np.int16)
    r = upload(client, "/api/mic-check", data=wav_bytes(silent, 44_100), ts={"sampleRate": 44100})
    assert r.json["ok"] is False and "couldn't hear" in r.json["message"]

    noisy, _ = sing("alto", snr_db=8.0)
    r = upload(client, "/api/mic-check", data=wav_bytes(noisy, 44_100), ts={"sampleRate": 44100})
    assert r.json["ok"] is False and "noise_high" in r.json["flags"]
    assert r.json["session"]["mic_check"] is None

    # Only part of the arpeggio: completeness doesn't matter for the mic check.
    partial, _ = sing("alto", stop_after_beats=4)
    r = upload(client, "/api/mic-check", data=wav_bytes(partial, 44_100), ts={"sampleRate": 44100})
    assert r.json["ok"] is True, r.json
    assert not {"incomplete", "intervals_off", "key_off", "note_short"} & set(r.json["flags"])
    assert r.json["session"]["mic_check"]["passed_at"]
    assert store.list("takes/") == []


def test_real_take_end_to_end(tmp_path):
    store = LocalStore(tmp_path / "b")
    app = create_app(store=store, now=lambda: T0, payment_form_url=FORM)
    client = app.test_client()
    create_json(store, f"invites/{token_hash(TOKEN)}.json",
                {"singer_id": "s007", "created_at": T0.isoformat(), "revoked": False})
    setup_session(client, store)
    good, _ = sing("alto")
    r = upload(client, "/api/takes/1", data=wav_bytes(good, 44_100), ts={"sampleRate": 44100})
    assert r.json["accepted"] is True, r.json
    short, _ = sing("alto", stop_after_beats=4)
    r = upload(client, "/api/takes/2", data=wav_bytes(short, 44_100), ts={"sampleRate": 44100})
    assert r.json["accepted"] is False and "incomplete" in r.json["flags"]
    assert len(store.list("takes/")) == 2


# ---------------------------------------------------------------- storage, image


def test_local_store_generations():
    import tempfile

    st = LocalStore(tempfile.mkdtemp())
    g1 = create_json(st, "sessions/x.json", {"a": 1})
    with pytest.raises(Conflict):
        create_json(st, "sessions/x.json", {"a": 2})
    g2 = put_json(st, "sessions/x.json", {"a": 2}, if_generation=g1)
    with pytest.raises(Conflict):
        put_json(st, "sessions/x.json", {"a": 3}, if_generation=g1)  # stale tab
    assert get_json(st, "sessions/x.json") == ({"a": 2}, g2)
    with pytest.raises(ValueError):
        st.get_bytes("../outside")


def test_service_imports_stay_slim():
    """The Cloud Run image installs only the `collect` group: the app must not need anything else."""
    code = (
        "import sys; from ml.collect.app import create_app; "
        "bad = {'matplotlib','sklearn','lightgbm','opensmile','fastembed','qdrant_client','groq','pandas','tensorflow','librosa'}; "
        "print(sorted(bad & {m.split('.')[0] for m in sys.modules}))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
