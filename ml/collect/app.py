"""Collection site: Flask JSON API + one HTML page (spec Section 2.8).

    uv run flask --app ml.collect.app run          # local, LocalStore in data/collect_dev/
    gunicorn "ml.collect.app:create_app()"         # Cloud Run (COLLECT_BUCKET set)

The singer's link is /s/<token>. The page reads the token from its URL and
sends it as the X-Invite-Token header on every API call; the server only
ever sees a SHA-256 of it in storage (invites/<hash>.json).

Storage layout (spec 2.8): invites/, consent/<sid>.json, sessions/<sid>.json,
takes/<sid>/<clip_id>.wav + .json (accepted takes only), codes/<code>.
Rejected takes and mic-check takes are judged in memory and never stored.
"""

from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from scipy.io import wavfile
from werkzeug.exceptions import HTTPException

from api import quality
from ml.arpeggios import ARPEGGIOS, VOICE_TYPES, target_hz
from ml.collect import consent, session
from ml.collect.session import FLOW, SLOTS, SessionError, token_hash
from ml.collect.storage import Conflict, Store, create_json, get_json, put_json, store_from_env

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = ROOT / "api" / "static" / "audio"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_TAKE_S = 30.0  # longer uploads aren't takes (the recorder stops at FLOW["max_take_s"])
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
MIC_CHECKS = {"bandwidth", "clipping", "noise", "pre_roll"}  # spec 2.8 step 4
NO_VOICE_MESSAGE = "We couldn't hear you singing. Sing the whole arpeggio after \"Sing now\", then stop."
HTTP_OF = {"rate_limited": 429, "submitted": 409, "bad_slot": 404}
RETRIES = 5

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _arpeggio_info() -> dict[str, Any]:
    return {
        v: {"notes": ARPEGGIOS[v], "hz": target_hz(v), "range": f"{ARPEGGIOS[v][0]}–{ARPEGGIOS[v][3]}"}
        for v in VOICE_TYPES
    }


def create_app(
    store: Store | None = None,
    now: Callable[[], datetime] = _utcnow,
    payment_form_url: str | None = None,
    check: Callable[..., dict[str, Any]] | None = None,
) -> Flask:
    """store: defaults to store_from_env(). check: the take check, defaults to api.quality.check."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    store = store if store is not None else store_from_env()
    payment_url = payment_form_url if payment_form_url is not None else os.environ.get("PAYMENT_FORM_URL")
    consent_version, consent_html = consent.load()
    check_take = check or quality.check

    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    # ------------------------------------------------------------ plumbing

    @app.after_request
    def _headers(resp: Response) -> Response:
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["Permissions-Policy"] = "microphone=(self)"
        return resp

    @app.errorhandler(SessionError)
    def _session_error(e: SessionError):
        return jsonify(error=e.code, message=str(e)), HTTP_OF.get(e.code, 400)

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        if request.path.startswith("/api/"):
            msg = "That recording is too big." if e.code == 413 else e.description
            return jsonify(error=e.name.lower().replace(" ", "_"), message=msg), e.code
        return e

    def _err(code: str, message: str, status: int):
        return jsonify(error=code, message=message), status

    def _invite(allow_expired: bool = False) -> tuple[dict[str, Any], bool]:
        """(invite, expired) for the request's token, or abort 403."""
        token = request.headers.get("X-Invite-Token", "")
        if not TOKEN_RE.match(token):
            abort(403, "This link isn't valid.")
        got = get_json(store, f"invites/{token_hash(token)}.json")
        if got is None or got[0].get("revoked"):
            abort(403, "This link isn't valid. Reply to your invite email for a new one.")
        inv = got[0]
        expired = not session.invite_valid(inv, now())
        if expired and not allow_expired:
            abort(403, "This link has expired. Reply to your invite email for a new one.")
        return inv, expired

    def _session_path(sid: str) -> str:
        return f"sessions/{sid}.json"

    def _load(sid: str) -> tuple[dict[str, Any], int]:
        got = get_json(store, _session_path(sid))
        if got is not None:
            return got
        s = session.new_session(sid, now())
        try:
            return s, create_json(store, _session_path(sid), s)
        except Conflict:
            return get_json(store, _session_path(sid))  # another tab created it first

    def _mutate(sid: str, fn: Callable[[dict[str, Any]], Any]) -> tuple[dict[str, Any], Any]:
        """Read-modify-write the session with a generation precondition; retry on conflict."""
        for _ in range(RETRIES):
            s, gen = _load(sid)
            out = fn(s)
            try:
                put_json(store, _session_path(sid), s, if_generation=gen)
                return s, out
            except Conflict:
                continue
        abort(503, "Busy, please try again.")

    def _public(s: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in s.items() if k != "upload_times"}

    def _read_wav() -> tuple[np.ndarray, int, dict[str, Any]]:
        f = request.files.get("audio")
        if f is None:
            abort(400, "No recording in the upload.")
        try:
            sr, x = wavfile.read(io.BytesIO(f.read()))
        except Exception:
            abort(400, "The upload isn't a WAV file.")
        if x.dtype != np.int16 or x.ndim != 1:
            abort(400, "The recording must be mono 16-bit PCM WAV.")
        if not 8_000 <= sr <= 192_000 or len(x) / sr > MAX_TAKE_S:
            abort(400, "That recording has an unexpected length or sample rate.")
        try:
            ts = json.loads(request.form.get("track_settings") or "{}")
        except json.JSONDecodeError:
            ts = {}
        if not isinstance(ts, dict) or len(json.dumps(ts)) > 4000:
            ts = {}
        return x, int(sr), ts

    def _delete_clip(sid: str, clip_id: str | None) -> None:
        if clip_id:
            store.delete(f"takes/{sid}/{clip_id}.wav")
            store.delete(f"takes/{sid}/{clip_id}.json")

    def _slot_arg(slot: int) -> int:
        if not 1 <= slot <= session.N_SLOTS:
            abort(404, "No such slot.")
        return slot

    # ------------------------------------------------------------ pages

    @app.get("/")
    def home():
        return Response(
            "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>StrainlessAI recordings</title><body style='font:18px system-ui;margin:2rem;max-width:36rem'>"
            "<h1>StrainlessAI recordings</h1><p>This site is invite-only. Open the personal link from your "
            "invite email.</p>",
            mimetype="text/html",
        )

    @app.get("/s/<token>")
    def singer_page(token: str):
        return send_from_directory(STATIC, "index.html")

    @app.get("/audio/<voice>.wav")
    def example(voice: str):
        if voice not in VOICE_TYPES:
            abort(404)
        return send_from_directory(EXAMPLES, f"{voice}.wav", mimetype="audio/wav")

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    # ------------------------------------------------------------ API

    @app.get("/api/state")
    def state():
        inv, expired = _invite(allow_expired=True)
        sid = inv["singer_id"]
        got = get_json(store, _session_path(sid))
        if expired and (got is None or got[0]["status"] != "submitted"):
            abort(403, "This link has expired. Reply to your invite email for a new one.")
        s = got[0] if got else _load(sid)[0]
        return jsonify(
            session=_public(s),
            slots=SLOTS,
            flow=FLOW,
            arpeggios=_arpeggio_info(),
            options={"sexes": session.SEXES, "devices": session.DEVICES, "mics": session.MICS},
            consent={"version": consent_version, "html": consent_html},
            payment_form_url=payment_url if s["status"] == "submitted" else None,
        )

    @app.post("/api/consent")
    def give_consent():
        inv, _ = _invite()
        sid = inv["singer_id"]
        body = request.get_json(silent=True) or {}
        name = " ".join(str(body.get("name") or "").split())
        if body.get("version") != consent_version:
            return _err("bad_version", "The consent text changed. Reload the page and read it again.", 400)
        if body.get("age_ok") is not True:
            return _err("age", "You must be 18 or older to take part.", 400)
        if not 2 <= len(name) <= 200:
            return _err("name", "Type your full name to sign.", 400)
        if _load(sid)[0]["status"] != "open":
            raise SessionError("submitted", "This session is submitted and can't be changed.")
        t = now()
        put_json(
            store,
            f"consent/{sid}.json",
            {"singer_id": sid, "name": name, "consent_version": consent_version,
             "age_confirmed": True, "consented_at": t.isoformat(timespec="seconds")},
        )
        s, _ = _mutate(sid, lambda s: session.set_consent(s, consent_version, t))
        return jsonify(session=_public(s))

    @app.post("/api/profile")
    def profile():
        inv, _ = _invite()
        body = request.get_json(silent=True) or {}
        t = now()
        s, _ = _mutate(inv["singer_id"], lambda s: session.set_profile(s, body, t))
        return jsonify(session=_public(s))

    def _count_upload(sid: str, slot: int | None = None) -> dict[str, Any]:
        """Check the upload is allowed (slot recordable, or profile set for a mic check) and count it."""
        t = now()

        def fn(s):
            if slot is not None:
                session.require_record(s, slot)
            elif s["profile"] is None:
                raise SessionError("no_profile", "Fill in About you first.")
            session.note_upload(s, t)

        return _mutate(sid, fn)[0]

    @app.post("/api/mic-check")
    def mic_check():
        inv, _ = _invite()
        sid = inv["singer_id"]
        s = _count_upload(sid)
        x, sr, ts = _read_wav()
        m = quality.measure(x, sr, s["profile"]["voice_type"], ts)
        verdict = quality.judge(m)
        flags = [f for f in verdict["flags"] if quality.CHECK_OF[f] in MIC_CHECKS]
        rejects = [f for f in flags if f in quality.REJECT_MESSAGES]
        ok = not rejects and m["n_notes"] > 0
        message = quality.REJECT_MESSAGES[rejects[0]] if rejects else (None if ok else NO_VOICE_MESSAGE)
        t = now()
        s, _ = _mutate(sid, lambda s: session.set_mic_check(s, ok, flags, t))
        notes = [quality.FLAG_NOTES[f] for f in flags if f in quality.FLAG_NOTES]
        return jsonify(ok=ok, flags=flags, message=message, notes=notes, session=_public(s))

    @app.post("/api/takes/<int:slot>")
    def upload_take(slot: int):
        inv, _ = _invite()
        sid = inv["singer_id"]
        _slot_arg(slot)
        s = _count_upload(sid, slot=slot)
        x, sr, ts = _read_wav()
        result = check_take(x, sr, s["profile"]["voice_type"], track_settings=ts)
        t = now()
        if not result["ok"]:
            s, _ = _mutate(sid, lambda s: session.reject_take(s, slot, result["flags"], result["message"], t))
            return jsonify(accepted=False, flags=result["flags"], message=result["message"], session=_public(s))

        clip_id = session.new_clip_id()
        flags = [f for f in result["flags"] if f in quality.FLAG_NOTES]
        meta = session.take_metadata(
            s, slot, clip_id, t,
            user_agent=request.headers.get("User-Agent", "")[:500],
            track_settings=ts, sample_rate=sr, duration_s=len(x) / sr, quality_flags=flags,
        )
        buf = io.BytesIO()
        wavfile.write(buf, sr, x)
        store.put_bytes(f"takes/{sid}/{clip_id}.wav", buf.getvalue(), "audio/wav")
        put_json(store, f"takes/{sid}/{clip_id}.json", meta)
        try:
            s, old = _mutate(sid, lambda s: session.accept_take(s, slot, clip_id, flags, t))
        except Exception:
            _delete_clip(sid, clip_id)  # never leave an unreferenced take behind
            raise
        _delete_clip(sid, old)
        notes = [quality.FLAG_NOTES[f] for f in flags]
        return jsonify(accepted=True, flags=flags, message=None, notes=notes, session=_public(s))

    @app.get("/api/takes/<int:slot>/audio")
    def take_audio(slot: int):
        inv, _ = _invite(allow_expired=True)
        sid = inv["singer_id"]
        st = _load(sid)[0]["slots"][str(_slot_arg(slot))]
        got = store.get_bytes(f"takes/{sid}/{st['clip_id']}.wav") if st["status"] == "accepted" else None
        if got is None:
            abort(404, "No accepted take in this slot.")
        return Response(got[0], mimetype="audio/wav")

    @app.delete("/api/takes/<int:slot>")
    def delete_take(slot: int):
        inv, _ = _invite()
        sid = inv["singer_id"]
        _slot_arg(slot)
        t = now()
        s, old = _mutate(sid, lambda s: session.delete_take(s, slot, t))
        _delete_clip(sid, old)
        return jsonify(session=_public(s))

    @app.post("/api/submit")
    def submit():
        inv, _ = _invite()
        sid = inv["singer_id"]
        s = _load(sid)[0]
        if s["status"] == "submitted":
            return jsonify(completion_code=s["completion_code"], payment_form_url=payment_url, session=_public(s))
        if not session.can_submit(s):
            raise SessionError("incomplete", f"All {session.N_SLOTS} takes must be accepted before you submit.")
        for _ in range(RETRIES):
            code = session.new_completion_code()
            try:
                store.put_bytes(f"codes/{code}", b"", "text/plain", if_generation=0)
                break
            except Conflict:
                continue
        else:
            abort(503, "Busy, please try again.")
        t = now()

        def fn(s):
            if s["status"] == "submitted":
                return False  # another tab won; keep its code
            session.submit(s, code, t)
            return True

        try:
            s, mine = _mutate(sid, fn)
        except Exception:
            store.delete(f"codes/{code}")
            raise
        if not mine:
            store.delete(f"codes/{code}")
        return jsonify(completion_code=s["completion_code"], payment_form_url=payment_url, session=_public(s))

    return app

