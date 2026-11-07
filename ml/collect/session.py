"""Collection-site session logic (spec Section 2.8). Pure functions, no I/O.

A session is a JSON-friendly dict stored at sessions/<singer_id>.json:

    s = new_session("s007", now)
    set_consent(s, "consent-v1", now)
    set_profile(s, {"voice_type": "alto", "sex": "F", "device": "phone", "mic": "built-in"}, now)
    set_mic_check(s, ok=True, flags=[], now=now)
    note_upload(s, now)                         # rate limit, raises SessionError("rate_limited")
    old = accept_take(s, 3, "c1a2b3c4d5", [], now)   # returns the clip it replaced, or None
    submit(s, new_completion_code(), now)       # only when all 16 slots are accepted

Slot status is "empty", "accepted" or "rejected". "checking" exists only in the
browser while an upload is being judged. A rejected re-record of an accepted
slot leaves the accepted take in place. The session never holds a name,
email or IP; the typed name lives only in consent/<singer_id>.json.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any

from api.quality import QUALITY_CONFIG
from ml.arpeggios import BPM, VOICE_TYPES, arpeggio_id

N_SLOTS = 16
PUSHED_WARNING = "Stop if anything hurts or feels wrong. A few strained takes are enough; never push through pain."

# Spec 2.3: comfortable 1-6 (5-6 loud), edge 7-10, pushed 11-16 (15-16 moderate volume).
_COMFORTABLE = "Easy, healthy singing."
_LOUD = "Loud but free: a healthy belt, no squeezing."
_EDGE = "A little more effort than comfortable, but not fully pushed."
_PUSHED = 'Same notes with too much effort: squeezed, pressed, "shouty" throat.'
_PUSHED_MODERATE = "Pushed and squeezed, but not loud: moderate volume."


def _slot(n: int) -> dict[str, Any]:
    if n <= 6:
        cond, loud, text = "comfortable", n >= 5, _LOUD if n >= 5 else _COMFORTABLE
    elif n <= 10:
        cond, loud, text = "edge", False, _EDGE
    else:
        cond, loud, text = "pushed", False, _PUSHED_MODERATE if n >= 15 else _PUSHED
    return {
        "slot": n,
        "condition": cond,
        "loud": loud,
        "instruction": text,
        "warning": PUSHED_WARNING if cond == "pushed" else None,
    }


SLOTS: list[dict[str, Any]] = [_slot(n) for n in range(1, N_SLOTS + 1)]

# Recorder and site settings (spec 5.2, 2.8), served to the page by /api/state.
FLOW: dict[str, Any] = {
    "start_note_s": 1.0,
    "count_in_beats": 3,
    "bpm": BPM,
    "gap_s": 0.2,  # playback stops, then recording starts this much later
    "preroll_s": QUALITY_CONFIG["preroll_s"],
    "silence_stop_s": 1.5,  # [TUNE]
    "max_take_s": 15.0,  # [TUNE]
    "uploads_per_hour": 30,
    "invite_days": 30,  # [TUNE]
}

SEXES = ("F", "M", "unknown")
DEVICES = ("phone", "laptop", "tablet", "desktop")
MICS = ("built-in", "wired headset", "USB mic", "other")
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L
CODE_LEN = 8
CLIP_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
CLIP_LEN = 10
MAX_TRADITION_LEN = 100


class SessionError(Exception):
    """A request the session can't accept. `code` is a short machine-readable reason."""

    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


def _iso(now: datetime) -> str:
    return now.isoformat(timespec="seconds")


def new_session(singer_id: str, now: datetime) -> dict[str, Any]:
    return {
        "singer_id": singer_id,
        "session_id": f"{singer_id}-1",
        "status": "open",
        "created_at": _iso(now),
        "updated_at": _iso(now),
        "consent_version": None,
        "consented_at": None,
        "profile": None,
        "mic_check": None,
        "slots": {
            str(n): {
                "status": "empty",
                "clip_id": None,
                "recorded_at": None,
                "flags": [],
                "rejected_count": 0,
                "last_reject": None,
            }
            for n in range(1, N_SLOTS + 1)
        },
        "upload_times": [],
        "completion_code": None,
        "submitted_at": None,
    }


def _touch(s: dict[str, Any], now: datetime) -> None:
    s["updated_at"] = _iso(now)


def _require_open(s: dict[str, Any]) -> None:
    if s["status"] != "open":
        raise SessionError("submitted", "This session is submitted and can't be changed.")


def _slot_state(s: dict[str, Any], slot: int) -> dict[str, Any]:
    if not isinstance(slot, int) or not 1 <= slot <= N_SLOTS:
        raise SessionError("bad_slot", f"Slot must be 1-{N_SLOTS}.")
    return s["slots"][str(slot)]


def accepted_count(s: dict[str, Any]) -> int:
    return sum(v["status"] == "accepted" for v in s["slots"].values())


def accepted_clips(s: dict[str, Any]) -> dict[int, str]:
    """slot -> clip_id for every accepted slot."""
    return {int(k): v["clip_id"] for k, v in s["slots"].items() if v["status"] == "accepted"}


# ---------------------------------------------------------------- setup steps


def set_consent(s: dict[str, Any], version: str, now: datetime) -> None:
    _require_open(s)
    s["consent_version"] = version
    s["consented_at"] = _iso(now)
    _touch(s, now)


def clean_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate the About-you form. Raises SessionError("bad_profile")."""
    voice = raw.get("voice_type")
    sex = raw.get("sex") or "unknown"
    device, mic = raw.get("device"), raw.get("mic")
    tradition = (raw.get("tradition") or "").strip()
    if voice not in VOICE_TYPES:
        raise SessionError("bad_profile", "Choose a voice type.")
    if sex not in SEXES:
        raise SessionError("bad_profile", "Sex must be F, M or left blank.")
    if device not in DEVICES or mic not in MICS:
        raise SessionError("bad_profile", "Choose your device and microphone.")
    if len(tradition) > MAX_TRADITION_LEN:
        raise SessionError("bad_profile", f"Tradition: {MAX_TRADITION_LEN} characters at most.")
    return {"voice_type": voice, "sex": sex, "device": device, "mic": mic, "tradition": tradition or None}


def set_profile(s: dict[str, Any], raw: dict[str, Any], now: datetime) -> None:
    _require_open(s)
    if s["consent_version"] is None:
        raise SessionError("no_consent", "Consent comes first.")
    profile = clean_profile(raw)
    old = s["profile"]
    if old and old["voice_type"] != profile["voice_type"] and accepted_count(s):
        raise SessionError(
            "voice_locked", "Your voice type can't change after a take is accepted. Delete your takes first."
        )
    s["profile"] = profile
    _touch(s, now)


def set_mic_check(s: dict[str, Any], ok: bool, flags: list[str], now: datetime) -> None:
    _require_open(s)
    if s["profile"] is None:
        raise SessionError("no_profile", "Fill in About you first.")
    if ok:
        s["mic_check"] = {"passed_at": _iso(now), "flags": list(flags)}
    _touch(s, now)


# ---------------------------------------------------------------- takes


def can_record(s: dict[str, Any], slot: int) -> bool:
    """Slots can be recorded in any order once setup is done (decided 2026-11-07)."""
    try:
        _slot_state(s, slot)
    except SessionError:
        return False
    return s["status"] == "open" and s["profile"] is not None and s["mic_check"] is not None


def require_record(s: dict[str, Any], slot: int) -> dict[str, Any]:
    """The slot's state if a take may be recorded there now, else SessionError."""
    _require_open(s)
    st = _slot_state(s, slot)
    if s["profile"] is None:
        raise SessionError("no_profile", "Fill in About you first.")
    if s["mic_check"] is None:
        raise SessionError("no_mic_check", "Do the mic check first.")
    return st


def note_upload(s: dict[str, Any], now: datetime) -> None:
    """Count one upload against the hourly limit, or raise SessionError("rate_limited")."""
    _require_open(s)
    cutoff = now - timedelta(hours=1)
    recent = [t for t in s["upload_times"] if datetime.fromisoformat(t) > cutoff]
    if len(recent) >= FLOW["uploads_per_hour"]:
        s["upload_times"] = recent
        raise SessionError("rate_limited", "That's a lot of takes in an hour. Take a break and come back soon.")
    s["upload_times"] = recent + [_iso(now)]


def accept_take(s: dict[str, Any], slot: int, clip_id: str, flags: list[str], now: datetime) -> str | None:
    """Mark a slot accepted with a new clip. Returns the clip it replaced, if any."""
    st = require_record(s, slot)
    old = st["clip_id"] if st["status"] == "accepted" else None
    st.update(status="accepted", clip_id=clip_id, recorded_at=_iso(now), flags=list(flags), last_reject=None)
    _touch(s, now)
    return old


def reject_take(s: dict[str, Any], slot: int, flags: list[str], message: str | None, now: datetime) -> None:
    """Record a rejected try. An accepted slot keeps its take; an empty one shows the reason."""
    st = require_record(s, slot)
    st["rejected_count"] += 1
    st["last_reject"] = {"flags": list(flags), "message": message, "at": _iso(now)}
    if st["status"] != "accepted":
        st["status"] = "rejected"
    _touch(s, now)


def delete_take(s: dict[str, Any], slot: int, now: datetime) -> str | None:
    """Empty a slot. Returns the deleted clip_id (None if there was no accepted take)."""
    _require_open(s)
    st = _slot_state(s, slot)
    old = st["clip_id"] if st["status"] == "accepted" else None
    st.update(status="empty", clip_id=None, recorded_at=None, flags=[], last_reject=None)
    _touch(s, now)
    return old


# ---------------------------------------------------------------- submit


def can_submit(s: dict[str, Any]) -> bool:
    return s["status"] == "open" and accepted_count(s) == N_SLOTS


def submit(s: dict[str, Any], code: str, now: datetime) -> None:
    """Lock the session and store its completion code."""
    _require_open(s)
    if not can_submit(s):
        raise SessionError("incomplete", f"All {N_SLOTS} takes must be accepted before you submit.")
    if not is_completion_code(code):
        raise SessionError("bad_code")
    s.update(status="submitted", completion_code=code, submitted_at=_iso(now))
    _touch(s, now)


def new_completion_code(rng: secrets.SystemRandom | None = None) -> str:
    r = rng or secrets.SystemRandom()
    return "".join(r.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def is_completion_code(code: str) -> bool:
    return isinstance(code, str) and len(code) == CODE_LEN and all(c in CODE_ALPHABET for c in code)


def new_clip_id() -> str:
    return "c" + "".join(secrets.choice(CLIP_ALPHABET) for _ in range(CLIP_LEN))


# ---------------------------------------------------------------- invites


def token_hash(token: str) -> str:
    """invites/<token_hash>.json: only the hash of an invite token is ever stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def invite_valid(invite: dict[str, Any] | None, now: datetime) -> bool:
    """Not revoked and younger than FLOW["invite_days"]."""
    if not invite or invite.get("revoked"):
        return False
    created = datetime.fromisoformat(invite["created_at"])
    return now < created + timedelta(days=FLOW["invite_days"])


def take_metadata(
    s: dict[str, Any],
    slot: int,
    clip_id: str,
    now: datetime,
    *,
    user_agent: str,
    track_settings: dict[str, Any],
    sample_rate: int,
    duration_s: float,
    quality_flags: list[str],
) -> dict[str, Any]:
    """Spec 2.6 clip metadata for an accepted take."""
    p = s["profile"]
    info = SLOTS[slot - 1]
    meta = {
        "clip_id": clip_id,
        "singer_id": s["singer_id"],
        "session_id": s["session_id"],
        "recorded_at": _iso(now),
        "voice_type": p["voice_type"],
        "arpeggio_id": arpeggio_id(p["voice_type"]),
        "condition": info["condition"],
        "slot": slot,
        "loud": info["loud"],
        "sex": p["sex"],
        "device": f"{p['device']} / {p['mic']}",
        "user_agent": user_agent,
        "track_settings": track_settings,
        "sample_rate": int(sample_rate),
        "duration_s": round(float(duration_s), 3),
        "quality_flags": list(quality_flags),
    }
    if p.get("tradition"):
        meta["tradition"] = p["tradition"]
    return meta
