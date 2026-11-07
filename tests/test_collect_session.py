"""Session logic of the collection site (ml/collect/session.py, spec 2.8)."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from ml.collect import session as S
from ml.collect.session import SessionError

T0 = datetime(2026, 11, 20, 14, 0, tzinfo=UTC)
PROFILE = {"voice_type": "alto", "sex": "F", "device": "phone", "mic": "built-in", "tradition": "Carnatic"}


def ready(now: datetime = T0) -> dict:
    """A session through consent, profile and mic check."""
    s = S.new_session("s007", now)
    S.set_consent(s, "consent-v1", now)
    S.set_profile(s, PROFILE, now)
    S.set_mic_check(s, ok=True, flags=[], now=now)
    return s


def fill(s: dict, upto: int = S.N_SLOTS) -> None:
    for n in range(1, upto + 1):
        S.accept_take(s, n, f"c{n:010d}", [], T0)


# ---------------------------------------------------------------- slots definition


def test_slots_follow_spec_2_3():
    conds = [x["condition"] for x in S.SLOTS]
    assert conds == ["comfortable"] * 6 + ["edge"] * 4 + ["pushed"] * 6
    assert [x["slot"] for x in S.SLOTS if x["loud"]] == [5, 6]
    assert all(x["warning"] == S.PUSHED_WARNING for x in S.SLOTS if x["condition"] == "pushed")
    assert all(x["warning"] is None for x in S.SLOTS if x["condition"] != "pushed")
    assert "not loud" in S.SLOTS[14]["instruction"] and "not loud" in S.SLOTS[15]["instruction"]


def test_new_session_is_empty_and_has_no_personal_fields():
    s = S.new_session("s007", T0)
    assert s["session_id"] == "s007-1" and s["status"] == "open"
    assert {v["status"] for v in s["slots"].values()} == {"empty"}
    assert len(s["slots"]) == 16
    assert s["completion_code"] is None
    flat = repr(s).lower()
    assert "name" not in flat.replace("singer_id", "") and "email" not in flat


# ---------------------------------------------------------------- slot states


def test_accept_reject_delete_cycle():
    s = ready()
    st = s["slots"]["3"]
    S.reject_take(s, 3, ["noise_high"], "Too much background noise.", T0)
    assert st["status"] == "rejected" and st["rejected_count"] == 1
    assert st["last_reject"]["message"].startswith("Too much")

    assert S.accept_take(s, 3, "cAAAAAAAAAA", ["noise_moderate"], T0) is None
    assert st["status"] == "accepted" and st["clip_id"] == "cAAAAAAAAAA"
    assert st["flags"] == ["noise_moderate"] and st["last_reject"] is None

    # A rejected re-record keeps the accepted take.
    S.reject_take(s, 3, ["clipping"], "Too loud.", T0)
    assert st["status"] == "accepted" and st["clip_id"] == "cAAAAAAAAAA"
    assert st["rejected_count"] == 2

    # An accepted re-record replaces it and reports the old clip for deletion.
    assert S.accept_take(s, 3, "cBBBBBBBBBB", [], T0) == "cAAAAAAAAAA"
    assert st["clip_id"] == "cBBBBBBBBBB"

    assert S.delete_take(s, 3, T0) == "cBBBBBBBBBB"
    assert st["status"] == "empty" and st["clip_id"] is None
    assert st["rejected_count"] == 2  # history of rejections is kept
    assert S.delete_take(s, 3, T0) is None  # deleting an empty slot is harmless


def test_slots_can_be_recorded_in_any_order():
    s = ready()
    assert all(S.can_record(s, n) for n in range(1, 17))
    S.accept_take(s, 16, "c0000000016", [], T0)
    S.accept_take(s, 1, "c0000000001", [], T0)
    assert S.accepted_clips(s) == {1: "c0000000001", 16: "c0000000016"}


@pytest.mark.parametrize("slot", [0, 17, -1])
def test_bad_slot(slot):
    s = ready()
    assert not S.can_record(s, slot)
    with pytest.raises(SessionError) as e:
        S.accept_take(s, slot, "cX", [], T0)
    assert e.value.code == "bad_slot"


def test_recording_needs_profile_and_mic_check():
    s = S.new_session("s007", T0)
    assert not S.can_record(s, 1)
    with pytest.raises(SessionError, match="no_consent|Consent"):
        S.set_profile(s, PROFILE, T0)
    S.set_consent(s, "consent-v1", T0)
    with pytest.raises(SessionError) as e:
        S.accept_take(s, 1, "cX", [], T0)
    assert e.value.code == "no_profile"
    S.set_profile(s, PROFILE, T0)
    with pytest.raises(SessionError) as e:
        S.accept_take(s, 1, "cX", [], T0)
    assert e.value.code == "no_mic_check"
    S.set_mic_check(s, ok=False, flags=["noise_high"], now=T0)
    assert s["mic_check"] is None and not S.can_record(s, 1)
    S.set_mic_check(s, ok=True, flags=["noise_moderate"], now=T0)
    assert S.can_record(s, 1)


def test_profile_validation_and_voice_lock():
    s = ready()
    for bad in ({**PROFILE, "voice_type": "bass"}, {**PROFILE, "device": "watch"},
                {**PROFILE, "sex": "X"}, {**PROFILE, "tradition": "x" * 101}):
        with pytest.raises(SessionError) as e:
            S.set_profile(s, bad, T0)
        assert e.value.code == "bad_profile"
    S.set_profile(s, {**PROFILE, "sex": "", "tradition": "  "}, T0)
    assert s["profile"]["sex"] == "unknown" and s["profile"]["tradition"] is None

    S.set_profile(s, {**PROFILE, "voice_type": "soprano"}, T0)  # no takes yet: allowed
    S.accept_take(s, 1, "c0000000001", [], T0)
    with pytest.raises(SessionError) as e:
        S.set_profile(s, {**PROFILE, "voice_type": "alto"}, T0)
    assert e.value.code == "voice_locked"
    S.set_profile(s, {**PROFILE, "voice_type": "soprano", "mic": "USB mic"}, T0)  # other fields still editable
    S.delete_take(s, 1, T0)
    S.set_profile(s, PROFILE, T0)  # unlocked again once no takes remain


# ---------------------------------------------------------------- submit


def test_submit_only_when_all_16_accepted():
    s = ready()
    fill(s, 15)
    S.reject_take(s, 16, ["incomplete"], "msg", T0)
    assert not S.can_submit(s)
    with pytest.raises(SessionError) as e:
        S.submit(s, "ABCDEFGH", T0)
    assert e.value.code == "incomplete"
    assert s["status"] == "open" and s["completion_code"] is None

    S.accept_take(s, 16, "c0000000016", [], T0)
    assert S.can_submit(s)
    S.delete_take(s, 4, T0)
    assert not S.can_submit(s)
    S.accept_take(s, 4, "c0000000004", [], T0)
    assert S.can_submit(s)


def test_completion_code_only_after_submit():
    s = ready()
    fill(s)
    assert s["completion_code"] is None and s["submitted_at"] is None
    code = S.new_completion_code()
    S.submit(s, code, T0 + timedelta(minutes=5))
    assert s["status"] == "submitted" and s["completion_code"] == code
    assert s["submitted_at"] == (T0 + timedelta(minutes=5)).isoformat(timespec="seconds")
    assert not S.can_submit(s)


def test_submitted_session_is_locked():
    s = ready()
    fill(s)
    S.submit(s, S.new_completion_code(), T0)
    attempts = [
        lambda: S.accept_take(s, 1, "cNEW", [], T0),
        lambda: S.reject_take(s, 1, [], None, T0),
        lambda: S.delete_take(s, 1, T0),
        lambda: S.set_profile(s, PROFILE, T0),
        lambda: S.set_consent(s, "consent-v1", T0),
        lambda: S.set_mic_check(s, True, [], T0),
        lambda: S.note_upload(s, T0),
        lambda: S.submit(s, "ABCDEFGH", T0),
    ]
    for f in attempts:
        with pytest.raises(SessionError) as e:
            f()
        assert e.value.code == "submitted"
    assert not S.can_record(s, 1)
    assert S.accepted_count(s) == 16


def test_completion_code_format():
    rng = random.Random(1)
    codes = {S.new_completion_code(rng) for _ in range(5000)}
    assert len(codes) == 5000
    for c in codes:
        assert len(c) == 8 and S.is_completion_code(c)
        assert not set(c) & set("0O1IL")
    s = ready()
    fill(s)
    with pytest.raises(SessionError):
        S.submit(s, "ABCDEFG0", T0)  # ambiguous character


def test_clip_ids_are_random_and_opaque():
    ids = {S.new_clip_id() for _ in range(2000)}
    assert len(ids) == 2000
    assert all(len(i) == 11 and i[0] == "c" for i in ids)


# ---------------------------------------------------------------- limits, invites


def test_upload_rate_limit_30_per_hour():
    s = ready()
    for k in range(30):
        S.note_upload(s, T0 + timedelta(seconds=k))
    with pytest.raises(SessionError) as e:
        S.note_upload(s, T0 + timedelta(minutes=30))
    assert e.value.code == "rate_limited"
    # An hour after the first uploads, room frees up again and old entries are pruned.
    S.note_upload(s, T0 + timedelta(hours=1, seconds=5))
    assert len(s["upload_times"]) <= 30


def test_invite_validity():
    inv = {"singer_id": "s007", "created_at": T0.isoformat(), "revoked": False}
    assert S.invite_valid(inv, T0 + timedelta(days=29, hours=23))
    assert not S.invite_valid(inv, T0 + timedelta(days=30))
    assert not S.invite_valid({**inv, "revoked": True}, T0)
    assert not S.invite_valid(None, T0)


def test_take_metadata_has_spec_2_6_fields():
    s = ready()
    meta = S.take_metadata(s, 5, "cABCDEFGHJK", T0, user_agent="UA", track_settings={"sampleRate": 48000},
                           sample_rate=48000, duration_s=9.81234, quality_flags=["noise_moderate"])
    required = {"clip_id", "singer_id", "session_id", "recorded_at", "voice_type", "arpeggio_id", "condition",
                "slot", "loud", "sex", "device", "user_agent", "track_settings", "sample_rate", "duration_s"}
    assert required <= set(meta)
    assert meta["arpeggio_id"] == "alto_F4" and meta["condition"] == "comfortable" and meta["loud"] is True
    assert meta["device"] == "phone / built-in" and meta["tradition"] == "Carnatic"
    assert meta["duration_s"] == 9.812
    assert "name" not in meta and "email" not in meta
