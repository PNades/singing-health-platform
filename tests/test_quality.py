"""Tests for api/quality.py (spec Section 5.3).

All fixtures are generated: a voice-like legato arpeggio (harmonics up to
near Nyquist with a natural roll-off), recorded as in spec 5.2 with a 1 s
pre-roll, a pink room-noise floor, and 16-bit quantisation. Each case then
breaks one thing: clipping, a sharp 8 kHz low-pass (Bluetooth call mic),
heavy or moderate room noise, a digitally silent pre-roll, a 16 kHz sample
rate, a cut-off take, a take sung too fast, and a singer starting early.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, fftconvolve, firwin, sosfilt

from api.quality import FLAG_NOTES, QUALITY_CONFIG, REJECT_MESSAGES, check, judge, measure
from ml.arpeggios import BEATS, BPM, VOICE_TYPES, target_hz

SR = 44_100
CLEAN_SNR_DB = 45.0  # a quiet room
PRE_S = QUALITY_CONFIG["preroll_s"]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS 1/f noise: most of its energy is low, like room noise."""
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / np.sqrt(np.mean(x**2))


def sing(
    voice: str = "alto",
    sr: int = SR,
    bpm: float = BPM,
    start_s: float = PRE_S,
    snr_db: float | None = CLEAN_SNR_DB,
    stop_after_beats: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A take as int16 samples, plus a boolean mask of the sung samples.

    start_s: when singing starts (the pre-roll is always the first PRE_S).
    stop_after_beats: the recording stops this many beats into the arpeggio.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bounds = start_s + beat * np.concatenate([[0], np.cumsum(BEATS)])
    end_s = bounds[-1] + 0.5 if stop_after_beats is None else start_s + stop_after_beats * beat
    n = int(round(end_s * sr))
    t = np.arange(n) / sr

    # Legato notes with a 40 ms glide, a little vibrato.
    st_notes = 12 * np.log2(target_hz(voice))
    st = np.interp(t, np.repeat(bounds[1:-1], 2) + np.tile([-0.02, 0.02], 6), np.repeat(st_notes, 2)[1:-1])
    st = st + 0.2 * np.sin(2 * np.pi * 5.5 * t)
    f0 = 2 ** (st / 12)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    n_harm = int(0.45 * sr // f0.max())
    voice_sig = sum(k**-1.2 * np.sin(k * phase) for k in range(1, n_harm + 1))

    sung = (t >= bounds[0]) & (t < bounds[-1])
    env = sung.astype(float)
    ramp = int(0.04 * sr)
    a, b = np.flatnonzero(sung)[[0, -1]]
    env[a : a + ramp] = np.linspace(0, 1, ramp)
    env[max(a, b - ramp + 1) : b + 1] = np.linspace(1, 0, len(env[max(a, b - ramp + 1) : b + 1]))
    x = voice_sig * env
    x *= 0.5 / np.max(np.abs(x))  # peaks at -6 dBFS
    if snr_db is not None:
        rms = np.sqrt(np.mean(x[sung] ** 2))
        x = x + rms * 10 ** (-snr_db / 20) * pink_noise(n, rng)
    return to_int16(x), sung


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)


def as_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) / 32767


@pytest.fixture(scope="module")
def clean() -> np.ndarray:
    return sing()[0]


def assert_rejected(result: dict, code: str) -> None:
    assert not result["ok"], result
    assert code in result["flags"], result
    assert result["message"] == REJECT_MESSAGES[code], result


# ---------------------------------------------------------------- the fixtures


@pytest.mark.parametrize("voice", VOICE_TYPES)
def test_clean_take_passes(voice: str) -> None:
    x, _ = sing(voice)
    result = check(x, SR, voice, {"sampleRate": SR})
    assert result == {"ok": True, "flags": [], "message": None}


def test_clean_take_measures(clean: np.ndarray) -> None:
    m = measure(clean, SR, "alto")
    assert m["n_notes"] == 7
    assert abs(m["snr_db"] - CLEAN_SNR_DB) < 3
    assert m["clip_share"] == 0
    assert m["preroll_voice_s"] == 0
    assert m["cliff_db"] < QUALITY_CONFIG["cliff_min_drop_db"] / 2  # gradual roll-off
    assert min(m["core_s"]) > QUALITY_CONFIG["min_core_s"]


def test_clipped(clean: np.ndarray) -> None:
    x = to_int16(as_float(clean) * 4)  # 12 dB too hot, flat-topped at full scale
    assert_rejected(check(x, SR, "alto"), "clipping")


def test_lowpassed_at_8khz_is_a_cliff(clean: np.ndarray) -> None:
    """A Bluetooth call mic: everything (voice and room) above ~8 kHz is gone."""
    lp = firwin(1023, 8_000, fs=SR)
    x = to_int16(fftconvolve(as_float(clean), lp, mode="same"))
    result = check(x, SR, "alto", {"sampleRate": SR})
    assert result["flags"] == ["bandwidth_cliff"], result
    assert result["message"] == REJECT_MESSAGES["bandwidth_cliff"]


def test_dark_voice_is_not_a_cliff(clean: np.ndarray) -> None:
    """A gradual roll-off (dark voice or mic) leaves little above 8.5 kHz but passes."""
    sos = butter(2, 2_000, fs=SR, output="sos")
    x = to_int16(sosfilt(sos, as_float(clean)) * 1.5)
    m = measure(x, SR, "alto")
    assert m["high_band_db"] < QUALITY_CONFIG["high_band_max_db"]  # dark...
    assert m["cliff_db"] < QUALITY_CONFIG["cliff_min_drop_db"]  # ...but no cliff
    assert check(x, SR, "alto")["ok"]


def test_heavy_room_noise_rejects() -> None:
    x, _ = sing(snr_db=QUALITY_CONFIG["snr_reject_db"] - 7)
    assert_rejected(check(x, SR, "alto"), "noise_high")


def test_moderate_room_noise_flags() -> None:
    snr = (QUALITY_CONFIG["snr_reject_db"] + QUALITY_CONFIG["snr_flag_db"]) / 2
    x, _ = sing(snr_db=snr)
    assert check(x, SR, "alto") == {"ok": True, "flags": ["noise_moderate"], "message": None}
    assert "noise_moderate" in FLAG_NOTES


def test_digitally_silent_preroll_passes(clean: np.ndarray) -> None:
    x = clean.copy()
    x[: int(PRE_S * SR)] = 0
    assert measure(x, SR, "alto")["snr_db"] == float("inf")
    assert check(x, SR, "alto") == {"ok": True, "flags": [], "message": None}


def test_fully_noiseless_take_passes() -> None:
    x, _ = sing(snr_db=None)
    assert check(x, SR, "alto") == {"ok": True, "flags": [], "message": None}


def test_16khz_sample_rate_rejects() -> None:
    x, _ = sing(sr=16_000)
    assert_rejected(check(x, 16_000, "alto", {"sampleRate": 16_000}), "bandwidth_low_rate")


def test_16khz_track_setting_rejects(clean: np.ndarray) -> None:
    """The browser reports a 16 kHz track even though the WAV header says 44.1 kHz."""
    assert_rejected(check(clean, SR, "alto", {"sampleRate": 16_000}), "bandwidth_low_rate")


def test_cut_off_take_is_incomplete() -> None:
    x, _ = sing(stop_after_beats=5.0)  # stopped as note 5 starts
    result = check(x, SR, "alto")
    assert result["flags"] == ["incomplete"], result
    assert result["message"] == REJECT_MESSAGES["incomplete"]


def test_sung_too_fast_is_note_short() -> None:
    x, _ = sing(bpm=180)  # 0.33 s notes: steady cores under 250 ms
    m = measure(x, SR, "alto")
    assert m["n_notes"] == 7  # every note found...
    assert min(m["core_s"]) < QUALITY_CONFIG["min_core_s"]  # ...but too short to analyse
    result = check(x, SR, "alto")
    assert result["flags"] == ["note_short"], result
    assert result["message"] == "Slow down a little and hold each note."


def test_singer_starts_during_preroll() -> None:
    x, _ = sing(start_s=0.4)
    result = check(x, SR, "alto")
    assert result == {"ok": True, "flags": ["preroll_voice"], "message": None}


def test_early_start_does_not_hide_clipping() -> None:
    x, _ = sing(start_s=0.4)
    x = to_int16(as_float(x) * 4)
    result = check(x, SR, "alto")
    assert_rejected(result, "clipping")
    assert "preroll_voice" in result["flags"]


def test_shorter_than_preroll_is_incomplete() -> None:
    assert_rejected(check(np.zeros(SR // 2, dtype=np.int16), SR, "alto"), "incomplete")


# ---------------------------------------------------------------- thresholds


def good() -> dict:
    return {
        "sample_rate": 48_000,
        "clip_share": 0.0,
        "preroll_voice_s": 0.0,
        "snr_db": 40.0,
        "high_band_db": -20.0,
        "cliff_db": 3.0,
        "n_notes": 7,
        "interval_err_st": [0.1] * 7,
        "key_offset_st": 0.3,
        "core_s": [0.4] * 7,
    }


@pytest.mark.parametrize(
    ("change", "flags"),
    [
        ({}, []),
        ({"snr_db": QUALITY_CONFIG["snr_reject_db"]}, ["noise_moderate"]),  # boundary is a flag
        ({"snr_db": QUALITY_CONFIG["snr_flag_db"]}, []),
        ({"snr_db": float("inf")}, []),
        ({"snr_db": None}, []),  # no voiced notes to measure: other checks reject
        ({"snr_db": 5.0, "preroll_voice_s": 0.5}, ["preroll_voice"]),  # noise skipped
        ({"clip_share": QUALITY_CONFIG["clip_max_share"]}, []),
        ({"clip_share": 2 * QUALITY_CONFIG["clip_max_share"]}, ["clipping"]),
        ({"sample_rate": 16_000}, ["bandwidth_low_rate"]),
        ({"sample_rate": 22_050}, []),
        ({"cliff_db": 40.0}, []),  # steep, but real energy above 8.5 kHz
        ({"cliff_db": 40.0, "high_band_db": -60.0}, ["bandwidth_cliff"]),
        ({"n_notes": 6, "interval_err_st": [0.1] * 6, "core_s": [0.4] * 6}, ["incomplete"]),
        ({"interval_err_st": [0.1] * 6 + [1.2]}, ["intervals_off"]),
        ({"key_offset_st": -2.5}, ["key_off"]),
        ({"key_offset_st": 12.0}, ["key_off"]),  # whole take an octave up
        ({"core_s": [0.4] * 6 + [0.2]}, ["note_short"]),
    ],
)
def test_judge_thresholds(change: dict, flags: list[str]) -> None:
    result = judge(good() | change)
    assert result["flags"] == flags
    assert result["ok"] == all(f in FLAG_NOTES for f in flags)
    assert (result["message"] is None) == result["ok"]


def test_message_follows_priority() -> None:
    """Fast singing often drops notes too; "slow down" is the useful message."""
    result = judge(good() | {"n_notes": 5, "core_s": [0.2] * 5, "interval_err_st": [0.1] * 5})
    assert result["flags"] == ["note_short", "incomplete"]
    assert result["message"] == REJECT_MESSAGES["note_short"]


# ---------------------------------------------------------------- report helpers


def test_test_singers(tmp_path) -> None:
    from ml.splits import test_singers

    assert test_singers(tmp_path / "missing.json") == set()
    p = tmp_path / "splits.json"
    p.write_text('{"test": ["s009", "s010"], "train": ["s001"]}', encoding="utf-8")
    assert test_singers(p) == {"s009", "s010"}
