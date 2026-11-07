"""Recording quality checks (spec Section 5.3).

    result = check(x, sr, "alto", track_settings={"sampleRate": 48000})
    # {"ok": False, "flags": ["note_short"], "message": "Slow down a little and hold each note."}

`flags` lists every check that failed, in priority order; reject codes and
flag-only codes are mixed. `ok` is False if any reject code is present, and
`message` is then the re-record instruction for the first one; otherwise it
is None.

| code                | check        | effect | what is measured                               |
|---------------------|--------------|--------|------------------------------------------------|
| bandwidth_low_rate  | bandwidth    | reject | sample rate <= min_sample_rate_hz              |
| bandwidth_cliff     | bandwidth    | reject | sharp spectral cliff at <= cliff_max_hz        |
| clipping            | clipping     | reject | share of samples with |x| >= clip_level        |
| noise_high          | noise        | reject | SNR (voiced notes vs pre-roll) < snr_reject_db |
| noise_moderate      | noise        | flag   | snr_reject_db <= SNR < snr_flag_db             |
| note_short          | note_length  | reject | a note's steady core < min_core_s              |
| incomplete          | completeness | reject | fewer than 7 notes found                       |
| intervals_off       | completeness | reject | a note > max_interval_err_st off the pattern   |
| key_off             | completeness | reject | key offset > max_key_offset_st from target     |
| preroll_voice       | pre_roll     | flag   | voice in the pre-roll; noise check skipped     |

A pre-roll of pure digital silence (RMS exactly 0) counts as infinite SNR:
a pass, not an error.

Every threshold lives in QUALITY_CONFIG below and nowhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.signal import welch

from ml.arpeggios import INTERVALS_ST, N_NOTES
from ml.features.segment import Segmentation, segment

QUALITY_CONFIG: dict[str, Any] = {
    # clipping
    "clip_level": 0.99,  # |x| at or above this counts as clipped
    "clip_max_share": 0.001,  # reject if more than 0.1% of samples clip [TUNE]
    # noise
    "preroll_s": 1.0,  # silent lead-in that measures the room (spec 5.2)
    "snr_reject_db": 15.0,  # [TUNE]
    "snr_flag_db": 25.0,  # [TUNE]
    # bandwidth
    "min_sample_rate_hz": 16_000,  # reject at or below this
    "ref_band_hz": (1_000.0, 4_000.0),  # voice reference band
    "high_band_hz": (8_500.0, 11_000.0),  # band above a call-mic cutoff
    "high_band_min_width_hz": 500.0,  # skip the cliff test if Nyquist leaves less than this
    "cliff_min_hz": 2_000.0,  # search range for the cliff centre
    "cliff_max_hz": 8_000.0,  # cliffs above this are fine (spec: "at <= 8 kHz") [TUNE]
    "cliff_step_hz": 125.0,
    "cliff_window_hz": (250.0, 1_000.0),  # compare [f-1000, f-250] with [f+250, f+1000]
    "cliff_min_drop_db": 25.0,  # a drop this steep is a cliff, not a roll-off [TUNE]
    "high_band_max_db": -35.0,  # ...and the high band is this far below the ref band [TUNE]
    "psd_window_s": 0.046,  # Welch segment length (rounded up to a power of 2)
    # completeness
    "max_interval_err_st": 1.0,
    "max_key_offset_st": 2.0,
    # note length
    "min_core_s": 0.25,  # [TUNE]
    # pre-roll
    "preroll_guard_s": 0.05,  # ignore the last bit of the pre-roll (pitch window lead-in)
    "preroll_voice_min_s": 0.10,  # this much stable pitch in the pre-roll = voice
}

# Re-record instructions, in priority order: the first failing code's text is shown.
REJECT_MESSAGES: dict[str, str] = {
    "bandwidth_low_rate": "Your mic is cutting off high sounds. Switch to your device's built-in mic or a wired mic and record again.",
    "bandwidth_cliff": "Your mic is cutting off high sounds (Bluetooth headsets do this). Switch to your device's built-in mic or a wired mic and record again.",
    "clipping": "Too loud for the mic. Move a little further away and record again.",
    "noise_high": "Too much background noise. Find a quieter spot and record again.",
    "note_short": "Slow down a little and hold each note.",
    "incomplete": "We didn't hear all 7 notes. Sing the whole arpeggio, then stop.",
    "intervals_off": "Some notes were off the pattern. Listen to the example again, then record.",
    "key_off": "Start on the note the app plays before the count-in, then record again.",
}
# Notes for flag-only codes (shown with the result, not as a re-record message).
FLAG_NOTES: dict[str, str] = {
    "noise_moderate": "Recording was noisy, so this result is less certain.",
    "preroll_voice": 'Next time, wait for "Sing now" before you start.',
}
CHECK_OF: dict[str, str] = {
    "bandwidth_low_rate": "bandwidth",
    "bandwidth_cliff": "bandwidth",
    "clipping": "clipping",
    "noise_high": "noise",
    "noise_moderate": "noise",
    "note_short": "note_length",
    "incomplete": "completeness",
    "intervals_off": "completeness",
    "key_off": "completeness",
    "preroll_voice": "pre_roll",
}
FLAG_ORDER: list[str] = list(REJECT_MESSAGES) + list(FLAG_NOTES)
_EPS = 1e-20


def to_float_mono(wav: np.ndarray) -> np.ndarray:
    """int PCM or float audio, mono or (samples, channels), as float64 mono in [-1, 1]."""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


# ---------------------------------------------------------------- measures


def _clip_share(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) >= QUALITY_CONFIG["clip_level"])) if len(x) else 0.0


def _note_spans(seg: Segmentation, sr: int, n: int) -> list[tuple[int, int]]:
    """Sample spans of the found notes (whole notes, not just cores)."""
    spans = []
    for note in seg.notes:
        a, b = max(0, int(note.note_start * sr)), min(n, int(note.note_end * sr))
        if b > a:
            spans.append((a, b))
    return spans


def _preroll_voice_s(seg: Segmentation) -> float:
    """Seconds of stable pitch (segmentation candidates) inside the pre-roll."""
    end = QUALITY_CONFIG["preroll_s"] - QUALITY_CONFIG["preroll_guard_s"]
    voiced = sum(int(np.sum(seg.times[c.frames] < end)) for c in seg.candidates)
    return voiced * float(np.median(np.diff(seg.times))) if len(seg.times) > 1 else 0.0


def _snr_db(x: np.ndarray, sr: int, spans: list[tuple[int, int]]) -> float | None:
    """Voiced-note RMS vs pre-roll RMS, dB. inf for a digitally silent pre-roll."""
    pre_n = int(QUALITY_CONFIG["preroll_s"] * sr)
    voiced = [x[max(a, pre_n) : b] for a, b in spans if b > max(a, pre_n)]
    if len(x) < pre_n or not voiced:
        return None
    pre_rms = float(np.sqrt(np.mean(x[:pre_n] ** 2)))
    v = np.concatenate(voiced)
    voice_rms = float(np.sqrt(np.mean(v**2)))
    if pre_rms == 0.0:
        return float("inf")
    if voice_rms == 0.0:
        return float("-inf")
    return 20.0 * np.log10(voice_rms / pre_rms)


def _spectrum(x: np.ndarray, sr: int, spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Long-term power spectrum over the voiced notes (Welch per note, length-weighted)."""
    nper = 1 << int(np.ceil(np.log2(QUALITY_CONFIG["psd_window_s"] * sr)))
    total, weight, freqs = None, 0, None
    for a, b in spans:
        if b - a < nper:
            continue
        freqs, p = welch(x[a:b], fs=sr, nperseg=nper)
        total = p * (b - a) if total is None else total + p * (b - a)
        weight += b - a
    return None if total is None else (freqs, total / weight)


def _band_db(freqs: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    """Mean power per Hz in [lo, hi), dB."""
    m = (freqs >= lo) & (freqs < hi)
    return 10.0 * np.log10(np.mean(p[m]) + _EPS)


def _bandwidth(freqs: np.ndarray, p: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    """(high band vs ref band dB, steepest drop dB across a cliff at <= cliff_max_hz).

    None, None when Nyquist leaves too little of the high band to judge.
    """
    c = QUALITY_CONFIG
    hi_lo, hi_hi = c["high_band_hz"]
    hi_hi = min(hi_hi, 0.98 * sr / 2)
    if hi_hi - hi_lo < c["high_band_min_width_hz"]:
        return None, None
    high_db = _band_db(freqs, p, hi_lo, hi_hi) - _band_db(freqs, p, *c["ref_band_hz"])
    near, far = c["cliff_window_hz"]
    drops = [
        _band_db(freqs, p, f - far, f - near) - _band_db(freqs, p, f + near, f + far)
        for f in np.arange(c["cliff_min_hz"], c["cliff_max_hz"] + c["cliff_step_hz"] / 2, c["cliff_step_hz"])
    ]
    return high_db, float(max(drops))


def _track_rate(track_settings: Mapping[str, Any] | None) -> int | None:
    if not track_settings:
        return None
    rate = track_settings.get("sampleRate", track_settings.get("sample_rate"))
    try:
        return int(rate) if rate is not None else None
    except (TypeError, ValueError):
        return None


def measure(
    wav: np.ndarray, sr: int, voice_type: str, track_settings: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Raw quality measurements, before thresholds on flags (used by check() and reports)."""
    x = to_float_mono(wav)
    seg = segment(x, sr, voice_type)
    spans = _note_spans(seg, sr, len(x))
    preroll_voice_s = _preroll_voice_s(seg)
    track_rate = _track_rate(track_settings)
    spec = _spectrum(x, sr, spans)
    high_db, cliff_db = _bandwidth(*spec, sr) if spec else (None, None)
    return {
        "sample_rate": min(r for r in (sr, track_rate) if r is not None),
        "clip_share": _clip_share(x),
        "preroll_voice_s": preroll_voice_s,
        "snr_db": _snr_db(x, sr, spans),
        "high_band_db": high_db,
        "cliff_db": cliff_db,
        "n_notes": len(seg.notes),
        "interval_err_st": [
            abs(n.f0_st - seg.key_offset_st - INTERVALS_ST[n.index - 1]) for n in seg.notes
        ],
        "key_offset_st": seg.key_offset_st,
        "core_s": [n.core_s for n in seg.notes],
    }


# ---------------------------------------------------------------- verdict


def judge(m: Mapping[str, Any]) -> dict[str, Any]:
    """Turn measurements from measure() into {ok, flags, message}."""
    c = QUALITY_CONFIG
    found: set[str] = set()

    if m["sample_rate"] <= c["min_sample_rate_hz"]:
        found.add("bandwidth_low_rate")
    elif (
        m["cliff_db"] is not None
        and m["cliff_db"] >= c["cliff_min_drop_db"]
        and m["high_band_db"] <= c["high_band_max_db"]
    ):
        found.add("bandwidth_cliff")

    if m["clip_share"] > c["clip_max_share"]:
        found.add("clipping")

    if m["preroll_voice_s"] >= c["preroll_voice_min_s"]:
        found.add("preroll_voice")  # pre-roll isn't room noise: noise unmeasured
    elif m["snr_db"] is not None:
        if m["snr_db"] < c["snr_reject_db"]:
            found.add("noise_high")
        elif m["snr_db"] < c["snr_flag_db"]:
            found.add("noise_moderate")

    if any(s < c["min_core_s"] for s in m["core_s"]):
        found.add("note_short")
    if m["n_notes"] < N_NOTES:
        found.add("incomplete")
    if any(e > c["max_interval_err_st"] for e in m["interval_err_st"]):
        found.add("intervals_off")
    if m["key_offset_st"] is not None and abs(m["key_offset_st"]) > c["max_key_offset_st"]:
        found.add("key_off")

    flags = [f for f in FLAG_ORDER if f in found]
    rejects = [f for f in flags if f in REJECT_MESSAGES]
    return {"ok": not rejects, "flags": flags, "message": REJECT_MESSAGES[rejects[0]] if rejects else None}


def check(
    wav: np.ndarray, sr: int, voice_type: str, track_settings: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Quality-check one take: {"ok": bool, "flags": [code, ...], "message": str | None}.

    wav: mono (or multi-channel) samples, int PCM or float in [-1, 1].
    sr: sample rate of `wav`. voice_type: the arpeggio the singer was given.
    track_settings: the browser's track.getSettings(); its sampleRate is checked too.
    """
    x = to_float_mono(wav)
    if len(x) < int(QUALITY_CONFIG["preroll_s"] * sr):
        return {"ok": False, "flags": ["incomplete"], "message": REJECT_MESSAGES["incomplete"]}
    return judge(measure(x, sr, voice_type, track_settings))
