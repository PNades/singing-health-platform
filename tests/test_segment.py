"""Tests for ml/features/segment.py (spec Section 6.1).

There are no real recordings yet, so these use the four synthesized examples
plus generated "sung" arpeggios: legato notes with a short transition, a
little slow pitch drift, and optionally vibrato, slides, background noise, a
wrong key or a skipped note.

Every case also writes a plot to reports/segments/<case>.png so the note
boundaries can be checked by eye.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from ml.arpeggios import BEATS, BPM, INTERVALS_ST, VOICE_TYPES, target_hz
from ml.features.plot_segments import plot_segmentation
from ml.features.segment import (
    GLIDE_MAX_ST,
    TIME_STEP_S,
    Segmentation,
    fix_octave_errors,
    load_mono,
    segment,
    segment_pitch,
    track_pitch,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "api" / "static" / "audio"
REPORT_DIR = ROOT / "reports" / "segments"

SR = 44_100
CLEAN_CENTS = 5.0  # median-F0 tolerance on clean generated cases
ROUGH_CENTS = 15.0  # vibrato and noise cases
TEMPO_TOL = 0.03  # +/- 3%
CONTOUR_STEP_S = 0.001  # ground-truth pitch contour resolution


# ---------------------------------------------------------------- generator


@dataclass
class Take:
    """A generated take and its ground truth."""

    x: np.ndarray
    indices: list[int]  # notes actually sung, in order
    f0_hz: list[float]  # true pitch of each sung note
    steady: list[tuple[float, float]]  # true steady span of each sung note (s)
    glides: list[tuple[float, float]] = field(default_factory=list)  # transition spans (s)
    bpm: float = BPM
    contour_st: np.ndarray = field(default_factory=lambda: np.zeros(0))  # true pitch, st re 1 Hz,
    # every CONTOUR_STEP_S from t = 0: notes, glides and drift, without vibrato

    def true_st(self, t0: float, t1: float) -> np.ndarray:
        i0, i1 = int(np.ceil(t0 / CONTOUR_STEP_S)), int(np.floor(t1 / CONTOUR_STEP_S))
        return self.contour_st[i0 : i1 + 1]


def sing(
    voice: str,
    bpm: float = BPM,
    cents: float = 0.0,
    key_st: float = 0.0,
    skip: int | None = None,
    slide_s: float = 0.03,
    vibrato_cents: float = 0.0,
    vibrato_hz: float = 5.5,
    snr_db: float | None = None,
    seed: int = 0,
) -> Take:
    """A voice-like arpeggio take: 1 s pre-roll, legato notes, 0.5 s tail.

    slide_s: length of the pitch transition between notes (straddles the
    boundary). cents/key_st: detuning of every note. skip: 1-based index of a
    note the singer leaves out (the rest close up, back to back).
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    order = [i for i in range(1, 8) if i != skip]
    targets = target_hz(voice)
    hz = [targets[i - 1] * 2 ** ((cents + 100 * key_st) / 1200) for i in order]
    durs = [BEATS[i - 1] * beat for i in order]

    pre, tail = 1.0, 0.5
    bounds = pre + np.concatenate([[0], np.cumsum(durs)])
    n = int(round((bounds[-1] + tail) * SR))
    t = np.arange(n) / SR

    # Pitch contour in semitones: steps between notes, linear glide of slide_s.
    st_notes = 12 * np.log2(np.array(hz))
    st = np.full(n, st_notes[0])
    for k in range(len(order)):
        st[t >= bounds[k]] = st_notes[k]
    glides = []
    for k in range(1, len(order)):
        g0, g1 = bounds[k] - slide_s / 2, bounds[k] + slide_s / 2
        m = (t >= g0) & (t < g1)
        st[m] = st_notes[k - 1] + (st_notes[k] - st_notes[k - 1]) * (t[m] - g0) / slide_s
        glides.append((g0, g1))
    # Slow random drift (+/- ~3 cents) and optional vibrato.
    drift = np.interp(t, np.arange(0, t[-1] + 0.25, 0.25), rng.normal(0, 0.03, len(np.arange(0, t[-1] + 0.25, 0.25))))
    st = st + drift
    contour = st[:: int(round(CONTOUR_STEP_S * SR))]
    st = st + (vibrato_cents / 100) * np.sin(2 * np.pi * vibrato_hz * t)
    f0 = 2 ** (st / 12)

    phase = 2 * np.pi * np.cumsum(f0) / SR
    n_harm = int(8000 // f0.max())
    voice_sig = sum((k ** -1.5) * np.sin(k * phase) for k in range(1, n_harm + 1))

    env = np.zeros(n)
    sung = (t >= bounds[0]) & (t < bounds[-1])
    env[sung] = 1.0
    ramp = int(0.04 * SR)
    a, b = np.flatnonzero(sung)[[0, -1]]
    env[a : a + ramp] = np.linspace(0, 1, ramp)
    env[b - ramp + 1 : b + 1] = np.linspace(1, 0, ramp)
    x = voice_sig * env
    x *= 0.5 / np.max(np.abs(x))
    if snr_db is not None:
        rms = np.sqrt(np.mean(x[sung] ** 2))
        x = x + rng.normal(0, rms * 10 ** (-snr_db / 20), n)

    half = slide_s / 2
    steady = [(bounds[k] + (half if k else 0), bounds[k + 1] - (half if k < len(order) - 1 else 0)) for k in range(len(order))]
    return Take(x=x, indices=order, f0_hz=hz, steady=steady, glides=glides, bpm=bpm, contour_st=contour)


# ---------------------------------------------------------------- helpers


def save_plot(name: str, seg: Segmentation, x: np.ndarray, sr: int, take: Take | None = None) -> None:
    plot_segmentation(seg, x, sr, REPORT_DIR / f"{name}.png", title=name, truth=take.steady if take else None)


def cents_off(measured_hz: float, true_hz: float) -> float:
    return 1200 * np.log2(measured_hz / true_hz)


def check_take(seg: Segmentation, take: Take, cents_tol: float, key_st: float = 0.0, cents: float = 0.0) -> None:
    assert seg.indices == take.indices
    root_st = 12 * np.log2(target_hz(seg.voice_type)[0])
    for n, (s0, s1) in zip(seg.notes, take.steady):
        # True pitch at the core's frame centres (the span reaches half a frame further).
        true = take.true_st(n.start + TIME_STEP_S / 2, n.end - TIME_STEP_S / 2)
        true_med = float(np.median(true))
        # Median F0 matches the true pitch over the core.
        err_c = 100 * (12 * np.log2(n.f0_hz) - true_med)
        assert abs(err_c) <= cents_tol, (n.index, err_c)
        # Semitones are relative to the target root.
        assert abs(n.f0_st - (true_med - root_st)) <= cents_tol / 100
        # Spec 6.1 glide rule on the ground truth: nothing in the core strays
        # more than GLIDE_MAX_ST from the note's pitch.
        assert np.max(np.abs(true - true_med)) <= GLIDE_MAX_ST, (n.index, np.max(np.abs(true - true_med)))
        # The core is a real stretch of the note, not a sliver.
        assert n.core_s >= 0.5 * (s1 - s0), (n.index, n.core_s, s1 - s0)
    assert seg.tempo_bpm is not None
    assert abs(seg.tempo_bpm / take.bpm - 1) <= TEMPO_TOL, seg.tempo_bpm
    assert abs(seg.key_offset_st - (key_st + cents / 100)) <= cents_tol / 100


def run(name: str, voice: str, take: Take) -> Segmentation:
    seg = segment(take.x, SR, voice)
    save_plot(name, seg, take.x, SR, take)
    return seg


# ---------------------------------------------------------------- cases


@pytest.mark.parametrize("voice", VOICE_TYPES)
def test_synth_examples(voice: str) -> None:
    x, sr = load_mono(AUDIO_DIR / f"{voice}.wav")
    seg = segment(x, sr, voice)
    save_plot(f"example_{voice}", seg, x, sr)
    assert seg.indices == list(range(1, 8))
    for n, f in zip(seg.notes, target_hz(voice)):
        assert abs(cents_off(n.f0_hz, f)) <= 1.0
        assert n.core_s >= 0.25
    assert abs(seg.tempo_bpm / BPM - 1) <= TEMPO_TOL
    assert abs(seg.key_offset_st) < 0.01


@pytest.mark.parametrize("bpm, voice", [(70, "baritone"), (90, "tenor"), (120, "soprano")])
def test_tempos(bpm: float, voice: str) -> None:
    take = sing(voice, bpm=bpm, seed=bpm)
    check_take(run(f"tempo_{bpm}_{voice}", voice, take), take, CLEAN_CENTS)


def test_flat_30_cents() -> None:
    take = sing("baritone", cents=-30, seed=1)
    seg = run("flat_30c_baritone", "baritone", take)
    check_take(seg, take, CLEAN_CENTS, cents=-30)


def test_wrong_key_3_semitones() -> None:
    take = sing("alto", key_st=3, seed=2)
    seg = run("wrong_key_+3st_alto", "alto", take)
    check_take(seg, take, CLEAN_CENTS, key_st=3)
    assert seg.notes[0].f0_st == pytest.approx(3, abs=0.05)


def test_missing_note() -> None:
    take = sing("tenor", skip=6, seed=3)
    seg = run("missing_note6_tenor", "tenor", take)
    assert seg.indices == [1, 2, 3, 4, 5, 7]
    check_take(seg, take, CLEAN_CENTS)


def test_slides_between_notes() -> None:
    take = sing("tenor", slide_s=0.25, seed=4)
    seg = run("slides_250ms_tenor", "tenor", take)
    check_take(seg, take, CLEAN_CENTS)
    # The slides are cut out of the cores. A core may keep only the first or
    # last few tens of ms of a slide, where its pitch is still within
    # GLIDE_MAX_ST of the note (checked in check_take).
    for n in seg.notes:
        for g0, g1 in take.glides:
            overlap = max(0.0, min(n.end, g1) - max(n.start, g0))
            assert overlap <= 0.2 * (g1 - g0), (n.index, g0, g1, overlap)
    # A slow 5 st slide holds a level long enough to look like a note; it must
    # be dropped as a stray, and its frames must not skew the note onsets.
    assert len(seg.candidates) > 7
    assert abs(seg.tempo_bpm / take.bpm - 1) <= 0.01


def test_vibrato() -> None:
    # +/-60 cents is wider than the 0.5 st glide limit, so the core only
    # survives because of the >= one-cycle smoothing.
    take = sing("alto", vibrato_cents=60, vibrato_hz=5.5, seed=5)
    seg = run("vibrato_60c_5.5hz_alto", "alto", take)
    check_take(seg, take, ROUGH_CENTS)


def test_background_noise() -> None:
    take = sing("baritone", snr_db=20, seed=6)  # in the 15-25 dB "flag" zone (spec 5.3)
    seg = run("noise_snr20db_baritone", "baritone", take)
    check_take(seg, take, ROUGH_CENTS)


def test_brief_octave_errors_are_repaired() -> None:
    take = sing("tenor", seed=7)
    times, f0 = track_pitch(take.x, SR)
    clean = f0.copy()
    glitched = f0.copy()
    # A few frames an octave off inside note 2 (down) and note 4 (up).
    for (s0, s1), factor in [(take.steady[1], 0.5), (take.steady[3], 2.0)]:
        mid = np.flatnonzero((times > (s0 + s1) / 2) & (glitched > 0))[:6]
        glitched[mid] *= factor
    fixed = fix_octave_errors(times, glitched)
    np.testing.assert_allclose(fixed, clean, rtol=1e-9)

    seg = segment_pitch(times, glitched, "tenor")
    save_plot("octave_glitch_tenor", seg, take.x, SR, take)
    check_take(seg, take, CLEAN_CENTS)


def test_whole_take_octave_off_is_not_corrected() -> None:
    # A tenor singing an octave up (e.g. picked the wrong voice type).
    take = sing("tenor", key_st=12, seed=8)
    seg = run("octave_off_+12st_tenor", "tenor", take)
    check_take(seg, take, CLEAN_CENTS, key_st=12)
    assert seg.key_offset_st == pytest.approx(12, abs=0.1)


def test_core_has_no_glide() -> None:
    """Within every core, the smoothed pitch stays within GLIDE_MAX_ST of the median."""
    take = sing("tenor", slide_s=0.25, seed=9)
    seg = segment(take.x, SR, "tenor")
    for n in seg.notes:
        m = (seg.times >= n.start) & (seg.times <= n.end) & (seg.f0_hz > 0)
        st = 12 * np.log2(seg.f0_hz[m] / n.f0_hz)
        assert np.max(np.abs(st)) <= GLIDE_MAX_ST


def test_silence_has_no_notes() -> None:
    seg = segment(np.zeros(SR * 2), SR, "alto")
    assert seg.notes == [] and seg.tempo_bpm is None and seg.key_offset_st is None


def test_unknown_voice_type() -> None:
    with pytest.raises(ValueError):
        segment(np.zeros(SR), SR, "bass")
