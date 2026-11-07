"""Split an arpeggio take into its 7 notes by pitch (spec Section 6.1).

    seg = segment_file("take.wav", "alto")
    for n in seg.notes:
        print(n.index, n.start, n.end, n.f0_hz, n.f0_st)
    seg.tempo_bpm, seg.key_offset_st

Steps:
1. Track pitch with Praat (parselmouth), 60-1100 Hz, 10 ms steps.
2. Fold brief tracker octave errors back (a few frames ~12 st off the
   surrounding pitch). A whole take an octave off is left alone and shows up
   as key_offset_st ~ +/-12.
3. Find stable pitch levels: a frame joins the current level while it stays
   within LEVEL_TOL_ST of the level's running median. Short levels (glides,
   blips) are dropped; neighbouring levels at the same pitch are merged.
4. Line the levels up against the 1-3-5-8-5-3-1 pattern at whatever key the
   singer used, so a skipped note keeps the other notes' indices and stray
   sounds are dropped.
5. Between sung-through (legato) neighbours, the note boundary is where the
   pitch first gets nearer the next note. Tempo comes from these onsets.
6. Steady core of each note: drop the first and last CORE_TRIM, smooth the
   pitch over SMOOTH_S (>= one vibrato cycle), keep the longest unbroken
   stretch within GLIDE_MAX_ST of the note's median.

Segmentation only reports what it found. Whether a take is complete, in key
or long enough is decided by api/quality.py.

This module must not import matplotlib (api/quality.py imports it). Plots are
in ml/features/plot_segments.py.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import parselmouth
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

from ml.arpeggios import ARPEGGIOS, BEATS, INTERVALS_ST, N_NOTES, root_hz

# Pitch tracking (spec 6.1)
PITCH_FLOOR_HZ = 60.0
PITCH_CEILING_HZ = 1100.0
TIME_STEP_S = 0.01

# Octave-error repair
OCTAVE_CONTEXT_S = 0.15  # look this far either side for the surrounding pitch
OCTAVE_TOL_ST = 2.0  # a frame 12 +/- this many st off its context is an octave error

# Finding notes
LEVEL_TOL_ST = 1.5  # half the smallest arpeggio interval (3 st)
MAX_GAP_S = 0.05  # an unvoiced gap longer than this ends a level
MIN_NOTE_S = 0.12  # shorter levels are glides or blips [TUNE]
MERGE_GAP_S = 0.30  # same-pitch levels this close are one note

# Lining up with the pattern
MATCH_MAX_ST = 2.5  # a note further than this from its slot can't fill it
SKIP_COST_ST = 2.0  # cost of a missing note or a stray sound

# Steady core (spec 6.1) [TUNE]
CORE_TRIM = 0.15  # dropped from each end of a note
SMOOTH_S = 0.21  # moving mean; >= one vibrato cycle (~0.18 s at 5.5 Hz)
GLIDE_MAX_ST = 0.5  # smoothed pitch further than this from the median is a glide


@dataclass
class Note:
    index: int  # 1-7 in sung order; 4 is the held top note
    start: float  # steady core, s
    end: float
    note_start: float  # whole note, s
    note_end: float
    f0_hz: float  # median F0 of the core (after vibrato-cycle smoothing)
    f0_st: float  # f0_hz in semitones above the target root of the voice type

    @property
    def core_s(self) -> float:
        return self.end - self.start


@dataclass
class Candidate:
    """A stable pitch level found in the take, before lining up with the pattern."""

    frames: np.ndarray  # indices into the pitch track
    f0_st: float  # median pitch, st above the target root


@dataclass
class Segmentation:
    voice_type: str
    notes: list[Note]  # found notes, by index; missing notes are absent
    tempo_bpm: float | None  # None if no two consecutive notes were found
    key_offset_st: float | None  # median of sung minus target pattern; None if no notes
    times: np.ndarray = field(repr=False)  # pitch-track frame centres, s
    f0_hz: np.ndarray = field(repr=False)  # after octave repair; 0 = unvoiced
    f0_raw_hz: np.ndarray = field(repr=False)  # as tracked, before octave repair
    candidates: list[Candidate] = field(repr=False, default_factory=list)

    @property
    def indices(self) -> list[int]:
        return [n.index for n in self.notes]

    def note(self, index: int) -> Note | None:
        return next((n for n in self.notes if n.index == index), None)

    def to_dict(self) -> dict:
        """JSON-friendly summary (no pitch track)."""
        return {
            "voice_type": self.voice_type,
            "tempo_bpm": self.tempo_bpm,
            "key_offset_st": self.key_offset_st,
            "notes": [asdict(n) for n in self.notes],
        }


# ---------------------------------------------------------------- audio, pitch


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as float64 mono in [-1, 1]."""
    sr, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64), int(sr)


def track_pitch(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Praat autocorrelation pitch. Returns (frame times s, F0 Hz with 0 = unvoiced)."""
    pitch = parselmouth.Sound(x, sampling_frequency=sr).to_pitch_ac(
        time_step=TIME_STEP_S, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ
    )
    return pitch.xs(), pitch.selected_array["frequency"].copy()


def fix_octave_errors(times: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Fold frames that sit ~12 st off the surrounding pitch back by an octave.

    The surrounding pitch is the median over +/- OCTAVE_CONTEXT_S of voiced
    frames, so only errors shorter than about that long are repaired; a whole
    take (or a whole note) an octave off is left as sung.
    """
    out = f0.copy()
    voiced = np.flatnonzero(f0 > 0)
    if len(voiced) < 3:
        return out
    st = 12 * np.log2(f0[voiced])
    t = times[voiced]
    lo = np.searchsorted(t, t - OCTAVE_CONTEXT_S, side="left")
    hi = np.searchsorted(t, t + OCTAVE_CONTEXT_S, side="right")
    for k in range(len(voiced)):
        diff = st[k] - np.median(st[lo[k] : hi[k]])
        octaves = np.round(diff / 12)
        if octaves != 0 and abs(diff - 12 * octaves) <= OCTAVE_TOL_ST:
            out[voiced[k]] = f0[voiced[k]] * 2.0 ** (-octaves)
    return out


# ---------------------------------------------------------------- notes


def _levels(times: np.ndarray, st: np.ndarray, voiced: np.ndarray) -> list[np.ndarray]:
    """Group voiced frames into stable pitch levels (lists of frame indices)."""
    runs: list[list[int]] = []
    cur: list[int] = []
    for i in np.flatnonzero(voiced):
        if cur and (
            times[i] - times[cur[-1]] > MAX_GAP_S + TIME_STEP_S / 2
            or abs(st[i] - np.median(st[cur])) > LEVEL_TOL_ST
        ):
            runs.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        runs.append(cur)

    min_frames = int(round(MIN_NOTE_S / TIME_STEP_S))
    runs = [r for r in runs if len(r) >= min_frames]

    merged: list[list[int]] = []
    for r in runs:
        if (
            merged
            and abs(np.median(st[r]) - np.median(st[merged[-1]])) <= LEVEL_TOL_ST
            and times[r[0]] - times[merged[-1][-1]] <= MERGE_GAP_S
        ):
            merged[-1].extend(r)
        else:
            merged.append(r)
    return [np.array(r) for r in merged]


def _join_legato(
    times: np.ndarray, st: np.ndarray, voiced: np.ndarray, levels: list[np.ndarray]
) -> list[np.ndarray]:
    """Give the frames between legato neighbours to the nearer note.

    A level only starts once the pitch has left the previous one, at a point
    that depends on the interval (and a slow slide can leave a stray level of
    its own), so the boundary between two sung-through notes is put where the
    pitch first gets nearer the next note than the current one. Called on the
    matched notes only, so frames of stray levels go to the nearer note.
    Notes separated by an unvoiced gap keep their own edges.
    """
    out = [lv.copy() for lv in levels]
    for k in range(len(out) - 1):
        a, b = out[k], out[k + 1]
        between = np.arange(a[-1] + 1, b[0])
        between = between[voiced[between]]
        span = np.concatenate([[a[-1]], between, [b[0]]])
        if np.any(np.diff(times[span]) > MAX_GAP_S + TIME_STEP_S / 2):
            continue  # not legato
        a_st, b_st = np.median(st[a]), np.median(st[b])
        nearer_b = np.abs(st[between] - b_st) < np.abs(st[between] - a_st)
        cut = int(np.argmax(nearer_b)) if nearer_b.any() else len(between)
        out[k] = np.concatenate([a, between[:cut]])
        out[k + 1] = np.concatenate([between[cut:], b])
    return out


def _steady_core(times: np.ndarray, st: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longest unbroken stretch of the note's middle within GLIDE_MAX_ST of its median.

    Returns (core frame indices, smoothed pitch in st over the core).
    """
    n = len(frames)
    trim = int(n * CORE_TRIM)
    mid = np.arange(trim, n - trim) if n - 2 * trim > 0 else np.arange(n)
    width = max(1, int(round(SMOOTH_S / TIME_STEP_S)) | 1)  # odd
    smooth = uniform_filter1d(st[frames], size=width, mode="nearest")
    ref = np.median(st[frames[mid]])
    ok = np.abs(smooth[mid] - ref) <= GLIDE_MAX_ST

    best: tuple[int, int] = (0, 0)
    run_start = None
    for k in range(len(mid) + 1):
        joined = (
            k < len(mid)
            and ok[k]
            and run_start is not None
            and times[frames[mid[k]]] - times[frames[mid[k - 1]]] <= 1.5 * TIME_STEP_S
        )
        if joined:
            continue
        if run_start is not None and k - run_start > best[1] - best[0]:
            best = (run_start, k)
        run_start = k if k < len(mid) and ok[k] else None
    if best[1] == best[0]:  # nothing steady: keep the frame nearest the median
        k = int(np.argmin(np.abs(smooth[mid] - ref)))
        best = (k, k + 1)
    sel = mid[best[0] : best[1]]
    return frames[sel], smooth[sel]


def _align(cand_st: list[float]) -> tuple[list[tuple[int, int]], float]:
    """Line candidates up with INTERVALS_ST at a free key offset.

    Returns ([(candidate i, note slot j)], key offset st). Monotonic alignment
    (dynamic programming) with a cost per missing note or stray sound; the key
    offset is tried at every candidate/slot pairing, then refined to the
    median residual of the best alignment.
    """
    m = len(cand_st)
    if m == 0:
        return [], 0.0
    intervals = np.array(INTERVALS_ST, dtype=float)
    cs = np.array(cand_st)

    def dp(offset: float) -> tuple[float, list[tuple[int, int]]]:
        inf = float("inf")
        cost = np.full((m + 1, N_NOTES + 1), inf)
        back = np.zeros((m + 1, N_NOTES + 1), dtype=int)  # 0 diag, 1 skip cand, 2 skip slot
        cost[0, :] = SKIP_COST_ST * np.arange(N_NOTES + 1)
        cost[:, 0] = SKIP_COST_ST * np.arange(m + 1)
        back[1:, 0], back[0, 1:] = 1, 2
        for i in range(1, m + 1):
            for j in range(1, N_NOTES + 1):
                r = abs(cs[i - 1] - intervals[j - 1] - offset)
                options = (
                    cost[i - 1, j - 1] + r if r <= MATCH_MAX_ST else inf,
                    cost[i - 1, j] + SKIP_COST_ST,
                    cost[i, j - 1] + SKIP_COST_ST,
                )
                back[i, j] = int(np.argmin(options))
                cost[i, j] = options[back[i, j]]
        pairs: list[tuple[int, int]] = []
        i, j = m, N_NOTES
        while i > 0 or j > 0:
            b = back[i, j]
            if b == 0:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
            elif b == 1:
                i -= 1
            else:
                j -= 1
        return float(cost[m, N_NOTES]), pairs[::-1]

    best: tuple[float, float, list[tuple[int, int]]] | None = None
    for offset in sorted({float(c - iv) for c in cs for iv in set(INTERVALS_ST)}):
        _, pairs = dp(offset)
        if not pairs:
            continue
        offset = float(np.median([cs[i] - intervals[j] for i, j in pairs]))
        total, pairs = dp(offset)
        key = (total, -len(pairs), abs(offset))
        if best is None or key < (best[0], -len(best[2]), abs(best[1])):
            best = (total, offset, pairs)
    if best is None:
        return [], 0.0
    _, _, pairs = best
    offset = float(np.median([cs[i] - intervals[j] for i, j in pairs]))
    return pairs, offset


def _tempo_bpm(notes: list[Note]) -> float | None:
    """Median seconds per beat over consecutive found notes (top note = 2 beats)."""
    spb = [
        (b.note_start - a.note_start) / BEATS[a.index - 1]
        for a, b in zip(notes, notes[1:])
        if b.index == a.index + 1
    ]
    return 60.0 / float(np.median(spb)) if spb else None


def segment_pitch(times: np.ndarray, f0: np.ndarray, voice_type: str) -> Segmentation:
    """Segment a pitch track (times s, F0 Hz with 0 = unvoiced) for a voice type."""
    if voice_type not in ARPEGGIOS:
        raise ValueError(f"voice_type must be one of {', '.join(ARPEGGIOS)}, not {voice_type!r}")
    f0_raw = np.asarray(f0, dtype=float)
    f0 = fix_octave_errors(times, f0_raw)
    root = root_hz(voice_type)
    voiced = f0 > 0
    st = np.full(len(f0), np.nan)
    st[voiced] = 12 * np.log2(f0[voiced] / root)

    candidates = [Candidate(fr, float(np.median(st[fr]))) for fr in _levels(times, st, voiced)]
    pairs, _ = _align([c.f0_st for c in candidates])
    note_frames = _join_legato(times, st, voiced, [candidates[i].frames for i, _ in pairs])

    half = TIME_STEP_S / 2
    notes = []
    for (_, j), frames in zip(pairs, note_frames):
        core, smooth = _steady_core(times, st, frames)
        # Median of the cycle-smoothed pitch: the centre of any vibrato. The raw
        # median is biased when the core isn't a whole number of vibrato cycles.
        core_st = float(np.median(smooth))
        notes.append(
            Note(
                index=j + 1,
                start=float(times[core[0]] - half),
                end=float(times[core[-1]] + half),
                note_start=float(times[frames[0]] - half),
                note_end=float(times[frames[-1]] + half),
                f0_hz=float(root * 2 ** (core_st / 12)),
                f0_st=core_st,
            )
        )
    key_offset = float(np.median([n.f0_st - INTERVALS_ST[n.index - 1] for n in notes])) if notes else None
    return Segmentation(
        voice_type=voice_type,
        notes=notes,
        tempo_bpm=_tempo_bpm(notes),
        key_offset_st=key_offset,
        times=times,
        f0_hz=f0,
        f0_raw_hz=f0_raw,
        candidates=candidates,
    )


def segment(x: np.ndarray, sr: int, voice_type: str) -> Segmentation:
    """Segment mono audio `x` at sample rate `sr` sung on `voice_type`'s arpeggio."""
    times, f0 = track_pitch(x, sr)
    return segment_pitch(times, f0, voice_type)


def segment_file(path: str | Path, voice_type: str) -> Segmentation:
    x, sr = load_mono(path)
    return segment(x, sr, voice_type)
