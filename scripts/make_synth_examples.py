"""Generate the synthesized example arpeggios (spec Section 2.2).

Writes api/static/audio/{tenor,baritone,alto,soprano}.wav: a synthesized
1-3-5-8-5-3-1 major arpeggio per voice type at the exact pitches and tempo in
spec Section 2.1 (90 BPM, one beat per note, top note held 2 beats), with
0.5 s of silence before and after. Mono, 16-bit PCM, 44.1 kHz, peak -6 dBFS.

The tone is a soft additive harmonic tone (falling harmonic amplitudes, a
gentle brightness decay over each note), not a pure sine. No vibrato and no
glides, so every note sits exactly on its target pitch.

These are the examples for all of data collection; don't change them once
collection starts. Sung examples for the app come later (Step 8a) and live in
api/static/audio/sung/.

    uv run python scripts/make_synth_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.arpeggios import ARPEGGIOS, BEATS, BPM, note_to_hz  # noqa: E402  (spec 2.1)

OUT_DIR = ROOT / "api" / "static" / "audio"

SILENCE_S = 0.5

SAMPLE_RATE = 44_100
PEAK_DBFS = -6.0

# Tone shape
ATTACK_S = 0.03
RELEASE_S = 0.05
MAX_HARMONIC_HZ = 10_000.0
HARMONIC_ROLLOFF = 1.6  # amplitude of harmonic n is n**-ROLLOFF


def synth_note(f0: float, dur_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """One note of a soft harmonic tone with a smooth attack and release."""
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    tone = np.zeros(n)
    for k in range(1, int(MAX_HARMONIC_HZ // f0) + 1):
        # Upper harmonics fade a little over the note (softer, less buzzy).
        brightness = np.exp(-t * 0.6 * (k - 1))
        tone += (k ** -HARMONIC_ROLLOFF) * brightness * np.sin(2 * np.pi * k * f0 * t)

    env = np.ones(n)
    a, r = int(ATTACK_S * sr), int(RELEASE_S * sr)
    env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    env[-r:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
    env *= 1.0 - 0.15 * (t / dur_s)  # slight overall decay
    return tone * env


def synth_arpeggio(notes: list[str], sr: int = SAMPLE_RATE) -> np.ndarray:
    beat_s = 60.0 / BPM
    silence = np.zeros(int(round(SILENCE_S * sr)))
    body = np.concatenate(
        [synth_note(note_to_hz(n), b * beat_s, sr) for n, b in zip(notes, BEATS)]
    )
    body *= 10 ** (PEAK_DBFS / 20) / np.max(np.abs(body))
    return np.concatenate([silence, body, silence])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for voice, notes in ARPEGGIOS.items():
        audio = synth_arpeggio(notes)
        pcm = np.round(audio * 32767).astype(np.int16)
        path = OUT_DIR / f"{voice}.wav"
        wavfile.write(path, SAMPLE_RATE, pcm)
        print(f"{path.relative_to(ROOT)}  {len(audio) / SAMPLE_RATE:.2f} s  {' '.join(notes)}")


if __name__ == "__main__":
    main()
