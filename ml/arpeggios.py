"""Arpeggio definitions (spec Section 2.1). The one home for these values.

Every voice type sings a major arpeggio 1-3-5-8-5-3-1 on "ah", one beat per
note except the top note (note 4), which is held 2 beats: 8 beats in all, at
90 BPM in the examples and the count-in. Basses use the baritone arpeggio.

Notes are indexed 1-7 in sung order everywhere outside this module's lists.
"""

from __future__ import annotations

import math

VOICE_TYPES: tuple[str, ...] = ("tenor", "baritone", "alto", "soprano")

# Note names in sung order. List position 3 (note 4) is the held top note.
ARPEGGIOS: dict[str, list[str]] = {
    "baritone": ["F3", "A3", "C4", "F4", "C4", "A3", "F3"],
    "tenor": ["Ab3", "C4", "Eb4", "Ab4", "Eb4", "C4", "Ab3"],
    "alto": ["F4", "A4", "C5", "F5", "C5", "A4", "F4"],
    "soprano": ["Ab4", "C5", "Eb5", "Ab5", "Eb5", "C5", "Ab4"],
}
BEATS: list[int] = [1, 1, 1, 2, 1, 1, 1]  # top note held 2 beats; 8 beats in all
INTERVALS_ST: list[int] = [0, 4, 7, 12, 7, 4, 0]  # semitones above the root
N_NOTES = 7
TOP_NOTE = 4  # 1-based index of the held top note
BPM = 90.0

_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_hz(name: str) -> float:
    """Equal-tempered frequency, A4 = 440 Hz. Accepts e.g. 'F3', 'Ab4', 'C#5'."""
    letter, rest = name[0], name[1:]
    accidental = 0
    while rest and rest[0] in "b#":
        accidental += -1 if rest[0] == "b" else 1
        rest = rest[1:]
    midi = 12 * (int(rest) + 1) + _PITCH_CLASS[letter] + accidental
    return 440.0 * 2.0 ** ((midi - 69) / 12)


def target_hz(voice_type: str) -> list[float]:
    """Target frequency of each of the 7 notes for a voice type."""
    return [note_to_hz(n) for n in ARPEGGIOS[voice_type]]


def root_hz(voice_type: str) -> float:
    """Frequency of the arpeggio's root (note 1)."""
    return note_to_hz(ARPEGGIOS[voice_type][0])


def hz_to_st(hz: float, ref_hz: float) -> float:
    """Semitones of `hz` above `ref_hz`."""
    return 12.0 * math.log2(hz / ref_hz)


def arpeggio_id(voice_type: str) -> str:
    """Spec 2.6 arpeggio_id, e.g. 'alto_F4'."""
    return f"{voice_type}_{ARPEGGIOS[voice_type][0]}"


# Sanity: the note names and the interval pattern must agree.
for _v, _names in ARPEGGIOS.items():
    _r = note_to_hz(_names[0])
    assert [round(hz_to_st(note_to_hz(n), _r)) for n in _names] == INTERVALS_ST, _v
assert len(BEATS) == N_NOTES and sum(BEATS) == 8
