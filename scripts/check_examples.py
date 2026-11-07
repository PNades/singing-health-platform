"""Check the example arpeggios in api/static/audio/ against their target pitches.

For each of tenor/baritone/alto/soprano.wav: segment the take with
ml/features/segment.py (the same note splitter the feature pipeline uses,
spec Section 6.1) and report each note's steady-core median F0 error in cents
against its target. Fails if any of the 7 notes is missing or off by more
than 20 cents (spec 2.2).

Exit code 0 if every file passes, 1 otherwise.

    uv run python scripts/check_examples.py [files...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.arpeggios import ARPEGGIOS, N_NOTES, target_hz  # noqa: E402
from ml.features.segment import load_mono, segment  # noqa: E402

OUT_DIR = ROOT / "api" / "static" / "audio"
MAX_CENTS = 20.0


def check_file(path: Path, voice: str) -> bool:
    notes = ARPEGGIOS[voice]
    targets = target_hz(voice)
    x, sr = load_mono(path)
    seg = segment(x, sr, voice)

    tempo = f"{seg.tempo_bpm:.1f} BPM" if seg.tempo_bpm else "tempo n/a"
    key = f"key {seg.key_offset_st:+.2f} st" if seg.key_offset_st is not None else "no notes"
    print(f"\n{path.name}  ({voice}, {sr} Hz, {len(x) / sr:.2f} s, {tempo}, {key})")
    expected = list(range(1, N_NOTES + 1))
    if seg.indices != expected:
        print(f"  FAIL  found notes {seg.indices}, expected {expected}")
        return False

    ok = True
    print(f"  {'#':>2}  {'note':<4}  {'start':>6}  {'end':>6}  {'target':>8}  {'measured':>8}  {'cents':>7}")
    for name, target, n in zip(notes, targets, seg.notes):
        cents = 1200 * np.log2(n.f0_hz / target)
        bad = abs(cents) > MAX_CENTS
        ok &= not bad
        print(
            f"  {n.index:>2}  {name:<4}  {n.start:6.2f}  {n.end:6.2f}  "
            f"{target:8.2f}  {n.f0_hz:8.2f}  {cents:+7.2f}  {'FAIL' if bad else 'ok'}"
        )
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or [OUT_DIR / f"{v}.wav" for v in ARPEGGIOS]
    all_ok = True
    for p in paths:
        voice = p.stem
        if voice not in ARPEGGIOS:
            print(f"\n{p}: file name must be one of {', '.join(ARPEGGIOS)}.wav")
            all_ok = False
            continue
        if not p.exists():
            print(f"\n{p}: missing")
            all_ok = False
            continue
        all_ok &= check_file(p, voice)
    print(f"\n{'All examples within' if all_ok else 'Some notes missing or off by more than'} +/-{MAX_CENTS:.0f} cents.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
