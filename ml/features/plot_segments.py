"""Diagnostic plot of a segmentation, for checking note boundaries by eye.

    from ml.features.plot_segments import plot_segmentation
    plot_segmentation(seg, x, sr, "reports/segments/take.png", title="take")

Top panel: the pitch track in semitones above the target root, the expected
note levels at the measured key, each whole note (light band) and its steady
core (blue band) with the core's median F0 (orange line), thin lines at note
boundaries, and red crosses where a tracker octave error was repaired (at the
pitch it was tracked at). Bottom panel: the waveform envelope with the same
bands. Reports only; never import from api/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ml.arpeggios import INTERVALS_ST, root_hz  # noqa: E402
from ml.features.segment import Segmentation  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e6e5e0"
NOTE_BAND = "#e9e8e3"
CORE = "#2a78d6"
MEDIAN = "#eb6834"
TRUTH = "#6250d6"
REPAIRED = "#e34948"
DEGREE_NAMES = {0: "root", 4: "3rd", 7: "5th", 12: "octave"}


def plot_segmentation(
    seg: Segmentation,
    x: np.ndarray,
    sr: int,
    path: str | Path,
    title: str = "",
    truth: list[tuple[float, float]] | None = None,
) -> Path:
    """Save the plot to `path` (PNG). `truth`: optional true steady spans per note, drawn as ticks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = root_hz(seg.voice_type)
    voiced = seg.f0_hz > 0
    st = 12 * np.log2(seg.f0_hz[voiced] / root)
    t_end = len(x) / sr

    fig, (ax, ax_w) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURFACE
    )
    for a in (ax, ax_w):
        a.set_facecolor(SURFACE)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(MUTED)
        a.tick_params(colors=INK_2, labelsize=9)

    key = seg.key_offset_st or 0.0
    for level in sorted(set(INTERVALS_ST)):
        ax.axhline(level + key, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(t_end, level + key, f" {DEGREE_NAMES[level]}", color=INK_2, fontsize=8, va="center")

    for n in seg.notes:
        for a in (ax, ax_w):
            a.axvspan(n.note_start, n.note_end, color=NOTE_BAND, lw=0, zorder=0)
            a.axvspan(n.start, n.end, color=CORE, alpha=0.18, lw=0, zorder=0)
        ax.hlines(n.f0_st, n.start, n.end, color=MEDIAN, lw=2, zorder=4)
        for edge in (n.note_start, n.note_end):
            for a in (ax, ax_w):
                a.axvline(edge, color=INK_2, lw=0.6, alpha=0.5, zorder=1)
        cents = 100 * (n.f0_st - INTERVALS_ST[n.index - 1] - key)
        ax.annotate(
            f"{n.index}\n{n.core_s * 1000:.0f} ms\n{cents:+.0f} c",
            (0.5 * (n.start + n.end), n.f0_st),
            xytext=(0, 10), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=INK,
        )
    if truth:
        for a0, a1 in truth:
            ax_w.hlines(-1.08, a0, a1, color=TRUTH, lw=3)

    ax.scatter(seg.times[voiced], st, s=4, color=INK_2, zorder=3, linewidths=0)
    repaired = voiced & (np.abs(seg.f0_raw_hz - seg.f0_hz) > 1e-9)
    raw_st = 12 * np.log2(seg.f0_raw_hz[repaired] / root)
    if repaired.any():
        ax.scatter(seg.times[repaired], raw_st, s=14, marker="x", color=REPAIRED, zorder=5, linewidths=1)
    all_st = np.concatenate([st, raw_st])
    lo = min(all_st.min() if all_st.size else 0, key) - 2
    hi = max(all_st.max() if all_st.size else 12, key + 12) + 4
    ax.set_ylim(lo, hi)
    ax.set_ylabel("semitones above target root", color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)

    hop = max(1, sr // 200)  # 5 ms envelope
    n_hops = len(x) // hop
    env = np.abs(x[: n_hops * hop]).reshape(n_hops, hop).max(axis=1)
    tt = (np.arange(n_hops) + 0.5) * hop / sr
    ax_w.fill_between(tt, -env, env, color=INK_2, lw=0)
    ax_w.set_ylim(-1.15, 1.05)
    ax_w.set_yticks([])
    ax_w.set_xlabel("time (s)", color=INK_2, fontsize=9)
    ax_w.set_xlim(0, t_end)

    tempo = f"{seg.tempo_bpm:.1f} BPM" if seg.tempo_bpm else "tempo n/a"
    key_s = f"key {seg.key_offset_st:+.2f} st" if seg.key_offset_st is not None else "no notes"
    ax.set_title(
        f"{title}    {seg.voice_type} · {len(seg.notes)} notes {seg.indices} · {tempo} · {key_s}",
        loc="left", color=INK, fontsize=11,
    )
    handles = [
        plt.Line2D([], [], color=INK_2, marker="o", ls="", ms=3, label="pitch frames"),
        Patch(color=NOTE_BAND, label="whole note"),
        Patch(color=CORE, alpha=0.3, label="steady core"),
        plt.Line2D([], [], color=MEDIAN, lw=2, label="core median F0"),
        plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3)), label="expected levels at measured key"),
    ]
    if repaired.any():
        handles.append(plt.Line2D([], [], color=REPAIRED, marker="x", ls="", ms=5, label="octave error, as tracked (repaired)"))
    if truth:
        handles.append(plt.Line2D([], [], color=TRUTH, lw=3, label="true steady span (generated)"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path
