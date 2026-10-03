"""Optimization history chart from docs/e2e_results.json -> docs/optimization.png

uv run --no-project --with matplotlib python bench/chart.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOCS = Path(__file__).resolve().parents[1] / "docs"
R = json.loads((DOCS / "e2e_results.json").read_text())

# (label, x real time, fidelity dB or None). First row: the user's first run (audio-separator CLI, fp32, RAM busy).
STEPS = [("Start: audio-separator fp32\n(first run, RAM busy)", 180 / 232, None)]
NAMES = [
    ("ref32", "Same model, fp32, RAM free"),
    ("ref16", "fp16"),
    ("patch", "+ RoPE tables, fused gates, fused norm"),
    ("gpu", "+ custom Metal kernels (GPU)"),
    ("ane_g1", "+ transformer on the Neural Engine"),
    ("ane", "+ attention tiled for the Neural Engine"),
    ("ane2", "2 workers overlapping GPU and ANE (no gain)"),
]
for key, label in NAMES:
    if key in R:
        STEPS.append((label, R[key]["x_real_time"], None if key == "ref32" else R[key]["fidelity_db"]))

BLUE, GRAY, INK, INK2, SURF = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(9, 0.62 * len(STEPS) + 1.6), facecolor=SURF)
ax.set_facecolor(SURF)
ys = range(len(STEPS))[::-1]
best = max(v for _, v, _ in STEPS)
for y, (label, v, fid) in zip(ys, STEPS):
    ax.barh(y, v, height=0.62, color=BLUE if v == best else GRAY, edgecolor=SURF, linewidth=2)
    mins = 45 / v
    note = f"{v:.2f}x   45 min episode in {mins:.0f} min" + (f"   match {fid:.0f} dB" if fid else "")
    ax.text(v + best * 0.015, y, note, va="center", fontsize=9, color=INK2, bbox=dict(facecolor=SURF, edgecolor="none", pad=1), zorder=3)
ax.axvline(1.0, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
ax.text(1.0, len(STEPS) - 0.35, " real time", color=INK2, fontsize=8, va="bottom")
ax.set_yticks(list(ys), [s[0] for s in STEPS], fontsize=9, color=INK)
ax.set_xlim(0, best * 2.15)
ax.set_xlabel("Speed: seconds of audio processed per second (higher is better)", color=INK2, fontsize=9)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRAY)
ax.tick_params(axis="y", length=0)
ax.tick_params(axis="x", colors=INK2, labelsize=8)
ax.set_title("Kim Mel-RoFormer on an 8 GB M3 MacBook Air: 3 min TV clip, end to end",
             loc="left", fontsize=11, color=INK, pad=12)
fig.text(0.01, 0.01, "Same weights throughout. 'match' = SDR of the vocals vs the fp32 output (above ~40 dB is inaudible).",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(DOCS / "optimization.png", dpi=160, facecolor=SURF)
print("wrote", DOCS / "optimization.png")
