"""Step 11 (Person C): figures for the evaluation -> outputs/figures/eval_*.png

  python scripts/11_eval_figures.py     (reads outputs/eval_*.csv written by 10_evaluate.py)
"""
import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUTPUTS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HILITE, MUTED, WARN = "#2a78d6", "#a9a8a2", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
FIG = OUTPUTS / "figures"
OURS = ["5 Controller (no queue)", "6 Controller + priority queue"]


def label(s):
    return s[2:].replace(" (ResNet-18)", "")


def systems(out):
    r = pd.read_csv(OUTPUTS / "eval_summary.csv", index_col=0)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for a, col, title, ci in [(ax[0], "dangerous_FN_%", "Dangerous false negatives (%), lower is better", "dangerous_FN_CI95"),
                              (ax[1], "macro_F1", "Macro-F1, higher is better", "macro_F1_CI95"),
                              (ax[2], "lat_mean_ms", "Mean immediate latency (ms), lower is better", None)]:
        y = np.arange(len(r))[::-1]
        a.barh(y, r[col], color=[HILITE if s in OURS else MUTED for s in r.index], height=0.65, zorder=3)
        if ci:
            lo_hi = np.array([ast.literal_eval(v) for v in r[ci]])
            a.errorbar(r[col], y, xerr=[r[col] - lo_hi[:, 0], lo_hi[:, 1] - r[col]], fmt="none", ecolor=INK2, capsize=3, zorder=4)
        a.set_yticks(y, [label(s) for s in r.index] if a is ax[0] else [""] * len(r))
        a.set_title(title, fontsize=10, loc="left", color=INK)
        a.grid(True, axis="x", color=GRID, zorder=0)
    ax[1].set_xlim(0.55, 0.67)
    fig.tight_layout()
    fig.savefig(out / "eval_systems.png", dpi=170)
    plt.close(fig)


def scenarios(out):
    b = pd.read_csv(OUTPUTS / "eval_by_scenario.csv")
    order = ["normal", "tight_latency", "low_memory", "low_battery", "degraded_network", "disaster_surge"]
    keep = {"1 Small only": MUTED, "3 Fixed-confidence cascade": "#d8a93b", "4 Resource-aware, no risk": "#1baf7a",
            "6 Controller + priority queue": HILITE}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    w = 0.2
    for a, col, ttl in [(ax[0], "dangerous_FN_%", "Dangerous false negatives (%) by scenario"),
                        (ax[1], "violations_%", "Resource violations (%) by scenario")]:
        for i, (s, c) in enumerate(keep.items()):
            v = b[b.system == s].set_index("scenario").loc[order, col]
            a.bar(np.arange(len(order)) + (i - 1.5) * w, v, w, color=c, label=label(s), zorder=3)
        a.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order], fontsize=8.5)
        a.set_title(ttl, fontsize=10, loc="left", color=INK)
        a.grid(True, axis="y", color=GRID, zorder=0)
    ax[0].set_ylim(0, 9.5)
    ax[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "eval_scenarios.png", dpi=170)
    plt.close(fig)


def queue(out):
    q = pd.read_csv(OUTPUTS / "eval_queue.csv")
    cols = {"priority": HILITE, "fifo": MUTED, "random": "#d8a93b"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for pol, c in cols.items():
        d = q[q.policy == pol]
        ax[0].plot(d.rate, d["severe_resolved_%"], color=c, marker="o", label=pol, linewidth=2 if pol == "priority" else 1.5)
        ax[1].plot(d.rate, d["severe_median_wait"], color=c, marker="o", label=pol, linewidth=2 if pol == "priority" else 1.5)
    ax[0].set_ylabel("truly-severe deferred cases resolved (%)")
    ax[1].set_ylabel("median wait for severe cases (arrivals)")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("spare capacity (deferred items finished per arriving case)")
        a.grid(True, color=GRID)
    ax[0].legend(frameon=False, title="queue order")
    fig.tight_layout()
    fig.savefig(out / "eval_queue.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    systems(FIG), scenarios(FIG), queue(FIG)
    print("wrote", *sorted(p.name for p in FIG.glob("eval_*.png")))
