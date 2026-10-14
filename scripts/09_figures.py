"""Step 9: figures for the report -> outputs/figures/*.png (reads router_results.csv, router_by_budget.csv)

  python scripts/09_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import OUTPUTS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HILITE, MUTED = "#2a78d6", "#a9a8a2"
ACTION_COLORS = {"accept_small": "#2a78d6", "escalate_effnet": "#eb6834", "escalate_resnet18": "#1baf7a",
                 "defer": "#eda100", "human_review": "#e87ba4"}
OURS = "ResQRoute controller (ours)"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def short(p):
    return p.split(" (t=")[0].replace(" (MobileNetV3-S)", "")


def cost_vs_quality(res, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for r in res.itertuples():
        ours = r.policy == OURS
        ax.scatter(r.avg_time_ms, r.macro_F1, s=90 if ours else 60, color=HILITE if ours else MUTED,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(short(r.policy), (r.avg_time_ms, r.macro_F1), xytext=(8, -3 if not ours else 6),
                    textcoords="offset points", fontsize=9, color=INK if ours else INK2,
                    fontweight="bold" if ours else "normal")
    ax.set_xlabel("Average CPU time per image (ms, lower is better)")
    ax.set_ylabel("Macro-F1 on test (higher is better)")
    ax.set_title("Accuracy vs computation: the controller is top-left", loc="left", fontsize=12)
    ax.set_xlim(0, res.avg_time_ms.max() * 1.45)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def safety(res, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    r = res.iloc[::-1]
    colors = [HILITE if p == OURS else MUTED for p in r.policy]
    ax.barh([short(p) for p in r.policy], r["dangerous_FN_%"], color=colors, height=0.55, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(r["dangerous_FN_%"]):
        ax.text(v + 0.15, y, f"{v:.1f}%", va="center", fontsize=9, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Severe buildings predicted as 'no-damage' (%, lower is safer)")
    ax.set_title("Dangerous false negatives on test", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def actions_by_budget(bb, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    acts = [a for a in ACTION_COLORS if a in bb.columns]
    labels = {60: "tight (60 ms)", 180: "normal (180 ms)", 400: "relaxed (400 ms)"}
    ys = [labels.get(b, str(b)) for b in bb.index]
    left = [0.0] * len(bb)
    for a in acts:
        vals = bb[a].tolist()
        ax.barh(ys, vals, left=left, color=ACTION_COLORS[a], height=0.55, edgecolor=SURFACE, linewidth=2, label=a)
        for i, v in enumerate(vals):
            if v >= 4:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=INK)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of routing decisions (%)")
    ax.set_title("The controller adapts to the device's latency budget", loc="left", fontsize=12)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    figs = OUTPUTS / "figures"
    figs.mkdir(exist_ok=True)
    res = pd.read_csv(OUTPUTS / "router_results.csv")
    bb = pd.read_csv(OUTPUTS / "router_by_budget.csv", index_col=0)
    cost_vs_quality(res, figs / "cost_vs_quality.png")
    safety(res, figs / "dangerous_fn.png")
    actions_by_budget(bb, figs / "actions_by_budget.png")
    print("wrote", sorted(p.name for p in figs.iterdir()))


if __name__ == "__main__":
    main()
