"""
Figure 39: every correction tried on NeuralFoil's drag, and what the best one leaves behind.

Left : change in mean drag error (percentage points, below zero is better) for each correction in
       plan/improve-neuralfoil.md, in the three splits of the test bench, with 95% intervals from
       resampling whole airfoils.
Right: what the error is made of after correction A (out of fold): an offset per airfoil, an offset
       per polar, and scatter inside a polar.

Inputs : data/drag_correction_bench.csv, drag_correction_gated_bench.csv, drag_correction_E_bench.csv,
         bl_state_bench.csv, residual_scatter.csv
Output : figures/39_drag_correction.png
Run    : python benchmark/drag_correction_figure.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.7, "grid.alpha": 0.7,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.axisbelow": True,
})

SPLITS = [("UIUC->Princeton", "Learn on UIUC, score Princeton", "#2166ac"),
          ("Princeton->UIUC", "Learn on Princeton, score UIUC", "#e08214"),
          ("airfoil-CV", "10-fold, whole airfoils held out", "#444444")]
ROWS = [("correction_A", "A  speed and lift"),
        ("correction_B", "B  A + camber, thickness"),
        ("correction_C", "C  B + NeuralFoil's confidence"),
        ("correction_A_Re<=150k", "A, below Re 150,000 only"),
        ("correction_E", "E  more flexible speed and lift"),
        ("correction_F", "F  A + boundary-layer state"),
        ("correction_G", "G  F + how fast transition moves")]


def collect():
    parts = [pd.read_csv(os.path.join(DATA, f)) for f in
             ["drag_correction_bench.csv", "drag_correction_gated_bench.csv",
              "drag_correction_E_bench.csv", "bl_state_bench.csv"]]
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["candidate", "split"])
    return d.set_index(["candidate", "split"])


def main():
    d = collect()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [2.3, 1]})
    for i, (cand, label) in enumerate(ROWS):
        for j, (split, _, color) in enumerate(SPLITS):
            r = d.loc[(cand, split)]
            y = i + (j - 1) * 0.24
            ax.plot([r.vs_nf_ci_lo * 100, r.vs_nf_ci_hi * 100], [y, y], color=color, lw=1.8, alpha=0.8, solid_capstyle="butt")
            ax.plot(r.vs_neuralfoil * 100, y, "o", color=color, ms=6.5, zorder=3)
    ax.axvline(0, color="#222222", lw=1.2)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS])
    ax.invert_yaxis()
    ax.set_xlim(-2.3, 1.0)
    ax.set_xlabel("Change in mean drag error, percentage points (below zero is better)")
    ax.set_title("Seven corrections, three tests each", fontsize=11)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.8, label=lab) for _, lab, c in SPLITS]
    ax.legend(handles=handles, loc="lower right", fontsize=8.8)

    t = pd.read_csv(os.path.join(DATA, "residual_scatter.csv"))
    t = t[t.analysis == "1 share of residual sum of squares"].set_index("item").value * 100
    parts = [("between_airfoil", "An offset per airfoil", "#2166ac"),
             ("between_polar", "An offset per polar", "#9ecae1"),
             ("within_polar", "Scatter inside a polar", "#b2182b")]
    bottom = 0
    for key, label, color in parts:
        bx.bar(0, t[key], bottom=bottom, color=color, width=0.55)
        bx.text(0, bottom + t[key] / 2, f"{label}\n{t[key]:.0f}%", ha="center", va="center", fontsize=9.5,
                color="white" if key != "between_polar" else "#222222", fontweight="bold")
        bottom += t[key]
    bx.set_xlim(-0.5, 0.5)
    bx.set_ylim(0, 100)
    bx.set_xticks([])
    bx.set_ylabel("Share of squared error left after A (%)")
    bx.set_title("What is left", fontsize=11)
    bx.grid(False)
    for s in ("top", "right", "bottom"):
        bx.spines[s].set_visible(False)

    fig.suptitle("A smooth correction is worth about 1.2 points, and nothing tried since has added a reliable second one",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "39_drag_correction.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
