"""
replot_with_uncertainty.py
=============================================================================
Re-plot the headline A-vs-B figures with the EMPIRICALLY MEASURED uncertainty
band from the wind-tunnel validation (e387_neuralfoil_validation.py), so no
performance number is shown without its error bar.

Error model (data/error_model.json), all measured against real wind-tunnel data:
  - NeuralFoil L/D error vs experiment = ~16% (CL ~7%, CD ~15%) at the
    confidence range where it was validated (analysis_confidence 0.70-0.92).
  - This is a FLOOR, not a symmetric estimate: NeuralFoil under-predicts
    laminar-separation-bubble drag at low Re, so it OVER-predicts L/D. True L/D
    therefore trends toward the LOWER edge of the band.
  - Airfoils A and B operate at analysis_confidence BELOW the validated range
    (A ~ 0.00, B ~ 0.02-0.37). Their true error is therefore >= the measured
    band. Airfoil A in particular is disclaimed by NeuralFoil itself
    (confidence ~ 0) and is drawn with a hatched "unvalidated" band.

Outputs:
  figures/18_LD_vs_AoA_uncertainty.png
  figures/19_tradeoff_uncertainty.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
EM  = json.load(open(os.path.join(OUT, "data", "error_model.json")))
BAND = EM["LD_band_pct"] / 100.0
CONF_LO = EM["validated_conf_range"][0]

grid = pd.read_csv(os.path.join(OUT, "data", "grid_evaluation.csv"))
trade = pd.read_csv(os.path.join(OUT, "data", "tradeoff_family.csv"))

A = "A_single_point"
B = "B_robust"
COL = {A: "#1f77b4", B: "#d62728"}
LABEL = {A: "A (single-point)", B: "B (robust)"}

# Nearest available grid Re to the canonical 50k / 200k / 500k targets.
_RE_AVAIL = sorted(grid.Re.unique())
PLOT_RE = [min(_RE_AVAIL, key=lambda r: abs(r - t)) for t in (50e3, 200e3, 500e3)]


def band_is_floor(conf_median):
    """A point whose confidence is below the validated range gets a hatched
    'lower-bound only' band; otherwise a solid measured band."""
    return conf_median < CONF_LO


# ─────────────────────────────────────────────────────────────────────────────
# Fig 18: L/D vs AoA at three Re, with measured uncertainty band
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, Re in zip(axes, PLOT_RE):
    for name in (B, A):  # plot B first so A (low conf) sits on top, visibly hatched
        s = grid[(grid.airfoil == name) & (grid.Re == Re)].sort_values("AoA")
        if s.empty:
            continue
        ld = s.L_over_D.to_numpy()
        aoa = s.AoA.to_numpy()
        conf_med = s.analysis_confidence.median()
        floor = band_is_floor(conf_med)
        tag = "below validated conf" if floor else "validated band"
        ax.plot(aoa, ld, "-o", color=COL[name], lw=1.9, ms=4,
                label=f"{LABEL[name]}  (conf≈{conf_med:.2f}, {tag})")
        # one-sided emphasis: band extends mostly downward (true L/D <= predicted)
        lo = ld * (1 - BAND)
        hi = ld * (1 + BAND * 0.4)   # smaller upward (bias is one-sided)
        if floor:
            ax.fill_between(aoa, lo, hi, facecolor="none", hatch="////",
                            edgecolor=COL[name], linewidth=0.0, alpha=0.55, zorder=1)
        else:
            ax.fill_between(aoa, lo, hi, color=COL[name], alpha=0.18,
                            linewidth=0, zorder=1)
    ax.set_title(f"Re = {Re/1e3:.0f}k")
    ax.set_xlabel("AoA [deg]"); ax.grid(alpha=0.25)
axes[0].set_ylabel("L / D")
axes[0].legend(fontsize=8.5, framealpha=0.9, loc="upper left")
fig.suptitle(
    "Airfoil A vs B: L/D with measured uncertainty band (±16% from wind-tunnel validation)\n"
    "Shaded = validated band; hatched = NeuralFoil confidence below validated range "
    "(band is a LOWER bound, true L/D trends to lower edge)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "18_LD_vs_AoA_uncertainty.png"), dpi=160)
print("wrote figures/18_LD_vs_AoA_uncertainty.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 19: peak vs worst-case tradeoff, with error bars on both axes
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 6.5))
peak = trade.peak_LD.to_numpy()
worst = trade.worst_case_LD.to_numpy()
lam = trade.lam.to_numpy()

# Asymmetric error bars: mostly downward (true values <= predicted at low Re)
def asym_err(vals):
    return np.vstack([vals * BAND, vals * BAND * 0.4])   # [down, up]

ax.errorbar(peak, worst,
            xerr=asym_err(peak), yerr=asym_err(worst),
            fmt="none", ecolor="gray", elinewidth=1, capsize=3, alpha=0.7,
            zorder=1, label="±16% measured band (one-sided)")
sc = ax.scatter(peak, worst, c=lam, cmap="viridis", s=90, zorder=3,
                edgecolors="k", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="λ  (1 = peak-seeking A, 0 = robust B)")
for x, y, l in zip(peak, worst, lam):
    ax.annotate(f"λ={l:.1f}", (x, y), fontsize=7.5,
                xytext=(5, 4), textcoords="offset points")
# Mark A and B
ax.scatter([peak[lam == 1.0][0]], [worst[lam == 1.0][0]], s=240,
           facecolors="none", edgecolors=COL[A], linewidths=2.2, zorder=4,
           label="A (conf≈0, unvalidated)")
ax.scatter([peak[lam == 0.0][0]], [worst[lam == 0.0][0]], s=240,
           facecolors="none", edgecolors=COL[B], linewidths=2.2, zorder=4,
           label="B (robust)")
ax.set_xlabel("Peak L/D  (surrogate, optimistic ceiling)")
ax.set_ylabel("Worst-case L/D over envelope")
ax.set_title("Peak-vs-robustness tradeoff with measured uncertainty\n"
             "A's peak (~233) sits where NeuralFoil reports ~zero confidence; "
             "treat as an upper bound, not a value", fontsize=10)
ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="center right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "19_tradeoff_uncertainty.png"), dpi=160)
print("wrote figures/19_tradeoff_uncertainty.png")

print("\nDone. Both headline figures now carry the empirically measured band.")
print(f"  band = ±{BAND*100:.0f}% (one-sided, true L/D <= predicted at low Re)")
print(f"  A confidence ~0 -> drawn as unvalidated lower bound")
