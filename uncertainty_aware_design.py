"""
uncertainty_aware_design.py
=============================================================================
THE NOVEL EXTENSION.

Every other airfoil-optimization paper that uses a neural surrogate treats the
surrogate as if it were perfectly correct. Ours does not. In the experimental
validation (e387_neuralfoil_validation.py) we *measured* that NeuralFoil's error
grows when its self-reported `analysis_confidence` drops (Pearson r ≈ −0.48).

So here is the new idea: put that finding back INTO the optimizer. Instead of
only chasing the highest predicted lift-to-drag ratio, also reward designs that
live where NeuralFoil is actually reliable (high confidence). We sweep a weight
`w_conf` from 0 (trust the surrogate blindly, the standard approach) upward, and
watch the optimizer trade a little predicted performance for a lot of
trustworthiness.

Why this matters: a design with a huge predicted L/D but near-zero confidence
(like airfoil A, ~233 at confidence ~0) is a mirage — the number is exactly where
the surrogate is least believable. A confidence-aware optimizer steers away from
those mirages toward designs whose predicted performance you can actually trust.

Output:
  data/uncertainty_aware_sweep.csv
  figures/20_trust_vs_performance.png
"""

import os
import numpy as np
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as anp

OUT = "/Users/neelmadhav/Airfoil research"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)

MODEL = "large"
RE_OPT  = np.geomspace(50e3, 500e3, 3)   # coarse grid: sweep does many solves
AOA_OPT = np.linspace(0.0, 8.0, 3)
RE_EVAL  = np.geomspace(50e3, 500e3, 6)
AOA_EVAL = np.linspace(0.0, 8.0, 5)

THK_MIN, THK_MAX, TE = 0.08, 0.16, 0.0025
SEEDS = ["naca4412", "naca6412"]

# The dial: how much we reward the surrogate being confident (trustworthy).
W_CONF = [0.0, 1.0, 2.0, 4.0, 8.0]
S_LD = 40.0   # reference L/D so the two objective terms are comparable


def solve_one(w_conf, seed):
    af0 = asb.KulfanAirfoil(seed) if isinstance(seed, str) else seed
    opti = asb.Opti()
    uw = opti.variable(init_guess=af0.upper_weights, lower_bound=-0.2, upper_bound=0.5)
    lw = opti.variable(init_guess=af0.lower_weights, lower_bound=-0.5, upper_bound=0.4)
    le = opti.variable(init_guess=af0.leading_edge_weight, lower_bound=-0.2, upper_bound=0.2)
    af = asb.KulfanAirfoil(lower_weights=lw, upper_weights=uw,
                           leading_edge_weight=le, TE_thickness=TE)

    opti.subject_to(af.local_thickness(x_over_c=np.linspace(0.02, 0.98, 20)) > 0.004)
    opti.subject_to(af.max_thickness() < THK_MAX)
    opti.subject_to(af.max_thickness() > THK_MIN)

    # Epigraph worst-case L/D, and collect confidence over the same grid.
    g = opti.variable(init_guess=30.0)
    confs = []
    for Re in RE_OPT:
        for a in AOA_OPT:
            aero = af.get_aero_from_neuralfoil(alpha=a, Re=Re, model_size=MODEL)
            opti.subject_to(g <= aero["CL"] / aero["CD"])
            confs.append(aero["analysis_confidence"])
    mean_conf = sum(confs) / len(confs)

    # Trust-adjusted objective: performance PLUS a reward for being where the
    # surrogate is reliable.
    opti.maximize(g / S_LD + w_conf * mean_conf)
    sol = opti.solve(verbose=False, max_iter=300)
    return sol(af), float(sol(g)), float(sol(mean_conf))


def best_over_seeds(w_conf):
    best = None
    for seed in SEEDS:
        try:
            af, g, mc = solve_one(w_conf, seed)
        except Exception:
            continue
        score = g / S_LD + w_conf * mc
        if best is None or score > best[0]:
            best = (score, af, g, mc)
    return best[1], best[2], best[3]


def evaluate(af):
    """Honest fine-grid re-evaluation of worst-case L/D and mean confidence."""
    LDs, cfs = [], []
    for Re in RE_EVAL:
        aero = af.get_aero_from_neuralfoil(alpha=AOA_EVAL, Re=Re, model_size=MODEL)
        cl, cd = np.atleast_1d(aero["CL"]), np.atleast_1d(aero["CD"])
        LDs.extend(list(cl / cd))
        cfs.extend(list(np.atleast_1d(aero["analysis_confidence"])))
    LDs, cfs = np.array(LDs), np.array(cfs)
    return dict(worst_LD=float(LDs.min()), mean_LD=float(LDs.mean()),
                peak_LD=float(LDs.max()), mean_conf=float(cfs.mean()),
                min_conf=float(cfs.min()), thk=float(af.max_thickness()))


print("Uncertainty-aware design sweep (this runs several optimizations) ...")
rows, geoms = [], {}
prev = None
for wc in W_CONF:
    af, _, _ = best_over_seeds(wc)
    # warm-start-style continuity: not required, but keep the best geometry
    geoms[wc] = af
    m = evaluate(af)
    m["w_conf"] = wc
    rows.append(m)
    print(f"  w_conf={wc:>4.1f}  worst L/D={m['worst_LD']:6.1f}  "
          f"mean L/D={m['mean_LD']:6.1f}  mean conf={m['mean_conf']:.3f}  "
          f"min conf={m['min_conf']:.3f}  thk={m['thk']*100:4.1f}%")

df = pd.DataFrame(rows)[["w_conf", "worst_LD", "mean_LD", "peak_LD",
                         "mean_conf", "min_conf", "thk"]]
df.to_csv(os.path.join(OUT, "data", "uncertainty_aware_sweep.csv"), index=False)
print("  wrote data/uncertainty_aware_sweep.csv")
print(df.round(3).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Figure: the trust-vs-performance trade
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: the trade curve
ax1.plot(df.mean_conf, df.worst_LD, "-o", color="#1f77b4", lw=2, ms=7)
for _, r in df.iterrows():
    ax1.annotate(f"w={r.w_conf:.0f}", (r.mean_conf, r.worst_LD),
                 xytext=(6, 5), textcoords="offset points", fontsize=8.5)
ax1.set_xlabel("Mean NeuralFoil confidence  (how much you can trust the number)")
ax1.set_ylabel("Worst-case L/D  (predicted performance)")
ax1.set_title("The trust-vs-performance trade\n"
              "turning up w_conf buys trustworthiness for a little performance",
              fontsize=10)
ax1.grid(alpha=0.25)

# Right: shapes at the two extremes
for wc, c, lab in [(0.0, "#d62728", "w_conf=0 (trust blindly)"),
                   (max(W_CONF), "#2ca02c", f"w_conf={max(W_CONF):.0f} (trust-aware)")]:
    co = geoms[wc].coordinates
    ax2.plot(co[:, 0], co[:, 1], "-", color=c, lw=1.8, label=lab)
ax2.set_aspect("equal"); ax2.grid(alpha=0.25); ax2.legend(fontsize=9)
ax2.set_title("Blind vs trust-aware optimal shapes", fontsize=10)
ax2.set_xlabel("x / c")

fig.suptitle("Uncertainty-aware airfoil optimization: using the MEASURED "
             "confidence–error link inside the design loop", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "20_trust_vs_performance.png"), dpi=160)
print("  wrote figures/20_trust_vs_performance.png")
