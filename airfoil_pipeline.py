"""
Robust multi-point airfoil optimization at low Reynolds number
==============================================================

Pipeline
--------
  A) Single-point optimum   : maximize L/D at the design point (Re=200k, AoA=4 deg)
  B) Robust optimum         : maximize the WORST-CASE L/D over the operating
                              envelope  Re in [50k, 500k],  AoA in [0, 8] deg
  C) Evaluate A and B on a fine (Re, AoA) grid                 -> CSV
  D) Trace the peak-vs-robustness tradeoff family             -> CSV
  E) Figures: shapes, L/D vs AoA, L/D heatmaps, tradeoff curve

Tooling
-------
  * Geometry  : Kulfan / CST parameterization (AeroSandbox KulfanAirfoil).
                Smooth, low-dimensional, manufacturable, and -- crucially --
                analytically differentiable, so it plays well with a
                gradient-based optimizer.
  * Aero model: NeuralFoil (a differentiable neural surrogate trained on XFoil).
                Gives fast, smooth gradients of CL/CD w.r.t. the shape, which is
                what makes shape optimization over a 25-point grid tractable.
  * Optimizer : AeroSandbox Opti -> IPOPT (gradient-based NLP with exact,
                automatically-differentiated derivatives).
"""

import os
import numpy as np
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as anp   # casadi-aware numpy for use inside Opti

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)

# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
MODEL = "large"            # NeuralFoil fidelity used for BOTH opt and eval
DESIGN_RE = 200e3          # single-point design Reynolds number
DESIGN_AOA = 4.0           # single-point design angle of attack [deg]

# Operating envelope the robust design must survive
RE_LO, RE_HI = 50e3, 500e3
AOA_LO, AOA_HI = 0.0, 8.0

# Coarse grid used *inside* the optimizer (keeps the NLP small & fast).
# Re is geometric (log) spaced because it spans a full decade.
RE_OPT = np.geomspace(RE_LO, RE_HI, 5)        # 50k,100k,158k,316k,500k-ish
AOA_OPT = np.linspace(AOA_LO, AOA_HI, 5)      # 0,2,4,6,8

# Fine grid used to *evaluate* the finished airfoils (tests generalization to
# operating points the optimizer never saw).
RE_EVAL = np.geomspace(RE_LO, RE_HI, 11)
AOA_EVAL = np.linspace(AOA_LO, AOA_HI, 9)

# Thickness band -- keep airfoils structurally viable and comparable.
THK_MIN, THK_MAX = 0.09, 0.13
TE_THICKNESS = 0.0025


# Seed airfoils for multi-start. The problem is non-convex, so a single start
# can land in a weak local optimum; we solve from several seeds and keep the
# best. Low-to-high camber covers the sensible low-Re design space.
SEEDS = ["naca2412", "naca4412", "naca6412"]


# --------------------------------------------------------------------------- #
# Core solver
# --------------------------------------------------------------------------- #
def solve_one(lam, seed):
    """
    Solve one airfoil with a blended objective:

        maximize   lam * (L/D at design point)  +  (1 - lam) * g
        subject to g <= L/D(Re_i, AoA_j)   for every grid point   (epigraph)
                   geometry / thickness constraints

    lam = 1  -> pure single-point design  (airfoil A)
    lam = 0  -> pure worst-case (max-min) robust design  (airfoil B)
    0<lam<1  -> intermediate points on the tradeoff curve

    The epigraph trick (introduce g, push every operating point above it,
    maximize g) is the standard smooth way to turn a non-smooth max-min
    objective into something a gradient-based NLP solver can handle.

    `seed` may be a NACA name (str) or an existing KulfanAirfoil to warm-start
    from.  Returns (solved_airfoil, objective_value).
    """
    af0 = asb.KulfanAirfoil(seed) if isinstance(seed, str) else seed
    opti = asb.Opti()

    # ---- design variables: the shape ----
    uw = opti.variable(init_guess=af0.upper_weights, lower_bound=-0.2, upper_bound=0.5)
    lw = opti.variable(init_guess=af0.lower_weights, lower_bound=-0.5, upper_bound=0.4)
    le = opti.variable(init_guess=af0.leading_edge_weight, lower_bound=-0.2, upper_bound=0.2)

    af = asb.KulfanAirfoil(
        lower_weights=lw, upper_weights=uw,
        leading_edge_weight=le, TE_thickness=TE_THICKNESS,
    )

    # ---- geometry constraints ----
    x_stn = np.linspace(0.02, 0.98, 20)
    opti.subject_to(af.local_thickness(x_over_c=x_stn) > 0.004)  # no self-intersection
    opti.subject_to(af.max_thickness() < THK_MAX)
    opti.subject_to(af.max_thickness() > THK_MIN)

    # ---- worst-case epigraph variable ----
    g = opti.variable(init_guess=30.0)

    for Re in RE_OPT:
        for a in AOA_OPT:
            aero = af.get_aero_from_neuralfoil(alpha=a, Re=Re, model_size=MODEL)
            LD = aero["CL"] / aero["CD"]
            opti.subject_to(g <= LD)        # g is a lower bound on every point

    # ---- single design point ----
    aero_sp = af.get_aero_from_neuralfoil(alpha=DESIGN_AOA, Re=DESIGN_RE, model_size=MODEL)
    LD_sp = aero_sp["CL"] / aero_sp["CD"]

    obj = lam * LD_sp + (1.0 - lam) * g
    opti.maximize(obj)

    sol = opti.solve(verbose=False, max_iter=300)
    return sol(af), float(sol(obj))


def solve_airfoil(lam, label, extra_seeds=()):
    """Multi-start wrapper: solve from every seed (NACA names + any warm-start
    airfoils passed in `extra_seeds`), keep the best objective."""
    best_af, best_obj = None, -np.inf
    for seed in list(SEEDS) + list(extra_seeds):
        try:
            af, obj = solve_one(lam, seed)
        except Exception:
            continue  # IPOPT failed from this seed; try the next
        if obj > best_obj:
            best_af, best_obj = af, obj
    if best_af is None:
        raise RuntimeError(f"[{label}] lam={lam}: every seed failed to converge")
    aero_sp = best_af.get_aero_from_neuralfoil(alpha=DESIGN_AOA, Re=DESIGN_RE, model_size=MODEL)
    print(f"  [{label:>14}] lam={lam:.2f}  "
          f"design L/D={float(aero_sp['CL']/aero_sp['CD']):6.1f}   "
          f"max thk={float(best_af.max_thickness()):.3f}")
    return best_af


# --------------------------------------------------------------------------- #
# Evaluation over a grid
# --------------------------------------------------------------------------- #
def evaluate_grid(af, re_grid, aoa_grid, name):
    """Return a tidy DataFrame of aero coefficients across (Re, AoA)."""
    rows = []
    for Re in re_grid:
        aero = af.get_aero_from_neuralfoil(
            alpha=aoa_grid, Re=Re, model_size=MODEL
        )
        for k, a in enumerate(aoa_grid):
            cl = float(np.atleast_1d(aero["CL"])[k])
            cd = float(np.atleast_1d(aero["CD"])[k])
            cm = float(np.atleast_1d(aero["CM"])[k])
            conf = float(np.atleast_1d(aero["analysis_confidence"])[k])
            rows.append(dict(airfoil=name, Re=float(Re), AoA=float(a),
                             CL=cl, CD=cd, CM=cm, L_over_D=cl / cd,
                             analysis_confidence=conf))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1) Solve the whole family by continuation, then pick out A and B.
#    Order matters: start robust (lam=0), walk lam upward warm-starting each
#    solve from the previous one, and finally solve the peak design (lam=1)
#    seeded from every family member. Continuation keeps the non-convex solves
#    on a consistent branch and stops the peak design from getting stuck.
# --------------------------------------------------------------------------- #
print("Optimizing airfoils (continuation over lambda) ...")
LAMS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
family_geoms = {}
prev = None
for lam in LAMS:
    if lam == 1.0:
        label = "A single-pt"
        seeds = list(family_geoms.values())          # seed peak from all family
    elif lam == 0.0:
        label = "B robust"
        seeds = ()
    else:
        label = "family"
        seeds = (prev,) if prev is not None else ()   # continuation
    af = solve_airfoil(lam=lam, label=label, extra_seeds=seeds)
    family_geoms[lam] = af
    prev = af

afA = family_geoms[1.0]
afB = family_geoms[0.0]

# Save geometry (Kulfan params + coordinates)
def save_geometry(af, name):
    pd.DataFrame({
        "param": (["upper_w%d" % i for i in range(len(af.upper_weights))]
                  + ["lower_w%d" % i for i in range(len(af.lower_weights))]
                  + ["leading_edge_weight", "TE_thickness"]),
        "value": list(np.array(af.upper_weights)) + list(np.array(af.lower_weights))
                 + [float(af.leading_edge_weight), float(af.TE_thickness)],
    }).to_csv(os.path.join(OUT, "data", f"airfoil_{name}_kulfan.csv"), index=False)
    coords = af.coordinates
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}).to_csv(
        os.path.join(OUT, "data", f"airfoil_{name}_coords.csv"), index=False)

save_geometry(afA, "A")
save_geometry(afB, "B")

# --------------------------------------------------------------------------- #
# 2) Evaluate both on the fine grid -> CSV (BEFORE plotting)
# --------------------------------------------------------------------------- #
print("Evaluating on fine grid ...")
dfA = evaluate_grid(afA, RE_EVAL, AOA_EVAL, "A_single_point")
dfB = evaluate_grid(afB, RE_EVAL, AOA_EVAL, "B_robust")
df_eval = pd.concat([dfA, dfB], ignore_index=True)
df_eval.to_csv(os.path.join(OUT, "data", "grid_evaluation.csv"), index=False)
print(f"  wrote data/grid_evaluation.csv  ({len(df_eval)} rows)")

# Headline robustness comparison
for name, df in [("A (single-point)", dfA), ("B (robust)", dfB)]:
    print(f"    {name:>18}:  peak L/D={df.L_over_D.max():6.1f}   "
          f"worst-case L/D={df.L_over_D.min():6.1f}   "
          f"mean L/D={df.L_over_D.mean():6.1f}")

# --------------------------------------------------------------------------- #
# 3) Tradeoff family (peak vs robustness) -> CSV
# --------------------------------------------------------------------------- #
print("Tracing peak-vs-robustness tradeoff family ...")
family_rows = []
for lam in LAMS:
    af = family_geoms[lam]
    d = evaluate_grid(af, RE_EVAL, AOA_EVAL, f"lam_{lam}")
    family_rows.append(dict(lam=lam,
                            peak_LD=d.L_over_D.max(),
                            worst_case_LD=d.L_over_D.min(),
                            mean_LD=d.L_over_D.mean()))
df_family = pd.DataFrame(family_rows).sort_values("worst_case_LD").reset_index(drop=True)

# Pareto frontier: an airfoil is non-dominated if no other has BOTH higher
# peak L/D and higher worst-case L/D. This is the honest tradeoff curve even
# when individual (non-convex) solves are imperfect.
is_pareto = []
for i, r in df_family.iterrows():
    dominated = ((df_family.peak_LD >= r.peak_LD) &
                 (df_family.worst_case_LD >= r.worst_case_LD) &
                 ((df_family.peak_LD > r.peak_LD) |
                  (df_family.worst_case_LD > r.worst_case_LD))).any()
    is_pareto.append(not dominated)
df_family["pareto"] = is_pareto
df_family.to_csv(os.path.join(OUT, "data", "tradeoff_family.csv"), index=False)
print(df_family.to_string(index=False))

# --------------------------------------------------------------------------- #
# 4) Figures (matplotlib AFTER all data is written)
# --------------------------------------------------------------------------- #
print("Rendering figures ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(OUT, "figures")
cA, cB = "#1f77b4", "#d62728"

# (i) Overlaid shapes
fig, ax = plt.subplots(figsize=(9, 3.2))
for af, name, c in [(afA, "A  single-point (Re=200k)", cA),
                    (afB, "B  robust (max-min L/D)", cB)]:
    co = af.coordinates
    ax.plot(co[:, 0], co[:, 1], color=c, lw=2, label=name)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
ax.set_title("Optimized airfoil shapes"); ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_shapes.png"), dpi=160); plt.close(fig)

# (ii) L/D vs AoA at three representative Re
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, Re in zip(axes, [50e3, 200e3, 500e3]):
    for df, name, c in [(dfA, "A single-pt", cA), (dfB, "B robust", cB)]:
        sub = df[np.isclose(df.Re, Re, rtol=1e-3)].sort_values("AoA")
        if sub.empty:  # nearest Re actually present
            nearest = df.Re.iloc[(df.Re - Re).abs().argmin()]
            sub = df[df.Re == nearest].sort_values("AoA")
        ax.plot(sub.AoA, sub.L_over_D, "-o", color=c, ms=4, label=name)
    ax.set_title(f"Re = {Re/1e3:.0f}k"); ax.set_xlabel("AoA [deg]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("L / D"); axes[0].legend()
fig.suptitle("L/D vs angle of attack"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "2_LD_vs_AoA.png"), dpi=160); plt.close(fig)

# (iii) L/D heatmaps (shared color scale) + difference
def pivot(df):
    return df.pivot_table(index="AoA", columns="Re", values="L_over_D")

pA, pB = pivot(dfA), pivot(dfB)
vmin = min(pA.values.min(), pB.values.min())
vmax = max(pA.values.max(), pB.values.max())
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, p, name in [(axes[0], pA, "A single-point"), (axes[1], pB, "B robust")]:
    im = ax.imshow(p.values, origin="lower", aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax,
                   extent=[0, len(p.columns), 0, len(p.index)])
    ax.set_xticks(np.arange(len(p.columns)) + 0.5)
    ax.set_xticklabels([f"{c/1e3:.0f}k" for c in p.columns], rotation=45, fontsize=8)
    ax.set_yticks(np.arange(len(p.index)) + 0.5)
    ax.set_yticklabels([f"{i:.0f}" for i in p.index], fontsize=8)
    ax.set_xlabel("Re"); ax.set_ylabel("AoA [deg]"); ax.set_title(f"L/D  -  {name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
diff = pB.values - pA.values
m = np.abs(diff).max()
im = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r",
                    vmin=-m, vmax=m, extent=[0, len(pB.columns), 0, len(pB.index)])
axes[2].set_xticks(np.arange(len(pB.columns)) + 0.5)
axes[2].set_xticklabels([f"{c/1e3:.0f}k" for c in pB.columns], rotation=45, fontsize=8)
axes[2].set_yticks(np.arange(len(pB.index)) + 0.5)
axes[2].set_yticklabels([f"{i:.0f}" for i in pB.index], fontsize=8)
axes[2].set_xlabel("Re"); axes[2].set_ylabel("AoA [deg]")
axes[2].set_title("L/D difference  (B - A)\nblue = robust wins")
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.suptitle("L/D across the operating envelope"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "3_LD_heatmap.png"), dpi=160); plt.close(fig)

# (iv) Peak-vs-robustness tradeoff curve
fig, ax = plt.subplots(figsize=(7.5, 6))
front = df_family[df_family.pareto].sort_values("worst_case_LD")
ax.plot(front.worst_case_LD, front.peak_LD, "-", color="#555", zorder=1,
        label="Pareto frontier")
ax.scatter(df_family.worst_case_LD, df_family.peak_LD, s=35,
           facecolors="none", edgecolors="#999", zorder=2)
for _, r in df_family.iterrows():
    ax.annotate(f"$\\lambda$={r.lam:g}", (r.worst_case_LD, r.peak_LD),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
A = df_family[df_family.lam == 1.0].iloc[0]
B = df_family[df_family.lam == 0.0].iloc[0]
ax.scatter([A.worst_case_LD], [A.peak_LD], s=130, color=cA, zorder=3,
           label="A  single-point ($\\lambda$=1)")
ax.scatter([B.worst_case_LD], [B.peak_LD], s=130, color=cB, zorder=3,
           label="B  robust ($\\lambda$=0)")
ax.set_xlabel("Worst-case L/D  (robustness)  -->")
ax.set_ylabel("Peak L/D  (best-case performance)  -->")
ax.set_title("Peak vs robustness tradeoff\n(each point = one optimized airfoil)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_tradeoff.png"), dpi=160); plt.close(fig)

print("\nDone. Figures in figures/, data in data/.")
