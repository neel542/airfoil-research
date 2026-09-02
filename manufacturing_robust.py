"""
Manufacturing-tolerance robustness + fidelity validation
========================================================

A field-built or 3D-printed wing never matches the CAD shape. A design that is
only operating-condition-robust can still lose most of its performance to build
error. This module:

  1. Models manufacturing error as bounded random perturbations of the Kulfan
     weights (camber + thickness + LE errors).
  2. Optimizes TWO designs on the same operating envelope:
        B_nominal : robust to operating conditions only (no build error).
        B_mfg     : robust to operating conditions AND a fixed ensemble of
                    manufacturing-error realizations (worst-case / max-min).
  3. VALIDATES out-of-sample: re-tests both designs against a *fresh* set of
     manufacturing perturbations the optimizer never saw, and compares the
     distribution of realized worst-case L/D.
  4. FIDELITY cross-check: compares the optimization surrogate (NeuralFoil
     `large`) against a higher-fidelity model (`xxlarge`) and, if an XFoil
     binary is available, against true XFoil -- quantifying the surrogate gap.

The manufacturing-robust formulation is sample-based robust optimization: the
build-error distribution is approximated by a fixed ensemble drawn once (fixed
RNG seed), so the inner worst-case is deterministic and differentiable.
"""

import os
import shutil
import numpy as np
import pandas as pd
import aerosandbox as asb
import aerosandbox.numpy as anp

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)

MODEL = "large"
RE_LO, RE_HI, AOA_LO, AOA_HI = 50e3, 500e3, 0.0, 8.0
# Coarse grids inside the optimizer (the manufacturing ensemble multiplies cost).
RE_OPT = np.geomspace(RE_LO, RE_HI, 3)
AOA_OPT = np.linspace(AOA_LO, AOA_HI, 3)
# Fine grid for evaluation / validation.
RE_EVAL = np.geomspace(RE_LO, RE_HI, 11)
AOA_EVAL = np.linspace(AOA_LO, AOA_HI, 9)

THK_MIN, THK_MAX, TE_THICKNESS = 0.08, 0.14, 0.0025
SEEDS = ["naca4412", "naca6412"]

# Manufacturing-error model: zero-mean Gaussian on every Kulfan weight.
SIGMA_W = 0.040            # per-weight std-dev (calibrated to ~0.5% chord below)
N_TRAIN = 8                # ensemble size used inside the optimizer
N_TEST = 40               # fresh out-of-sample realizations for validation

N_WEIGHTS = 8             # per surface


def make_perturbations(n, seed):
    """n perturbation dicts, each with upper/lower/le offsets. Fixed seed."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(dict(
            up=rng.normal(0, SIGMA_W, N_WEIGHTS),
            lo=rng.normal(0, SIGMA_W, N_WEIGHTS),
            le=rng.normal(0, SIGMA_W * 0.5),
        ))
    return out


TRAIN_PERTS = [dict(up=np.zeros(N_WEIGHTS), lo=np.zeros(N_WEIGHTS), le=0.0)] \
              + make_perturbations(N_TRAIN, seed=1)        # include nominal
TEST_PERTS = make_perturbations(N_TEST, seed=999)          # out-of-sample


def perturbed_airfoil(uw, lw, le, p):
    return asb.KulfanAirfoil(
        upper_weights=uw + p["up"], lower_weights=lw + p["lo"],
        leading_edge_weight=le + p["le"], TE_thickness=TE_THICKNESS,
    )


def rms_surface_error(p, base="naca4412"):
    """Physical size of a perturbation: RMS y-deviation in % chord."""
    af0 = asb.KulfanAirfoil(base)
    af1 = perturbed_airfoil(af0.upper_weights, af0.lower_weights,
                            af0.leading_edge_weight, p)
    x = np.linspace(0.01, 0.99, 100)
    d_up = np.array(af1.upper_coordinates(x)[:, 1]) - np.array(af0.upper_coordinates(x)[:, 1])
    d_lo = np.array(af1.lower_coordinates(x)[:, 1]) - np.array(af0.lower_coordinates(x)[:, 1])
    return float(np.sqrt(np.mean(np.concatenate([d_up, d_lo]) ** 2)) * 100)


def solve(robust_to_mfg, seed):
    af0 = asb.KulfanAirfoil(seed) if isinstance(seed, str) else seed
    opti = asb.Opti()
    uw = opti.variable(init_guess=af0.upper_weights, lower_bound=-0.2, upper_bound=0.5)
    lw = opti.variable(init_guess=af0.lower_weights, lower_bound=-0.5, upper_bound=0.4)
    le = opti.variable(init_guess=af0.leading_edge_weight, lower_bound=-0.2, upper_bound=0.2)

    # Geometry constraints on the NOMINAL (as-drawn) shape.
    nominal = asb.KulfanAirfoil(upper_weights=uw, lower_weights=lw,
                                leading_edge_weight=le, TE_thickness=TE_THICKNESS)
    opti.subject_to(nominal.local_thickness(x_over_c=np.linspace(0.02, 0.98, 20)) > 0.004)
    opti.subject_to(nominal.max_thickness() < THK_MAX)
    opti.subject_to(nominal.max_thickness() > THK_MIN)

    perts = TRAIN_PERTS if robust_to_mfg else [TRAIN_PERTS[0]]  # nominal only
    g = opti.variable(init_guess=20.0)
    for p in perts:
        af = perturbed_airfoil(uw, lw, le, p)
        for Re in RE_OPT:
            aero = af.get_aero_from_neuralfoil(alpha=AOA_OPT, Re=Re, model_size=MODEL)
            opti.subject_to(g <= aero["CL"] / aero["CD"])   # vector over AoA
    opti.maximize(g)
    sol = opti.solve(verbose=False, max_iter=300)
    return sol(nominal), float(sol(g))


def solve_best(robust_to_mfg, extra_seeds=()):
    best_af, best = None, -np.inf
    for s in list(SEEDS) + list(extra_seeds):
        try:
            af, g = solve(robust_to_mfg, s)
        except Exception:
            continue
        if g > best:
            best_af, best = af, g
    if best_af is None:
        raise RuntimeError(f"robust_to_mfg={robust_to_mfg}: every seed failed to converge")
    return best_af


def worst_case_LD(af, perts=None):
    """Worst-case L/D over the fine envelope; if perts given, over the worst of
    all (perturbation, Re, AoA) too. Returns scalar worst-case."""
    realizations = perts if perts is not None else [TRAIN_PERTS[0]]
    worst = np.inf
    for p in realizations:
        a = perturbed_airfoil(af.upper_weights, af.lower_weights, af.leading_edge_weight, p)
        for Re in RE_EVAL:
            aero = a.get_aero_from_neuralfoil(alpha=AOA_EVAL, Re=Re, model_size=MODEL)
            ld = np.atleast_1d(aero["CL"]) / np.atleast_1d(aero["CD"])
            worst = min(worst, float(np.min(ld)))
    return worst


# --------------------------------------------------------------------------- #
print(f"Manufacturing-error model: per-weight sigma={SIGMA_W}")
rms = np.mean([rms_surface_error(p) for p in TEST_PERTS])
print(f"  => mean RMS surface error ~ {rms:.2f}% chord across the test ensemble")

print("Optimizing B_nominal (operating-robust only) ...")
B_nom = solve_best(robust_to_mfg=False)
print("Optimizing B_mfg (operating + manufacturing-robust) ...")
B_mfg = solve_best(robust_to_mfg=True, extra_seeds=(B_nom,))

# --------------------------------------------------------------------------- #
# Out-of-sample validation: realized worst-case L/D under FRESH build errors.
# --------------------------------------------------------------------------- #
print(f"Validating against {N_TEST} fresh (out-of-sample) build-error samples ...")
rows = []
for name, af in [("B_nominal", B_nom), ("B_mfg", B_mfg)]:
    as_designed = worst_case_LD(af)                       # perfect build
    realized = []
    for p in TEST_PERTS:
        a = perturbed_airfoil(af.upper_weights, af.lower_weights, af.leading_edge_weight, p)
        wc = np.inf
        for Re in RE_EVAL:
            aero = a.get_aero_from_neuralfoil(alpha=AOA_EVAL, Re=Re, model_size=MODEL)
            ld = np.atleast_1d(aero["CL"]) / np.atleast_1d(aero["CD"])
            wc = min(wc, float(np.min(ld)))
        realized.append(wc)
        rows.append(dict(design=name, sample="perturbed", worst_case_LD=wc))
    realized = np.array(realized)
    rows.append(dict(design=name, sample="as_designed", worst_case_LD=as_designed))
    print(f"  {name:>10}: as-designed worst-case L/D={as_designed:6.1f} | "
          f"built  mean={realized.mean():6.1f}  5th-pct={np.percentile(realized,5):6.1f}  "
          f"min={realized.min():6.1f}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, "data", "manufacturing_validation.csv"), index=False)
print("  wrote data/manufacturing_validation.csv")

# --------------------------------------------------------------------------- #
# Fidelity cross-check: large vs xxlarge (vs XFoil if available).
# --------------------------------------------------------------------------- #
print("Fidelity cross-check (NeuralFoil large vs xxlarge ...) ...")
have_xfoil = shutil.which("xfoil") is not None
print(f"  XFoil binary available: {have_xfoil}")
a_sweep = np.linspace(0, 9, 10)
CHECK_RE = 200e3
fid_rows = []
for name, af in [("B_nominal", B_nom), ("B_mfg", B_mfg)]:
    for ms in ["large", "xxlarge"]:
        aero = af.get_aero_from_neuralfoil(alpha=a_sweep, Re=CHECK_RE, model_size=ms)
        cl = np.atleast_1d(aero["CL"]); cd = np.atleast_1d(aero["CD"])
        for k, a in enumerate(a_sweep):
            fid_rows.append(dict(design=name, source=f"NF_{ms}", alpha=float(a),
                                 CL=float(cl[k]), CD=float(cd[k]), L_over_D=float(cl[k] / cd[k])))
    if have_xfoil:
        try:
            xf = asb.XFoil(airfoil=asb.Airfoil(coordinates=af.coordinates),
                           Re=CHECK_RE, max_iter=100, timeout=60)
            res = xf.alpha(a_sweep)
            for k in range(len(np.atleast_1d(res["alpha"]))):
                cl = float(np.atleast_1d(res["CL"])[k]); cd = float(np.atleast_1d(res["CD"])[k])
                fid_rows.append(dict(design=name, source="XFoil",
                                     alpha=float(np.atleast_1d(res["alpha"])[k]),
                                     CL=cl, CD=cd, L_over_D=cl / cd))
        except Exception as e:
            print(f"    XFoil run failed for {name}: {e}")
df_fid = pd.DataFrame(fid_rows)
df_fid.to_csv(os.path.join(OUT, "data", "fidelity_check.csv"), index=False)
print("  wrote data/fidelity_check.csv")

# Quantify large-vs-xxlarge gap
for name in ["B_nominal", "B_mfg"]:
    lg = df_fid[(df_fid.design == name) & (df_fid.source == "NF_large")].set_index("alpha")
    xl = df_fid[(df_fid.design == name) & (df_fid.source == "NF_xxlarge")].set_index("alpha")
    gap = (lg.L_over_D - xl.L_over_D).abs().mean()
    print(f"  {name}: mean |L/D large - xxlarge| = {gap:.1f} "
          f"({100*gap/xl.L_over_D.mean():.1f}% of mean L/D)")

# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
print("Rendering figures ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = os.path.join(OUT, "figures")
cN, cM = "#1f77b4", "#d62728"
df_val = pd.read_csv(os.path.join(OUT, "data", "manufacturing_validation.csv"))

# (8) Manufacturing robustness: distribution of realized worst-case L/D
fig, ax = plt.subplots(figsize=(8.5, 5))
for name, c in [("B_nominal", cN), ("B_mfg", cM)]:
    d = df_val[(df_val.design == name) & (df_val["sample"] == "perturbed")].worst_case_LD
    ax.hist(d, bins=14, alpha=0.5, color=c, label=f"{name} (built)")
    desg = df_val[(df_val.design == name) & (df_val["sample"] == "as_designed")].worst_case_LD.iloc[0]
    ax.axvline(desg, color=c, ls="--", lw=2, label=f"{name} as-designed")
ax.set_xlabel("Realized worst-case L/D under build error")
ax.set_ylabel("count (out of %d builds)" % N_TEST)
ax.set_title(f"Manufacturing-tolerance validation  (~{rms:.2f}% chord RMS error, out-of-sample)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "8_manufacturing_robustness.png"), dpi=160); plt.close(fig)

# (9) Fidelity cross-check: L/D vs alpha, large vs xxlarge (vs XFoil)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
styles = {"NF_large": dict(ls="-", marker="o"), "NF_xxlarge": dict(ls="--", marker="s"),
          "XFoil": dict(ls=":", marker="^")}
scolor = {"NF_large": "#1f77b4", "NF_xxlarge": "#ff7f0e", "XFoil": "#2ca02c"}
for ax, name in zip(axes, ["B_nominal", "B_mfg"]):
    for src in ["NF_large", "NF_xxlarge", "XFoil"]:
        d = df_fid[(df_fid.design == name) & (df_fid.source == src)].sort_values("alpha")
        if d.empty:
            continue
        ax.plot(d.alpha, d.L_over_D, color=scolor[src], ms=4, label=src, **styles[src])
    ax.set_title(f"{name}  @ Re=200k"); ax.set_xlabel("AoA [deg]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("L / D"); axes[0].legend()
fig.suptitle("Fidelity cross-check: optimization surrogate vs higher fidelity")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "9_fidelity_check.png"), dpi=160); plt.close(fig)

# geometries
for name, af in [("B_nominal", B_nom), ("B_mfg", B_mfg)]:
    co = af.coordinates
    pd.DataFrame({"x": co[:, 0], "y": co[:, 1]}).to_csv(
        os.path.join(OUT, "data", f"mfg_{name}_coords.csv"), index=False)

print("\nDone. Figures 8-9 in figures/, data in data/.")
