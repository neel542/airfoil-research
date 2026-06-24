"""
Multi-objective low-Re airfoil design: efficiency + safety + structure + noise
==============================================================================

L/D alone is a thin objective for an aircraft that flies over people, gets built
in the field, and must not stall. This extends the robust pipeline to four
*physically grounded, differentiable* objectives and shows the design dial from
"pure efficiency" to "community-friendly" (quiet + gentle-stall).

Objectives
----------
  1. EFFICIENCY  : worst-case L/D over the operating envelope (epigraph max-min).
                   -> maximize.   Re in [50k,500k], AoA in [0,8] deg.
  2. SAFETY      : lift retained at high AoA = CL at 12 deg, Re=50k (hardest
                   case). Higher => later, gentler stall, more margin.
                   -> maximize.   (NeuralFoil reports ~0.98 confidence here.)
  3. STRUCTURE   : max thickness = available spar depth. Bending stiffness grows
                   ~ t^3, so a thicker section means a lighter, stiffer wing.
                   -> maximize.
  4. NOISE       : trailing-edge displacement thickness delta* = theta * H at the
                   TE (NeuralFoil boundary-layer output). Brooks-Pope-Marcolini
                   trailing-edge self-noise scales with delta*_TE; it is the
                   dominant broadband source for small props/rotors.
                   -> minimize (penalty).

Scalarization: weighted sum of metrics normalized by reference scales, so the
weights are unit-free and comparable. Three weight profiles trace the dial.
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
RE_LO, RE_HI, AOA_LO, AOA_HI = 50e3, 500e3, 0.0, 8.0
RE_OPT = np.geomspace(RE_LO, RE_HI, 5)
AOA_OPT = np.linspace(AOA_LO, AOA_HI, 5)
RE_EVAL = np.geomspace(RE_LO, RE_HI, 11)
AOA_EVAL = np.linspace(AOA_LO, AOA_HI, 9)

CRUISE_RE, CRUISE_AOA = 200e3, 4.0     # representative cruise for the noise metric
STALL_AOA, STALL_RE = 12.0, 50e3       # safety check: lift retained at high AoA

# Reference scales (typical "good" values) so weights are unit-free.
S_LD, S_SAFE, S_STRUCT, S_NOISE = 40.0, 1.4, 0.12, 0.014

THK_MIN, THK_MAX, TE_THICKNESS = 0.08, 0.16, 0.0025
SEEDS = ["naca4412", "naca6412"]

# Weight profiles: the design dial.
PROFILES = {
    "efficiency":  dict(LD=1.0, safe=0.0, struct=0.0, noise=0.0),
    "balanced":    dict(LD=1.0, safe=0.5, struct=0.8, noise=0.5),
    "community":   dict(LD=0.6, safe=1.0, struct=1.2, noise=1.0),
}


def te_displacement_thickness(aero):
    """delta* = theta * H summed over both surfaces at the trailing edge."""
    return (aero["upper_bl_theta_31"] * aero["upper_bl_H_31"]
            + aero["lower_bl_theta_31"] * aero["lower_bl_H_31"])


def solve_one(w, seed):
    af0 = asb.KulfanAirfoil(seed) if isinstance(seed, str) else seed
    opti = asb.Opti()
    uw = opti.variable(init_guess=af0.upper_weights, lower_bound=-0.2, upper_bound=0.5)
    lw = opti.variable(init_guess=af0.lower_weights, lower_bound=-0.5, upper_bound=0.4)
    le = opti.variable(init_guess=af0.leading_edge_weight, lower_bound=-0.2, upper_bound=0.2)
    af = asb.KulfanAirfoil(lower_weights=lw, upper_weights=uw,
                           leading_edge_weight=le, TE_thickness=TE_THICKNESS)

    opti.subject_to(af.local_thickness(x_over_c=np.linspace(0.02, 0.98, 20)) > 0.004)
    opti.subject_to(af.max_thickness() < THK_MAX)
    opti.subject_to(af.max_thickness() > THK_MIN)

    # 1) efficiency: epigraph worst-case L/D
    g = opti.variable(init_guess=30.0)
    for Re in RE_OPT:
        for a in AOA_OPT:
            aero = af.get_aero_from_neuralfoil(alpha=a, Re=Re, model_size=MODEL)
            opti.subject_to(g <= aero["CL"] / aero["CD"])

    # 2) safety  3) structure  4) noise
    safety = af.get_aero_from_neuralfoil(alpha=STALL_AOA, Re=STALL_RE, model_size=MODEL)["CL"]
    struct = af.max_thickness()
    cruise = af.get_aero_from_neuralfoil(alpha=CRUISE_AOA, Re=CRUISE_RE, model_size=MODEL)
    noise = te_displacement_thickness(cruise)

    obj = (w["LD"] * g / S_LD
           + w["safe"] * safety / S_SAFE
           + w["struct"] * struct / S_STRUCT
           - w["noise"] * noise / S_NOISE)
    opti.maximize(obj)
    sol = opti.solve(verbose=False, max_iter=300)
    return sol(af), float(sol(obj))


def solve_profile(name, w, extra_seeds=()):
    best_af, best_obj = None, -np.inf
    for seed in list(SEEDS) + list(extra_seeds):
        try:
            af, obj = solve_one(w, seed)
        except Exception:
            continue
        if obj > best_obj:
            best_af, best_obj = af, obj
    return best_af


def metrics(af):
    """Numeric report of all four objectives + supporting quantities."""
    # efficiency over fine grid
    LDs = []
    for Re in RE_EVAL:
        aero = af.get_aero_from_neuralfoil(alpha=AOA_EVAL, Re=Re, model_size=MODEL)
        LDs.extend(list(np.atleast_1d(aero["CL"]) / np.atleast_1d(aero["CD"])))
    LDs = np.array(LDs)
    # safety: CLmax proxy + stall AoA over a sweep at low Re
    a_sweep = np.linspace(0, 16, 33)
    sweep = af.get_aero_from_neuralfoil(alpha=a_sweep, Re=STALL_RE, model_size=MODEL)
    cl_sweep = np.atleast_1d(sweep["CL"])
    cruise = af.get_aero_from_neuralfoil(alpha=CRUISE_AOA, Re=CRUISE_RE, model_size=MODEL)
    return dict(
        worst_case_LD=float(LDs.min()),
        mean_LD=float(LDs.mean()),
        peak_LD=float(LDs.max()),
        safety_CL12=float(np.atleast_1d(
            af.get_aero_from_neuralfoil(alpha=STALL_AOA, Re=STALL_RE, model_size=MODEL)["CL"])),
        CLmax=float(cl_sweep.max()),
        stall_AoA=float(a_sweep[int(np.argmax(cl_sweep))]),
        noise_TE_deltastar=float(te_displacement_thickness(cruise)),
        max_thickness=float(af.max_thickness()),
    )


# --------------------------------------------------------------------------- #
print("Solving multi-objective design profiles ...")
geoms, rows = {}, []
prev = None
for name, w in PROFILES.items():
    af = solve_profile(name, w, extra_seeds=(prev,) if prev is not None else ())
    geoms[name] = af
    prev = af
    m = metrics(af)
    m["profile"] = name
    rows.append(m)
    print(f"  [{name:>10}]  worstLD={m['worst_case_LD']:6.1f}  "
          f"CLmax={m['CLmax']:.2f}@{m['stall_AoA']:.0f}deg  "
          f"thk={m['max_thickness']*100:4.1f}%  "
          f"TEd*={m['noise_TE_deltastar']*1e3:5.2f}e-3")

df = pd.DataFrame(rows)[["profile", "worst_case_LD", "mean_LD", "peak_LD",
                         "safety_CL12", "CLmax", "stall_AoA",
                         "noise_TE_deltastar", "max_thickness"]]
df.to_csv(os.path.join(OUT, "data", "multiobjective_metrics.csv"), index=False)
print("  wrote data/multiobjective_metrics.csv")
print(df.round(3).to_string(index=False))

# Save geometries
for name, af in geoms.items():
    co = af.coordinates
    pd.DataFrame({"x": co[:, 0], "y": co[:, 1]}).to_csv(
        os.path.join(OUT, "data", f"multiobj_{name}_coords.csv"), index=False)

# --------------------------------------------------------------------------- #
print("Rendering figures ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(OUT, "figures")
colors = {"efficiency": "#1f77b4", "balanced": "#2ca02c", "community": "#d62728"}

# (1) Shapes
fig, ax = plt.subplots(figsize=(9, 3.2))
for name, af in geoms.items():
    co = af.coordinates
    ax.plot(co[:, 0], co[:, 1], color=colors[name], lw=2, label=name)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Multi-objective airfoil shapes"); ax.set_xlabel("x/c"); ax.set_ylabel("y/c")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "5_multiobj_shapes.png"), dpi=160); plt.close(fig)

# (2) Radar chart of the four normalized objectives (all "higher = better")
axes_labels = ["Efficiency\n(worst-case L/D)", "Safety\n(CLmax)",
               "Structure\n(thickness)", "Quietness\n(1/TE noise)"]
def scores(m):
    return np.array([
        m["worst_case_LD"], m["CLmax"], m["max_thickness"], 1.0 / m["noise_TE_deltastar"]
    ])
raw = {n: scores(metrics_row) for n, metrics_row in zip(df.profile, rows)}
stack = np.vstack(list(raw.values()))
norm = {n: raw[n] / stack.max(axis=0) for n in raw}     # normalize each axis to [0,1]

angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for name in PROFILES:
    vals = norm[name].tolist(); vals += vals[:1]
    ax.plot(angles, vals, color=colors[name], lw=2, label=name)
    ax.fill(angles, vals, color=colors[name], alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, fontsize=9)
ax.set_ylim(0, 1.05); ax.set_title("Four-objective tradeoff\n(each axis normalized to best design)")
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
fig.tight_layout(); fig.savefig(os.path.join(FIG, "6_multiobj_radar.png"), dpi=160); plt.close(fig)

# (3) Stall behavior at Re=50k: CL and L/D vs AoA out to deep stall
a_sweep = np.linspace(0, 16, 33)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for name, af in geoms.items():
    s = af.get_aero_from_neuralfoil(alpha=a_sweep, Re=STALL_RE, model_size=MODEL)
    cl = np.atleast_1d(s["CL"]); cd = np.atleast_1d(s["CD"])
    ax1.plot(a_sweep, cl, color=colors[name], lw=2, label=name)
    ax2.plot(a_sweep, cl / cd, color=colors[name], lw=2, label=name)
ax1.axvspan(AOA_LO, AOA_HI, color="gray", alpha=0.1, label="design envelope")
ax2.axvspan(AOA_LO, AOA_HI, color="gray", alpha=0.1)
ax1.set_title("Lift curve @ Re=50k  (safety / stall)"); ax1.set_xlabel("AoA [deg]"); ax1.set_ylabel("CL")
ax2.set_title("L/D @ Re=50k"); ax2.set_xlabel("AoA [deg]"); ax2.set_ylabel("L/D")
ax1.grid(alpha=0.3); ax2.grid(alpha=0.3); ax1.legend()
fig.suptitle("Off-design / stall behavior"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "7_multiobj_stall.png"), dpi=160); plt.close(fig)

print("\nDone. New figures 5-7 in figures/, metrics in data/multiobjective_metrics.csv")
