"""
rebuild_all_figures.py
=============================================================================
Re-render every figure in the project with one clean, consistent, presentation
style: readable fonts, a fixed color palette, light gridlines, no top/right
border clutter, and higher resolution. All 20 figures are rebuilt straight
from the already-saved CSV/coordinate data in data/ -- nothing is re-optimized,
so this runs in well under a minute except for one quick NeuralFoil forward
sweep (fast; not an optimization) needed for the multi-objective stall plot.
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
FIG = os.path.join(OUT, "figures")

# ─────────────────────────────────────────────────────────────────────────────
# One shared, clean style used by every figure below.
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.7,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.axisbelow": True,
})

# Consistent color roles, reused across every figure.
C_A, C_B = "#2166ac", "#b2182b"          # single-point / robust
C_EFF, C_BAL, C_COM = "#2166ac", "#1a9850", "#b2182b"   # efficiency/balanced/community
C_NF_L, C_NF_XL, C_XF, C_WT = "#2166ac", "#e08214", "#1a9850", "#b2182b"
C_NOM, C_MFG = "#2166ac", "#b2182b"


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


print("Rebuilding all figures with a single consistent style ...\n")

# ═════════════════════════════════════════════════════════════════════════
# 1-4: airfoil_pipeline.py  (single-point A vs robust B)
# ═════════════════════════════════════════════════════════════════════════
afA = pd.read_csv(os.path.join(DATA, "airfoil_A_coords.csv"))
afB = pd.read_csv(os.path.join(DATA, "airfoil_B_coords.csv"))
grid = pd.read_csv(os.path.join(DATA, "grid_evaluation.csv"))
fam = pd.read_csv(os.path.join(DATA, "tradeoff_family.csv"))
dfA = grid[grid.airfoil == "A_single_point"]
dfB = grid[grid.airfoil == "B_robust"]

# (1) Shapes
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(afA.x, afA.y, color=C_A, lw=2.2, label="A -- single-point (Re = 200k)")
ax.plot(afB.x, afB.y, color=C_B, lw=2.2, label="B -- robust (max-min L/D)")
ax.set_aspect("equal"); clean(ax); ax.legend(loc="upper right")
ax.set_title("Optimized airfoil shapes"); ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
save(fig, "1_shapes.png")

# (2) L/D vs AoA at three Re
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, Re in zip(axes, [50e3, 200e3, 500e3]):
    for d, name, c in [(dfA, "A single-pt", C_A), (dfB, "B robust", C_B)]:
        nearest = d.Re.iloc[(d.Re - Re).abs().argmin()]
        s = d[d.Re == nearest].sort_values("AoA")
        ax.plot(s.AoA, s.L_over_D, "-o", color=c, ms=4.5, lw=1.8, label=name)
    ax.set_title(f"Re = {Re/1e3:.0f}k"); ax.set_xlabel("AoA [deg]"); clean(ax)
axes[0].set_ylabel("L / D"); axes[0].legend()
save(fig, "2_LD_vs_AoA.png", "L/D vs angle of attack")

# (3) Heatmaps
def pivot(d):
    return d.pivot_table(index="AoA", columns="Re", values="L_over_D")
pA, pB = pivot(dfA), pivot(dfB)
vmin, vmax = min(pA.values.min(), pB.values.min()), max(pA.values.max(), pB.values.max())
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for ax, p, name in [(axes[0], pA, "A single-point"), (axes[1], pB, "B robust")]:
    im = ax.imshow(p.values, origin="lower", aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, extent=[0, len(p.columns), 0, len(p.index)])
    ax.set_xticks(np.arange(len(p.columns)) + 0.5)
    ax.set_xticklabels([f"{c/1e3:.0f}k" for c in p.columns], rotation=45, fontsize=8)
    ax.set_yticks(np.arange(len(p.index)) + 0.5)
    ax.set_yticklabels([f"{i:.0f}" for i in p.index], fontsize=8)
    ax.set_xlabel("Re"); ax.set_ylabel("AoA [deg]"); ax.set_title(f"L/D -- {name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
diff = pB.values - pA.values
m = np.abs(diff).max()
im = axes[2].imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m,
                    extent=[0, len(pB.columns), 0, len(pB.index)])
axes[2].set_xticks(np.arange(len(pB.columns)) + 0.5)
axes[2].set_xticklabels([f"{c/1e3:.0f}k" for c in pB.columns], rotation=45, fontsize=8)
axes[2].set_yticks(np.arange(len(pB.index)) + 0.5)
axes[2].set_yticklabels([f"{i:.0f}" for i in pB.index], fontsize=8)
axes[2].set_xlabel("Re"); axes[2].set_ylabel("AoA [deg]")
axes[2].set_title("L/D difference (B - A)\nblue = robust wins")
fig.colorbar(im, ax=axes[2], fraction=0.046)
save(fig, "3_LD_heatmap.png", "L/D across the operating envelope")

# (4) Tradeoff
fig, ax = plt.subplots(figsize=(7.5, 6))
front = fam[fam.pareto].sort_values("worst_case_LD")
ax.plot(front.worst_case_LD, front.peak_LD, "-", color="#888888", zorder=1, label="Pareto frontier")
ax.scatter(fam.worst_case_LD, fam.peak_LD, s=40, facecolors="none", edgecolors="#999999", zorder=2)
for _, r in fam.iterrows():
    ax.annotate(f"$\\lambda$={r.lam:g}", (r.worst_case_LD, r.peak_LD),
                textcoords="offset points", xytext=(6, 6), fontsize=8)
A, B = fam[fam.lam == 1.0].iloc[0], fam[fam.lam == 0.0].iloc[0]
ax.scatter([A.worst_case_LD], [A.peak_LD], s=140, color=C_A, zorder=3,
           edgecolors="k", linewidths=0.6, label="A single-point ($\\lambda$=1)")
ax.scatter([B.worst_case_LD], [B.peak_LD], s=140, color=C_B, zorder=3,
           edgecolors="k", linewidths=0.6, label="B robust ($\\lambda$=0)")
ax.set_xlabel("Worst-case L/D  (robustness)  -->")
ax.set_ylabel("Peak L/D  (best-case performance)  -->")
ax.set_title("Peak vs robustness tradeoff\n(each point is one optimized airfoil)")
clean(ax); ax.legend()
save(fig, "4_tradeoff.png")

# ═════════════════════════════════════════════════════════════════════════
# 5-7: multiobjective.py
# ═════════════════════════════════════════════════════════════════════════
mo = pd.read_csv(os.path.join(DATA, "multiobjective_metrics.csv"))
mo_colors = {"efficiency": C_EFF, "balanced": C_BAL, "community": C_COM}
mo_coords = {n: pd.read_csv(os.path.join(DATA, f"multiobj_{n}_coords.csv")) for n in mo_colors}

# (5) Shapes
fig, ax = plt.subplots(figsize=(9, 3.2))
for name, c in mo_colors.items():
    co = mo_coords[name]
    ax.plot(co.x, co.y, color=c, lw=2.2, label=name)
ax.set_aspect("equal"); clean(ax); ax.legend()
ax.set_title("Multi-objective airfoil shapes"); ax.set_xlabel("x / c"); ax.set_ylabel("y / c")
save(fig, "5_multiobj_shapes.png")

# (6) Radar
labels = ["Efficiency\n(worst-case L/D)", "Safety\n(CLmax)",
          "Structure\n(thickness)", "Quietness\n(1 / TE noise)"]
def scores(row):
    return np.array([row.worst_case_LD, row.CLmax, row.max_thickness, 1.0 / row.noise_TE_deltastar])
raw = {r.profile: scores(r) for _, r in mo.iterrows()}
stack = np.vstack(list(raw.values()))
norm = {n: raw[n] / stack.max(axis=0) for n in raw}
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for name, c in mo_colors.items():
    vals = norm[name].tolist(); vals += vals[:1]
    ax.plot(angles, vals, color=c, lw=2.2, label=name)
    ax.fill(angles, vals, color=c, alpha=0.12)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylim(0, 1.05); ax.set_title("Four-objective tradeoff\n(each axis normalized to the best design)")
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1))
save(fig, "6_multiobj_radar.png")

# (7) Stall behavior -- needs one fast NeuralFoil forward sweep (no optimization)
import aerosandbox as asb
STALL_RE, AOA_LO, AOA_HI = 50e3, 0.0, 8.0
a_sweep = np.linspace(0, 16, 33)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
for name, c in mo_colors.items():
    af = asb.Airfoil(coordinates=mo_coords[name][["x", "y"]].to_numpy()).to_kulfan_airfoil()
    s = af.get_aero_from_neuralfoil(alpha=a_sweep, Re=STALL_RE, model_size="large")
    cl, cd = np.atleast_1d(s["CL"]), np.atleast_1d(s["CD"])
    ax1.plot(a_sweep, cl, color=c, lw=2.2, label=name)
    ax2.plot(a_sweep, cl / cd, color=c, lw=2.2, label=name)
ax1.axvspan(AOA_LO, AOA_HI, color="gray", alpha=0.1, label="design envelope")
ax2.axvspan(AOA_LO, AOA_HI, color="gray", alpha=0.1)
ax1.set_title("Lift curve @ Re = 50k (safety / stall)"); ax1.set_xlabel("AoA [deg]"); ax1.set_ylabel("$C_L$")
ax2.set_title("L/D @ Re = 50k"); ax2.set_xlabel("AoA [deg]"); ax2.set_ylabel("L / D")
clean(ax1); clean(ax2); ax1.legend()
save(fig, "7_multiobj_stall.png", "Off-design / stall behavior")

# ═════════════════════════════════════════════════════════════════════════
# 8-9: manufacturing_robust.py
# ═════════════════════════════════════════════════════════════════════════
val = pd.read_csv(os.path.join(DATA, "manufacturing_validation.csv"))
fid = pd.read_csv(os.path.join(DATA, "fidelity_check.csv"))
xf9 = pd.read_csv(os.path.join(DATA, "xfoil_validation.csv"))
N_TEST = (val.design.eq("B_nominal") & val["sample"].eq("perturbed")).sum()
rms = 0.54  # documented manufacturing sigma, % chord RMS (see METHODS.md sec 6)

# (8) Distribution of realized worst-case L/D
fig, ax = plt.subplots(figsize=(8.5, 5))
for name, c in [("B_nominal", C_NOM), ("B_mfg", C_MFG)]:
    d = val[(val.design == name) & (val["sample"] == "perturbed")].worst_case_LD
    ax.hist(d, bins=14, alpha=0.55, color=c, edgecolor="white", linewidth=0.6,
            label=f"{name} (built)")
    desg = val[(val.design == name) & (val["sample"] == "as_designed")].worst_case_LD.iloc[0]
    ax.axvline(desg, color=c, ls="--", lw=2.2, label=f"{name} as-designed")
ax.set_xlabel("Realized worst-case L/D under build error")
ax.set_ylabel(f"count (out of {N_TEST} builds)")
ax.set_title(f"Manufacturing-tolerance validation (~{rms:.2f}% chord RMS error, out-of-sample)")
clean(ax); ax.legend()
save(fig, "8_manufacturing_robustness.png")

# (9) Fidelity cross-check
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
styles = {"NF_large": dict(ls="-", marker="o", color=C_NF_L),
          "NF_xxlarge": dict(ls="--", marker="s", color=C_NF_XL),
          "XFoil": dict(ls=":", marker="^", color=C_XF)}
for ax, name in zip(axes, ["B_nominal", "B_mfg"]):
    for src, sty in styles.items():
        d = fid[(fid.design == name) & (fid.source == src)].sort_values("alpha")
        if d.empty:
            continue
        ax.plot(d.alpha, d.L_over_D, ms=5, lw=1.8, label=src, **sty)
    # True-XFoil ground truth comes from xfoil_validate.py (headless binary,
    # PACC polar parsed directly); fidelity_check.csv only has it when the
    # AeroSandbox XFoil wrapper happened to work, which it does not on this build.
    if fid[(fid.design == name) & (fid.source == "XFoil")].empty:
        x9 = xf9[xf9.design == name].dropna(subset=["XFoil_LD"]).sort_values("alpha")
        if not x9.empty:
            ax.plot(x9.alpha, x9.XFoil_LD, ms=5, lw=1.8, label="XFoil", **styles["XFoil"])
    ax.set_title(f"{name} @ Re = 200k"); ax.set_xlabel("AoA [deg]"); clean(ax)
axes[0].set_ylabel("L / D"); axes[0].legend()
save(fig, "9_fidelity_check.png", "Fidelity cross-check: surrogate vs higher fidelity")

# ═════════════════════════════════════════════════════════════════════════
# 10-11: xfoil_validate_envelope.py
# ═════════════════════════════════════════════════════════════════════════
env = pd.read_csv(os.path.join(DATA, "xfoil_validation_envelope.csv"))
RE_GRID = sorted(env.Re.unique())
ALPHAS = sorted(env.alpha.unique())
designs = sorted(env.design.unique())

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, name in zip(axes, designs):
    d = env[env.design == name]
    piv = (d.pivot_table(index="alpha", columns="Re", values="pct_err_large")
           .reindex(index=ALPHAS, columns=RE_GRID))
    data = piv.values
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=20,
                   extent=[0, len(piv.columns), 0, len(piv.index)])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, hatch="////", fill=True,
                                           fc="#e5e5e5", ec="#999999", lw=0))
            else:
                ax.text(j + 0.5, i + 0.5, f"{data[i,j]:.0f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    ax.set_xticks(np.arange(len(piv.columns)) + 0.5)
    ax.set_xticklabels([f"{c/1e3:.0f}k" for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)) + 0.5)
    ax.set_yticklabels([f"{i:.0f}" for i in piv.index])
    ax.set_xlabel("Re"); ax.set_ylabel("AoA [deg]")
    ax.set_title(f"{name}: |L/D error| NeuralFoil vs XFoil [%]\n(hatched = XFoil did not converge)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% L/D error")
save(fig, "10_trust_map.png", "Surrogate trust map across the operating envelope")

d11 = env[env.design == "B_mfg"]
fig, axes = plt.subplots(1, len(RE_GRID), figsize=(17, 3.8), sharey=True)
for ax, Re in zip(axes, RE_GRID):
    s = d11[d11.Re == Re].sort_values("alpha")
    ax.plot(s.alpha, s.NF_large_LD, "-o", color=C_NF_L, ms=3.5, lw=1.6, label="NF large")
    ax.plot(s.alpha, s.NF_xxlarge_LD, "--s", color=C_NF_XL, ms=3.5, lw=1.6, label="NF xxlarge")
    ax.plot(s.alpha, s.XFoil_LD, ":^", color=C_XF, ms=5.5, lw=1.6, label="XFoil")
    ax.set_title(f"Re = {Re/1e3:.0f}k"); ax.set_xlabel("AoA [deg]"); clean(ax)
axes[0].set_ylabel("L / D"); axes[0].legend(fontsize=8)
save(fig, "11_envelope_polars.png", "B_mfg: surrogate vs true XFoil across Re")

# ═════════════════════════════════════════════════════════════════════════
# 12-17: e387_neuralfoil_validation.py  +  multifoil comparison
# ═════════════════════════════════════════════════════════════════════════
exp = pd.read_csv(os.path.join(DATA, "e387_experimental_NREL.csv"))
val12 = pd.read_csv(os.path.join(DATA, "e387_neuralfoil_validation.csv"))
PLOT_RE_NOM = [100, 200, 500]

def _panels(value_fn, ylabel, fname, title, logy=False):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, Re_nom in zip(axes, PLOT_RE_NOM):
        wt = exp[exp.Re_nom_k == Re_nom].sort_values("alpha")
        ax.scatter(wt.alpha, value_fn(wt.CL, wt.CD), s=44, color=C_WT, marker="D",
                   zorder=5, label="Wind tunnel (UIUC/NREL)", edgecolors="white", linewidths=0.4)
        for model, c, ls in [("large", C_NF_L, "-"), ("xxlarge", C_NF_XL, "--")]:
            s = (val12[(val12.Re_nom_k == Re_nom) & (val12.model == model)]
                 .sort_values("alpha").drop_duplicates("alpha"))
            ax.plot(s.alpha, value_fn(s.NF_CL, s.NF_CD), ls, color=c, lw=1.9,
                    label=f"NeuralFoil {model}")
        Re_exact = int(exp[exp.Re_nom_k == Re_nom].Re.iloc[0])
        s = val12[(val12.Re_nom_k == Re_nom) & (val12.model == "large")].dropna(subset=["XF_CL"])
        if not s.empty:
            s = s.sort_values("alpha")
            ax.plot(s.alpha, value_fn(s.XF_CL, s.XF_CD), ":", color=C_XF, lw=1.9, label="XFoil (this build)")
        ax.set_title(f"Re $\\approx$ {Re_nom}k"); ax.set_xlabel("AoA [deg]"); clean(ax)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8, framealpha=0.9)
    save(fig, fname, title)

_panels(lambda cl, cd: cl, "$C_L$", "12_e387_CL_polars.png",
        "Eppler E387 $C_L$: NeuralFoil vs XFoil vs experiment")
_panels(lambda cl, cd: cd, "$C_D$ (log)", "13_e387_CD_polars.png",
        "Eppler E387 $C_D$: NeuralFoil vs XFoil vs experiment", logy=True)
_panels(lambda cl, cd: cl / cd, "L / D", "14_e387_LD_polars.png",
        "Eppler E387 L/D: NeuralFoil vs XFoil vs experiment")

# (15) Error heatmap
sub_l = val12[(val12.model == "large") & (val12.WT_CL.abs() > 0.1)].copy()
sub_l["pct"] = sub_l.err_CL * 100
sub_l["a_bin"] = sub_l.alpha.round().astype(int)
RE_NOMS = sorted(exp.Re_nom_k.unique())
grid15 = (sub_l.groupby(["a_bin", "Re_nom_k"]).pct.mean().reset_index()
          .pivot(index="a_bin", columns="Re_nom_k", values="pct").reindex(columns=RE_NOMS))
fig, ax = plt.subplots(figsize=(9, 6))
vmax15 = max(np.nanmax(np.abs(grid15.values)), 10)
im = ax.imshow(grid15.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax15, vmax=vmax15)
ax.set_xticks(range(len(RE_NOMS))); ax.set_xticklabels([f"{r}k" for r in RE_NOMS])
ax.set_yticks(range(len(grid15.index))); ax.set_yticklabels([f"{a}°" for a in grid15.index])
ax.set_xlabel("Reynolds number"); ax.set_ylabel("Angle of attack")
plt.colorbar(im, ax=ax, label="(NF $C_L$ - WT $C_L$) / |WT $C_L$|  [%]")
for i in range(grid15.shape[0]):
    for j in range(grid15.shape[1]):
        v = grid15.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > vmax15 * 0.6 else "black")
save(fig, "15_e387_error_heatmap.png",
     "NeuralFoil (large) $C_L$ error vs experiment -- Eppler E387\n"
     "red = over-predicts lift, blue = under-predicts")

# (16) Confidence vs error, E387. Absolute CL error, so near-zero-lift points
# do not inflate the picture, and the alpha >= 0 correlation shown alongside
# the all-points one, because the two differ a lot.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, model in zip(axes, ["large", "xxlarge"]):
    s = val12[(val12.model == model) & (val12.WT_CL.abs() > 0.1)].copy()
    s["ae"] = (s.NF_CL - s.WT_CL).abs()
    sc = ax.scatter(s.NF_conf, s.ae, c=s.alpha, cmap="coolwarm", s=44, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="angle of attack [deg]")
    r_all = np.corrcoef(s.NF_conf, s.ae)[0, 1]
    sp = s[s.alpha >= 0]
    r_pos = np.corrcoef(sp.NF_conf, sp.ae)[0, 1]
    ax.text(0.97, 0.96, f"r = {r_all:+.2f} (all points)\nr = {r_pos:+.2f} (alpha >= 0)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#b2182b",
            bbox=dict(fc="white", ec="#b2182b", alpha=0.9, pad=3))
    ax.set_xlabel("NeuralFoil analysis_confidence")
    ax.set_ylabel("|$\\Delta C_L$|  (NeuralFoil - wind tunnel)")
    ax.set_title(f"NeuralFoil {model}"); clean(ax)
save(fig, "16_e387_confidence_error.png",
     "Does confidence predict error? E387 vs experiment (confidence drops only at the alpha extremes)")

# (17) Multi-airfoil parity: every clean UIUC point, NeuralFoil large, free transition
uv = pd.read_csv(os.path.join(DATA, "uiuc_neuralfoil_validation.csv"))
u17 = uv[(uv.config == "clean") & (uv.NF_mode == "free") & (uv.model == "large") & uv.fit_ok]
n_af, n_pt = u17.asb_name.nunique(), len(u17)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
ax = axes[0]
sc = ax.scatter(u17.WT_CL, u17.NF_CL, c=np.log10(u17.Re), cmap="viridis", s=14, alpha=0.6, edgecolors="none")
lim = [-0.6, 2.2]
ax.plot(lim, lim, "k--", lw=1, alpha=0.6, label="perfect")
ax.fill_between(lim, [l - 0.05 for l in lim], [l + 0.05 for l in lim], color="gray", alpha=0.15, label="±0.05 $C_L$")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); clean(ax); ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("Wind-tunnel $C_L$"); ax.set_ylabel("NeuralFoil $C_L$"); ax.set_title("Lift parity")
ax = axes[1]
sc = ax.scatter(u17.WT_CD, u17.NF_CD, c=np.log10(u17.Re), cmap="viridis", s=14, alpha=0.6, edgecolors="none")
lim = [0.004, 0.2]
ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
ax.plot(lim, [l * 1.2 for l in lim], ":", color="#b2182b", lw=1, alpha=0.7, label="±20%")
ax.plot(lim, [l * 0.8 for l in lim], ":", color="#b2182b", lw=1, alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Wind-tunnel $C_D$"); ax.set_ylabel("NeuralFoil $C_D$"); ax.set_title("Drag parity (log)")
clean(ax); ax.legend(fontsize=9, loc="upper left")
fig.colorbar(sc, ax=axes.ravel().tolist(), label="log$_{10}$(Re)", fraction=0.025, pad=0.02)
save(fig, "17_multifoil_parity.png",
     f"NeuralFoil vs wind tunnel: {n_af} airfoils, {n_pt} clean measured points, Re 40k-500k")

# ═════════════════════════════════════════════════════════════════════════
# 18-19: replot_with_uncertainty.py
# ═════════════════════════════════════════════════════════════════════════
import json
EM = json.load(open(os.path.join(DATA, "error_model.json")))
BAND, CONF_LO = EM["LD_band_pct"] / 100.0, EM["validated_conf_range"][0]
A_NAME, B_NAME = "A_single_point", "B_robust"
COL18 = {A_NAME: C_A, B_NAME: C_B}
LABEL18 = {A_NAME: "A (single-point)", B_NAME: "B (robust)"}
_RE_AVAIL = sorted(grid.Re.unique())
PLOT_RE = [min(_RE_AVAIL, key=lambda r: abs(r - t)) for t in (50e3, 200e3, 500e3)]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, Re in zip(axes, PLOT_RE):
    for name in (B_NAME, A_NAME):
        s = grid[(grid.airfoil == name) & (grid.Re == Re)].sort_values("AoA")
        if s.empty:
            continue
        ld, aoa = s.L_over_D.to_numpy(), s.AoA.to_numpy()
        conf_med = s.analysis_confidence.median()
        floor = conf_med < CONF_LO
        tag = "below validated conf" if floor else "validated band"
        ax.plot(aoa, ld, "-o", color=COL18[name], lw=2, ms=4.5,
                label=f"{LABEL18[name]} (conf$\\approx${conf_med:.2f}, {tag})")
        lo, hi = ld * (1 - BAND), ld * (1 + BAND * 0.4)
        if floor:
            ax.fill_between(aoa, lo, hi, facecolor="none", hatch="////",
                            edgecolor=COL18[name], linewidth=0.0, alpha=0.55, zorder=1)
        else:
            ax.fill_between(aoa, lo, hi, color=COL18[name], alpha=0.18, linewidth=0, zorder=1)
    ax.set_title(f"Re = {Re/1e3:.0f}k"); ax.set_xlabel("AoA [deg]"); clean(ax)
axes[0].set_ylabel("L / D"); axes[0].legend(fontsize=8, loc="upper left")
save(fig, "18_LD_vs_AoA_uncertainty.png",
     f"Airfoil A vs B -- L/D with measured uncertainty (±{BAND*100:.0f}%)")

fig, ax = plt.subplots(figsize=(8.5, 6.5))
peak, worst, lam = fam.peak_LD.to_numpy(), fam.worst_case_LD.to_numpy(), fam.lam.to_numpy()
def asym_err(vals):
    return np.vstack([vals * BAND, vals * BAND * 0.4])
ax.errorbar(peak, worst, xerr=asym_err(peak), yerr=asym_err(worst), fmt="none",
            ecolor="#999999", elinewidth=1, capsize=3, alpha=0.75, zorder=1,
            label=f"±{BAND*100:.0f}% measured band (one-sided)")
sc = ax.scatter(peak, worst, c=lam, cmap="viridis", s=95, zorder=3, edgecolors="k", linewidths=0.5)
plt.colorbar(sc, ax=ax, label="$\\lambda$ (1 = peak-seeking A, 0 = robust B)")
for x, y, l in zip(peak, worst, lam):
    ax.annotate(f"$\\lambda$={l:.1f}", (x, y), fontsize=7.5, xytext=(5, 4), textcoords="offset points")
ax.scatter([peak[lam == 1.0][0]], [worst[lam == 1.0][0]], s=250, facecolors="none",
           edgecolors=C_A, linewidths=2.2, zorder=4, label="A (conf$\\approx$0 -- unvalidated)")
ax.scatter([peak[lam == 0.0][0]], [worst[lam == 0.0][0]], s=250, facecolors="none",
           edgecolors=C_B, linewidths=2.2, zorder=4, label="B (robust)")
ax.set_xlabel("Peak L/D (surrogate, optimistic ceiling)")
ax.set_ylabel("Worst-case L/D over envelope")
clean(ax); ax.legend(fontsize=8.5, loc="center right")
save(fig, "19_tradeoff_uncertainty.png",
     "Peak-vs-robustness tradeoff with measured uncertainty")

# ═════════════════════════════════════════════════════════════════════════
# 20: uncertainty_aware_design.py
# ═════════════════════════════════════════════════════════════════════════
sweep = pd.read_csv(os.path.join(DATA, "uncertainty_aware_sweep.csv"))
wc_lo, wc_hi = sweep.w_conf.min(), sweep.w_conf.max()
co_lo = pd.read_csv(os.path.join(DATA, f"uncertainty_aware_wconf{wc_lo:.0f}_coords.csv"))
co_hi = pd.read_csv(os.path.join(DATA, f"uncertainty_aware_wconf{wc_hi:.0f}_coords.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(sweep.mean_conf, sweep.worst_LD, "-o", color=C_A, lw=2.2, ms=8)
for _, r in sweep.iterrows():
    ax1.annotate(f"w={r.w_conf:g}", (r.mean_conf, r.worst_LD), xytext=(6, 5),
                 textcoords="offset points", fontsize=9)
ax1.set_xlabel("Mean NeuralFoil confidence (how much you can trust the number)")
ax1.set_ylabel("Worst-case L/D (predicted performance)")
ax1.set_title("The trust-vs-performance trade\nturning up w_conf buys trustworthiness", fontsize=10.5)
clean(ax1)

ax2.plot(co_lo.x, co_lo.y, color=C_B, lw=2.2, label=f"w_conf={wc_lo:.0f} (trust blindly)")
ax2.plot(co_hi.x, co_hi.y, color="#1a9850", lw=2.2, label=f"w_conf={wc_hi:.0f} (trust-aware)")
ax2.set_aspect("equal"); clean(ax2); ax2.legend(fontsize=9)
ax2.set_title("Blind vs trust-aware optimal shapes", fontsize=10.5)
ax2.set_xlabel("x / c")

save(fig, "20_trust_vs_performance.png",
     "Uncertainty-aware airfoil optimization: using the measured\n"
     "confidence-error link inside the design loop")

# ═════════════════════════════════════════════════════════════════════════
# 21-23: uiuc_neuralfoil_validation.py  (22-airfoil benchmark)
# ═════════════════════════════════════════════════════════════════════════
from scipy import stats as _stats
ba = pd.read_csv(os.path.join(DATA, "uiuc_validation_by_airfoil.csv")).sort_values("mean_abs_err_CD")

# (21) Error by airfoil
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
y = np.arange(len(ba))
bar_c = [C_NF_L if ok else "#bbbbbb" for ok in ba.fit_ok]
axes[0].barh(y, ba.mean_abs_err_CD * 100, color=bar_c, alpha=0.85)
axes[0].set_xlabel("mean |$\\Delta C_D$ / $C_D$|  [%]"); axes[0].set_title("Drag error by airfoil")
axes[1].barh(y, ba.mean_abs_dCL, color=[C_NF_XL if ok else "#bbbbbb" for ok in ba.fit_ok], alpha=0.85)
axes[1].set_xlabel("mean |$\\Delta C_L$|"); axes[1].set_title("Lift error by airfoil")
for ax in axes:
    clean(ax)
axes[0].set_yticks(y)
axes[0].set_yticklabels([f"{a}  (n={int(n)})" + ("" if ok else "  [poor Kulfan fit]") for a, n, ok in zip(ba.asb_name, ba.n, ba.fit_ok)], fontsize=9)
save(fig, "21_uiuc_error_by_airfoil.png",
     "NeuralFoil (large) vs wind tunnel, clean runs, by airfoil")

# (22) What analysis_confidence tracks: binned calibration + airfoil level
cb = pd.read_csv(os.path.join(DATA, "uiuc_validation_confidence_bins.csv"))
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
x = np.arange(len(cb))
axes[0].bar(x, cb.mean_abs_dCL, color=C_NF_L, alpha=0.85)
axes[0].set_ylabel("mean |$\\Delta C_L$|"); axes[0].set_title("Lift error by confidence bin")
axes[1].bar(x, cb.mean_abs_err_CD * 100, color=C_NF_XL, alpha=0.85)
axes[1].set_ylabel("mean |$\\Delta C_D$ / $C_D$|  [%]"); axes[1].set_title("Drag error by confidence bin")
for ax in axes[:2]:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={int(n)}" for b, n in zip(cb.conf_bin, cb.n)], fontsize=8.5)
    ax.set_xlabel("analysis_confidence"); clean(ax)
ax = axes[2]
bo = ba[ba.fit_ok]
ax.scatter(bo.mean_conf, bo.mean_abs_err_CD * 100, s=55, color=C_A, edgecolors="k", linewidths=0.5)
for _, r in bo.iterrows():
    ax.annotate(r.asb_name, (r.mean_conf, r.mean_abs_err_CD * 100), fontsize=7.5,
                xytext=(4, 3), textcoords="offset points")
rho = _stats.spearmanr(bo.mean_conf, bo.mean_abs_err_CD).correlation
ax.set_xlabel("mean confidence for the airfoil"); ax.set_ylabel("mean |$\\Delta C_D$ / $C_D$|  [%]")
ax.set_title(f"Airfoil level (Spearman rho = {rho:+.2f})"); clean(ax)
save(fig, "22_uiuc_confidence_calibration.png",
     "What NeuralFoil's confidence actually tracks (clean runs, large model)")

# (23) Tripped runs: transition forced at the trip location, Vol 4 airfoils near Re = 200k
tr = uv[(uv.volume == "vol4") & (uv.model == "large")]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
for ax, name in zip(axes, ["e387", "sd2030", "fx63137"]):
    d = tr[tr.asb_name == name]
    rc = min(d[d.config == "clean"].Re.unique(), key=lambda r: abs(r - 200e3))
    rt = min(d[d.config == "tripped"].Re.unique(), key=lambda r: abs(r - 200e3))
    c = d[(d.config == "clean") & (d.Re == rc) & (d.NF_mode == "free")].sort_values("alpha")
    tf = d[(d.config == "tripped") & (d.Re == rt) & (d.NF_mode == "free")].sort_values("alpha")
    tx = d[(d.config == "tripped") & (d.Re == rt) & (d.NF_mode == "forced")].sort_values("alpha")
    ax.scatter(c.alpha, c.WT_CD, marker="D", s=30, color=C_WT, zorder=5, label="wind tunnel, clean")
    ax.scatter(tf.alpha, tf.WT_CD, marker="s", s=30, color="#762a83", zorder=5, label="wind tunnel, tripped")
    ax.plot(c.alpha, c.NF_CD, "-", color=C_NF_L, lw=1.8, label="NeuralFoil, free transition")
    ax.plot(tx.alpha, tx.NF_CD, "--", color="#762a83", lw=1.8, label="NeuralFoil, transition forced at trip")
    ax.set_yscale("log"); ax.set_title(f"{name.upper()}   Re $\\approx$ {rc/1e3:.0f}k")
    ax.set_xlabel("AoA [deg]"); clean(ax)
axes[0].set_ylabel("$C_D$ (log)"); axes[0].legend(fontsize=8)
save(fig, "23_uiuc_tripped_vs_forced_transition.png",
     "Boundary-layer trip runs: NeuralFoil with transition forced at the trip location")

print("\nAll figures rebuilt with a single consistent style.")
