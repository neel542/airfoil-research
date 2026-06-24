"""
Envelope-wide XFoil validation: a "trust map" for the surrogate.
================================================================

The single-point (Re=200k) check showed NeuralFoil agrees with true XFoil to
~5%. But low Re is where viscous separation makes both XFoil hard to converge
and NeuralFoil least confident. This sweeps the FULL envelope and quantifies the
surrogate-vs-XFoil error as a function of (Re, AoA) -- so we know where the
optimization model can be trusted and where finalists need real CFD/wind-tunnel.
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import aerosandbox as asb

OUT = "/Users/neelmadhav/Airfoil research"
XFOIL = os.path.join(OUT, ".venv", "bin", "xfoil")
RE_GRID = np.array([50e3, 100e3, 200e3, 350e3, 500e3])
ALPHAS = np.arange(0, 8.5, 1.0)


def run_xfoil_polar(coords, Re, alphas):
    """True XFoil viscous polar -> DataFrame(alpha, CL, CD, CM). Empty if none
    converge. (Headless build; PACC polar parsed directly.)"""
    af = asb.Airfoil(coordinates=coords).repanel(n_points_per_side=80)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "af.dat"), "w") as f:
            f.write("design\n")
            for x, y in af.coordinates:
                f.write(f"{x:.6f} {y:.6f}\n")
        cmds = (["PLOP", "G F", "", "LOAD af.dat", "PANE",
                 "OPER", f"VISC {Re:.0f}", "ITER 200",
                 "PACC", "polar.txt", ""]
                + [f"ALFA {a:.3f}" for a in alphas]
                + ["PACC", "", "QUIT"])
        try:
            subprocess.run([XFOIL], input="\n".join(cmds) + "\n",
                           capture_output=True, text=True, timeout=180, cwd=d)
        except Exception:
            return pd.DataFrame()
        pol = os.path.join(d, "polar.txt")
        if not os.path.exists(pol):
            return pd.DataFrame()
        rows = []
        for line in open(pol):
            p = line.split()
            if len(p) >= 5:
                try:
                    rows.append((float(p[0]), float(p[1]), float(p[2]), float(p[4])))
                except ValueError:
                    continue
        return pd.DataFrame(rows, columns=["alpha", "CL", "CD", "CM"])


designs = {
    "B_nominal": pd.read_csv(os.path.join(OUT, "data", "mfg_B_nominal_coords.csv")).values,
    "B_mfg": pd.read_csv(os.path.join(OUT, "data", "mfg_B_mfg_coords.csv")).values,
}

print("Sweeping the full Re envelope in XFoil ...")
rows = []
for name, coords in designs.items():
    af = asb.Airfoil(coordinates=coords)
    for Re in RE_GRID:
        xf = run_xfoil_polar(coords, Re, ALPHAS)
        nf_l = af.get_aero_from_neuralfoil(alpha=ALPHAS, Re=Re, model_size="large")
        nf_x = af.get_aero_from_neuralfoil(alpha=ALPHAS, Re=Re, model_size="xxlarge")
        conf = np.atleast_1d(nf_l["analysis_confidence"])
        for k, a in enumerate(ALPHAS):
            ld_l = float(np.atleast_1d(nf_l["CL"])[k] / np.atleast_1d(nf_l["CD"])[k])
            ld_x = float(np.atleast_1d(nf_x["CL"])[k] / np.atleast_1d(nf_x["CD"])[k])
            m = xf[np.isclose(xf.alpha, a, atol=0.05)] if not xf.empty else pd.DataFrame()
            ld_xf = float(m.CL.iloc[0] / m.CD.iloc[0]) if len(m) else np.nan
            rows.append(dict(design=name, Re=float(Re), alpha=float(a),
                             XFoil_LD=ld_xf, NF_large_LD=ld_l, NF_xxlarge_LD=ld_x,
                             NF_confidence=float(conf[k])))
        nconv = xf.shape[0] if not xf.empty else 0
        print(f"  {name:>10}  Re={Re/1e3:5.0f}k : XFoil converged {nconv}/{len(ALPHAS)}")

df = pd.DataFrame(rows)
df["pct_err_large"] = 100 * (df.NF_large_LD - df.XFoil_LD).abs() / df.XFoil_LD
df["pct_err_xxlarge"] = 100 * (df.NF_xxlarge_LD - df.XFoil_LD).abs() / df.XFoil_LD
df.to_csv(os.path.join(OUT, "data", "xfoil_validation_envelope.csv"), index=False)
print("  wrote data/xfoil_validation_envelope.csv")

print("\nSurrogate error vs true XFoil, by Re (mean over converged AoA):")
summ = (df.dropna(subset=["XFoil_LD"])
        .groupby(["design", "Re"])
        .agg(n_conv=("XFoil_LD", "size"),
             mean_err_large=("pct_err_large", "mean"),
             mean_err_xxlarge=("pct_err_xxlarge", "mean"),
             mean_conf=("NF_confidence", "mean"))
        .reset_index())
for _, r in summ.iterrows():
    print(f"  {r.design:>10}  Re={r.Re/1e3:5.0f}k : "
          f"|err| large={r.mean_err_large:5.1f}%  xxlarge={r.mean_err_xxlarge:5.1f}%  "
          f"(n={int(r.n_conv)}, NF-conf={r.mean_conf:.2f})")

# --------------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (10) Trust map: % |L/D error| NeuralFoil-large vs XFoil over (Re, AoA).
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, name in zip(axes, designs):
    d = df[df.design == name]
    piv = (d.pivot_table(index="alpha", columns="Re", values="pct_err_large")
           .reindex(index=ALPHAS, columns=RE_GRID))   # keep full grid; missing -> NaN
    data = piv.values
    im = ax.imshow(data, origin="lower", aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=20, extent=[0, len(piv.columns), 0, len(piv.index)])
    # gray-out non-converged (NaN) cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, hatch="xx",
                                           fill=True, fc="lightgray", ec="gray", lw=0))
            else:
                ax.text(j + 0.5, i + 0.5, f"{data[i,j]:.0f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xticks(np.arange(len(piv.columns)) + 0.5)
    ax.set_xticklabels([f"{c/1e3:.0f}k" for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)) + 0.5)
    ax.set_yticklabels([f"{i:.0f}" for i in piv.index])
    ax.set_xlabel("Re"); ax.set_ylabel("AoA [deg]")
    ax.set_title(f"{name}: |L/D error| NeuralFoil vs XFoil [%]\n(gray = XFoil did not converge)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="% L/D error")
fig.suptitle("Surrogate trust map across the operating envelope")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "10_trust_map.png"), dpi=160)
print("  wrote figures/10_trust_map.png")

# (11) L/D vs AoA across Re for B_mfg (the buildable design).
d = df[df.design == "B_mfg"]
fig, axes = plt.subplots(1, len(RE_GRID), figsize=(17, 3.8), sharey=True)
for ax, Re in zip(axes, RE_GRID):
    s = d[d.Re == Re].sort_values("alpha")
    ax.plot(s.alpha, s.NF_large_LD, "-o", color="#1f77b4", ms=3, label="NF large")
    ax.plot(s.alpha, s.NF_xxlarge_LD, "--s", color="#ff7f0e", ms=3, label="NF xxlarge")
    ax.plot(s.alpha, s.XFoil_LD, ":^", color="#2ca02c", ms=5, label="XFoil")
    ax.set_title(f"Re={Re/1e3:.0f}k"); ax.set_xlabel("AoA [deg]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("L/D"); axes[0].legend(fontsize=8)
fig.suptitle("B_mfg: surrogate vs true XFoil across Re")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "11_envelope_polars.png"), dpi=160)
print("  wrote figures/11_envelope_polars.png")
