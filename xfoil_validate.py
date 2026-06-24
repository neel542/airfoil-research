"""
True-XFoil validation of the finished designs.

Drives a headless XFoil binary directly (bypassing AeroSandbox's polar parser,
which is finicky about output-column formatting) and compares true viscous XFoil
polars against the NeuralFoil surrogate used for optimization. This is the
independent ground-truth check that closes the surrogate-confidence gap.
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import aerosandbox as asb

OUT = "/Users/neelmadhav/Airfoil research"
RE = 200e3
ALPHAS = np.arange(0, 9.5, 1.0)
XFOIL = os.path.join(OUT, ".venv", "bin", "xfoil")


def run_xfoil_polar(coords, Re, alphas):
    """Return DataFrame(alpha, CL, CD, CM) from a true XFoil viscous sweep.

    XFoil quirks handled: panel-node limit (repanel to ~160 pts), the airfoil
    name prompt (write a name header line), and inline path parsing (run from
    the temp dir with a relative filename). The alpha sweep is sequential so
    each point warm-starts from the previous -- best for viscous convergence.
    """
    af = asb.Airfoil(coordinates=coords).repanel(n_points_per_side=80)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "af.dat"), "w") as f:
            f.write("design\n")
            for x, y in af.coordinates:
                f.write(f"{x:.6f} {y:.6f}\n")
        cmds = (["PLOP", "G F", "",                 # graphics off (headless)
                 "LOAD af.dat", "PANE",
                 "OPER", f"VISC {Re:.0f}", "ITER 200",
                 "PACC", "polar.txt", ""]
                + [f"ALFA {a:.3f}" for a in alphas]
                + ["PACC", "", "QUIT"])
        try:
            subprocess.run([XFOIL], input="\n".join(cmds) + "\n",
                           capture_output=True, text=True, timeout=180, cwd=d)
        except Exception as e:
            print("  xfoil call failed:", e); return pd.DataFrame()
        pol = os.path.join(d, "polar.txt")
        if not os.path.exists(pol):
            return pd.DataFrame()
        rows = []
        for line in open(pol):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append((float(parts[0]), float(parts[1]),
                                 float(parts[2]), float(parts[4])))
                except ValueError:
                    continue   # header / separator lines
        return pd.DataFrame(rows, columns=["alpha", "CL", "CD", "CM"])


designs = {
    "B_nominal": pd.read_csv(os.path.join(OUT, "data", "mfg_B_nominal_coords.csv")).values,
    "B_mfg": pd.read_csv(os.path.join(OUT, "data", "mfg_B_mfg_coords.csv")).values,
}

print(f"Validating in XFoil at Re={RE:.0f} ...")
all_rows = []
for name, coords in designs.items():
    af = asb.Airfoil(coordinates=coords)
    xf = run_xfoil_polar(coords, RE, ALPHAS)
    nf_l = af.get_aero_from_neuralfoil(alpha=ALPHAS, Re=RE, model_size="large")
    nf_x = af.get_aero_from_neuralfoil(alpha=ALPHAS, Re=RE, model_size="xxlarge")
    for k, a in enumerate(ALPHAS):
        row = dict(design=name, alpha=float(a),
                   NF_large_LD=float(np.atleast_1d(nf_l["CL"])[k] / np.atleast_1d(nf_l["CD"])[k]),
                   NF_xxlarge_LD=float(np.atleast_1d(nf_x["CL"])[k] / np.atleast_1d(nf_x["CD"])[k]))
        m = xf[np.isclose(xf.alpha, a, atol=0.05)] if not xf.empty else pd.DataFrame()
        row["XFoil_LD"] = float(m.CL.iloc[0] / m.CD.iloc[0]) if len(m) else np.nan
        all_rows.append(row)
    n_conv = xf.shape[0] if not xf.empty else 0
    print(f"  {name}: XFoil converged at {n_conv}/{len(ALPHAS)} alphas")

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(OUT, "data", "xfoil_validation.csv"), index=False)
print("  wrote data/xfoil_validation.csv")

# Quantify the surrogate gap vs ground truth
for name in designs:
    d = df[(df.design == name)].dropna(subset=["XFoil_LD"])
    if len(d):
        gap_l = (d.NF_large_LD - d.XFoil_LD).abs().mean()
        gap_x = (d.NF_xxlarge_LD - d.XFoil_LD).abs().mean()
        mean_xf = d.XFoil_LD.mean()
        print(f"  {name}: NeuralFoil vs true XFoil  |large|={gap_l:.1f} "
              f"({100*gap_l/mean_xf:.0f}%)  |xxlarge|={gap_x:.1f} ({100*gap_x/mean_xf:.0f}%)  "
              f"[mean XFoil L/D={mean_xf:.0f}]")

# Figure: surrogate vs ground truth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, name in zip(axes, designs):
    d = df[df.design == name].sort_values("alpha")
    ax.plot(d.alpha, d.NF_large_LD, "-o", color="#1f77b4", ms=4, label="NeuralFoil large (opt model)")
    ax.plot(d.alpha, d.NF_xxlarge_LD, "--s", color="#ff7f0e", ms=4, label="NeuralFoil xxlarge")
    ax.plot(d.alpha, d.XFoil_LD, ":^", color="#2ca02c", ms=6, label="XFoil (ground truth)")
    ax.set_title(f"{name}  @ Re=200k"); ax.set_xlabel("AoA [deg]"); ax.grid(alpha=0.3)
axes[0].set_ylabel("L / D"); axes[0].legend()
fig.suptitle("Fidelity validation: NeuralFoil surrogate vs true XFoil")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "9_fidelity_check.png"), dpi=160)
print("  updated figures/9_fidelity_check.png with true XFoil")
