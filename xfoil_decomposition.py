"""
Where does NeuralFoil's error come from: XFoil's physics, or the network?

NeuralFoil is trained to reproduce XFoil, so its disagreement with a wind
tunnel is the sum of two parts: XFoil's own disagreement with the tunnel, and
the network's disagreement with XFoil. Only the second part is the network's.
This script runs a headless XFoil 6.99 (see THIRD_PARTY_XFOIL.md) on the same
17-parameter Kulfan geometry NeuralFoil saw, at every clean measured condition
of both wind-tunnel benchmarks, and splits the error three ways:

    NeuralFoil vs tunnel   (the number the paper reports elsewhere)
    XFoil      vs tunnel   (the physics model's own error)
    NeuralFoil vs XFoil    (the emulation error, the only part that is the network's)

It also asks what NeuralFoil's confidence score tracks: its distance from XFoil
(what it was trained on) or its distance from the experiment.

XFoil settings match the NeuralFoil run: viscous, n_crit = 9, free transition,
the same Reynolds number, the same angles. Each polar is swept from 0 deg
outward in 0.5 deg steps with the measured angles inserted, so every point is
warm-started from its neighbour (XFoil's viscous solver needs that at low Re).
Points where XFoil fails to converge are kept as NaN and counted.

Inputs  : data/uiuc_neuralfoil_validation.csv, data/soartech8_neuralfoil_validation.csv,
          data/soartech8_kulfan_fit.csv, data/soartech8_measured_coords.csv,
          data/soartech8_design_coords.csv
Outputs : data/xfoil_decomposition.csv             every clean point, three-way
          data/xfoil_decomposition_by_Re.csv       by tunnel and Re bin
          data/xfoil_decomposition_summary.csv     by tunnel and pooled
          data/xfoil_decomposition_confidence.csv  confidence bins, three errors
"""

import os
import subprocess
import tempfile
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
XFOIL = os.path.join(OUT, ".venv", "bin", "xfoil")
N_CRIT = 9
N_PROC = max(1, min(8, (os.cpu_count() or 2) - 2))
RE_ORDER = ["40-60k", "60k", "100k", "150k", "200k", "300-400k", "300k", "400-500k"]
CONF_BINS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0000001]


# ─────────────────────────────────────────────────────────────────────────────
# XFoil driver (one fresh process per sweep, one temp dir per polar)
# ─────────────────────────────────────────────────────────────────────────────
def _sweep(d, Re, alphas, tag):
    pol = f"polar_{tag}.txt"
    cmds = (["PLOP", "G F", "",
             "LOAD af.dat", "PANE",
             "OPER", f"VISC {Re:.0f}", "ITER 300",
             "VPAR", f"N {N_CRIT}", "",
             "PACC", pol, ""]
            + [f"ALFA {a:.3f}" for a in alphas]
            + ["PACC", "", "QUIT"])
    try:
        subprocess.run([XFOIL], input="\n".join(cmds) + "\n",
                       capture_output=True, text=True, timeout=900, cwd=d)
    except Exception:
        pass
    rows = []
    path = os.path.join(d, pol)
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) >= 5:
                try:
                    rows.append((float(p[0]), float(p[1]), float(p[2])))
                except ValueError:
                    pass
    return rows


def run_polar(job):
    """job = (key, coords, Re, alphas). Returns (key, {alpha: (CL, CD)})."""
    key, coords, Re, alphas = job
    alphas = np.asarray(alphas, float)
    up = np.unique(np.round(np.concatenate([np.arange(0.0, alphas.max() + 0.25, 0.5),
                                            alphas[alphas >= 0]]), 3))
    down = np.unique(np.round(np.concatenate([np.arange(-0.5, alphas.min() - 0.25, -0.5),
                                              alphas[alphas < 0]]), 3))[::-1]
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "af.dat"), "w") as f:
            f.write("kulfan\n")
            for x, y in coords:
                f.write(f"{x:.6f} {y:.6f}\n")
        rows = _sweep(d, Re, up, "up")
        if len(down):
            rows += _sweep(d, Re, down, "down")
    out = {}
    for a, cl, cd in rows:
        for a0 in alphas:
            if abs(a - a0) < 2e-3:
                out[float(a0)] = (cl, cd)
    return key, out


def panel_coords(kf):
    import aerosandbox as asb
    return asb.Airfoil(coordinates=kf.coordinates).repanel(n_points_per_side=80).coordinates


# ─────────────────────────────────────────────────────────────────────────────
# Build the point table and the job list
# ─────────────────────────────────────────────────────────────────────────────
def load_points():
    import aerosandbox as asb
    u = pd.read_csv(os.path.join(DATA, "uiuc_neuralfoil_validation.csv"))
    u = u[(u.config == "clean") & (u.NF_mode == "free") & (u.model == "large") & u.fit_ok].copy()
    u["tunnel"] = "UIUC"
    u["airfoil"] = u.asb_name
    u["polar"] = u.asb_name + "|" + u.file + "|" + u.Re.astype(str)
    u["geom_key"] = u.asb_name

    s = pd.read_csv(os.path.join(DATA, "soartech8_neuralfoil_validation.csv"))
    s = s[(s.config == "clean") & (s.NF_mode == "free") & (s.nf == "large") & s.primary & s.fit_ok].copy()
    s["tunnel"] = "Princeton"
    s["airfoil"] = s.family
    s["polar"] = s.airfoil_label + "|" + s.Re.astype(str)
    s["geom_key"] = s.model

    cols = ["tunnel", "airfoil", "polar", "geom_key", "Re", "Re_bin", "alpha",
            "WT_CL", "WT_CD", "NF_CL", "NF_CD", "NF_conf", "dCL", "err_CD"]
    pts = pd.concat([u[cols], s[cols]], ignore_index=True)

    # Geometries: exactly what the two validation scripts handed to NeuralFoil.
    geoms = {}
    for name in sorted(u.asb_name.unique()):
        geoms[("UIUC", name)] = panel_coords(asb.Airfoil(name).to_kulfan_airfoil())
    fit = pd.read_csv(os.path.join(DATA, "soartech8_kulfan_fit.csv")).set_index("model")
    mc = pd.read_csv(os.path.join(DATA, "soartech8_measured_coords.csv"))
    dc = pd.read_csv(os.path.join(DATA, "soartech8_design_coords.csv"))
    for model in sorted(s.model.unique()):
        r = fit.loc[model]
        if r.primary_source == "measured":
            pts_xy = mc[mc.dap_name == r.dap_name][["x", "y"]].values
        else:
            pts_xy = dc[dc.cor_name == r.cor_name][["x", "y"]].values
        kf = asb.Airfoil(name=model, coordinates=np.asarray(pts_xy, float)).normalize().to_kulfan_airfoil()
        geoms[("Princeton", model)] = panel_coords(kf)
    return pts, geoms


def summarise(d):
    ok = d.dropna(subset=["XF_CD"])
    r = dict(n=len(d), n_xfoil_converged=len(ok), frac_converged=len(ok) / max(len(d), 1),
             mean_conf=d.NF_conf.mean())
    for tag, col in [("NF_WT", "err_CD_NF_WT"), ("XF_WT", "err_CD_XF_WT"), ("NF_XF", "err_CD_NF_XF")]:
        r[f"mean_abs_errCD_{tag}"] = ok[col].abs().mean()
        r[f"median_abs_errCD_{tag}"] = ok[col].abs().median()
        r[f"bias_CD_{tag}"] = ok[col].mean()
    for tag, col in [("NF_WT", "dCL_NF_WT"), ("XF_WT", "dCL_XF_WT"), ("NF_XF", "dCL_NF_XF")]:
        r[f"mean_abs_dCL_{tag}"] = ok[col].abs().mean()
        r[f"bias_CL_{tag}"] = ok[col].mean()
    r["frac_NF_closer_than_XF"] = (ok.err_CD_NF_WT.abs() < ok.err_CD_XF_WT.abs()).mean()
    # share of NF's error already present in XFoil: how much of the NF-vs-WT signed
    # error is explained by the XF-vs-WT signed error (slope-free R^2 of a 1:1 line)
    if len(ok) > 10:
        r["corr_signed_NF_WT_vs_XF_WT"] = np.corrcoef(ok.err_CD_NF_WT, ok.err_CD_XF_WT)[0, 1]
        r["corr_conf_abs_NF_XF"] = np.corrcoef(ok.NF_conf, ok.err_CD_NF_XF.abs())[0, 1]
        r["corr_conf_abs_NF_WT"] = np.corrcoef(ok.NF_conf, ok.err_CD_NF_WT.abs())[0, 1]
        r["corr_conf_abs_XF_WT"] = np.corrcoef(ok.NF_conf, ok.err_CD_XF_WT.abs())[0, 1]
    return pd.Series(r)


def main():
    pts, geoms = load_points()
    print(f"{len(pts)} clean points, {pts.polar.nunique()} polars, {len(geoms)} geometries; "
          f"XFoil on {N_PROC} processes")

    jobs = []
    for polar, g in pts.groupby("polar"):
        tunnel = g.tunnel.iloc[0]
        jobs.append((polar, geoms[(tunnel, g.geom_key.iloc[0])], float(g.Re.iloc[0]),
                     np.sort(g.alpha.unique())))
    results = {}
    with Pool(N_PROC) as pool:
        for i, (key, out) in enumerate(pool.imap_unordered(run_polar, jobs, chunksize=2), 1):
            results[key] = out
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} polars done", flush=True)

    xf_cl, xf_cd = [], []
    for _, r in pts.iterrows():
        v = results.get(r.polar, {}).get(float(r.alpha))
        xf_cl.append(v[0] if v else np.nan)
        xf_cd.append(v[1] if v else np.nan)
    pts["XF_CL"] = xf_cl
    pts["XF_CD"] = xf_cd
    pts["xf_converged"] = pts.XF_CD.notna()
    pts["err_CD_NF_WT"] = (pts.NF_CD - pts.WT_CD) / pts.WT_CD
    pts["err_CD_XF_WT"] = (pts.XF_CD - pts.WT_CD) / pts.WT_CD
    pts["err_CD_NF_XF"] = (pts.NF_CD - pts.XF_CD) / pts.XF_CD
    pts["dCL_NF_WT"] = pts.NF_CL - pts.WT_CL
    pts["dCL_XF_WT"] = pts.XF_CL - pts.WT_CL
    pts["dCL_NF_XF"] = pts.NF_CL - pts.XF_CL
    pts.to_csv(os.path.join(DATA, "xfoil_decomposition.csv"), index=False)
    print(f"\nXFoil converged at {pts.xf_converged.sum()} of {len(pts)} points "
          f"({100 * pts.xf_converged.mean():.1f}%)")

    by_re = pts.groupby(["tunnel", "Re_bin"]).apply(summarise).reset_index()
    by_re["Re_order"] = by_re.Re_bin.map({b: i for i, b in enumerate(RE_ORDER)})
    by_re = by_re.sort_values(["tunnel", "Re_order"]).drop(columns="Re_order")
    by_re.to_csv(os.path.join(DATA, "xfoil_decomposition_by_Re.csv"), index=False)

    summ = pd.concat([pts.groupby("tunnel").apply(summarise),
                      summarise(pts).to_frame("pooled").T]).rename_axis("tunnel").reset_index()
    summ.to_csv(os.path.join(DATA, "xfoil_decomposition_summary.csv"), index=False)

    att = pts[pts.WT_CL.abs() > 0.1].dropna(subset=["XF_CD"]).copy()
    att["conf_bin"] = pd.cut(att.NF_conf, CONF_BINS)
    cb = att.groupby("conf_bin", observed=True).apply(summarise).reset_index()
    cb.to_csv(os.path.join(DATA, "xfoil_decomposition_confidence.csv"), index=False)

    pd.set_option("display.width", 200)
    show = ["tunnel", "Re_bin", "n", "frac_converged", "mean_abs_errCD_NF_WT", "mean_abs_errCD_XF_WT",
            "mean_abs_errCD_NF_XF", "bias_CD_NF_WT", "bias_CD_XF_WT", "bias_CD_NF_XF", "frac_NF_closer_than_XF"]
    print("\nDrag error three ways, by tunnel and Re bin (converged points):")
    print(by_re[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSummary:")
    print(summ.T.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nConfidence bins (attached flow, converged):")
    print(cb[["conf_bin", "n", "mean_abs_errCD_NF_WT", "mean_abs_errCD_XF_WT", "mean_abs_errCD_NF_XF",
              "mean_abs_dCL_NF_WT", "mean_abs_dCL_NF_XF"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
