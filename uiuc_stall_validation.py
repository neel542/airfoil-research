"""
uiuc_stall_validation.py
=============================================================================
Lift curves through stall: NeuralFoil against the UIUC lift runs.

The archive's lift files (data/uiuc_experimental_lift.csv, from
uiuc_lsat_parse.py) sweep alpha well past maximum lift, stepping up and then
back down, on the same 22 airfoils as the drag benchmark. That makes them a
test of three things the drag runs cannot check and that the multi-objective
"safety" objective in this project relies on:

  * maximum lift coefficient (CLmax) and the angle it occurs at;
  * the lift curve before and after stall;
  * the pitching-moment coefficient CM, which NeuralFoil also predicts.

Stall hysteresis (different lift on the way up and the way down) is a real
low-Re effect that a steady model cannot represent; its size is reported as
a measured quantity. CLmax is taken from the increasing-alpha sweep, the
conventional definition. Tripped runs are compared with transition forced at
the trip locations.

Outputs
  data/uiuc_lift_validation.csv      point by point (increasing sweep)
  data/uiuc_stall_validation.csv     one row per (file, Re): CLmax, stall angle, errors
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import aerosandbox as asb

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
MODEL = "large"
N_CRIT = 9.0

lf = pd.read_csv(os.path.join(DATA, "uiuc_experimental_lift.csv"))
fit = pd.read_csv(os.path.join(DATA, "uiuc_kulfan_fit.csv")).set_index("asb_name")
geoms = {}
for name in sorted(lf.asb_name.unique()):
    try:
        geoms[name] = asb.Airfoil(name).to_kulfan_airfoil()
    except Exception as e:
        print(f"  {name}: geometry failed ({type(e).__name__})")
print(f"{len(lf)} lift points, {lf.file.nunique()} files, {len(geoms)} airfoils")

rows, pts = [], []
for (file, Re), g in lf.groupby(["file", "Re"]):
    name, cfg = g.asb_name.iloc[0], g.config.iloc[0]
    if name not in geoms:
        continue
    xu = g.xtr_upper.iloc[0] if cfg == "tripped" else 1.0
    xl = g.xtr_lower.iloc[0] if cfg == "tripped" else 1.0
    xu, xl = (xu if np.isfinite(xu) else 1.0), (xl if np.isfinite(xl) else 1.0)
    up = g[g.sweep == "up"].sort_values("alpha").drop_duplicates("alpha")
    dn = g[g.sweep == "down"].sort_values("alpha").drop_duplicates("alpha")
    if len(up) < 5:
        continue
    i = up.CL.idxmax()
    wt_clmax, wt_a = float(up.loc[i, "CL"]), float(up.loc[i, "alpha"])
    captured = bool(up.alpha.max() >= wt_a + 1.5)     # sweep went past CLmax

    a_fine = np.arange(up.alpha.min(), up.alpha.max() + 0.05, 0.25)
    nf = geoms[name].get_aero_from_neuralfoil(alpha=a_fine, Re=Re, model_size=MODEL,
                                              n_crit=N_CRIT, xtr_upper=xu, xtr_lower=xl)
    cl_f = np.atleast_1d(nf["CL"])
    j = int(cl_f.argmax())
    nf_clmax, nf_a = float(cl_f[j]), float(a_fine[j])

    nfp = geoms[name].get_aero_from_neuralfoil(alpha=up.alpha.to_numpy(float), Re=Re,
                                               model_size=MODEL, n_crit=N_CRIT,
                                               xtr_upper=xu, xtr_lower=xl)
    cl_p, cm_p, cf_p = (np.atleast_1d(nfp[k]) for k in ("CL", "CM", "analysis_confidence"))
    up = up.assign(NF_CL=cl_p, NF_CM=cm_p, NF_conf=cf_p)
    up["region"] = np.where(up.alpha <= wt_a - 2, "pre-stall",
                            np.where(up.alpha > wt_a, "post-stall", "near CLmax"))
    pre, post = up[up.region == "pre-stall"], up[up.region == "post-stall"]

    hyst = np.nan
    if len(dn) >= 4:
        lo, hi = max(up.alpha.min(), dn.alpha.min()), min(up.alpha.max(), dn.alpha.max())
        aa = up[(up.alpha >= lo) & (up.alpha <= hi)].alpha.to_numpy()
        if len(aa) >= 3:
            cl_dn = np.interp(aa, dn.alpha, dn.CL)
            cl_up = up[(up.alpha >= lo) & (up.alpha <= hi)].CL.to_numpy()
            hyst = float(np.abs(cl_up - cl_dn).max())

    rows.append(dict(volume=g.volume.iloc[0], file=file, asb_name=name, config=cfg,
                     fit_ok=bool(fit.loc[name, "fit_ok"]) if name in fit.index else False,
                     Re=int(Re), n_up=len(up), alpha_min=float(up.alpha.min()),
                     alpha_max=float(up.alpha.max()), stall_captured=captured,
                     WT_CLmax=wt_clmax, WT_alpha_CLmax=wt_a,
                     NF_CLmax=nf_clmax, NF_alpha_CLmax=nf_a,
                     dCLmax=nf_clmax - wt_clmax, dalpha_CLmax=nf_a - wt_a,
                     err_CL_prestall=float((pre.NF_CL - pre.CL).abs().mean()) if len(pre) else np.nan,
                     err_CL_poststall=float((post.NF_CL - post.CL).abs().mean()) if len(post) else np.nan,
                     err_CM_prestall=float((pre.NF_CM - pre.CM).abs().mean()) if len(pre) else np.nan,
                     conf_prestall=float(pre.NF_conf.mean()) if len(pre) else np.nan,
                     conf_poststall=float(post.NF_conf.mean()) if len(post) else np.nan,
                     hysteresis_dCL=hyst))
    for _, r in up.iterrows():
        pts.append(dict(volume=r.volume, file=file, asb_name=name, config=cfg, Re=int(Re),
                        alpha=r.alpha, WT_CL=r.CL, WT_CM=r.CM, NF_CL=r.NF_CL, NF_CM=r.NF_CM,
                        NF_conf=r.NF_conf, region=r.region))

st = pd.DataFrame(rows)
st["Re_bin"] = pd.cut(st.Re, [0, 45e3, 80e3, 150e3, 250e3, 400e3, 600e3],
                      labels=["30k", "40-60k", "100k", "200k", "300-350k", "460-500k"])
st.to_csv(os.path.join(DATA, "uiuc_stall_validation.csv"), index=False)
pd.DataFrame(pts).to_csv(os.path.join(DATA, "uiuc_lift_validation.csv"), index=False)
print(f"  wrote data/uiuc_stall_validation.csv ({len(st)} lift runs) and uiuc_lift_validation.csv ({len(pts)} points)")

b = st[st.fit_ok & (st.config == "clean") & st.stall_captured]
print(f"\nBENCHMARK: clean, Kulfan-fit OK, stall captured: {len(b)} runs, {b.asb_name.nunique()} airfoils, "
      f"Re {b.Re.min()}-{b.Re.max()}")
print(f"  CLmax:      mean dCLmax = {b.dCLmax.mean():+.3f}   mean |dCLmax| = {b.dCLmax.abs().mean():.3f}"
      f"   median |dCLmax| = {b.dCLmax.abs().median():.3f}   (relative {100*(b.dCLmax.abs()/b.WT_CLmax).mean():.1f}%)")
print(f"  stall angle: mean dalpha = {b.dalpha_CLmax.mean():+.2f} deg   mean |dalpha| = {b.dalpha_CLmax.abs().mean():.2f} deg"
      f"   median |dalpha| = {b.dalpha_CLmax.abs().median():.2f} deg")
print(f"  CL error:   pre-stall {b.err_CL_prestall.mean():.3f}   post-stall {b.err_CL_poststall.mean():.3f}"
      f"   | confidence pre-stall {b.conf_prestall.mean():.2f}, post-stall {b.conf_poststall.mean():.2f}")
print(f"  CM error, pre-stall: {b.err_CM_prestall.mean():.4f}   (measured CM magnitude ~{pd.DataFrame(pts).WT_CM.abs().mean():.3f})")
print(f"  measured hysteresis (max CL gap up vs down): median {b.hysteresis_dCL.median():.3f}, "
      f"90th pct {b.hysteresis_dCL.quantile(0.9):.3f}")

print("\nBy Reynolds-number bin (benchmark runs):")
print(b.groupby("Re_bin", observed=True).agg(runs=("Re", "size"), dCLmax=("dCLmax", "mean"),
      abs_dCLmax=("dCLmax", lambda x: x.abs().mean()), dalpha=("dalpha_CLmax", "mean"),
      abs_dalpha=("dalpha_CLmax", lambda x: x.abs().mean()), pre=("err_CL_prestall", "mean"),
      post=("err_CL_poststall", "mean"), hyst=("hysteresis_dCL", "median")).round(3).to_string())

print("\nBy airfoil (benchmark runs):")
print(b.groupby("asb_name").agg(runs=("Re", "size"), WT_CLmax=("WT_CLmax", "mean"), NF_CLmax=("NF_CLmax", "mean"),
      dCLmax=("dCLmax", "mean"), dalpha=("dalpha_CLmax", "mean"), pre=("err_CL_prestall", "mean"),
      post=("err_CL_poststall", "mean")).round(3).to_string())

t = st[st.fit_ok & (st.config == "tripped") & st.stall_captured]
if len(t):
    print(f"\nTripped runs (forced transition), {len(t)} runs: mean dCLmax = {t.dCLmax.mean():+.3f}, "
          f"|dCLmax| = {t.dCLmax.abs().mean():.3f}, |dalpha| = {t.dalpha_CLmax.abs().mean():.2f} deg")
print("\nDone.")
