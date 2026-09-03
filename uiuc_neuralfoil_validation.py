"""
uiuc_neuralfoil_validation.py
=============================================================================
Systematic check of NeuralFoil against measured wind-tunnel polars from the
UIUC Low-Speed Airfoil Tests archive: 57 airfoils (Vols 1-4), Re 40k-500k, clean and
boundary-layer-tripped, 5,441 measured drag-run points
(data/uiuc_experimental.csv, built by uiuc_lsat_parse.py).

For every measured (airfoil, configuration, Re, alpha):
  * NeuralFoil `large` and `xxlarge` at the same condition, from the Kulfan
    fit of the design coordinates in the AeroSandbox airfoil database;
  * clean runs   -> free transition (n_crit = 9, xtr_upper = xtr_lower = 1);
  * tripped runs -> transition forced at the trip locations given in the
    file header (xtr_upper / xtr_lower), AND free transition, so the effect
    of modelling the trip is visible.
Then: error by airfoil, by Reynolds number and by configuration; how well
NeuralFoil's analysis_confidence tracks its own error (absolute and
relative, point level and airfoil level, with a calibration table); and the
Kulfan-fit geometry error for every airfoil.

Outputs
  data/uiuc_neuralfoil_validation.csv         point by point
  data/uiuc_kulfan_fit.csv                    geometry-fit error per airfoil
  data/uiuc_validation_by_airfoil.csv         clean runs, NeuralFoil large
  data/uiuc_validation_by_Re.csv              clean runs, both models
  data/uiuc_validation_tripped.csv            tripped runs, free vs forced
  data/uiuc_validation_confidence_bins.csv    calibration table
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import aerosandbox as asb
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
N_CRIT = 9.0          # standard for the UIUC tunnel (turbulence ~0.1%)
MODELS = ["large", "xxlarge"]
ATTACHED = 0.1        # |CL| threshold for relative-CL statistics

exp = pd.read_csv(os.path.join(DATA, "uiuc_experimental.csv"))
print(f"{len(exp)} measured points, {exp.asb_name.nunique()} airfoils, "
      f"Re {exp.Re.min()}-{exp.Re.max()}")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: Kulfan fit of every airfoil, with its fit error
# ─────────────────────────────────────────────────────────────────────────────
def kulfan_fit(name):
    base = asb.Airfoil(name)
    kf = base.to_kulfan_airfoil()
    devs = []
    for surf in ("upper", "lower"):
        oc = np.asarray(getattr(base, f"{surf}_coordinates")())
        x = np.clip(oc[:, 0], 0.0, 1.0)          # a few database files overshoot [0, 1] slightly
        yk = np.asarray(getattr(kf, f"{surf}_coordinates")(x_over_c=x))[:, 1]
        ok = np.isfinite(yk)
        devs.append((oc[:, 1] - yk)[ok])
    d = np.concatenate(devs)
    return kf, dict(asb_name=name, n_coords=int(base.coordinates.shape[0]),
                    kulfan_rms_pct=float(np.sqrt(np.mean(d ** 2)) * 100),
                    kulfan_max_pct=float(np.abs(d).max() * 100),
                    max_thickness=float(kf.max_thickness()),
                    max_camber=float(kf.max_camber()))


print("\nKulfan fit of each airfoil (RMS / max, % chord):")
geoms, fit_rows = {}, []
for name in sorted(exp.asb_name.unique()):
    try:
        kf, row = kulfan_fit(name)
    except Exception as e:
        print(f"  {name:>9}: geometry failed ({type(e).__name__}); skipped")
        continue
    geoms[name] = kf
    fit_rows.append(row)
    print(f"  {name:>9}: {row['kulfan_rms_pct']:.3f} / {row['kulfan_max_pct']:.3f}"
          f"   t={row['max_thickness']*100:4.1f}%  camber={row['max_camber']*100:4.1f}%")
fit = pd.DataFrame(fit_rows)
# The 17-parameter Kulfan basis cannot reproduce every shape. Where its fit
# error exceeds the project's manufacturing sigma (0.5% chord), any
# NeuralFoil-vs-experiment gap is geometry-limited, so those airfoils are
# listed but kept out of the benchmark statistics.
FIT_OK_PCT = 0.5
fit["fit_ok"] = fit.kulfan_rms_pct < FIT_OK_PCT
OK = set(fit[fit.fit_ok].asb_name)
print(f"  benchmark set: {len(OK)} of {len(fit)} airfoils with Kulfan RMS < {FIT_OK_PCT}% chord; "
      f"geometry-limited: {sorted(set(fit.asb_name) - OK)}")
fit.to_csv(os.path.join(DATA, "uiuc_kulfan_fit.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# NeuralFoil at every measured condition
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning NeuralFoil at every measured condition ...")
rows = []
for (vol, file, Re), g in exp.groupby(["volume", "file", "Re"]):
    g = g.sort_values("alpha")
    name, cfg = g.asb_name.iloc[0], g.config.iloc[0]
    if name not in geoms:
        continue
    alphas = g.alpha.to_numpy(float)
    modes = [("free", 1.0, 1.0)]
    if cfg == "tripped":
        xu, xl = g.xtr_upper.iloc[0], g.xtr_lower.iloc[0]
        modes.append(("forced", xu if np.isfinite(xu) else 1.0, xl if np.isfinite(xl) else 1.0))
    for model in MODELS:
        for mode, xu, xl in modes:
            aero = geoms[name].get_aero_from_neuralfoil(
                alpha=alphas, Re=Re, model_size=model, n_crit=N_CRIT,
                xtr_upper=xu, xtr_lower=xl)
            cl, cd, cf = (np.atleast_1d(aero[k]) for k in ("CL", "CD", "analysis_confidence"))
            for k, (_, r) in enumerate(g.iterrows()):
                rows.append(dict(volume=r.volume, file=file, asb_name=name,
                                 airfoil_label=r.airfoil_label, config=cfg, NF_mode=mode,
                                 xtr_upper=xu, xtr_lower=xl, Re=int(Re), alpha=float(r.alpha),
                                 WT_CL=float(r.CL), WT_CD=float(r.CD), model=model,
                                 NF_CL=float(cl[k]), NF_CD=float(cd[k]), NF_conf=float(cf[k])))
df = pd.DataFrame(rows)
df["dCL"] = df.NF_CL - df.WT_CL
df["err_CL"] = df.dCL / (df.WT_CL.abs() + 1e-6)
df["err_CD"] = (df.NF_CD - df.WT_CD) / df.WT_CD
df["Re_bin"] = pd.cut(df.Re, [0, 80e3, 150e3, 250e3, 400e3, 600e3],
                      labels=["40-60k", "100k", "200k", "300-400k", "400-500k"])
df["fit_ok"] = df.asb_name.isin(OK)
df.to_csv(os.path.join(DATA, "uiuc_neuralfoil_validation.csv"), index=False)
print(f"  wrote data/uiuc_neuralfoil_validation.csv ({len(df)} rows)")


def summarise(d):
    a = d[d.WT_CL.abs() > ATTACHED]
    return pd.Series(dict(n=len(d), Re_min=d.Re.min(), Re_max=d.Re.max(),
                          mean_abs_dCL=d.dCL.abs().mean(),
                          mean_abs_err_CL=a.err_CL.abs().mean(),
                          mean_abs_err_CD=d.err_CD.abs().mean(),
                          median_abs_err_CD=d.err_CD.abs().median(),
                          mean_err_CD=d.err_CD.mean(),
                          mean_conf=d.NF_conf.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Clean runs: by airfoil, by Reynolds number
# ─────────────────────────────────────────────────────────────────────────────
clean_all = df[(df.config == "clean") & (df.NF_mode == "free") & (df.model == "large")]
by_af = clean_all.groupby("asb_name").apply(summarise).reset_index()
by_af = by_af.merge(fit[["asb_name", "kulfan_rms_pct", "fit_ok", "max_thickness", "max_camber"]], on="asb_name")
clean_L = clean_all[clean_all.fit_ok]        # the benchmark set
by_af.to_csv(os.path.join(DATA, "uiuc_validation_by_airfoil.csv"), index=False)
print("\nClean runs, NeuralFoil large, by airfoil:")
print(by_af.round(3).to_string(index=False))

by_Re = (df[(df.config == "clean") & (df.NF_mode == "free") & df.fit_ok]
         .groupby(["model", "Re_bin"], observed=True).apply(summarise).reset_index())
by_Re.to_csv(os.path.join(DATA, "uiuc_validation_by_Re.csv"), index=False)
print("\nClean runs, by Reynolds-number bin:")
print(by_Re.round(3).to_string(index=False))

all_clean = summarise(clean_L)
print(f"\nHEADLINE (clean, large, {int(all_clean.n)} points, {clean_L.asb_name.nunique()} airfoils):"
      f"  mean|dCL| = {all_clean.mean_abs_dCL:.3f}   mean|dCL/CL| = {all_clean.mean_abs_err_CL:.1%}"
      f"   mean|dCD/CD| = {all_clean.mean_abs_err_CD:.1%} (median {all_clean.median_abs_err_CD:.1%})"
      f"   bias dCD/CD = {all_clean.mean_err_CD:+.1%}   conf = {all_clean.mean_conf:.3f}")
every = summarise(clean_all)
print(f"          (all {clean_all.asb_name.nunique()} airfoils incl. geometry-limited, {int(every.n)} pts: "
      f"mean|dCL| = {every.mean_abs_dCL:.3f}  mean|dCD/CD| = {every.mean_abs_err_CD:.1%})")
xl = summarise(df[(df.config == "clean") & (df.NF_mode == "free") & (df.model == "xxlarge") & df.fit_ok])
print(f"          (xxlarge: mean|dCL| = {xl.mean_abs_dCL:.3f}  mean|dCD/CD| = {xl.mean_abs_err_CD:.1%}"
      f"  bias {xl.mean_err_CD:+.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# Tripped runs: free transition vs transition forced at the trip
# ─────────────────────────────────────────────────────────────────────────────
trip = df[(df.config == "tripped") & (df.model == "large")]
t_rows = []
for (name, vol), d in trip.groupby(["asb_name", "volume"]):
    for mode in ("free", "forced"):
        s = summarise(d[d.NF_mode == mode]); s["asb_name"], s["volume"], s["NF_mode"] = name, vol, mode
        s["xtr_upper"], s["xtr_lower"] = d.xtr_upper.max(), d.xtr_lower.min()
        t_rows.append(s)
    c = summarise(clean_L[(clean_L.asb_name == name) & (clean_L.volume == vol)])
    c["asb_name"], c["volume"], c["NF_mode"] = name, vol, "clean-run reference"
    t_rows.append(c)
tripped = pd.DataFrame(t_rows)
tripped.to_csv(os.path.join(DATA, "uiuc_validation_tripped.csv"), index=False)
print("\nTripped runs (large): NeuralFoil free vs forced transition, and the clean run for reference:")
print(tripped[["asb_name", "volume", "NF_mode", "n", "mean_abs_dCL", "mean_abs_err_CD",
               "mean_err_CD", "mean_conf"]].round(3).to_string(index=False))
v4 = trip[trip.volume == "vol4"]
both = trip[np.isfinite(trip.xtr_upper) & np.isfinite(trip.xtr_lower) & trip.asb_name.isin(OK)]
for lab, d in [("Vol 4 tripped, three airfoils", v4), ("all both-surface trips, benchmark airfoils", both)]:
    for mode in ("free", "forced"):
        s = summarise(d[d.NF_mode == mode])
        print(f"  {lab}, {mode:>6}: n={int(s.n)} airfoils={d.asb_name.nunique()} mean|dCD/CD| = {s.mean_abs_err_CD:.1%}"
              f"  bias {s.mean_err_CD:+.1%}   mean|dCL| = {s.mean_abs_dCL:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Does analysis_confidence track the error?
# ─────────────────────────────────────────────────────────────────────────────
print("\nConfidence vs error (clean, large, |CL| > 0.1):")
s = clean_L[clean_L.WT_CL.abs() > ATTACHED].copy()
s["abs_dCL"], s["abs_err_CL"], s["abs_err_CD"] = s.dCL.abs(), s.err_CL.abs(), s.err_CD.abs()


def corr_line(d, tag):
    out = [f"  {tag:<22} n={len(d):4d} conf[{d.NF_conf.min():.2f},{d.NF_conf.max():.2f}]"]
    for col, lab in [("abs_dCL", "|dCL|"), ("abs_err_CL", "|dCL/CL|"), ("abs_err_CD", "|dCD/CD|")]:
        r = np.corrcoef(d.NF_conf, d[col])[0, 1]
        rho = stats.spearmanr(d.NF_conf, d[col]).correlation
        out.append(f"{lab}: r={r:+.2f} rho={rho:+.2f}")
    print("  |  ".join(out))


corr_line(s, "all points")
corr_line(s[s.alpha >= 0], "alpha >= 0")
corr_line(s[s.alpha < 0], "alpha < 0")
for b, d in s.groupby("Re_bin", observed=True):
    corr_line(d, f"Re {b}")
rng = np.random.default_rng(0)
bs = [np.corrcoef(*s.sample(len(s), replace=True, random_state=int(rng.integers(1e9)))
                  [["NF_conf", "abs_dCL"]].values.T)[0, 1] for _ in range(2000)]
print(f"  bootstrap 95% CI for Pearson r(conf, |dCL|), all points: "
      f"[{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")

bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
s["conf_bin"] = pd.cut(s.NF_conf, bins)
cb = (s.groupby("conf_bin", observed=True)
      .agg(n=("abs_dCL", "size"), mean_alpha=("alpha", "mean"),
           mean_abs_dCL=("abs_dCL", "mean"), mean_abs_err_CL=("abs_err_CL", "mean"),
           mean_abs_err_CD=("abs_err_CD", "mean"), median_abs_err_CD=("abs_err_CD", "median"))
      .reset_index())
cb["conf_bin"] = cb.conf_bin.astype(str)
cb.to_csv(os.path.join(DATA, "uiuc_validation_confidence_bins.csv"), index=False)
print("\nCalibration table (clean, large):")
print(cb.round(3).to_string(index=False))

print("\nAirfoil level: does an airfoil's mean confidence predict its mean error?")
bo = by_af[by_af.fit_ok]
for col, lab in [("mean_abs_dCL", "|dCL|"), ("mean_abs_err_CD", "|dCD/CD|")]:
    rho = stats.spearmanr(bo.mean_conf, bo[col])
    print(f"  Spearman rho(mean conf, mean {lab}) over {len(bo)} airfoils = {rho.correlation:+.2f}"
          f"  (p = {rho.pvalue:.3f})")
print("\nDone.")
