"""
soartech8_neuralfoil_validation.py
=============================================================================
Independent replication of the NeuralFoil benchmark on a second wind tunnel:
the SoarTech 8 data set (Selig, Donovan and Fraser, "Airfoils at Low
Speeds", 1989), measured in the Princeton University 3 x 4 ft smoke tunnel
in 1986-1989, Re 60k-300k. Parsed by soartech8_parse.py into
data/soartech8_experimental.csv (drag polars) and
data/soartech8_experimental_lift.csv (lift curves).

What this adds to the UIUC benchmark (uiuc_neuralfoil_validation.py):
  1. A second, older, independent tunnel with different instrumentation.
     Same metrics, same Kulfan gate, same models, so the two benchmarks
     are directly comparable.
  2. Measured model geometry. SoarTech 8 published profiler measurements
     of the actual models (ALL.DAP) as well as the design coordinates, so
     NeuralFoil can be run on the shape that was really in the tunnel. The
     difference between the two runs is the part of the error that is the
     model-builder's, not the surrogate's.
  3. Tunnel-to-tunnel comparison. 15 benchmark airfoils were tested in both
     tunnels. At matched Reynolds number and angle of attack, the
     disagreement between the two experiments is a floor on what any
     prediction can be validated to; NeuralFoil's error is compared with
     that floor on the same points.
  4. n_crit sensitivity. The Princeton tunnel's turbulence level is not
     stated in the archive, so the transition parameter is swept.

Outputs
  data/soartech8_neuralfoil_validation.csv        point by point
  data/soartech8_kulfan_fit.csv                   fit error, measured and design geometry
  data/soartech8_validation_by_Re.csv             clean runs, both models, measured geometry
  data/soartech8_validation_by_airfoil.csv        clean runs, large, measured geometry
  data/soartech8_geometry_effect.csv              measured vs design geometry, per model
  data/soartech8_validation_tripped.csv           tripped runs, free vs forced transition
  data/soartech8_validation_confidence_bins.csv   calibration table
  data/soartech8_ncrit_sensitivity.csv            n_crit sweep
  data/cross_tunnel_comparison.csv                UIUC vs Princeton vs NeuralFoil, matched points
  data/soartech8_stall_validation.csv             CLmax and stall angle from the lift files
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
N_CRIT = 9.0
MODELS = ["large", "xxlarge"]
ATTACHED = 0.1
FIT_OK_PCT = 0.5
RE_BINS = [0, 80e3, 125e3, 175e3, 250e3, 400e3]
RE_LABELS = ["60k", "100k", "150k", "200k", "300k"]

exp = pd.read_csv(os.path.join(DATA, "soartech8_experimental.csv"))
exp = exp[exp.config.isin(["clean", "tripped"]) & (exp.geom_source != "none")]
mc = pd.read_csv(os.path.join(DATA, "soartech8_measured_coords.csv"))
dc = pd.read_csv(os.path.join(DATA, "soartech8_design_coords.csv"))
print(f"{len(exp)} clean/tripped points, {exp.model.nunique()} models, "
      f"{exp.family.nunique()} airfoil families, Re {exp.Re.min()}-{exp.Re.max()}")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: Kulfan fit of every model, measured and design coordinates
# ─────────────────────────────────────────────────────────────────────────────
def kulfan_from_coords(name, pts):
    af = asb.Airfoil(name=name, coordinates=np.asarray(pts, float)).normalize()
    kf = af.to_kulfan_airfoil()
    devs = []
    for surf in ("upper", "lower"):
        oc = np.asarray(getattr(af, f"{surf}_coordinates")())
        x = np.clip(oc[:, 0], 0.0, 1.0)
        yk = np.asarray(getattr(kf, f"{surf}_coordinates")(x_over_c=x))[:, 1]
        ok = np.isfinite(yk)
        devs.append((oc[:, 1] - yk)[ok])
    d = np.concatenate(devs)
    return kf, float(np.sqrt(np.mean(d ** 2)) * 100), float(np.abs(d).max() * 100)


def surface_deviation(k1, k2):
    """RMS and max |y1 - y2| over both surfaces, % chord, on a common x grid."""
    x = np.linspace(0, 1, 201)
    d = []
    for surf in ("upper", "lower"):
        y1 = np.asarray(getattr(k1, f"{surf}_coordinates")(x_over_c=x))[:, 1]
        y2 = np.asarray(getattr(k2, f"{surf}_coordinates")(x_over_c=x))[:, 1]
        d.append(y1 - y2)
    d = np.concatenate(d)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2)) * 100), float(np.abs(d).max() * 100)


geoms, fit_rows = {}, []          # geoms[(model, source)] -> KulfanAirfoil
models = exp[["model", "family", "dap_name", "cor_name", "asb_name"]].drop_duplicates("model")
print("\nKulfan fit of each model (RMS % chord): measured / design, and measured-vs-design deviation")
for _, m in models.iterrows():
    row = dict(model=m.model, family=m.family, asb_name=m.asb_name, dap_name=m.dap_name,
               cor_name=m.cor_name)
    if isinstance(m.dap_name, str) and m.dap_name:
        kf, rms, mx = kulfan_from_coords(m.model, mc[mc.dap_name == m.dap_name][["x", "y"]].values)
        geoms[(m.model, "measured")] = kf
        row.update(meas_rms_pct=rms, meas_max_pct=mx, meas_thickness=float(kf.max_thickness()),
                   meas_camber=float(kf.max_camber()))
    if isinstance(m.cor_name, str) and m.cor_name:
        kf, rms, mx = kulfan_from_coords(m.model, dc[dc.cor_name == m.cor_name][["x", "y"]].values)
        geoms[(m.model, "design")] = kf
        row.update(design_rms_pct=rms, design_max_pct=mx, design_thickness=float(kf.max_thickness()),
                   design_camber=float(kf.max_camber()))
    if (m.model, "measured") in geoms and (m.model, "design") in geoms:
        row["build_rms_pct"], row["build_max_pct"] = surface_deviation(
            geoms[(m.model, "measured")], geoms[(m.model, "design")])
    fit_rows.append(row)
    print(f"  {m.model:>18}: {row.get('meas_rms_pct', np.nan):.3f} / {row.get('design_rms_pct', np.nan):.3f}"
          f"   build deviation RMS {row.get('build_rms_pct', np.nan):.3f} max {row.get('build_max_pct', np.nan):.3f}")
fit = pd.DataFrame(fit_rows)
fit["primary_source"] = np.where(fit.meas_rms_pct.notna(), "measured", "design")
fit["primary_rms_pct"] = fit.meas_rms_pct.fillna(fit.design_rms_pct)
fit["fit_ok"] = fit.primary_rms_pct < FIT_OK_PCT
OK = set(fit[fit.fit_ok].model)
print(f"  benchmark set: {len(OK)} of {len(fit)} models with Kulfan RMS < {FIT_OK_PCT}% chord; "
      f"geometry-limited: {sorted(set(fit.model) - OK)}")
bd = fit.build_rms_pct.dropna()
print(f"  measured-vs-design deviation over {len(bd)} models: median RMS {bd.median():.3f}% chord, "
      f"mean {bd.mean():.3f}%, max {bd.max():.3f}% ({fit.loc[bd.idxmax(), 'model']})")
fit.to_csv(os.path.join(DATA, "soartech8_kulfan_fit.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# NeuralFoil at every measured condition
# ─────────────────────────────────────────────────────────────────────────────
def run_nf(kf, alphas, Re, model, n_crit=N_CRIT, xu=1.0, xl=1.0):
    aero = kf.get_aero_from_neuralfoil(alpha=alphas, Re=Re, model_size=model, n_crit=n_crit,
                                       xtr_upper=xu, xtr_lower=xl)
    return (np.atleast_1d(aero[k]) for k in ("CL", "CD", "analysis_confidence"))


print("\nRunning NeuralFoil at every measured condition ...")
rows = []
for (label, Re), g in exp.groupby(["airfoil_label", "Re"]):
    g = g.sort_values("alpha")
    model, cfg = g.model.iloc[0], g.config.iloc[0]
    alphas = g.alpha.to_numpy(float)
    sources = [s for s in ("measured", "design") if (model, s) in geoms]
    modes = [("free", 1.0, 1.0)]
    if cfg == "tripped":
        xu, xl = g.xtr_upper.iloc[0], g.xtr_lower.iloc[0]
        modes.append(("forced", xu if np.isfinite(xu) else 1.0, xl if np.isfinite(xl) else 1.0))
    for src in sources:
        for nf_model in MODELS:
            for mode, xu, xl in modes:
                cl, cd, cf = run_nf(geoms[(model, src)], alphas, Re, nf_model, xu=xu, xl=xl)
                for k, (_, r) in enumerate(g.iterrows()):
                    rows.append(dict(airfoil_label=label, model=model, family=r.family,
                                     asb_name=r.asb_name if isinstance(r.asb_name, str) else "",
                                     builder=r.builder, config=cfg, geom=src, NF_mode=mode,
                                     xtr_upper=xu, xtr_lower=xl, Re=int(Re), alpha=float(r.alpha),
                                     WT_CL=float(r.CL), WT_CD=float(r.CD), nf=nf_model,
                                     NF_CL=float(cl[k]), NF_CD=float(cd[k]), NF_conf=float(cf[k])))
df = pd.DataFrame(rows)
df["dCL"] = df.NF_CL - df.WT_CL
df["err_CL"] = df.dCL / (df.WT_CL.abs() + 1e-6)
df["err_CD"] = (df.NF_CD - df.WT_CD) / df.WT_CD
df["Re_bin"] = pd.cut(df.Re, RE_BINS, labels=RE_LABELS)
df["fit_ok"] = df.model.isin(OK)
# "primary" = measured geometry where it exists, design otherwise
primary = fit.set_index("model").primary_source
df["primary"] = df.geom == df.model.map(primary)
df.to_csv(os.path.join(DATA, "soartech8_neuralfoil_validation.csv"), index=False)
print(f"  wrote data/soartech8_neuralfoil_validation.csv ({len(df)} rows)")


def summarise(d):
    a = d[d.WT_CL.abs() > ATTACHED]
    return pd.Series(dict(n=len(d), n_models=d.model.nunique(), Re_min=d.Re.min(), Re_max=d.Re.max(),
                          mean_abs_dCL=d.dCL.abs().mean(), mean_dCL=d.dCL.mean(),
                          mean_abs_err_CL=a.err_CL.abs().mean(),
                          mean_abs_err_CD=d.err_CD.abs().mean(),
                          median_abs_err_CD=d.err_CD.abs().median(),
                          mean_err_CD=d.err_CD.mean(), mean_conf=d.NF_conf.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Clean runs, primary geometry: by Re, by airfoil
# ─────────────────────────────────────────────────────────────────────────────
clean = df[(df.config == "clean") & (df.NF_mode == "free") & df.primary]
clean_L = clean[(clean.nf == "large") & clean.fit_ok]
by_Re = (clean[clean.fit_ok].groupby(["nf", "Re_bin"], observed=True).apply(summarise).reset_index())
by_Re.to_csv(os.path.join(DATA, "soartech8_validation_by_Re.csv"), index=False)
print("\nClean runs, benchmark models, by Reynolds-number bin:")
print(by_Re.round(3).to_string(index=False))
h = summarise(clean_L)
print(f"\nHEADLINE (clean, large, {int(h.n)} points, {int(h.n_models)} models, "
      f"{clean_L.family.nunique()} families):  mean|dCL| = {h.mean_abs_dCL:.3f} (bias {h.mean_dCL:+.3f})"
      f"   mean|dCD/CD| = {h.mean_abs_err_CD:.1%} (median {h.median_abs_err_CD:.1%})"
      f"   bias dCD/CD = {h.mean_err_CD:+.1%}   conf = {h.mean_conf:.3f}")
xl_ = summarise(clean[(clean.nf == "xxlarge") & clean.fit_ok])
print(f"          (xxlarge: mean|dCL| = {xl_.mean_abs_dCL:.3f}  mean|dCD/CD| = {xl_.mean_abs_err_CD:.1%}"
      f"  bias {xl_.mean_err_CD:+.1%})")
u_bins = clean_L[clean_L.Re_bin.isin(["60k", "100k", "200k", "300k"])]
by_af = clean_L.groupby(["model", "family", "asb_name"]).apply(summarise).reset_index()
by_af = by_af.merge(fit[["model", "primary_rms_pct", "build_rms_pct", "meas_thickness", "meas_camber",
                         "design_thickness", "design_camber"]], on="model")
by_af.to_csv(os.path.join(DATA, "soartech8_validation_by_airfoil.csv"), index=False)
print("\nClean runs, large, by model:")
print(by_af.round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Measured vs design geometry
# ─────────────────────────────────────────────────────────────────────────────
both = df[(df.config == "clean") & (df.NF_mode == "free") & (df.nf == "large") & df.fit_ok]
have_both = set(both[both.geom == "measured"].model) & set(both[both.geom == "design"].model)
both = both[both.model.isin(have_both)]
piv = (both.pivot_table(index=["model", "Re", "alpha"], columns="geom",
                        values=["NF_CL", "NF_CD", "WT_CL", "WT_CD", "NF_conf"]).reset_index())
piv.columns = ["_".join(c).rstrip("_") for c in piv.columns]
piv["dCL_design"] = piv.NF_CL_design - piv.WT_CL_design
piv["dCL_measured"] = piv.NF_CL_measured - piv.WT_CL_measured
piv["errCD_design"] = (piv.NF_CD_design - piv.WT_CD_design) / piv.WT_CD_design
piv["errCD_measured"] = (piv.NF_CD_measured - piv.WT_CD_measured) / piv.WT_CD_measured
piv["dCL_geom"] = piv.NF_CL_measured - piv.NF_CL_design      # what the build deviation does to NF
piv["dCD_geom"] = (piv.NF_CD_measured - piv.NF_CD_design) / piv.NF_CD_design
ge = (piv.groupby("model")
      .agg(n=("alpha", "size"),
           abs_dCL_design=("dCL_design", lambda s: s.abs().mean()),
           abs_dCL_measured=("dCL_measured", lambda s: s.abs().mean()),
           abs_errCD_design=("errCD_design", lambda s: s.abs().mean()),
           abs_errCD_measured=("errCD_measured", lambda s: s.abs().mean()),
           bias_CD_design=("errCD_design", "mean"), bias_CD_measured=("errCD_measured", "mean"),
           geom_effect_CL=("dCL_geom", lambda s: s.abs().mean()),
           geom_effect_CD=("dCD_geom", lambda s: s.abs().mean()))
      .reset_index()
      .merge(fit[["model", "build_rms_pct", "build_max_pct"]], on="model"))
ge.to_csv(os.path.join(DATA, "soartech8_geometry_effect.csv"), index=False)
print(f"\nMeasured vs design geometry ({len(ge)} models, {len(piv)} points, clean, large):")
print(ge.round(3).to_string(index=False))
tot = dict(n=len(piv),
           dCL_d=piv.dCL_design.abs().mean(), dCL_m=piv.dCL_measured.abs().mean(),
           CD_d=piv.errCD_design.abs().mean(), CD_m=piv.errCD_measured.abs().mean(),
           bias_d=piv.errCD_design.mean(), bias_m=piv.errCD_measured.mean(),
           gCL=piv.dCL_geom.abs().mean(), gCD=piv.dCD_geom.abs().mean())
print(f"  ALL: mean|dCL| design {tot['dCL_d']:.3f} -> measured {tot['dCL_m']:.3f};"
      f"  mean|dCD/CD| design {tot['CD_d']:.1%} -> measured {tot['CD_m']:.1%}"
      f"  (bias {tot['bias_d']:+.1%} -> {tot['bias_m']:+.1%});"
      f"  NF shift from geometry alone: |dCL| {tot['gCL']:.3f}, |dCD/CD| {tot['gCD']:.1%}")
n_better = int((ge.abs_errCD_measured < ge.abs_errCD_design).sum())
print(f"  measured geometry lowers the drag error for {n_better} of {len(ge)} models")
rho = stats.spearmanr(ge.build_rms_pct, ge.geom_effect_CD)
print(f"  Spearman rho(build deviation, |NF drag shift|) = {rho.correlation:+.2f} (p = {rho.pvalue:.3f})")
rho = stats.spearmanr(ge.build_rms_pct, ge.abs_errCD_design - ge.abs_errCD_measured)
print(f"  Spearman rho(build deviation, improvement in |dCD/CD|) = {rho.correlation:+.2f} (p = {rho.pvalue:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Tripped runs
# ─────────────────────────────────────────────────────────────────────────────
trip = df[(df.config == "tripped") & (df.nf == "large") & df.primary & df.fit_ok]
t_rows = []
for label, d in trip.groupby("airfoil_label"):
    for mode in ("free", "forced"):
        s = summarise(d[d.NF_mode == mode]); s["airfoil_label"], s["NF_mode"] = label, mode
        s["xtr_upper"], s["xtr_lower"] = d.xtr_upper.max(), d.xtr_lower.min()
        t_rows.append(s)
tripped = pd.DataFrame(t_rows)
tripped.to_csv(os.path.join(DATA, "soartech8_validation_tripped.csv"), index=False)
print("\nTripped runs (large, benchmark models): free vs forced transition")
print(tripped[["airfoil_label", "NF_mode", "n", "xtr_upper", "xtr_lower", "mean_abs_dCL",
               "mean_abs_err_CD", "mean_err_CD", "mean_conf"]].round(3).to_string(index=False))
for mode in ("free", "forced"):
    s = summarise(trip[trip.NF_mode == mode])
    print(f"  all tripped runs, {mode:>6}: n={int(s.n)} models={int(s.n_models)} "
          f"mean|dCD/CD| = {s.mean_abs_err_CD:.1%}  bias {s.mean_err_CD:+.1%}  mean|dCL| = {s.mean_abs_dCL:.3f}")
# the UIUC finding was for two-surface trips at 2%/5% chord; here most trips are single upper-surface
# strips well aft (20-70% chord), a different regime, so report the aft-trip subset separately
trip_loc = exp[exp.config == "tripped"].drop_duplicates("airfoil_label").set_index("airfoil_label")
trip = trip.assign(trip_xu=trip.airfoil_label.map(trip_loc.xtr_upper),
                   trip_xl=trip.airfoil_label.map(trip_loc.xtr_lower))
for lab, d in [("upper trips at <= 30% chord", trip[trip.trip_xu <= 0.3]),
               ("upper trips at >= 40% chord", trip[trip.trip_xu >= 0.4])]:
    for mode in ("free", "forced"):
        s = summarise(d[d.NF_mode == mode])
        print(f"  {lab}, {mode:>6}: n={int(s.n)} runs={d.airfoil_label.nunique()} "
              f"mean|dCD/CD| = {s.mean_abs_err_CD:.1%}  bias {s.mean_err_CD:+.1%}  mean|dCL| = {s.mean_abs_dCL:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Confidence calibration (same bins as the UIUC benchmark)
# ─────────────────────────────────────────────────────────────────────────────
s = clean_L[clean_L.WT_CL.abs() > ATTACHED].copy()
s["abs_dCL"], s["abs_err_CD"] = s.dCL.abs(), s.err_CD.abs()
print(f"\nConfidence vs error (clean, large, |CL| > 0.1, n={len(s)}):")
for col, lab in [("abs_dCL", "|dCL|"), ("abs_err_CD", "|dCD/CD|")]:
    r = np.corrcoef(s.NF_conf, s[col])[0, 1]
    rho = stats.spearmanr(s.NF_conf, s[col]).correlation
    print(f"  {lab}: Pearson r = {r:+.2f}, Spearman rho = {rho:+.2f}")
s["conf_bin"] = pd.cut(s.NF_conf, [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
cb = (s.groupby("conf_bin", observed=True)
      .agg(n=("abs_dCL", "size"), mean_alpha=("alpha", "mean"), mean_abs_dCL=("abs_dCL", "mean"),
           mean_abs_err_CD=("abs_err_CD", "mean"), median_abs_err_CD=("abs_err_CD", "median"))
      .reset_index())
cb["conf_bin"] = cb.conf_bin.astype(str)
cb.to_csv(os.path.join(DATA, "soartech8_validation_confidence_bins.csv"), index=False)
print(cb.round(3).to_string(index=False))
bo = by_af
for col, lab in [("mean_abs_dCL", "|dCL|"), ("mean_abs_err_CD", "|dCD/CD|")]:
    rho = stats.spearmanr(bo.mean_conf, bo[col])
    print(f"  airfoil level: Spearman rho(mean conf, mean {lab}) over {len(bo)} models = "
          f"{rho.correlation:+.2f} (p = {rho.pvalue:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# n_crit sensitivity (clean, large, primary geometry, benchmark models)
# ─────────────────────────────────────────────────────────────────────────────
print("\nn_crit sensitivity (clean, large, benchmark models):")
nc_rows = []
base = exp[(exp.config == "clean") & exp.model.isin(OK)]
for n_crit in (5.0, 7.0, 9.0, 11.0):
    parts = []
    for (label, Re), g in base.groupby(["airfoil_label", "Re"]):
        g = g.sort_values("alpha")
        model = g.model.iloc[0]
        src = primary[model]
        cl, cd, cf = run_nf(geoms[(model, src)], g.alpha.to_numpy(float), Re, "large", n_crit=n_crit)
        parts.append(pd.DataFrame(dict(model=model, Re=Re, alpha=g.alpha.values, WT_CL=g.CL.values,
                                       WT_CD=g.CD.values, NF_CL=cl, NF_CD=cd, NF_conf=cf)))
    p = pd.concat(parts)
    p["dCL"], p["err_CL"], p["err_CD"] = p.NF_CL - p.WT_CL, (p.NF_CL - p.WT_CL) / (p.WT_CL.abs() + 1e-6), (p.NF_CD - p.WT_CD) / p.WT_CD
    p["Re_bin"] = pd.cut(p.Re, RE_BINS, labels=RE_LABELS)
    s_ = summarise(p); s_["n_crit"] = n_crit
    for b, d in p.groupby("Re_bin", observed=True):
        s_[f"abs_err_CD_{b}"] = d.err_CD.abs().mean()
        s_[f"bias_CD_{b}"] = d.err_CD.mean()
    nc_rows.append(s_)
nc = pd.DataFrame(nc_rows)
nc.to_csv(os.path.join(DATA, "soartech8_ncrit_sensitivity.csv"), index=False)
print(nc[["n_crit", "n", "mean_abs_dCL", "mean_dCL", "mean_abs_err_CD", "mean_err_CD"]
         + [c for c in nc.columns if c.startswith("abs_err_CD_") or c.startswith("bias_CD_")]]
      .round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Tunnel-to-tunnel comparison on the shared airfoils
# ─────────────────────────────────────────────────────────────────────────────
print("\nTunnel-to-tunnel: UIUC (1995-2005) vs Princeton (1986-89) on the same airfoils")
uiuc = pd.read_csv(os.path.join(DATA, "uiuc_neuralfoil_validation.csv"))
uiuc = uiuc[(uiuc.config == "clean") & (uiuc.NF_mode == "free") & (uiuc.model == "large") & uiuc.fit_ok]
pr = clean_L[clean_L.asb_name != ""]
shared = sorted(set(pr.asb_name) & set(uiuc.asb_name))
print(f"  shared benchmark airfoils: {len(shared)}: {shared}")
ct_rows = []
for name in shared:
    for (label, Re_p), gp in pr[pr.asb_name == name].groupby(["airfoil_label", "Re"]):
        gu_all = uiuc[uiuc.asb_name == name]
        cand = gu_all[(gu_all.Re > 0.85 * Re_p) & (gu_all.Re < 1.15 * Re_p)]
        if cand.empty:
            continue
        # nearest UIUC Re, one file (prefer the largest overlap in alpha)
        for (vol, file, Re_u), gu in cand.groupby(["volume", "file", "Re"]):
            gu = gu.sort_values("alpha").drop_duplicates("alpha")
            gp_ = gp.sort_values("alpha")
            lo, hi = max(gu.alpha.min(), gp_.alpha.min()), min(gu.alpha.max(), gp_.alpha.max())
            sel = gp_[(gp_.alpha >= lo) & (gp_.alpha <= hi)]
            if len(sel) < 3:
                continue
            cl_u = np.interp(sel.alpha, gu.alpha, gu.WT_CL)
            cd_u = np.interp(sel.alpha, gu.alpha, gu.WT_CD)
            nfcl_u = np.interp(sel.alpha, gu.alpha, gu.NF_CL)   # NF on the AeroSandbox design coords
            nfcd_u = np.interp(sel.alpha, gu.alpha, gu.NF_CD)
            for k, (_, r) in enumerate(sel.iterrows()):
                ct_rows.append(dict(asb_name=name, princeton_label=label, Re_princeton=int(Re_p),
                                    uiuc_volume=vol, uiuc_file=file, Re_uiuc=int(Re_u), alpha=r.alpha,
                                    CL_princeton=r.WT_CL, CD_princeton=r.WT_CD,
                                    CL_uiuc=float(cl_u[k]), CD_uiuc=float(cd_u[k]),
                                    NF_CL_measured_geom=r.NF_CL, NF_CD_measured_geom=r.NF_CD,
                                    NF_CL_design_geom=float(nfcl_u[k]), NF_CD_design_geom=float(nfcd_u[k])))
ct = pd.DataFrame(ct_rows)
ct["dCL_tunnels"] = ct.CL_uiuc - ct.CL_princeton
ct["errCD_tunnels"] = (ct.CD_uiuc - ct.CD_princeton) / ct.CD_princeton
ct["dCL_NF_vs_uiuc"] = ct.NF_CL_design_geom - ct.CL_uiuc
ct["dCL_NF_vs_princeton"] = ct.NF_CL_measured_geom - ct.CL_princeton
ct["errCD_NF_vs_uiuc"] = (ct.NF_CD_design_geom - ct.CD_uiuc) / ct.CD_uiuc
ct["errCD_NF_vs_princeton"] = (ct.NF_CD_measured_geom - ct.CD_princeton) / ct.CD_princeton
ct["Re_bin"] = pd.cut(ct.Re_princeton, RE_BINS, labels=RE_LABELS)
ct.to_csv(os.path.join(DATA, "cross_tunnel_comparison.csv"), index=False)


def ct_summary(d):
    return pd.Series(dict(n=len(d), airfoils=d.asb_name.nunique(),
                          tunnels_abs_dCL=d.dCL_tunnels.abs().mean(), tunnels_dCL=d.dCL_tunnels.mean(),
                          tunnels_abs_errCD=d.errCD_tunnels.abs().mean(), tunnels_errCD=d.errCD_tunnels.mean(),
                          NF_vs_uiuc_abs_dCL=d.dCL_NF_vs_uiuc.abs().mean(),
                          NF_vs_uiuc_abs_errCD=d.errCD_NF_vs_uiuc.abs().mean(),
                          NF_vs_uiuc_errCD=d.errCD_NF_vs_uiuc.mean(),
                          NF_vs_princeton_abs_dCL=d.dCL_NF_vs_princeton.abs().mean(),
                          NF_vs_princeton_abs_errCD=d.errCD_NF_vs_princeton.abs().mean(),
                          NF_vs_princeton_errCD=d.errCD_NF_vs_princeton.mean()))


cts = pd.concat([ct.groupby("Re_bin", observed=True).apply(ct_summary),
                 ct_summary(ct).to_frame("all").T])
cts.to_csv(os.path.join(DATA, "cross_tunnel_summary.csv"))
print(f"  matched points: {len(ct)} on {ct.asb_name.nunique()} airfoils, "
      f"{ct.groupby(['asb_name', 'Re_princeton', 'uiuc_file']).ngroups} polar pairs")
print(cts.round(3).to_string())
per_af = ct.groupby("asb_name").apply(ct_summary).reset_index()
per_af.to_csv(os.path.join(DATA, "cross_tunnel_by_airfoil.csv"), index=False)
print(per_af[["asb_name", "n", "tunnels_abs_dCL", "tunnels_abs_errCD", "tunnels_errCD",
              "NF_vs_uiuc_abs_errCD", "NF_vs_princeton_abs_errCD"]].round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Lift curves: CLmax and stall angle
# ─────────────────────────────────────────────────────────────────────────────
print("\nLift curves: CLmax and stall angle (clean lift files, measured geometry where available)")
lift = pd.read_csv(os.path.join(DATA, "soartech8_experimental_lift.csv"))
lift = lift[lift.config == "clean"]
st_rows = []
for file, g in lift.groupby("file"):
    model = g.model.iloc[0]
    src = primary.get(model)
    if src is None or (model, src) not in geoms or model not in OK:
        continue
    # up-sweep only: the files run alpha up then (sometimes) back down
    a = g.alpha.to_numpy(float); cl = g.CL.to_numpy(float)
    turn = int(np.argmax(a)) + 1
    au, clu = a[:turn], cl[:turn]
    k = int(np.argmax(clu))
    captured = au.max() >= au[k] + 1.5
    grid = np.arange(-6, 20.01, 0.25)
    ncl, ncd, ncf = run_nf(geoms[(model, src)], grid, int(g.Re.iloc[0]), "large")
    j = int(np.argmax(ncl))
    st_rows.append(dict(file=file, model=model, family=g.family.iloc[0], Re=int(g.Re.iloc[0]),
                        n_points=len(g), alpha_max_tested=au.max(), stall_captured=captured,
                        WT_CLmax=clu[k], WT_alpha_CLmax=au[k], NF_CLmax=float(ncl[j]),
                        NF_alpha_CLmax=float(grid[j]), NF_conf_at_CLmax=float(ncf[j]),
                        NF_CL_at_WT_stall=float(np.interp(au[k], grid, ncl))))
st = pd.DataFrame(st_rows)
st["dCLmax"] = st.NF_CLmax - st.WT_CLmax
st["dalpha_CLmax"] = st.NF_alpha_CLmax - st.WT_alpha_CLmax
st.to_csv(os.path.join(DATA, "soartech8_stall_validation.csv"), index=False)
sc = st[st.stall_captured]
print(f"  {len(st)} lift files, {len(sc)} with stall captured (alpha tested >= alpha_CLmax + 1.5 deg):"
      f"  mean dCLmax {sc.dCLmax.mean():+.3f} (mean |dCLmax| {sc.dCLmax.abs().mean():.3f}),"
      f"  median |dalpha| {sc.dalpha_CLmax.abs().median():.2f} deg, mean dalpha {sc.dalpha_CLmax.mean():+.2f} deg")
print("\nDone.")
