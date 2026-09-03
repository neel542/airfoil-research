"""
Pin the numbers quoted in PAPER.md / README.md to the data files, so prose
and data cannot drift apart silently. Run with `python tests/test_claims.py`
or `pytest tests/`.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _csv(name):
    return pd.read_csv(os.path.join(DATA, name))


def _close(actual, expected, tol, what):
    assert abs(actual - expected) <= tol, f"{what}: {actual:.4f} vs quoted {expected} (tol {tol})"


def test_benchmark_set_and_kulfan_gate():
    fit = _csv("uiuc_kulfan_fit.csv")
    assert set(fit[~fit.fit_ok].asb_name) == {"a18", "be50"}
    assert fit.fit_ok.sum() == 55
    _close(float(fit[fit.asb_name == "e387"].kulfan_rms_pct.iloc[0]), 0.15, 0.01, "E387 Kulfan RMS %")


def test_headline_by_Re():
    r = _csv("uiuc_validation_by_Re.csv")
    L = r[r.model == "large"].set_index("Re_bin")
    assert int(L.n.sum()) == 4428
    v = _csv("uiuc_neuralfoil_validation.csv")
    c = v[(v.config == "clean") & (v.NF_mode == "free") & (v.model == "large") & v.fit_ok]
    _close(c.dCL.abs().mean(), 0.072, 0.002, "mean |dCL|")
    _close(c.err_CD.abs().mean(), 0.123, 0.004, "mean |dCD/CD|")
    _close(c.err_CD.abs().median(), 0.081, 0.004, "median |dCD/CD|")
    _close(c.err_CD.mean(), 0.004, 0.005, "CD bias")
    _close(L.loc["40-60k", "mean_abs_err_CD"], 0.22, 0.01, "60k drag error")
    _close(L.loc["40-60k", "mean_err_CD"], 0.13, 0.02, "60k drag bias")
    _close(L.loc["400-500k", "mean_err_CD"], -0.065, 0.01, "400-500k drag bias")
    _close(L.loc["200k", "median_abs_err_CD"], 0.07, 0.01, "200k median drag error")


def test_lift_error_camber_split():
    ba = _csv("uiuc_validation_by_airfoil.csv")
    b = ba[ba.fit_ok]
    hi = b[b.max_camber >= 0.05]
    assert len(hi) == 10 and {"s1223", "s1210", "fx63137", "naca6409"} <= set(hi.asb_name)
    _close(b[b.max_camber < 0.05].mean_abs_dCL.mean(), 0.066, 0.004, "|dCL| low camber")
    _close(hi.mean_abs_dCL.mean(), 0.113, 0.005, "|dCL| high camber")


def test_LD_error_band():
    v = _csv("uiuc_neuralfoil_validation.csv")
    d = v[(v.config == "clean") & (v.NF_mode == "free") & (v.model == "large") & v.fit_ok & (v.WT_CL > 0.2)]
    e = (d.NF_CL / d.NF_CD - d.WT_CL / d.WT_CD) / (d.WT_CL / d.WT_CD)
    assert len(d) == 3151
    _close(e.abs().median(), 0.15, 0.01, "median |dLD/LD|")
    _close(e.abs().mean(), 0.21, 0.01, "mean |dLD/LD|")
    _close(e.mean(), 0.15, 0.01, "L/D bias")
    em = pd.read_json(os.path.join(DATA, "error_model.json"), typ="series")
    _close(float(em.LD_band_pct) / 100, e.abs().median(), 0.01, "error_model LD_band_pct")


def test_tripped_forced_transition():
    v = _csv("uiuc_neuralfoil_validation.csv")
    t = v[(v.volume == "vol4") & (v.config == "tripped") & (v.model == "large")]
    assert len(t[t.NF_mode == "free"]) == 247
    _close(t[t.NF_mode == "free"].err_CD.mean(), -0.19, 0.01, "tripped, free-transition CD bias")
    _close(t[t.NF_mode == "forced"].err_CD.mean(), -0.02, 0.01, "tripped, forced-transition CD bias")
    assert t[t.NF_mode == "forced"].dCL.abs().mean() < 0.6 * t[t.NF_mode == "free"].dCL.abs().mean()


def test_confidence_tracks_drag_not_lift():
    v = _csv("uiuc_neuralfoil_validation.csv")
    s = v[(v.config == "clean") & (v.NF_mode == "free") & (v.model == "large") & v.fit_ok & (v.WT_CL.abs() > 0.1)]
    assert len(s) == 3995
    _close(np.corrcoef(s.NF_conf, s.err_CD.abs())[0, 1], -0.43, 0.03, "r(conf, |dCD/CD|)")
    _close(np.corrcoef(s.NF_conf, s.dCL.abs())[0, 1], 0.00, 0.03, "r(conf, |dCL|)")
    for _, d in s.groupby("Re_bin", observed=True):
        assert np.corrcoef(d.NF_conf, d.err_CD.abs())[0, 1] < -0.25, "drag correlation sign by Re band"
    cb = _csv("uiuc_validation_confidence_bins.csv")
    _close(cb.mean_abs_err_CD.iloc[0], 0.32, 0.01, "drag error, confidence < 0.5")
    _close(cb.mean_abs_err_CD.iloc[-1], 0.09, 0.01, "drag error, confidence > 0.95")
    assert (np.diff(cb.mean_abs_err_CD) < 0).all(), "drag error should fall monotonically with confidence"
    ba = _csv("uiuc_validation_by_airfoil.csv"); bo = ba[ba.fit_ok]
    from scipy import stats
    assert stats.spearmanr(bo.mean_conf, bo.mean_abs_err_CD).correlation < -0.5, "airfoil-level drag correlation"


def test_e387_per_Re():
    v = _csv("uiuc_neuralfoil_validation.csv")
    e = v[(v.asb_name == "e387") & (v.volume == "vol4") & (v.config == "clean") & (v.NF_mode == "free") & (v.model == "large")]
    assert len(e) == 112 and e.Re.nunique() == 6
    a = e[e.WT_CL.abs() > 0.1]
    r100 = a[a.Re < 150e3]; r200 = a[(a.Re > 150e3) & (a.Re < 250e3)]
    _close(r100.err_CL.abs().mean(), 0.11, 0.01, "E387 100k relative CL error")
    _close(e[e.Re < 150e3].err_CD.abs().mean(), 0.22, 0.01, "E387 100k CD error")
    _close(r200.err_CL.abs().mean(), 0.054, 0.005, "E387 200k relative CL error")
    _close(e[(e.Re > 150e3) & (e.Re < 250e3)].err_CD.abs().mean(), 0.12, 0.01, "E387 200k CD error")
    _close(e.err_CD.mean(), -0.03, 0.01, "E387 CD bias")


def test_stall():
    st = _csv("uiuc_stall_validation.csv")
    b = st[st.fit_ok & (st.config == "clean") & st.stall_captured]
    assert len(b) == 298 and b.asb_name.nunique() == 55
    _close(b.dCLmax.mean(), 0.06, 0.01, "mean dCLmax")
    _close(b.dalpha_CLmax.abs().median(), 1.35, 0.2, "median |dalpha|")
    _close(b.dalpha_CLmax.mean(), -0.9, 0.2, "mean stall-angle offset")
    _close(b.err_CL_prestall.mean(), 0.08, 0.01, "pre-stall CL error")
    _close(b.err_CL_poststall.mean(), 0.15, 0.01, "post-stall CL error")
    _close(b.conf_prestall.mean(), 0.89, 0.02, "pre-stall confidence")
    _close(b.conf_poststall.mean(), 0.52, 0.03, "post-stall confidence")
    _close(b.err_CM_prestall.mean(), 0.018, 0.003, "pre-stall CM error")
    _close(b.hysteresis_dCL.median(), 0.047, 0.01, "median hysteresis")
    assert b.hysteresis_dCL.quantile(0.9) > 0.4


def test_uncertainty_sweep():
    s = _csv("uncertainty_aware_sweep.csv").set_index("w_conf")
    if 0.5 not in s.index:
        print("  (skip: uncertainty_aware_sweep.csv is the old 3x3-grid sweep)"); return
    _close(s.loc[0.0, "worst_LD"], 38.5, 0.3, "w_conf=0 worst L/D")
    _close(s.loc[0.0, "mean_conf"], 0.16, 0.02, "w_conf=0 confidence")
    _close(s.loc[0.5, "worst_LD"], 37.7, 0.3, "w_conf=0.5 worst L/D")
    _close(s.loc[0.5, "mean_conf"], 0.96, 0.01, "w_conf=0.5 confidence")
    _close(s.loc[8.0, "worst_LD"], 35.2, 0.3, "w_conf=8 worst L/D")
    fam = _csv("tradeoff_family.csv").set_index("lam")
    _close(fam.loc[0.0, "worst_case_LD"], 38.2, 0.1, "airfoil B worst-case L/D")


def test_soartech8_second_tunnel():
    fit = _csv("soartech8_kulfan_fit.csv")
    assert fit.fit_ok.all() and len(fit) == 68 and fit.primary_rms_pct.max() < 0.07
    bd = fit.build_rms_pct.dropna()
    assert len(bd) == 56
    _close(bd.median(), 0.22, 0.02, "median build deviation % chord")
    _close(bd.max(), 0.68, 0.02, "max build deviation % chord")
    assert fit.loc[bd.idxmax(), "model"] == "E387B"
    v = _csv("soartech8_neuralfoil_validation.csv")
    c = v[(v.config == "clean") & (v.NF_mode == "free") & (v.nf == "large") & v.primary & v.fit_ok]
    assert len(c) == 4702 and c.model.nunique() == 68 and c.family.nunique() == 54
    _close(c.dCL.abs().mean(), 0.086, 0.002, "Princeton mean |dCL|")
    _close(c.dCL.mean(), 0.046, 0.005, "Princeton CL bias")
    _close(c.err_CD.abs().mean(), 0.111, 0.004, "Princeton mean |dCD/CD|")
    _close(c.err_CD.abs().median(), 0.075, 0.004, "Princeton median |dCD/CD|")
    _close(c.err_CD.mean(), 0.013, 0.005, "Princeton CD bias")
    r = _csv("soartech8_validation_by_Re.csv"); L = r[r.nf == "large"].set_index("Re_bin")
    _close(L.loc["60k", "mean_abs_err_CD"], 0.17, 0.01, "Princeton 60k drag error")
    _close(L.loc["60k", "mean_err_CD"], 0.12, 0.01, "Princeton 60k drag bias")
    _close(L.loc["300k", "mean_err_CD"], -0.04, 0.01, "Princeton 300k drag bias")
    d = c[c.WT_CL > 0.2]
    e = (d.NF_CL / d.NF_CD - d.WT_CL / d.WT_CD) / (d.WT_CL / d.WT_CD)
    _close(e.abs().median(), 0.15, 0.01, "Princeton median |dLD/LD|")
    _close(e.abs().mean(), 0.21, 0.01, "Princeton mean |dLD/LD|")
    _close(e.mean(), 0.15, 0.01, "Princeton L/D bias")
    nc = _csv("soartech8_ncrit_sensitivity.csv").set_index("n_crit")
    assert nc.mean_abs_err_CD.idxmin() == 9.0, "n_crit = 9 should be the best setting"
    _close(nc.loc[7.0, "mean_err_CD"], -0.05, 0.01, "n_crit 7 bias")
    assert nc.loc[11.0, "mean_abs_err_CD"] > 0.19


def test_soartech8_geometry_effect():
    ge = _csv("soartech8_geometry_effect.csv")
    assert len(ge) == 56 and int((ge.abs_errCD_measured < ge.abs_errCD_design).sum()) == 46
    v = _csv("soartech8_neuralfoil_validation.csv")
    b = v[(v.config == "clean") & (v.NF_mode == "free") & (v.nf == "large") & v.fit_ok & v.model.isin(ge.model)]
    piv = b.pivot_table(index=["model", "Re", "alpha"], columns="geom", values=["err_CD", "dCL", "NF_CD", "NF_CL"])
    assert len(piv) == 4010
    _close(piv["err_CD"]["design"].abs().mean(), 0.128, 0.004, "drag error, design coords")
    _close(piv["err_CD"]["measured"].abs().mean(), 0.111, 0.004, "drag error, measured coords")
    _close(piv["dCL"]["design"].abs().mean(), 0.091, 0.003, "lift error, design coords")
    _close(piv["dCL"]["measured"].abs().mean(), 0.081, 0.003, "lift error, measured coords")
    shift = ((piv["NF_CD"]["measured"] - piv["NF_CD"]["design"]) / piv["NF_CD"]["design"]).abs().mean()
    _close(shift, 0.051, 0.005, "NF drag shift from geometry alone")
    e = ge[ge.model == "E387B"].iloc[0]
    _close(e.abs_errCD_design, 0.13, 0.01, "E387B design"); _close(e.abs_errCD_measured, 0.08, 0.01, "E387B measured")


def test_cross_tunnel():
    ct = _csv("cross_tunnel_comparison.csv")
    assert len(ct) == 2139 and ct.asb_name.nunique() == 15
    assert ct.groupby(["asb_name", "Re_princeton", "uiuc_file"]).ngroups == 131
    _close(ct.errCD_tunnels.abs().mean(), 0.118, 0.005, "tunnel-vs-tunnel |dCD/CD|")
    _close(ct.errCD_tunnels.mean(), 0.063, 0.005, "tunnel-vs-tunnel CD bias (UIUC higher)")
    _close(ct.dCL_tunnels.abs().mean(), 0.048, 0.003, "tunnel-vs-tunnel |dCL|")
    _close(ct.errCD_NF_vs_uiuc.abs().mean(), 0.111, 0.005, "NF vs UIUC on matched points")
    _close(ct.errCD_NF_vs_princeton.abs().mean(), 0.101, 0.005, "NF vs Princeton on matched points")
    _close(ct.dCL_NF_vs_uiuc.abs().mean(), 0.066, 0.003, "NF vs UIUC |dCL|")
    _close(ct.dCL_NF_vs_princeton.abs().mean(), 0.080, 0.003, "NF vs Princeton |dCL|")
    _close((ct.errCD_NF_vs_princeton.abs() < ct.errCD_tunnels.abs()).mean(), 0.53, 0.02, "fraction NF closer than other tunnel")
    hi = ct[ct.Re_princeton >= 175e3]
    assert hi.errCD_tunnels.abs().mean() < 0.09 and hi.errCD_NF_vs_princeton.abs().mean() < 0.08


def test_soartech8_trips_stall_confidence():
    v = _csv("soartech8_neuralfoil_validation.csv")
    t = v[(v.config == "tripped") & (v.nf == "large") & v.primary & v.fit_ok]
    assert len(t[t.NF_mode == "free"]) == 1744 and t.airfoil_label.nunique() == 34 and t.model.nunique() == 18
    _close(t[t.NF_mode == "free"].err_CD.abs().mean(), 0.135, 0.005, "tripped free drag error")
    _close(t[t.NF_mode == "forced"].err_CD.abs().mean(), 0.118, 0.005, "tripped forced drag error")
    _close(t[t.NF_mode == "free"].dCL.abs().mean(), 0.084, 0.003, "tripped free |dCL|")
    _close(t[t.NF_mode == "forced"].dCL.abs().mean(), 0.071, 0.003, "tripped forced |dCL|")
    st = _csv("soartech8_stall_validation.csv")
    b = st[st.stall_captured & (st.Re >= 55e3)]
    assert len(b) == 114 and b.family.nunique() == 35
    _close(b.dCLmax.mean(), 0.11, 0.01, "Princeton mean dCLmax")
    _close(b.dalpha_CLmax.abs().median(), 1.4, 0.2, "Princeton median |dalpha|")
    s = v[(v.config == "clean") & (v.NF_mode == "free") & (v.nf == "large") & v.primary & v.fit_ok & (v.WT_CL.abs() > 0.1)]
    assert len(s) == 4216
    _close(np.corrcoef(s.NF_conf, s.err_CD.abs())[0, 1], -0.26, 0.03, "Princeton r(conf, |dCD/CD|)")
    _close(np.corrcoef(s.NF_conf, s.dCL.abs())[0, 1], -0.03, 0.03, "Princeton r(conf, |dCL|)")
    cb = _csv("soartech8_validation_confidence_bins.csv")
    _close(cb.mean_abs_err_CD.iloc[0], 0.30, 0.01, "Princeton drag error, confidence < 0.5")
    _close(cb.mean_abs_err_CD.iloc[-1], 0.10, 0.01, "Princeton drag error, confidence > 0.95")
    assert int(cb.n.iloc[-1]) == 3742
    from scipy import stats
    ba = _csv("soartech8_validation_by_airfoil.csv")
    rho = stats.spearmanr(ba.mean_conf, ba.mean_abs_err_CD)
    assert abs(rho.correlation) < 0.2 and rho.pvalue > 0.05, "airfoil-level correlation does not replicate on Princeton"


def test_xfoil_decomposition():
    d = _csv("xfoil_decomposition.csv")
    assert len(d) == 9130 and d.polar.nunique() == 667
    _close(d.xf_converged.mean(), 0.965, 0.01, "XFoil convergence fraction")
    s = _csv("xfoil_decomposition_summary.csv").set_index("tunnel")
    p = s.loc["pooled"]
    assert int(p.n_xfoil_converged) == 8814
    _close(p.mean_abs_errCD_NF_WT, 0.112, 0.004, "NF vs tunnel drag error")
    _close(p.mean_abs_errCD_XF_WT, 0.121, 0.004, "XFoil vs tunnel drag error")
    _close(p.mean_abs_errCD_NF_XF, 0.028, 0.003, "NF vs XFoil drag error (network only)")
    _close(p.median_abs_errCD_NF_XF, 0.017, 0.003, "median NF vs XFoil drag error")
    _close(p.corr_signed_NF_WT_vs_XF_WT, 0.946, 0.02, "correlation of signed NF and XFoil errors")
    _close(p.frac_NF_closer_than_XF, 0.549, 0.02, "fraction of points where NF is closer to tunnel than XFoil")
    _close(p.mean_abs_dCL_NF_WT, 0.079, 0.003, "NF lift error"); _close(p.mean_abs_dCL_XF_WT, 0.082, 0.003, "XFoil lift error")
    _close(p.mean_abs_dCL_NF_XF, 0.011, 0.002, "NF vs XFoil lift error")
    _close(p.corr_conf_abs_XF_WT, -0.384, 0.03, "r(conf, |XFoil error|)")
    _close(p.corr_conf_abs_NF_XF, -0.328, 0.03, "r(conf, |NF - XFoil|)")
    _close(p.corr_conf_abs_NF_WT, -0.322, 0.03, "r(conf, |NF error|)")
    for t, nfwt, xfwt in [("UIUC", 0.115, 0.129), ("Princeton", 0.109, 0.114)]:
        _close(s.loc[t].mean_abs_errCD_NF_WT, nfwt, 0.004, f"{t} NF error"); _close(s.loc[t].mean_abs_errCD_XF_WT, xfwt, 0.004, f"{t} XFoil error")
    r = _csv("xfoil_decomposition_by_Re.csv").set_index(["tunnel", "Re_bin"])
    _close(r.loc[("UIUC", "100k")].mean_abs_errCD_XF_WT, 0.149, 0.005, "UIUC 100k XFoil error")
    _close(r.loc[("UIUC", "100k")].mean_abs_errCD_NF_WT, 0.125, 0.005, "UIUC 100k NF error")
    _close(r.loc[("UIUC", "40-60k")].mean_abs_errCD_NF_XF, 0.033, 0.005, "UIUC 60k network error")
    assert (r.mean_abs_errCD_NF_XF < 0.045).all() and (r.mean_abs_errCD_NF_WT <= r.mean_abs_errCD_XF_WT + 0.005).all()
    cb = _csv("xfoil_decomposition_confidence.csv")
    _close(cb.mean_abs_errCD_XF_WT.iloc[0], 0.405, 0.01, "XFoil error, confidence < 0.5")
    _close(cb.mean_abs_errCD_NF_XF.iloc[0], 0.101, 0.01, "network error, confidence < 0.5")
    _close(cb.mean_abs_errCD_XF_WT.iloc[-1], 0.097, 0.005, "XFoil error, confidence > 0.95")
    _close(cb.mean_abs_errCD_NF_XF.iloc[-1], 0.020, 0.003, "network error, confidence > 0.95")
    assert int(cb.n.iloc[-1]) == 6742


def test_clustered_statistics():
    cs = _csv("clustered_statistics.csv").set_index(["tunnel", "statistic"])
    u = cs.loc["UIUC"]; p = cs.loc["Princeton"]; a = cs.loc["pooled"]
    assert int(u.n_clusters.iloc[0]) == 55 and int(p.n_clusters.iloc[0]) == 54 and int(a.n_clusters.iloc[0]) == 109
    _close(u.loc["mean_abs_errCD"].cluster_ci_lo, 0.113, 0.004, "UIUC drag error CI low")
    _close(u.loc["mean_abs_errCD"].cluster_ci_hi, 0.134, 0.004, "UIUC drag error CI high")
    _close(p.loc["mean_abs_errCD"].cluster_ci_lo, 0.103, 0.004, "Princeton drag error CI low")
    _close(p.loc["mean_abs_errCD"].cluster_ci_hi, 0.120, 0.004, "Princeton drag error CI high")
    _close(a.loc["mean_abs_errCD"].cluster_ci_lo, 0.110, 0.004, "pooled drag error CI low")
    _close(a.loc["mean_abs_errCD"].cluster_ci_hi, 0.124, 0.004, "pooled drag error CI high")
    for t in (u, p):
        assert t.loc["mean_abs_errCD"].ci_width_ratio > 2.0 and t.loc["mean_abs_dCL"].ci_width_ratio > 3.0
        assert t.loc["bias_CD"].cluster_ci_lo < 0 < t.loc["bias_CD"].cluster_ci_hi + 0.001, "drag bias not distinguishable from zero"
        assert t.loc["bias_LD"].cluster_ci_lo > 0.10, "L/D over-prediction survives clustering"
        assert t.loc["r_conf_absDCL"].cluster_ci_lo < 0 < t.loc["r_conf_absDCL"].cluster_ci_hi, "lift correlation includes zero"
    assert u.loc["r_conf_absErrCD"].cluster_ci_hi < -0.30 and p.loc["r_conf_absErrCD"].cluster_ci_hi < -0.15
    assert u.loc["rho_airfoil_conf_errCD"].cluster_ci_hi < -0.4
    assert p.loc["rho_airfoil_conf_errCD"].cluster_ci_lo < 0 < p.loc["rho_airfoil_conf_errCD"].cluster_ci_hi
    d = _csv("confidence_correlation_decomposition.csv").set_index("tunnel")
    _close(d.loc["UIUC"].r_between_airfoil, -0.687, 0.03, "UIUC between-airfoil r")
    _close(d.loc["UIUC"].r_within_polar, -0.450, 0.03, "UIUC within-polar r")
    assert abs(d.loc["Princeton"].r_between_airfoil) < 0.15
    _close(d.loc["Princeton"].r_within_polar, -0.305, 0.03, "Princeton within-polar r")
    _close(d.loc["UIUC"].conf_var_share_between_airfoil, 0.26, 0.03, "UIUC confidence variance between airfoils")
    _close(d.loc["Princeton"].conf_var_share_between_airfoil, 0.07, 0.03, "Princeton confidence variance between airfoils")


def test_fitted_error_model():
    import json
    cv = _csv("error_model_cv.csv")
    m = cv.groupby(["scheme", "model"]).mean(numeric_only=True)
    f = m.loc[("airfoil_10fold", "full")]
    _close(f.mae, 0.0742, 0.002, "full model held-out MAE")
    _close(m.loc[("airfoil_10fold", "constant")].mae, 0.0867, 0.002, "constant MAE")
    _close(m.loc[("airfoil_10fold", "six_bin_table")].mae, 0.0794, 0.002, "six-bin table MAE")
    _close(f.spearman, 0.39, 0.03, "full model rank correlation")
    _close(f.coverage_80, 0.80, 0.03, "80th percentile coverage"); _close(f.coverage_95, 0.95, 0.02, "95th percentile coverage")
    _close(m.loc[("train_UIUC_test_Princeton", "full")].mae, 0.0711, 0.002, "UIUC -> Princeton MAE")
    _close(m.loc[("train_UIUC_test_Princeton", "constant")].mae, 0.0847, 0.002, "UIUC -> Princeton constant MAE")
    _close(m.loc[("train_Princeton_test_UIUC", "full")].mae, 0.0771, 0.002, "Princeton -> UIUC MAE")
    _close(m.loc[("train_Princeton_test_UIUC", "constant")].mae, 0.0889, 0.002, "Princeton -> UIUC constant MAE")
    em = json.load(open(os.path.join(DATA, "error_model_fit.json")))
    assert em["n_points"] == 8211 and em["n_airfoils"] == 94
    _close(em["beta"]["log_unconf"], 0.625, 0.03, "confidence coefficient"); _close(em["beta"]["log_Re"], -0.842, 0.03, "Re coefficient")
    _close(em["beta"]["camber"], 0.0345, 0.005, "camber coefficient"); assert abs(em["beta"]["thickness"]) < 0.01
    cal = _csv("error_model_calibration.csv")
    _close(cal.mean_pred.iloc[-1], 0.292, 0.01, "top-decile predicted"); _close(cal.mean_actual.iloc[-1], 0.287, 0.01, "top-decile actual")
    _close(cal.mean_pred.iloc[0], 0.057, 0.005, "bottom-decile predicted"); _close(cal.mean_actual.iloc[0], 0.069, 0.005, "bottom-decile actual")
    assert (cal.mean_actual.diff().dropna() > -0.01).all(), "actual error should rise with predicted error"
    ex = pd.DataFrame(em["worked_examples"]).set_index("NF_conf")
    _close(ex.loc[0.98].iloc[0].expected_abs_errCD if hasattr(ex.loc[0.98], "iloc") and ex.loc[0.98].ndim == 2 else ex.loc[0.98].expected_abs_errCD, 0.067, 0.005, "worked example, conf 0.98 Re 200k")
    top = json.load(open(os.path.join(DATA, "error_model.json")))
    assert top["fitted_error_model"] == "data/error_model_fit.json"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL  {name}: {e}")
    sys.exit(1 if fails else 0)
