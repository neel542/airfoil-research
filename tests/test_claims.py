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
    assert fit.fit_ok.sum() == 20
    _close(float(fit[fit.asb_name == "e387"].kulfan_rms_pct.iloc[0]), 0.15, 0.01, "E387 Kulfan RMS %")


def test_headline_by_Re():
    r = _csv("uiuc_validation_by_Re.csv")
    L = r[r.model == "large"].set_index("Re_bin")
    assert int(L.n.sum()) == 1408
    v = _csv("uiuc_neuralfoil_validation.csv")
    c = v[(v.config == "clean") & (v.NF_mode == "free") & (v.model == "large") & v.fit_ok]
    _close(c.dCL.abs().mean(), 0.078, 0.002, "mean |dCL|")
    _close(c.err_CD.abs().mean(), 0.132, 0.004, "mean |dCD/CD|")
    _close(c.err_CD.abs().median(), 0.082, 0.004, "median |dCD/CD|")
    _close(c.err_CD.mean(), 0.021, 0.005, "CD bias")
    _close(L.loc["40-60k", "mean_abs_err_CD"], 0.22, 0.01, "60k drag error")
    _close(L.loc["40-60k", "mean_err_CD"], 0.20, 0.02, "60k drag bias")
    _close(L.loc["460-500k", "mean_err_CD"], -0.08, 0.01, "460-500k drag bias")
    _close(L.loc["200k", "median_abs_err_CD"], 0.07, 0.01, "200k median drag error")


def test_lift_error_camber_split():
    ba = _csv("uiuc_validation_by_airfoil.csv")
    b = ba[ba.fit_ok]
    hi = b[b.max_camber >= 0.05]
    assert set(hi.asb_name) == {"fx63137", "naca6409", "s1210", "s1223"}
    _close(b[b.max_camber < 0.05].mean_abs_dCL.mean(), 0.058, 0.004, "|dCL| low camber")
    _close(hi.mean_abs_dCL.mean(), 0.143, 0.005, "|dCL| high camber")


def test_LD_error_band():
    v = _csv("uiuc_neuralfoil_validation.csv")
    d = v[(v.config == "clean") & (v.NF_mode == "free") & (v.model == "large") & v.fit_ok & (v.WT_CL > 0.2)]
    e = (d.NF_CL / d.NF_CD - d.WT_CL / d.WT_CD) / (d.WT_CL / d.WT_CD)
    assert len(d) == 1033
    _close(e.abs().median(), 0.16, 0.01, "median |dLD/LD|")
    _close(e.abs().mean(), 0.22, 0.01, "mean |dLD/LD|")
    _close(e.mean(), 0.14, 0.01, "L/D bias")
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
    assert len(s) == 1292
    _close(np.corrcoef(s.NF_conf, s.err_CD.abs())[0, 1], -0.46, 0.03, "r(conf, |dCD/CD|)")
    _close(np.corrcoef(s.NF_conf, s.dCL.abs())[0, 1], 0.07, 0.03, "r(conf, |dCL|)")
    for _, d in s.groupby("Re_bin", observed=True):
        assert np.corrcoef(d.NF_conf, d.err_CD.abs())[0, 1] < -0.3, "drag correlation sign by Re band"
    cb = _csv("uiuc_validation_confidence_bins.csv")
    _close(cb.mean_abs_err_CD.iloc[0], 0.37, 0.01, "drag error, confidence < 0.5")
    _close(cb.mean_abs_err_CD.iloc[-1], 0.10, 0.01, "drag error, confidence > 0.95")
    assert (np.diff(cb.mean_abs_err_CD) < 0).all(), "drag error should fall monotonically with confidence"


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


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS  {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL  {name}: {e}")
    sys.exit(1 if fails else 0)
