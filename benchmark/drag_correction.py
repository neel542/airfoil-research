"""
A small learned correction to NeuralFoil's drag, tested the way plan/improve-neuralfoil.md set out.

NeuralFoil's drag is multiplied by exp(f), where f is a ridge-regression model of
log(CD_tunnel / CD_NeuralFoil) fitted on the training data only. Inputs are things known
when you make a prediction (never the measured lift):
  A  log Re, NeuralFoil CL, CL^2, log Re * CL
  B  A + camber + thickness
  C  B + NeuralFoil's confidence
Ridge strength is fixed at 1.0 on standardized inputs, chosen before any result was seen.

Step 1: score A, B and C on the test bench (two cross-tunnel splits and 10-fold airfoil CV).
Step 2: pick the best model that LTPT can use (A or B), fit it on both tunnels, and score it
        once on the locked third tunnel, NASA Langley LTPT E387.

Inputs : data/xfoil_decomposition.csv (through testbench.py), data/uiuc_kulfan_fit.csv,
         data/soartech8_kulfan_fit.csv, data/ltpt_neuralfoil_validation.csv
Outputs: data/drag_correction_bench.csv     bench results for A, B, C and the baseline
         data/drag_correction_by_Re.csv     error by Reynolds-number band, before and after
         data/drag_correction_ltpt.csv      the one locked-test result
Run    : python benchmark/drag_correction.py
"""

import os
import numpy as np
import pandas as pd

import testbench as tb

DATA = tb.DATA
RIDGE = 1.0
FEATURES = {
    "A": ["logRe", "CL", "CL2", "logRe_CL"],
    "B": ["logRe", "CL", "CL2", "logRe_CL", "camber", "thickness"],
    "C": ["logRe", "CL", "CL2", "logRe_CL", "camber", "thickness", "log_unconf"],
}


def shapes():
    """Camber and thickness (fraction of chord) for every geometry key."""
    u = pd.read_csv(os.path.join(DATA, "uiuc_kulfan_fit.csv")).set_index("asb_name")
    s = pd.read_csv(os.path.join(DATA, "soartech8_kulfan_fit.csv")).set_index("model")
    meas = s.primary_source == "measured"
    sc = pd.Series(np.where(meas, s.meas_camber, s.design_camber), index=s.index)
    st = pd.Series(np.where(meas, s.meas_thickness, s.design_thickness), index=s.index)
    return pd.concat([u.max_camber, sc]), pd.concat([u.max_thickness, st])


CAMBER, THICK = shapes()


def design_matrix(d):
    cl = d.NF_CL.to_numpy(float)
    lr = np.log10(d.Re.to_numpy(float) / 1e5)
    cols = dict(logRe=lr, CL=cl, CL2=cl ** 2, logRe_CL=lr * cl,
                camber=d.camber.to_numpy(float) * 100, thickness=d.thickness.to_numpy(float) * 100)
    if "NF_conf" in d:
        cols["log_unconf"] = np.log10(1.001 - np.clip(d.NF_conf.to_numpy(float), 0, 1))
    return pd.DataFrame(cols)


def with_shape(d):
    d = d.copy()
    if "camber" not in d:
        d["camber"] = d.geom_key.map(CAMBER)
        d["thickness"] = d.geom_key.map(THICK)
    return d


class Correction:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, d):
        X = design_matrix(d)[self.feats].to_numpy()
        y = np.log(d.WT_CD.to_numpy(float) / d.NF_CD.to_numpy(float))
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        A = Z.T @ Z + RIDGE * np.eye(Z.shape[1])
        self.b0 = y.mean()
        self.b = np.linalg.solve(A, Z.T @ (y - self.b0))
        return self

    def predict(self, d):
        Z = (design_matrix(d)[self.feats].to_numpy() - self.mu) / self.sd
        return d.NF_CD.to_numpy(float) * np.exp(self.b0 + np.einsum("ij,j->i", Z, self.b))  # einsum: matmul raises false warnings on macOS numpy 2.0


def candidate(feats):
    def cand(train, test):
        return Correction(feats).fit(with_shape(train)).predict(with_shape(test))
    return cand


def by_re(d, model):
    """Error by Re band, cross-tunnel predictions for every point."""
    out = []
    for split, tr, te in tb.splits(d):
        te = te.assign(cd_corr=candidate(FEATURES[model])(tr, te))
        out.append(te)
    p = pd.concat(out)
    p["err_nf"] = (p.NF_CD - p.WT_CD).abs() / p.WT_CD
    p["err_corr"] = (p.cd_corr - p.WT_CD).abs() / p.WT_CD
    return (p.groupby(["tunnel", "Re_bin"]).agg(n=("err_nf", "size"), neuralfoil=("err_nf", "mean"),
                                                 corrected=("err_corr", "mean")).reset_index())


def ltpt_test(model):
    d = tb.load()
    m = Correction(FEATURES[model]).fit(with_shape(d))
    t = pd.read_csv(os.path.join(DATA, "ltpt_neuralfoil_validation.csv"))
    t = t[(t.cd > 0) & ~t.hysteresis.astype(bool) & ~t.turbulator.astype(bool)].copy()
    t["camber"], t["thickness"] = CAMBER["e387"], THICK["e387"]
    t["WT_CD"] = t.cd
    corr = m.predict(t)
    e_nf = np.abs(t.NF_CD - t.cd) / t.cd
    e_c = np.abs(corr - t.cd) / t.cd
    return pd.DataFrame([dict(model=model, n_points=len(t), neuralfoil=e_nf.mean(), corrected=e_c.mean(),
                              neuralfoil_median=e_nf.median(), corrected_median=e_c.median(),
                              share_improved=(e_c < e_nf).mean())])


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    cands = {"neuralfoil": tb.neuralfoil, **{f"correction_{k}": candidate(v) for k, v in FEATURES.items()}}
    res = tb.run(cands)
    tb.check_baseline(res)
    res.to_csv(os.path.join(DATA, "drag_correction_bench.csv"), index=False, float_format="%.6g")
    pct = ["mean_abs_errCD", "ci_lo", "ci_hi", "vs_neuralfoil", "vs_nf_ci_lo", "vs_nf_ci_hi"]
    print(res.assign(**{c: (res[c] * 100).round(2) for c in pct})[["candidate", "split"] + pct].to_string(index=False))

    cross = res[res.split.isin(["UIUC->Princeton", "Princeton->UIUC"])]
    usable = cross[cross.candidate.isin(["correction_A", "correction_B"])]
    best = usable.groupby("candidate").mean_abs_errCD.mean().idxmin().split("_")[1]
    rb = by_re(tb.load(), best)
    rb.to_csv(os.path.join(DATA, "drag_correction_by_Re.csv"), index=False, float_format="%.6g")
    print(f"\nBy Re band, model {best} (cross-tunnel):")
    print(rb.assign(neuralfoil=(rb.neuralfoil * 100).round(1), corrected=(rb.corrected * 100).round(1)).to_string(index=False))

    lt = ltpt_test(best)
    lt.to_csv(os.path.join(DATA, "drag_correction_ltpt.csv"), index=False, float_format="%.6g")
    print("\nLocked test, NASA Langley LTPT E387 (scored once):")
    print(lt.to_string(index=False))
