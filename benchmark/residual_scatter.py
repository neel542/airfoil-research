"""
What is the drag error made of after the best smooth correction? Plan entry 3.

Takes the out-of-fold residuals of correction A (10-fold by airfoil), r = log(CD_tunnel / CD_corrected),
and splits them into an offset per airfoil, an offset per polar (one airfoil at one Reynolds number),
and scatter inside a polar. Checks whether the airfoil offset follows camber or thickness, where in
lift the squared residual sits, and whether a more flexible smooth correction (E) beats A.

Inputs : data/xfoil_decomposition.csv, via testbench.py and drag_correction.py
Outputs: data/residual_scatter.csv   one row per measured quantity, in the order of the plan
         data/drag_correction_E_bench.csv   bench results for correction E
Run    : python benchmark/residual_scatter.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import drag_correction as dc
import testbench as tb

warnings.simplefilter("ignore")
DATA = tb.DATA


def design_E(d):
    X = dc.design_matrix(d)
    X["CL3"] = X.CL ** 3
    X["logRe2"] = X.logRe ** 2
    X["logRe2_CL"] = X.logRe ** 2 * X.CL
    X["logRe_CL2"] = X.logRe * X.CL ** 2
    return X


FEATURES_E = ["logRe", "CL", "CL2", "logRe_CL", "CL3", "logRe2", "logRe2_CL", "logRe_CL2"]


class CorrectionE(dc.Correction):
    """Correction A with four more smooth terms. Same ridge, same target."""

    def fit(self, d):
        X = design_E(d)[self.feats].to_numpy()
        y = np.log(d.WT_CD.to_numpy(float) / d.NF_CD.to_numpy(float))
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        self.b0 = y.mean()
        self.b = np.linalg.solve(Z.T @ Z + dc.RIDGE * np.eye(Z.shape[1]), Z.T @ (y - self.b0))
        return self

    def predict(self, d):
        Z = (design_E(d)[self.feats].to_numpy() - self.mu) / self.sd
        return d.NF_CD.to_numpy(float) * np.exp(self.b0 + np.einsum("ij,j->i", Z, self.b))


def candidate_E(train, test):
    return CorrectionE(FEATURES_E).fit(dc.with_shape(train)).predict(dc.with_shape(test))


def decompose(d):
    r = d.r.to_numpy()
    mu = r.mean()
    d = d.assign(r=r)
    a = d.groupby("airfoil").r.transform("mean")
    p = d.groupby("polar").r.transform("mean")
    ss_tot = ((r - mu) ** 2).sum()
    # polar means are taken inside an airfoil: an airfoil's polars all share its airfoil
    pa = d.assign(p=p).groupby("polar").agg(n=("r", "size"), pm=("p", "first"), am=("r", lambda x: 0)).pm
    ss_air = ((a - mu) ** 2).sum()
    ss_pol = ((p - a) ** 2).sum()
    ss_in = ((r - p) ** 2).sum()
    return dict(ss_total=ss_tot, between_airfoil=ss_air / ss_tot, between_polar=ss_pol / ss_tot,
                within_polar=ss_in / ss_tot, n_points=len(d), n_airfoils=d.airfoil.nunique(),
                n_polars=d.polar.nunique(), points_per_polar=len(d) / d.polar.nunique())


def main():
    d = tb.load()
    d = dc.with_shape(d)
    pred = tb.cv_predict(d, dc.candidate(dc.FEATURES["A"]))
    d["corrected"] = pred
    d["r"] = np.log(d.WT_CD / d.corrected)
    d = d[np.isfinite(d.r)].reset_index(drop=True)
    out = []

    # Analysis 1
    dec = decompose(d)
    for k in ["between_airfoil", "between_polar", "within_polar"]:
        out.append(dict(analysis="1 share of residual sum of squares", item=k, value=dec[k]))
    out.append(dict(analysis="1 context", item="points per polar", value=dec["points_per_polar"]))
    out.append(dict(analysis="1 context", item="sd of residual (log units)", value=d.r.std()))

    # Analysis 2
    am = d.groupby("airfoil").agg(r=("r", "mean"), camber=("camber", "first"), thickness=("thickness", "first"),
                                  n=("r", "size"))
    for col in ["camber", "thickness"]:
        rho = stats.spearmanr(am.r, am[col]).correlation
        out.append(dict(analysis="2 spearman of airfoil mean residual", item=col, value=rho))

    # Analysis 3
    ext = (d.WT_CL <= 0) | (d.WT_CL >= 1.2)
    ss = (d.r - d.r.mean()) ** 2
    out.append(dict(analysis="3 CL extremes", item="share of points", value=ext.mean()))
    out.append(dict(analysis="3 CL extremes", item="share of squared residual", value=ss[ext].sum() / ss.sum()))
    bands = pd.cut(d.WT_CL, [-5, 0, 0.3, 0.6, 0.9, 1.2, 5])
    for b, g in d.groupby(bands, observed=True):
        out.append(dict(analysis="3 mean |residual| by CL band (percent)", item=str(b), value=100 * g.r.abs().mean()))

    # Candidate E on the bench
    res = tb.run({"neuralfoil": tb.neuralfoil, "correction_A": dc.candidate(dc.FEATURES["A"]),
                  "correction_E": candidate_E})
    tb.check_baseline(res)
    res.to_csv(os.path.join(DATA, "drag_correction_E_bench.csv"), index=False, float_format="%.6g")
    for _, row in res[res.candidate != "neuralfoil"].iterrows():
        out.append(dict(analysis="4 change from NeuralFoil, points", item=f"{row.candidate} {row.split}",
                        value=100 * row.vs_neuralfoil))

    table = pd.DataFrame(out)
    table.to_csv(os.path.join(DATA, "residual_scatter.csv"), index=False, float_format="%.6g")
    pd.set_option("display.width", 140)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
