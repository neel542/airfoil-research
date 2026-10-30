"""
Does NeuralFoil's own boundary-layer state explain the scatter left after correction A? Plan entry 4.

Candidates (ridge 1.0 on standardized inputs, target log(CD_tunnel / CD_NeuralFoil), bench splits):
  F = A + xs, xs^2, xp, Hs, logtheta_s
  G = F + jump_s, jump_p, xs * logRe
with xs, xp the suction-side and pressure-side transition locations, Hs and logtheta_s the
trailing-edge shape factor and log momentum thickness on the suction side, and jump_s, jump_p
how fast transition moves per degree. All from NeuralFoil alone (data/nf_bl_state.csv).

Also: the within-polar correlations of the correction-A residual with xs and jump_s, the split of
the leftover squared error after F and G, and how much of the within-polar scatter
measurement noise could account for (from the repeat-model pairs).

Inputs : data/xfoil_decomposition.csv, data/nf_bl_state.csv, data/repeatability_same_model.csv
Outputs: data/bl_state_bench.csv        bench results for A, F, G
         data/bl_state_experiment.csv   the other measured quantities, in the order of the plan
Run    : python benchmark/bl_state_experiment.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import drag_correction as dc
import residual_scatter as rs
import testbench as tb

warnings.simplefilter("ignore")
DATA = tb.DATA

A_FEATS = dc.FEATURES["A"]
F_FEATS = A_FEATS + ["xs", "xs2", "xp", "Hs", "logtheta_s"]
G_FEATS = F_FEATS + ["jump_s", "jump_p", "xs_logRe"]


_bench_load = tb.load


def design_bl(d):
    X = dc.design_matrix(d)
    for c in ["xs", "xp", "Hs", "logtheta_s", "jump_s", "jump_p"]:
        X[c] = d[c].to_numpy(float)
    X["xs2"] = X["xs"] ** 2
    X["xs_logRe"] = X["xs"] * X["logRe"]
    return X


class CorrectionBL(dc.Correction):
    def fit(self, d):
        X = design_bl(d)[self.feats].to_numpy()
        y = np.log(d.WT_CD.to_numpy(float) / d.NF_CD.to_numpy(float))
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, y = X[ok], y[ok]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        self.b0 = y.mean()
        self.b = np.linalg.solve(Z.T @ Z + dc.RIDGE * np.eye(Z.shape[1]), Z.T @ (y - self.b0))
        return self

    def predict(self, d):
        Z = (design_bl(d)[self.feats].to_numpy() - self.mu) / self.sd
        return d.NF_CD.to_numpy(float) * np.exp(self.b0 + np.einsum("ij,j->i", Z, self.b))


def cand(feats):
    def f(train, test):
        return CorrectionBL(feats).fit(dc.with_shape(train)).predict(dc.with_shape(test))
    return f


def load():
    d = _bench_load()
    bl = pd.read_csv(os.path.join(DATA, "nf_bl_state.csv"))
    assert len(bl) == len(d)
    for c in bl.columns:
        d[c] = bl[c].to_numpy()
    return d


def residuals(d, fn):
    p = tb.cv_predict(d, fn)
    r = np.log(d.WT_CD.to_numpy(float) / p)
    return r, np.isfinite(r)


def noise_share(within_var):
    r = pd.read_csv(os.path.join(DATA, "repeatability_same_model.csv"))
    r = r[r.n >= 3]
    mean_abs_diff = np.average(r.mean_abs_errCD, weights=r.n)          # relative difference between two runs
    sigma_diff = mean_abs_diff * np.sqrt(np.pi / 2)                    # sd of a difference, if roughly normal
    sigma_single = sigma_diff / np.sqrt(2)                             # one run's noise
    return mean_abs_diff, sigma_single, sigma_single ** 2 / within_var


def main():
    tb.load = load   # the bench reads its points through tb.load; hand it the same points plus the state columns
    d = dc.with_shape(load())
    res = tb.run({"neuralfoil": tb.neuralfoil, "correction_A": cand(A_FEATS),
                  "correction_F": cand(F_FEATS), "correction_G": cand(G_FEATS)})
    tb.check_baseline(res)
    res.to_csv(os.path.join(DATA, "bl_state_bench.csv"), index=False, float_format="%.6g")
    pct = ["mean_abs_errCD", "ci_lo", "ci_hi", "vs_neuralfoil", "vs_nf_ci_lo", "vs_nf_ci_hi"]
    pd.set_option("display.width", 150)
    print(res.assign(**{c: (res[c] * 100).round(2) for c in pct})[["candidate", "split"] + pct].to_string(index=False))

    out = []
    cv = res[res.split == "airfoil-CV"].set_index("candidate")
    for k in ["correction_A", "correction_F", "correction_G"]:
        out.append(dict(analysis="1 change from NeuralFoil, 10-fold (points)", item=k, value=100 * cv.loc[k, "vs_neuralfoil"]))
    for k in ["F", "G"]:
        out.append(dict(analysis="1 gain beyond A, 10-fold (points)", item=f"correction_{k}",
                        value=100 * (cv.loc["correction_A", "vs_neuralfoil"] - cv.loc[f"correction_{k}", "vs_neuralfoil"])))
    cross = res[res.split.isin(["UIUC->Princeton", "Princeton->UIUC"])].pivot(index="split", columns="candidate", values="mean_abs_errCD")
    for k in ["F", "G"]:
        out.append(dict(analysis="1 lower than A in both directions", item=f"correction_{k}",
                        value=float((cross[f"correction_{k}"] < cross["correction_A"]).all())))
    sub = res[res.split.isin(["UIUC->Princeton", "Princeton->UIUC"])]
    for k in ["F", "G"]:
        out.append(dict(analysis="1 both cross-tunnel intervals below zero", item=f"correction_{k}",
                        value=float((sub[sub.candidate == f"correction_{k}"].vs_nf_ci_hi < 0).all())))

    # Within-polar share of the leftover squared error, A versus F and G
    for k, f in [("A", A_FEATS), ("F", F_FEATS), ("G", G_FEATS)]:
        r, ok = residuals(d, cand(f))
        dd = d[ok].assign(r=r[ok])
        dec = rs.decompose(dd)
        out.append(dict(analysis="3 within-polar share of leftover squared error", item=f"after {k}", value=dec["within_polar"]))
        out.append(dict(analysis="3 sd of leftover residual (log units)", item=f"after {k}", value=dd.r.std()))

    # Within-polar correlations of the correction-A residual with the boundary-layer state
    rA, ok = residuals(d, cand(A_FEATS))
    dd = d[ok].assign(r=rA[ok])
    dd["absr"] = dd.r.abs()
    for col in ["xs", "jump_s"]:
        x = dd[col] - dd.groupby("polar")[col].transform("mean")
        y = dd.absr - dd.groupby("polar").absr.transform("mean")
        out.append(dict(analysis="4 within-polar spearman of |residual| with", item=col, value=stats.spearmanr(x, y).correlation))

    # Measurement noise
    within_var = rs.decompose(dd)["within_polar"] * dd.r.var()
    mad, sig, share = noise_share(within_var)
    out.append(dict(analysis="5 repeat runs, mean relative difference", item="two runs of one model", value=mad))
    out.append(dict(analysis="5 single-measurement noise (log units)", item="from repeat pairs", value=sig))
    out.append(dict(analysis="5 share of within-polar variance noise could explain", item="noise var / within var", value=share))

    table = pd.DataFrame(out)
    table.to_csv(os.path.join(DATA, "bl_state_experiment.csv"), index=False, float_format="%.6g")
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
