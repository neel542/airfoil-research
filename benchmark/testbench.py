"""
The test bench for improving NeuralFoil's drag: fixed points, fixed splits, the baseline,
and a slot for any new prediction.

A candidate is a function candidate(train, test) that returns a predicted drag
coefficient for every row of test. It may learn from train and nothing else. The bench
scores it on test with the paper's metric, |CD_pred - CD_tunnel| / CD_tunnel, and gives
95 percent intervals by resampling whole airfoils (an airfoil's points stay together).
It also scores the improvement over NeuralFoil on the same resamples, so "better" comes
with its own interval.

Splits (fixed; never change them to suit a result):
  UIUC->Princeton   learn on the UIUC tunnel, score on Princeton
  Princeton->UIUC   learn on Princeton, score on UIUC
  airfoil-CV        10 folds over airfoils from both tunnels; an airfoil measured in
                    both tunnels sits in one fold, so it is never in train and test at once

Points  : data/xfoil_decomposition.csv, all 9,130 matched points (clean models, free
          transition, NeuralFoil "large"), with NeuralFoil and XFoil at each.
Output  : data/testbench_results.csv
Run     : python benchmark/testbench.py

To add an idea, write a function like neuralfoil() below and add it to CANDIDATES.
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
B = 2000
N_FOLDS = 10
SEED = 0


def load():
    d = pd.read_csv(os.path.join(DATA, "xfoil_decomposition.csv"))
    return d.reset_index(drop=True)


# ── Candidates ──────────────────────────────────────────────────────────────
def neuralfoil(train, test):
    """The baseline: NeuralFoil as it is."""
    return test.NF_CD.to_numpy(float)


def xfoil(train, test):
    """Reference: XFoil at the same points (missing where XFoil did not converge)."""
    return test.XF_CD.where(test.xf_converged.astype(bool)).to_numpy(float)


CANDIDATES = {"neuralfoil": neuralfoil, "xfoil": xfoil}


# ── Splits ──────────────────────────────────────────────────────────────────
def splits(d):
    yield "UIUC->Princeton", d[d.tunnel == "UIUC"], d[d.tunnel == "Princeton"]
    yield "Princeton->UIUC", d[d.tunnel == "Princeton"], d[d.tunnel == "UIUC"]


def cv_predict(d, cand):
    airfoils = np.array(sorted(d.airfoil.unique()))
    rng = np.random.default_rng(SEED)
    fold_of = dict(zip(airfoils, rng.permutation(len(airfoils)) % N_FOLDS))
    fold = d.airfoil.map(fold_of).to_numpy()
    pred = np.full(len(d), np.nan)
    for k in range(N_FOLDS):
        te = fold == k
        pred[te] = cand(d[~te], d[te])
    return pred


# ── Scoring ─────────────────────────────────────────────────────────────────
def abs_err(pred, wt):
    return np.abs(pred - wt) / wt


def cluster_boot(err, err_base, airfoil):
    """Mean |error|, and its difference from the baseline, resampling airfoils."""
    ok = np.isfinite(err) & np.isfinite(err_base)
    keys, inv = np.unique(airfoil[ok], return_inverse=True)
    s = np.bincount(inv, err[ok]); sb = np.bincount(inv, err_base[ok]); n = np.bincount(inv)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    m = s[idx].sum(1) / n[idx].sum(1)
    mb = sb[idx].sum(1) / n[idx].sum(1)
    return m, m - mb


def score(name, split, test, pred):
    wt = test.WT_CD.to_numpy(float)
    e = abs_err(pred, wt)
    eb = abs_err(test.NF_CD.to_numpy(float), wt)
    ok = np.isfinite(e)
    m, diff = cluster_boot(e, eb, test.airfoil.to_numpy())
    return dict(candidate=name, split=split,
                mean_abs_errCD=np.nanmean(e), ci_lo=np.percentile(m, 2.5), ci_hi=np.percentile(m, 97.5),
                median_abs_errCD=np.nanmedian(e),
                vs_neuralfoil=np.nanmean(e[ok]) - np.nanmean(eb[ok]),
                vs_nf_ci_lo=np.percentile(diff, 2.5), vs_nf_ci_hi=np.percentile(diff, 97.5),
                n_points=int(ok.sum()), coverage=ok.mean(), n_airfoils=test.airfoil[ok].nunique())


def run(candidates=CANDIDATES):
    d = load()
    rows = []
    for name, cand in candidates.items():
        for split, tr, te in splits(d):
            rows.append(score(name, split, te, cand(tr, te)))
        rows.append(score(name, "airfoil-CV", d, cv_predict(d, cand)))
    return pd.DataFrame(rows)


def check_baseline(res):
    """The bench must reproduce the paper's NeuralFoil numbers exactly."""
    paper = pd.read_csv(os.path.join(DATA, "clustered_statistics.csv"))
    paper = paper[paper.statistic == "mean_abs_errCD"].set_index("tunnel").estimate
    nf = res[res.candidate == "neuralfoil"].set_index("split").mean_abs_errCD
    for split, tunnel in [("UIUC->Princeton", "Princeton"), ("Princeton->UIUC", "UIUC")]:
        assert abs(nf[split] - paper[tunnel]) < 1e-12, (split, nf[split], paper[tunnel])


if __name__ == "__main__":
    res = run()
    check_baseline(res)
    res.to_csv(os.path.join(DATA, "testbench_results.csv"), index=False, float_format="%.6g")
    pd.set_option("display.width", 160)
    show = res.assign(**{c: (res[c] * 100).round(2) for c in
                         ["mean_abs_errCD", "ci_lo", "ci_hi", "median_abs_errCD", "vs_neuralfoil", "vs_nf_ci_lo", "vs_nf_ci_hi"]})
    print(show[["candidate", "split", "mean_abs_errCD", "ci_lo", "ci_hi", "median_abs_errCD",
                "vs_neuralfoil", "vs_nf_ci_lo", "vs_nf_ci_hi", "n_points", "n_airfoils"]].to_string(index=False))
    print("\nPercent. vs_neuralfoil < 0 means better than NeuralFoil; it counts only if the whole interval is below 0.")
    print("Baseline matches data/clustered_statistics.csv. Wrote data/testbench_results.csv")
