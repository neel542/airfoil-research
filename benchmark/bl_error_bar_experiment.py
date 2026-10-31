"""
Does NeuralFoil's boundary-layer state help predict how big its drag error is? Plan entry 5.

The target is the size of NeuralFoil's own relative drag error, y = max(|CD_NF - CD_tunnel| / CD_tunnel, 1e-4),
at the bench points with |CL_tunnel| > 0.1 (the domain of the published error model). The models are the
Gamma regressions of fit_error_model.py (log link, same fitting routine), with and without seven
boundary-layer inputs from NeuralFoil: xs, xs^2, xp, log10(Hs), logtheta_s, jump_s, jump_p.

  baselines   constant, confidence_only, confidence_Re, full           (as in fit_error_model.py)
  candidates  confidence_Re_BL, full_BL, BL_Re

Splits are the bench's: UIUC->Princeton, Princeton->UIUC, and 10 folds over airfoils (same fold assignment
as testbench.py, taken before the attached-flow filter). Primary score: mean Gamma deviance on held-out
points; intervals resample whole airfoils (2,000 times) on the paired difference from confidence_Re.

Inputs : data/xfoil_decomposition.csv, data/nf_bl_state.csv, and the shape tables drag_correction.py reads
Outputs: data/bl_error_bar_bench.csv        every model on every split
         data/bl_error_bar_experiment.csv   the quantities the plan's predictions and success rule ask for
Run    : python benchmark/bl_error_bar_experiment.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import drag_correction as dc
import fit_error_model as fem
import testbench as tb

warnings.simplefilter("ignore")
DATA = tb.DATA
BL = ["xs", "xs2", "xp", "log_Hs", "logtheta_s", "jump_s", "jump_p"]
SPECS = dict(fem.SPECS)
SPECS["confidence_Re_BL"] = fem.SPECS["confidence_Re"] + BL
SPECS["full_BL"] = fem.SPECS["full"] + BL
SPECS["BL_Re"] = ["one", "log_Re"] + BL
BAR, BASE = "confidence_Re", "confidence_only"   # the bar to beat, and the one the plan question names


def load():
    d = dc.with_shape(tb.load())
    bl = pd.read_csv(os.path.join(DATA, "nf_bl_state.csv"))
    assert len(bl) == len(d)
    for c in bl.columns:
        d[c] = bl[c].to_numpy()
    airfoils = np.array(sorted(d.airfoil.unique()))
    rng = np.random.default_rng(tb.SEED)
    fold_of = dict(zip(airfoils, rng.permutation(len(airfoils)) % tb.N_FOLDS))   # as tb.cv_predict, before filtering
    d["fold"] = d.airfoil.map(fold_of)
    d = d[(d.WT_CL.abs() > 0.1) & (d.WT_CD > 0)].reset_index(drop=True)
    d["y"] = np.maximum((d.NF_CD - d.WT_CD).abs() / d.WT_CD, fem.Y_FLOOR)
    need = ["NF_conf", "Re", "alpha", "camber", "thickness", "xs", "xp", "Hs", "logtheta_s", "jump_s", "jump_p", "y"]
    assert np.isfinite(d[need].to_numpy(float)).all()
    return d


def design(d, spec):
    base = [c for c in spec if c not in BL]
    cols = [fem.features(d, base)[0]]
    extra = dict(xs=d["xs"].to_numpy(float), xs2=d["xs"].to_numpy(float) ** 2, xp=d.xp.to_numpy(float),
                 log_Hs=np.log10(d.Hs.to_numpy(float)), logtheta_s=d.logtheta_s.to_numpy(float),
                 jump_s=d.jump_s.to_numpy(float), jump_p=d.jump_p.to_numpy(float))
    ex = [c for c in spec if c in BL]
    if ex:
        cols.append(np.column_stack([extra[c] for c in ex]))
    return np.column_stack(cols)


def fit_predict(train, test, spec):
    beta, phi, _ = fem.fit_gamma(design(train, SPECS[spec]), train.y.to_numpy())
    mu = fem.predict(design(test, SPECS[spec]), beta)
    return mu, phi


def deviance(y, mu):
    return 2 * ((y - mu) / mu - np.log(y / mu))


def boot_diff(delta, airfoil):
    """Mean of delta and its 95% interval, resampling whole airfoils."""
    keys, inv = np.unique(airfoil, return_inverse=True)
    s = np.bincount(inv, delta); n = np.bincount(inv)
    idx = np.random.default_rng(tb.SEED).integers(0, len(keys), size=(tb.B, len(keys)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return delta.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def split_sets(d):
    yield "UIUC->Princeton", d[d.tunnel == "UIUC"], d[d.tunnel == "Princeton"]
    yield "Princeton->UIUC", d[d.tunnel == "Princeton"], d[d.tunnel == "UIUC"]


def predictions(d, spec):
    """Out-of-sample predicted size and dispersion, for each split. 10-fold: one phi per fold."""
    out = {}
    for name, tr, te in split_sets(d):
        mu, phi = fit_predict(tr, te, spec)
        out[name] = (te, mu, np.full(len(te), phi))
    mu = np.full(len(d), np.nan); ph = np.full(len(d), np.nan)
    for k in range(tb.N_FOLDS):
        te = (d.fold == k).to_numpy()
        mu[te], ph[te] = fit_predict(d[~te], d[te], spec)
    out["airfoil-CV"] = (d, mu, ph)
    return out


def main():
    d = load()
    print(f"{len(d)} attached-flow points, {d.airfoil.nunique()} airfoils "
          f"({(d.tunnel == 'UIUC').sum()} UIUC, {(d.tunnel == 'Princeton').sum()} Princeton)")
    preds = {m: predictions(d, m) for m in SPECS}
    rows = []
    for m in SPECS:
        for split, (te, mu, phi) in preds[m].items():
            y = te.y.to_numpy()
            dev = deviance(y, mu)
            row = dict(model=m, split=split, n_points=len(te), n_airfoils=te.airfoil.nunique(),
                       mean_deviance=dev.mean(), mae=np.mean(np.abs(y - mu)),
                       spearman=stats.spearmanr(mu, y).correlation,
                       coverage_80=np.mean(y <= fem.gamma_quantile(mu, phi, 0.8)),
                       coverage_95=np.mean(y <= fem.gamma_quantile(mu, phi, 0.95)))
            for tag, ref in [("vs_bar", BAR), ("vs_conf_only", BASE)]:
                dref = deviance(y, preds[ref][split][1])
                dm, lo, hi = boot_diff(dev - dref, te.airfoil.to_numpy())
                row.update({f"{tag}": dm, f"{tag}_lo": lo, f"{tag}_hi": hi,
                            f"{tag}_rel": dm / dref.mean()})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA, "bl_error_bar_bench.csv"), index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    show = ["model", "split", "mean_deviance", "mae", "spearman", "coverage_80", "vs_bar", "vs_bar_lo", "vs_bar_hi", "vs_bar_rel"]
    print(res[show].round(4).to_string(index=False))

    g = lambda m, s, c: float(res[(res.model == m) & (res.split == s)][c].iloc[0])
    cross = ["UIUC->Princeton", "Princeton->UIUC"]
    out = []
    for s in cross + ["airfoil-CV"]:
        out.append(dict(analysis="1 deviance change, confidence_Re_BL minus confidence_Re", item=s, value=g("confidence_Re_BL", s, "vs_bar")))
        out.append(dict(analysis="1 same, as a share of confidence_Re's deviance", item=s, value=g("confidence_Re_BL", s, "vs_bar_rel")))
        out.append(dict(analysis="3 interval upper end of that change", item=s, value=g("confidence_Re_BL", s, "vs_bar_hi")))
    for s in cross + ["airfoil-CV"]:
        out.append(dict(analysis="4 deviance, BL_Re minus confidence_Re", item=s, value=g("BL_Re", s, "vs_bar")))
    for s in cross + ["airfoil-CV"]:
        out.append(dict(analysis="5 share gained: full_BL over full", item=s,
                        value=-(g("full_BL", s, "mean_deviance") - g("full", s, "mean_deviance")) / g("full", s, "mean_deviance")))
        out.append(dict(analysis="5 share gained: confidence_Re_BL over confidence_Re", item=s, value=-g("confidence_Re_BL", s, "vs_bar_rel")))
    for m in [BAR, "confidence_Re_BL", "full", "full_BL"]:
        for s in cross + ["airfoil-CV"]:
            out.append(dict(analysis="6 coverage of the 80 percent band", item=f"{m} {s}", value=g(m, s, "coverage_80")))
    for s in cross + ["airfoil-CV"]:
        out.append(dict(analysis="6 spearman, predicted vs actual size", item=f"confidence_Re_BL {s}", value=g("confidence_Re_BL", s, "spearman")))
        out.append(dict(analysis="6 spearman, predicted vs actual size", item=f"confidence_Re {s}", value=g(BAR, s, "spearman")))

    sign = all(g("confidence_Re_BL", s, "vs_bar") < 0 for s in cross + ["airfoil-CV"])
    below = all(g("confidence_Re_BL", s, "vs_bar_hi") < 0 for s in cross)
    two = g("confidence_Re_BL", "airfoil-CV", "vs_bar_rel") <= -0.02
    rank = all(g("confidence_Re_BL", s, "spearman") > g(BAR, s, "spearman") for s in cross)
    out += [dict(analysis="2 success rule part", item="both cross-tunnel intervals below zero", value=float(below)),
            dict(analysis="2 success rule part", item="10-fold at least 2 percent lower", value=float(two)),
            dict(analysis="2 success rule part", item="spearman higher in both cross-tunnel splits", value=float(rank)),
            dict(analysis="2 success rule", item="all three", value=float(below and two and rank)),
            dict(analysis="2 sign right in all three splits", item="confidence_Re_BL below confidence_Re", value=float(sign))]

    # Post hoc, not in the plan: is the state worth adding to the published model ("full"), and does it match it?
    for a, b in [("full_BL", "full"), ("confidence_Re_BL", "full")]:
        for s in cross + ["airfoil-CV"]:
            te, mu_a, _ = preds[a][s]
            y = te.y.to_numpy()
            dd = deviance(y, mu_a) - deviance(y, preds[b][s][1])
            dm, lo, hi = boot_diff(dd, te.airfoil.to_numpy())
            out.append(dict(analysis=f"8 post hoc (not in the plan): deviance, {a} minus {b}", item=f"{s} change", value=dm))
            out.append(dict(analysis=f"8 post hoc (not in the plan): deviance, {a} minus {b}", item=f"{s} interval low", value=lo))
            out.append(dict(analysis=f"8 post hoc (not in the plan): deviance, {a} minus {b}", item=f"{s} interval high", value=hi))

    # What the boundary-layer inputs do, fitted on everything (descriptive only; not scored)
    X = design(d, SPECS["confidence_Re_BL"]); beta, phi, _ = fem.fit_gamma(X, d.y.to_numpy())
    names = SPECS["confidence_Re_BL"]
    sd = X.std(0)
    for n, b, s in zip(names, beta, sd):
        if n in BL or n in ("log_unconf", "log_Re"):
            out.append(dict(analysis="7 full-data fit: change in expected error size per 1 sd of the input (percent)", item=n,
                            value=100 * (np.exp(b * s) - 1)))
    table = pd.DataFrame(out)
    table.to_csv(os.path.join(DATA, "bl_error_bar_experiment.csv"), index=False, float_format="%.6g")
    print()
    print(table.to_string(index=False))

    # Sanity: baselines on this point set against the published error-model scores (different points and folds)
    pub = pd.read_csv(os.path.join(DATA, "error_model_cv.csv"))
    pub = pub[pub.scheme == "airfoil_10fold"].groupby("model").mae.mean()
    mine = res[res.split == "airfoil-CV"].set_index("model").mae
    print("\n10-fold MAE of the predicted size, this point set vs fit_error_model.py's published run:")
    for m in ["constant", "confidence_only", "confidence_Re", "full"]:
        print(f"  {m:>16}: {mine[m]:.4f} vs {pub[m]:.4f}")


if __name__ == "__main__":
    main()
