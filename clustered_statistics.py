"""
Honest uncertainty for the benchmark statistics.

The benchmark points are not independent: each polar is an angle-of-attack
sweep on one model at one Reynolds number, and each airfoil contributes many
polars. A confidence interval that treats 4,428 points as 4,428 independent
draws is far too narrow. This script re-computes every headline statistic with
a cluster bootstrap that resamples *airfoils* (with replacement, all of their
points together), alongside the naive point bootstrap, so the two can be
compared.

It also splits the confidence-versus-drag-error correlation into the part that
lives between airfoils, within airfoils, and within single polars, to check
whether the point-level correlation is a real between-airfoil signal or mostly
alpha-sweep structure.

Inputs  : data/uiuc_neuralfoil_validation.csv, data/soartech8_neuralfoil_validation.csv
Outputs : data/clustered_statistics.csv                 estimate + cluster CI + naive CI
          data/confidence_correlation_decomposition.csv  total / between / within correlations
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
B = 2000
rng = np.random.default_rng(0)


def load():
    u = pd.read_csv(os.path.join(DATA, "uiuc_neuralfoil_validation.csv"))
    u = u[(u.config == "clean") & (u.NF_mode == "free") & (u.model == "large") & u.fit_ok].copy()
    u["airfoil"] = u.asb_name
    u["polar"] = u.asb_name + "|" + u.file + "|" + u.Re.astype(str)
    s = pd.read_csv(os.path.join(DATA, "soartech8_neuralfoil_validation.csv"))
    s = s[(s.config == "clean") & (s.NF_mode == "free") & (s.nf == "large") & s.primary & s.fit_ok].copy()
    s["airfoil"] = s.family
    s["polar"] = s.airfoil_label + "|" + s.Re.astype(str)
    cols = ["airfoil", "polar", "Re", "alpha", "WT_CL", "WT_CD", "NF_CL", "NF_CD", "NF_conf", "dCL", "err_CD"]
    return {"UIUC": u[cols].reset_index(drop=True), "Princeton": s[cols].reset_index(drop=True)}


# Statistics of interest, each a function of a point table -------------------
def stat_fns():
    def ld(d):
        d = d[d.WT_CL > 0.2]
        return (d.NF_CL / d.NF_CD - d.WT_CL / d.WT_CD) / (d.WT_CL / d.WT_CD)

    def att(d):
        return d[d.WT_CL.abs() > 0.1]

    def by_af(d):
        g = d.assign(abs_err=d.err_CD.abs()).groupby("airfoil")
        return pd.DataFrame(dict(conf=g.NF_conf.mean(), err=g.abs_err.mean()))

    return {
        "mean_abs_dCL": lambda d: d.dCL.abs().mean(),
        "bias_CL": lambda d: d.dCL.mean(),
        "mean_abs_errCD": lambda d: d.err_CD.abs().mean(),
        "median_abs_errCD": lambda d: d.err_CD.abs().median(),
        "bias_CD": lambda d: d.err_CD.mean(),
        "median_abs_errLD": lambda d: ld(d).abs().median(),
        "bias_LD": lambda d: ld(d).mean(),
        "r_conf_absErrCD": lambda d: np.corrcoef(att(d).NF_conf, att(d).err_CD.abs())[0, 1],
        "r_conf_absDCL": lambda d: np.corrcoef(att(d).NF_conf, att(d).dCL.abs())[0, 1],
        "rho_airfoil_conf_errCD": lambda d: stats.spearmanr(by_af(att(d)).conf, by_af(att(d)).err).correlation,
        "errCD_conf_below_0.5": lambda d: att(d)[att(d).NF_conf <= 0.5].err_CD.abs().mean(),
        "errCD_conf_above_0.95": lambda d: att(d)[att(d).NF_conf > 0.95].err_CD.abs().mean(),
    }


def cluster_bootstrap(d, fns, cluster_col):
    """Resample clusters with replacement; return {stat: array of B values}."""
    groups = {k: v for k, v in d.groupby(cluster_col)}
    keys = np.array(list(groups))
    out = {k: np.empty(B) for k in fns}
    for b in range(B):
        pick = rng.choice(keys, size=len(keys), replace=True)
        # give each resampled copy a distinct cluster id so airfoil-level stats
        # count duplicates as separate draws, the way a bootstrap should
        parts = []
        for i, k in enumerate(pick):
            g = groups[k]
            if cluster_col != "airfoil":
                parts.append(g)
            else:
                parts.append(g.assign(airfoil=f"{k}#{i}"))
        db = pd.concat(parts, ignore_index=True)
        for name, fn in fns.items():
            try:
                out[name][b] = fn(db)
            except Exception:
                out[name][b] = np.nan
    return out


def point_bootstrap(d, fns):
    out = {k: np.empty(B) for k in fns}
    n = len(d)
    for b in range(B):
        db = d.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        for name, fn in fns.items():
            try:
                out[name][b] = fn(db)
            except Exception:
                out[name][b] = np.nan
    return out


def ci(a):
    a = a[np.isfinite(a)]
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def decompose_correlation(d):
    """Total, between-airfoil, within-airfoil, between-polar and within-polar
    Pearson correlations of confidence with |drag error| (attached flow)."""
    a = d[d.WT_CL.abs() > 0.1].copy()
    a["e"] = a.err_CD.abs()
    r_total = np.corrcoef(a.NF_conf, a.e)[0, 1]
    af = a.groupby("airfoil")[["NF_conf", "e"]].mean()
    r_between_af = np.corrcoef(af.NF_conf, af.e)[0, 1]
    w = a[["NF_conf", "e"]] - a.groupby("airfoil")[["NF_conf", "e"]].transform("mean")
    r_within_af = np.corrcoef(w.NF_conf, w.e)[0, 1]
    pol = a.groupby("polar")[["NF_conf", "e"]].mean()
    r_between_polar = np.corrcoef(pol.NF_conf, pol.e)[0, 1]
    wp = a[["NF_conf", "e"]] - a.groupby("polar")[["NF_conf", "e"]].transform("mean")
    r_within_polar = np.corrcoef(wp.NF_conf, wp.e)[0, 1]
    # variance shares: how much of the confidence spread is between airfoils vs within
    var_total = a.NF_conf.var()
    var_between_af = a.groupby("airfoil").NF_conf.transform("mean").var()
    var_between_polar = a.groupby("polar").NF_conf.transform("mean").var()
    return dict(n_points=len(a), n_airfoils=a.airfoil.nunique(), n_polars=a.polar.nunique(),
                r_total=r_total, r_between_airfoil=r_between_af, r_within_airfoil=r_within_af,
                r_between_polar=r_between_polar, r_within_polar=r_within_polar,
                conf_var_share_between_airfoil=var_between_af / var_total,
                conf_var_share_between_polar=var_between_polar / var_total,
                rho_airfoil_spearman=stats.spearmanr(af.NF_conf, af.e).correlation,
                rho_airfoil_p=stats.spearmanr(af.NF_conf, af.e).pvalue)


def main():
    data = load()
    fns = stat_fns()
    rows, dec = [], []
    for tunnel, d in data.items():
        print(f"\n{tunnel}: {len(d)} points, {d.airfoil.nunique()} airfoils, {d.polar.nunique()} polars")
        est = {k: fn(d) for k, fn in fns.items()}
        cb = cluster_bootstrap(d, fns, "airfoil")
        pb = point_bootstrap(d, fns)
        for k in fns:
            lo, hi = ci(cb[k]); nlo, nhi = ci(pb[k])
            rows.append(dict(tunnel=tunnel, statistic=k, estimate=est[k],
                             cluster_ci_lo=lo, cluster_ci_hi=hi, naive_ci_lo=nlo, naive_ci_hi=nhi,
                             ci_width_ratio=(hi - lo) / max(nhi - nlo, 1e-12),
                             n_points=len(d), n_clusters=d.airfoil.nunique()))
            print(f"  {k:>24}: {est[k]:7.3f}   airfoil-cluster 95% [{lo:7.3f}, {hi:7.3f}]   "
                  f"naive [{nlo:7.3f}, {nhi:7.3f}]   width x{(hi - lo) / max(nhi - nlo, 1e-12):.1f}")
        r = decompose_correlation(d); r["tunnel"] = tunnel
        dec.append(r)
        print("  confidence vs |drag error| correlation: "
              + ", ".join(f"{k}={v:.3f}" for k, v in r.items() if k.startswith("r_")))
        print(f"  share of confidence variance between airfoils {r['conf_var_share_between_airfoil']:.2f}, "
              f"between polars {r['conf_var_share_between_polar']:.2f}")

    # Pooled two-tunnel estimate (airfoil clusters keyed by tunnel so the same
    # section in both tunnels counts as two independent measurements of it)
    pooled = pd.concat([d.assign(airfoil=t + ":" + d.airfoil, polar=t + ":" + d.polar)
                        for t, d in data.items()], ignore_index=True)
    est = {k: fn(pooled) for k, fn in fns.items()}
    cb = cluster_bootstrap(pooled, fns, "airfoil")
    pb = point_bootstrap(pooled, fns)
    print(f"\nPooled: {len(pooled)} points, {pooled.airfoil.nunique()} tunnel-airfoil clusters")
    for k in fns:
        lo, hi = ci(cb[k]); nlo, nhi = ci(pb[k])
        rows.append(dict(tunnel="pooled", statistic=k, estimate=est[k],
                         cluster_ci_lo=lo, cluster_ci_hi=hi, naive_ci_lo=nlo, naive_ci_hi=nhi,
                         ci_width_ratio=(hi - lo) / max(nhi - nlo, 1e-12),
                         n_points=len(pooled), n_clusters=pooled.airfoil.nunique()))
        print(f"  {k:>24}: {est[k]:7.3f}   cluster [{lo:7.3f}, {hi:7.3f}]   naive [{nlo:7.3f}, {nhi:7.3f}]")
    r = decompose_correlation(pooled); r["tunnel"] = "pooled"; dec.append(r)

    pd.DataFrame(rows).to_csv(os.path.join(DATA, "clustered_statistics.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(DATA, "confidence_correlation_decomposition.csv"), index=False)
    print("\nwrote data/clustered_statistics.csv, data/confidence_correlation_decomposition.csv")


if __name__ == "__main__":
    main()
