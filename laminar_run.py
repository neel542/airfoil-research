"""
Does the drag error track the length of the laminar run? No.

Nikhil Khobragade (IIT Madras) proposed this test. The background is the
n_crit result in section 3.2: dropping n_crit from 9 to 7 helps at Re =
60,000 and hurts everywhere else, so the flow appears to want more freestream
turbulence at low speed than at high speed. One reading is that a tunnel
running slower genuinely has a higher turbulence intensity, since intensity
is a fluctuation divided by a mean speed. The other is that e^N is being
asked for something it does not carry: it predicts where transition starts,
not how long a separation bubble runs or when it bursts, and at these speeds
the bubble is most of the drag. On the second reading the drag error should
grow with the length of the laminar run, which XFoil reports at every point
as Top_Xtr and Bot_Xtr.

Three traps, and the third decides the answer.

  1. Laminar run length and drag error are both governed by Reynolds number,
     so a raw correlation mostly rediscovers the Reynolds trend already in
     section 3.1. Correlate inside Reynolds bands and partial out log(Re).
  2. Laminar run length is very nearly a restatement of angle of attack, so
     partial out |alpha| too, and check inside narrow alpha windows where the
     remaining spread comes from airfoil shape rather than incidence.
  3. The bubble sits on the suction side, and below zero lift the suction
     side is the LOWER surface. Correlating against Top_Xtr throughout gives
     a positive result at negative incidence that reverses the moment the
     surface is picked by the sign of lift instead. That apparent result is
     an artefact of the variable rather than physics, and both definitions
     are reported here so the artefact stays visible.

Points inside one polar are nowhere near independent, so the bootstrap is
clustered on the airfoil.

Inputs  : data/xfoil_decomposition.csv
Outputs : data/laminar_run_length.csv       correlations, overall and by Re band
          data/laminar_run_bands.csv        mean error by laminar-run quartile
          data/laminar_run_alpha.csv        narrow alpha windows, both surface
                                            definitions side by side
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")

RE_BANDS = [(40e3, 80e3, "40-80k"), (80e3, 130e3, "80-130k"), (130e3, 180e3, "130-180k"),
            (180e3, 250e3, "180-250k"), (250e3, 550e3, "250-550k")]
ALPHA_WINDOWS = [(-3, 0), (0, 3), (3, 6), (6, 9)]
ALPHA_RE_BANDS = [(40e3, 130e3, "40-130k"), (130e3, 250e3, "130-250k"),
                  (250e3, 550e3, "250-550k")]
N_BOOT = 500
SEED = 20260915
MIN_N = 150


def _spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else np.nan


def _partial(y, x, C):
    """Spearman of y against x with the columns of C regressed out of both.

    A bootstrap draw can land on airfoils that were all measured at one
    Reynolds number, which makes that control column constant and the design
    matrix rank deficient. An SVD projection handles that case instead of
    returning a non-finite value that would then be dropped, which would bias
    the interval towards the draws that happened to be well conditioned."""
    ry, rx = rankdata(y), rankdata(x)
    M = np.column_stack([np.ones(len(ry))] + [rankdata(c) for c in C])
    u, sv, _ = np.linalg.svd(M, full_matrices=False)
    u = u[:, sv > sv.max() * 1e-10]
    return _spearman(ry - u @ (u.T @ ry), rx - u @ (u.T @ rx))


def stats_for(err, xtr, logre, aalpha):
    return _spearman(err, xtr), _partial(err, xtr, [logre, aalpha])


def rows_for(d, label, rng):
    """One row per error column, against the suction-side laminar run."""
    xtr = d.xtr_suction.to_numpy()
    logre, aalpha = np.log(d.Re.to_numpy()), np.abs(d.alpha.to_numpy())
    # index arrays per airfoil, so a bootstrap draw is a concatenate and a slice
    groups = [g.to_numpy() for _, g in
              pd.Series(np.arange(len(d))).groupby(d.airfoil.to_numpy())]
    out = []
    for tag, col in [("XF_vs_tunnel", "err_CD_XF_WT"), ("NF_vs_tunnel", "err_CD_NF_WT")]:
        err = np.abs(d[col].to_numpy())
        raw, par = stats_for(err, xtr, logre, aalpha)
        boots = np.empty((N_BOOT, 2))
        for b in range(N_BOOT):
            i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            boots[b] = stats_for(err[i], xtr[i], logre[i], aalpha[i])
        ok = boots[np.isfinite(boots).all(axis=1)]
        dropped = N_BOOT - len(ok)
        if dropped:
            print(f"  {label} {tag}: {dropped}/{N_BOOT} bootstrap draws non-finite", flush=True)
        lo, hi = np.percentile(ok, [2.5, 97.5], axis=0)
        out.append(dict(band=label, error=tag, n=len(d), airfoils=len(groups),
                        boot_dropped=dropped,
                        Re_median=float(d.Re.median()), xtr_median=float(np.median(xtr)),
                        rho_raw=raw, raw_lo=lo[0], raw_hi=hi[0],
                        rho_partial=par, partial_lo=lo[1], partial_hi=hi[1]))
    return out


def main():
    d = pd.read_csv(os.path.join(DATA, "xfoil_decomposition.csv"))
    d = d[d.xf_converged & d.XF_xtr_top.notna()].copy()
    # The bubble sits on the suction side. Below zero lift that is the lower
    # surface, so pick the surface by the sign of CL, not the sign of alpha.
    d["xtr_suction"] = np.where(d.XF_CL >= 0, d.XF_xtr_top, d.XF_xtr_bot)
    neg = int((d.XF_CL < 0).sum())
    print(f"{len(d)} converged points, {d.airfoil.nunique()} airfoils. {neg} sit "
          f"below zero lift, where the suction side is the lower surface.", flush=True)

    rng = np.random.default_rng(SEED)
    rows = rows_for(d, "all", rng)
    for lo, hi, label in RE_BANDS:
        b = d[(d.Re >= lo) & (d.Re < hi)]
        if len(b) >= MIN_N:
            rows += rows_for(b.reset_index(drop=True), label, rng)
            print(f"  {label} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(DATA, "laminar_run_length.csv"), index=False)

    show = ["band", "error", "n", "airfoils", "xtr_median", "rho_raw", "rho_partial",
            "partial_lo", "partial_hi"]
    print("\nSpearman of |drag error| against suction-side laminar run length.")
    print("rho_raw is confounded by Reynolds number; rho_partial holds log(Re)")
    print("and |alpha| fixed. CIs are 95% over an airfoil-clustered bootstrap.\n")
    print(r[show].round(3).to_string(index=False))

    bands = []
    for lo, hi, label in RE_BANDS:
        b = d[(d.Re >= lo) & (d.Re < hi)]
        if len(b) < MIN_N:
            continue
        q = pd.qcut(b.xtr_suction, 4, labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"],
                    duplicates="drop")
        for name, g in b.groupby(q, observed=True):
            bands.append(dict(band=label, quartile=str(name), n=len(g),
                              xtr_mean=float(g.xtr_suction.mean()),
                              alpha_mean=float(g.alpha.mean()),
                              CD_mean=float(g.WT_CD.mean()),
                              abs_CD_err=float((g.XF_CD - g.WT_CD).abs().mean()),
                              err_XF=float(g.err_CD_XF_WT.abs().mean()),
                              err_NF=float(g.err_CD_NF_WT.abs().mean())))
    bd = pd.DataFrame(bands)
    bd.to_csv(os.path.join(DATA, "laminar_run_bands.csv"), index=False)
    print("\nMean |drag error| by quartile of suction-side laminar run, per Re band:\n")
    print(bd.round(4).to_string(index=False))

    # The decisive test. Inside a narrow alpha window the spread in transition
    # location comes from airfoil shape rather than incidence. Both surface
    # definitions are reported, because they disagree at negative incidence and
    # that disagreement is the whole finding.
    win = []
    for alo, ahi in ALPHA_WINDOWS:
        for lo, hi, label in ALPHA_RE_BANDS:
            b = d[(d.alpha >= alo) & (d.alpha < ahi) & (d.Re >= lo) & (d.Re < hi)]
            if len(b) < 120:
                continue
            e = b.err_CD_XF_WT.abs()
            win.append(dict(alpha_lo=alo, alpha_hi=ahi, band=label, n=len(b),
                            airfoils=b.airfoil.nunique(),
                            n_negative_lift=int((b.XF_CL < 0).sum()),
                            xtr_p10=float(b.xtr_suction.quantile(0.10)),
                            xtr_p90=float(b.xtr_suction.quantile(0.90)),
                            rho_suction=_spearman(e, b.xtr_suction),
                            rho_upper=_spearman(e, b.XF_xtr_top),
                            rho_suction_NF=_spearman(b.err_CD_NF_WT.abs(), b.xtr_suction)))
    wd = pd.DataFrame(win)
    wd.to_csv(os.path.join(DATA, "laminar_run_alpha.csv"), index=False)
    print("\nInside a narrow alpha window, so the spread in laminar run length comes")
    print("from airfoil shape rather than incidence. rho_upper is the same test against")
    print("the upper surface regardless of the sign of lift, which is the wrong variable")
    print("below zero lift and is kept here to show the size of the difference.\n")
    print(wd.round(3).to_string(index=False))

    print("\nWrote data/laminar_run_length.csv, data/laminar_run_bands.csv and")
    print("data/laminar_run_alpha.csv")


if __name__ == "__main__":
    main()
