"""
The noise floor of a single wind tunnel: the same model, measured twice.

The two-tunnel comparison in this repository puts 12 percent between UIUC and
Princeton on drag. That number mixes two things: real disagreement between
facilities, and the plain repeatability of a slow-speed drag measurement. The
Princeton set contains models that were mounted and run a second time in the
same tunnel by the same builder, and those pairs isolate the second part.

Nikhil Khobragade (IIT Madras) asked whether the model errors sit inside
the measurement error bars. This is the tightest error bar the archives can
give, and the answer is no: the same model measured twice in the same tunnel
agrees to 3.7 percent, while NeuralFoil disagrees with the tunnels by 11.7.

Method: for each repeat pair, match the two runs at the same nominal Reynolds
number (within 6 percent, since the tunnel does not repeat a speed exactly),
interpolate the second run onto the first run's angles over the overlapping
range, and compare drag point by point.

Inputs  : data/soartech8_experimental.csv
Outputs : data/repeatability_same_model.csv
"""

import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")

# model name of the repeat run -> the original run of the same physical model
REPEATS = {"E387A REPEAT": "E387A", "SD7003 REPEAT": "SD7003", "S2055 REPEAT": "S2055"}
RE_TOL = 0.06          # the tunnel does not hit the same speed twice exactly
MIN_POINTS = 3


def main():
    d = pd.read_csv(os.path.join(DATA, "soartech8_experimental.csv"))
    clean = d[d.config == "clean"]
    rows = []
    for rep, orig in REPEATS.items():
        A, B = clean[clean.model == orig], clean[clean.model == rep]
        if A.empty or B.empty:
            print(f"  {orig}: no clean run to pair with {rep}, skipped")
            continue
        for re_a in sorted(A.Re.unique()):
            for re_b in B.Re.unique():
                if abs(re_b - re_a) / re_a >= RE_TOL:
                    continue
                x = A[A.Re == re_a].sort_values("alpha")
                y = B[B.Re == re_b].sort_values("alpha")
                lo = max(x.alpha.min(), y.alpha.min())
                hi = min(x.alpha.max(), y.alpha.max())
                m = x[(x.alpha >= lo) & (x.alpha <= hi)]
                if len(m) < MIN_POINTS:
                    continue
                cd_b = np.interp(m.alpha, y.alpha, y.CD)
                cl_b = np.interp(m.alpha, y.alpha, y.CL)
                rel = (m.CD.to_numpy() - cd_b) / cd_b
                rows.append(dict(model=orig, repeat=rep, Re_run1=int(re_a), Re_run2=int(re_b),
                                 n=len(m), alpha_min=lo, alpha_max=hi,
                                 mean_abs_errCD=np.abs(rel).mean(),
                                 median_abs_errCD=float(np.median(np.abs(rel))),
                                 bias_errCD=rel.mean(),
                                 mean_abs_dCL=float(np.abs(m.CL.to_numpy() - cl_b).mean())))
    r = pd.DataFrame(rows)
    pooled = pd.DataFrame([dict(
        model="pooled", repeat="all pairs", Re_run1=0, Re_run2=0, n=int(r.n.sum()),
        alpha_min=r.alpha_min.min(), alpha_max=r.alpha_max.max(),
        mean_abs_errCD=float(np.average(r.mean_abs_errCD, weights=r.n)),
        median_abs_errCD=float(np.average(r.median_abs_errCD, weights=r.n)),
        bias_errCD=float(np.average(r.bias_errCD, weights=r.n)),
        mean_abs_dCL=float(np.average(r.mean_abs_dCL, weights=r.n)))])
    out = pd.concat([r, pooled], ignore_index=True)
    out.to_csv(os.path.join(DATA, "repeatability_same_model.csv"), index=False)
    print(out.round(4).to_string(index=False))
    p = pooled.iloc[0]
    print(f"\nSame model, same tunnel, same builder, measured twice: "
          f"{100 * p.mean_abs_errCD:.1f}% mean drag disagreement over {int(p.n)} matched "
          f"points, and {p.mean_abs_dCL:.4f} in lift.")
    print("For comparison: 12% between the two tunnels, 11.7% between NeuralFoil "
          "and the tunnels.")


if __name__ == "__main__":
    main()
