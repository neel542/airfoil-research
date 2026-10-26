"""
NeuralFoil against the E387 in a third tunnel: the NASA Langley Low-Turbulence
Pressure Tunnel (McGhee, Walker and Millard 1988, NASA TM-4062, Table B1).

The benchmark's E387 result rests on UIUC (e387_neuralfoil_validation.py) and
the Princeton SoarTech 8 set. LTPT is the quietest of the three (turbulence
about 0.06 percent at 15 psi, 0.16 at 5 psi, 0.20 at 3 psi) and measures drag
with a wake rake, so it tests whether the E387 drag error belongs to the model
or to one facility.

Rows are taken from data/langley_ltpt_e387/e387_ltpt.csv (langley_ltpt_parse.py),
dropping hysteresis runs, turbulator runs, points without drag (above stall)
and any row with a digit the parser could not read. NeuralFoil is run exactly
as in uiuc_neuralfoil_validation.py: the asb "e387" geometry, free transition
at n_crit 9, no Mach correction. UIUC is set alongside from
data/e387_neuralfoil_validation.csv (same model size) at the Reynolds numbers
the two tunnels share.

    python ltpt_neuralfoil_validation.py

Output: data/ltpt_neuralfoil_validation.csv (per point)
"""
import os
import warnings

import numpy as np
import pandas as pd

import aerosandbox as asb

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paths below are relative to the repo root

warnings.filterwarnings("ignore")

N_CRIT = 9.0
MODEL = "large"


def main():
    d = pd.read_csv("data/langley_ltpt_e387/e387_ltpt.csv")
    d = d[~d.hysteresis & ~d.turbulator & d.cd.notna() & d.unread.isna()].copy()
    af = asb.Airfoil("e387")
    out = []
    for (re, pt), g in d.groupby(["Re", "ptinf_psi"]):
        a = af.get_aero_from_neuralfoil(alpha=g.alpha_deg.to_numpy(float), Re=float(re),
                                        model_size=MODEL, n_crit=N_CRIT)
        g = g.assign(NF_CL=np.atleast_1d(a["CL"]), NF_CD=np.atleast_1d(a["CD"]))
        out.append(g)
    p = pd.concat(out)
    p["err_CD"] = (p.NF_CD - p.cd) / p.cd
    p["dCL"] = p.NF_CL - p.cl
    p.to_csv("data/ltpt_neuralfoil_validation.csv", index=False, float_format="%.6g")

    s = (p.groupby(["Re", "ptinf_psi"])
          .agg(n=("cd", "size"), mean_abs_errCD=("err_CD", lambda e: e.abs().mean()),
               median_abs_errCD=("err_CD", lambda e: e.abs().median()),
               mean_errCD=("err_CD", "mean"), mean_abs_dCL=("dCL", lambda e: e.abs().mean()))
          .reset_index())
    u = pd.read_csv("data/e387_neuralfoil_validation.csv")
    u = u[(u.model == MODEL) & u.WT_CD.notna()]
    us = (u.groupby("Re_nom_k")
           .agg(uiuc_n=("err_CD", "size"), uiuc_mean_abs_errCD=("err_CD", lambda e: e.abs().mean()))
           .reset_index())
    s["Re_nom_k"] = (s.Re / 1000).round().astype(int)
    s = s.merge(us, on="Re_nom_k", how="left").drop(columns="Re_nom_k")
    pd.set_option("display.width", 200)
    print(f"{len(p)} LTPT points, NeuralFoil {MODEL}, free transition, n_crit {N_CRIT:g}\n")
    print(s.round(3).to_string(index=False))
    q = p[p.ptinf_psi == 15]
    print(f"\nAll 15 psi points: mean |CD error| {q.err_CD.abs().mean():.1%}, "
          f"median {q.err_CD.abs().median():.1%}, mean signed {q.err_CD.mean():+.1%}")


if __name__ == "__main__":
    main()
