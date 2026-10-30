"""
NeuralFoil's own boundary-layer state at every benchmark point (plan entry 4).

Re-runs NeuralFoil with exactly the settings the validation scripts used (large model, n_crit 9,
free transition, the same Kulfan geometries) and keeps what it says about the boundary layer:
transition location on each surface, the trailing-edge shape factor and momentum thickness, and
how fast transition moves with angle of attack (from NeuralFoil at alpha +/- 0.5 degrees).

Surfaces are named by the sign of NeuralFoil's own CL: suction side is the upper surface when
CL >= 0 and the lower surface when CL < 0. Choosing "upper" regardless reverses the sign of the
laminar-run effect below zero lift (PAPER.md, section 3.2).

Check: every re-computed CD must match the stored NF_CD to one part in a million, or this stops.

Inputs : data/xfoil_decomposition.csv, the Kulfan fits and coordinates used by the validations
Output : data/nf_bl_state.csv   one row per point, same order as xfoil_decomposition.csv
Run    : python benchmark/nf_boundary_layer.py
"""

import os
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
N_CRIT = 9.0
STENCIL = 0.5  # degrees either side, for how fast transition moves
LAST = 31      # NeuralFoil's last (trailing-edge) station of 32


def kulfan_geometries(pts):
    """The Kulfan airfoils the two validation scripts passed to NeuralFoil, keyed (tunnel, key)."""
    import aerosandbox as asb
    geoms = {}
    for name in sorted(pts[pts.tunnel == "UIUC"].geom_key.unique()):
        geoms[("UIUC", name)] = asb.Airfoil(name).to_kulfan_airfoil()
    fit = pd.read_csv(os.path.join(DATA, "soartech8_kulfan_fit.csv")).set_index("model")
    mc = pd.read_csv(os.path.join(DATA, "soartech8_measured_coords.csv"))
    dc = pd.read_csv(os.path.join(DATA, "soartech8_design_coords.csv"))
    for model in sorted(pts[pts.tunnel == "Princeton"].geom_key.unique()):
        r = fit.loc[model]
        if r.primary_source == "measured":
            xy = mc[mc.dap_name == r.dap_name][["x", "y"]].values
        else:
            xy = dc[dc.cor_name == r.cor_name][["x", "y"]].values
        geoms[("Princeton", model)] = asb.Airfoil(name=model, coordinates=np.asarray(xy, float)) \
            .normalize().to_kulfan_airfoil()
    return geoms


def run(kf, Re, alpha):
    a = kf.get_aero_from_neuralfoil(alpha=alpha, Re=Re, model_size="large", n_crit=N_CRIT,
                                    xtr_upper=1.0, xtr_lower=1.0)
    return {k: np.atleast_1d(v) for k, v in a.items()}


def main():
    pts = pd.read_csv(os.path.join(DATA, "xfoil_decomposition.csv"))
    geoms = kulfan_geometries(pts)
    out = []
    for polar, g in pts.groupby("polar", sort=False):
        kf = geoms[(g.tunnel.iloc[0], g.geom_key.iloc[0])]
        Re = float(g.Re.iloc[0])
        al = g.alpha.to_numpy(float)
        a0 = run(kf, Re, al)
        ap = run(kf, Re, al + STENCIL)
        am = run(kf, Re, al - STENCIL)
        up = a0["CL"] >= 0
        pick = lambda u, l, m=up: np.where(m, u, l)
        xs = pick(a0["Top_Xtr"], a0["Bot_Xtr"]); xp = pick(a0["Bot_Xtr"], a0["Top_Xtr"])
        jump_s = np.abs(pick(ap["Top_Xtr"] - am["Top_Xtr"], ap["Bot_Xtr"] - am["Bot_Xtr"])) / (2 * STENCIL)
        jump_p = np.abs(pick(ap["Bot_Xtr"] - am["Bot_Xtr"], ap["Top_Xtr"] - am["Top_Xtr"])) / (2 * STENCIL)
        Hs = pick(a0[f"upper_bl_H_{LAST}"], a0[f"lower_bl_H_{LAST}"])
        th = pick(a0[f"upper_bl_theta_{LAST}"], a0[f"lower_bl_theta_{LAST}"])
        out.append(pd.DataFrame(dict(index=g.index, NF_CD_recomputed=a0["CD"], xs=xs, xp=xp, Hs=Hs,
                                     logtheta_s=np.log10(np.maximum(th, 1e-12)), jump_s=jump_s, jump_p=jump_p)))
    bl = pd.concat(out).set_index("index").sort_index()
    rel = np.abs(bl.NF_CD_recomputed - pts.NF_CD) / pts.NF_CD
    print(f"Re-computed NF_CD vs stored: max relative difference {rel.max():.2e} over {len(pts)} points")
    if rel.max() > 1e-6:
        raise SystemExit("NeuralFoil does not reproduce the stored drag; the geometry or settings differ.")
    bl.drop(columns="NF_CD_recomputed").to_csv(os.path.join(DATA, "nf_bl_state.csv"), index=False, float_format="%.8g")
    print(bl.drop(columns="NF_CD_recomputed").describe().round(3).to_string())


if __name__ == "__main__":
    main()
