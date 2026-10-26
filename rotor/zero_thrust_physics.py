"""
Why measured zero-thrust profile power sits above free-transition drag:
three physical explanations, each tested against the measured rotors.

hover_lowre_validation.py part 2 found that at zero collective, measured
CP/sigma matches the free-transition drag integral at tip Re 139,528 but is
1.2 to 2.0 times it above tip Re 220,000, close to drag with transition forced
at 5 percent chord. Abhishek (IIT Kanpur) pointed out that a rotor at zero
pitch still has some induced velocity and may give a small negative thrust,
and that the physics has to be right before the data mean anything. So each
candidate is written down with what it predicts, then checked:

  1. Induced power from the residual thrust. Every measured zero-collective
     point has a small CT, positive or negative. Momentum theory gives the
     least induced power for it, |CT|^1.5 / sqrt(2), raised here by a hover
     induced-power factor of 1.5 so the bound is generous. If this is the
     gap, it should be a large fraction of measured CP.
  2. Hub and root torque. A tare that does not depend on blade count adds a
     constant to CP, so CP/sigma should fall as blades are added. At each
     (test, tip Re) with two or more blade counts, CP is fitted as
     a + b * sigma: a is the blade-count-free torque, b the per-blade profile
     power with that torque removed.
  3. Early transition. Lower n_crit is what e^N transition gives for more
     inflow turbulence (Mack: n_crit = -8.43 - 2.4 ln Tu). If turbulent
     inflow is the cause, some n_crit down to 1 (Tu about 2 percent) should
     reach the measured value. If it does not, transition has to be moved by
     something e^N does not model. The transition location that does match,
     forced on both surfaces, is reported as x_tr_equiv.

What this cannot do: name the mechanism behind 3 if it is not inflow
turbulence. Rotation (radial and crossflow in the boundary layer), the blade
running through its own viscous wake (at zero thrust nothing carries it out
of the rotor disk), and surface finish all move transition forward, and the
data here cannot tell them apart.

    python zero_thrust_physics.py

Inputs  : data/hover_zero_thrust.csv  (hover_lowre_validation.py part 2)
Outputs : data/zero_thrust_physics.csv        per measured point
          data/zero_thrust_hub_fit.csv        part 2 fits
"""
import os
import warnings

import numpy as np
import pandas as pd

import aerosandbox as asb
import hover_lowre_validation as hv

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # paths below are relative to the repo root

warnings.filterwarnings("ignore")

KAPPA = 1.5                                # generous hover induced-power factor
N_CRITS = [9.0, 5.0, 3.0, 2.0, 1.0]
XTR_GRID = np.linspace(0.02, 0.6, 30)      # NeuralFoil is not monotone below 2% chord


def section_power(r, **nf_kw):
    """1/2 integral Cd x^3 dx at zero incidence, as in hover_lowre_validation."""
    x = np.linspace(0.0, 1.0, 401)
    xs = x[x >= r.root_cut]
    a = asb.Airfoil(r.airfoil).get_aero_from_neuralfoil(
        alpha=np.zeros_like(xs), Re=r.Re_tip * xs, mach=r.M_tip * xs,
        model_size=hv.MODEL, **nf_kw)
    return hv.profile_integral(np.asarray(a["CD"]), xs)


def per_point(z):
    out = []
    for _, r in z.iterrows():
        row = dict(experiment=r.experiment, n_blades=r.n_blades, sigma=r.sigma,
                   Re_tip=r.Re_tip, M_tip=r.M_tip, CT=r.CT, CP=r.CP,
                   CP_sigma=r.CP_sigma)
        row["induced_bound_share"] = KAPPA * abs(r.CT) ** 1.5 / np.sqrt(2) / r.CP
        for n in N_CRITS:
            row[f"ratio_ncrit_{n:g}"] = r.CP_sigma / section_power(r, n_crit=n)
        p = np.array([section_power(r, xtr_upper=s, xtr_lower=s) for s in XTR_GRID])
        # p falls as transition moves aft; interpolate on the reversed arrays
        row["x_tr_equiv"] = (np.interp(r.CP_sigma, p[::-1], XTR_GRID[::-1])
                             if p.min() <= r.CP_sigma <= p.max() else np.nan)
        row["x_tr_equiv_note"] = ("" if np.isfinite(row["x_tr_equiv"]) else
                                  "above drag tripped at 2%" if r.CP_sigma > p.max()
                                  else "below drag tripped at 60%")
        out.append(row)
    return pd.DataFrame(out)


def hub_fit(z):
    rows = []
    for (exp, re), g in z.groupby(["experiment", "Re_tip"]):
        if g.n_blades.nunique() < 2:
            continue
        b, a = np.polyfit(g.sigma, g.CP, 1)
        rows.append(dict(experiment=exp, Re_tip=re, blade_counts=g.n_blades.nunique(),
                         hub_CP=a, per_blade_CP_sigma=b,
                         hub_share_at_min_sigma=a / g.loc[g.sigma.idxmin(), "CP"]))
    return pd.DataFrame(rows)


def main():
    z = pd.read_csv("data/hover_zero_thrust.csv")
    z = z[z.measured].reset_index(drop=True)
    p = per_point(z)
    h = hub_fit(z)
    p.to_csv("data/zero_thrust_physics.csv", index=False, float_format="%.6g")
    h.to_csv("data/zero_thrust_hub_fit.csv", index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    print(f"{len(p)} measured zero-collective points\n")
    print("1. Induced power bound (kappa 1.5) as a share of measured CP")
    print(f"   median {p.induced_bound_share.median():.2%}, max {p.induced_bound_share.max():.2%}\n")
    print("2. CP = hub + per_blade * sigma, at each test and tip Re")
    print(h.round(6).to_string(index=False), "\n")
    print("3. measured / predicted, by n_crit, and the transition location that matches")
    cols = ["experiment", "n_blades", "Re_tip"] + [f"ratio_ncrit_{n:g}" for n in N_CRITS] \
        + ["x_tr_equiv", "x_tr_equiv_note"]
    print(p[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
