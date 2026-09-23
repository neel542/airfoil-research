"""
Can a static thrust stand check profile power? Govindarajan's two suggestions,
tried on the UIUC propellers, plus the two questions he asked about the
validation numbers.

Bharath Govindarajan (IIT Madras), 23 September 2026, on why fitting
C_P = kappa C_T^1.5 / sqrt(2) + sigma Cd0 / 8 across a pitch family failed:

    "why not take the lower value of thrust (hopefully something close to
    zero), and reason that the power at that point is entirely profile --
    and hence must be sigma*cd0/8*(rho*area*vtip^3)? If that fails, you can
    find the zero-lift drag of the airfoil and use that for cd0."

and on the validation: "the mean value is 1,4% on thrust,power, but the MAE is
9,13% ... highlighting the unsteadiness in data. Am I interpreting this
correctly?" and "10 to 17 percent of what?"

Four parts, each printed and written to data/:

  1. Where the 9 and 13 percent mean absolute error comes from. Split into a
     per-propeller offset and the scatter inside each propeller's own sweep,
     and set against how noisy the measurement itself is, estimated from how
     far each measured point sits off a smooth curve through its own sweep.
  2. What the "10 to 17 percent" was, stated properly.
  3. His first suggestion. Take the lowest-thrust points in the database and
     read Cd0 off them as if all the power were profile. Then the same with
     the ideal induced power taken out, which momentum theory says is the
     least the induced power can be, so that version is an upper bound that
     does not depend on any rotor model.
  4. His fallback. The SDA1075's zero-lift drag at each propeller's Reynolds
     number, from NeuralFoil and from XFoil, put into sigma Cd0 / 8. What is
     left of the measured power is then the induced power, and dividing by
     the ideal gives the induced-power factor kappa that the section drag
     implies, propeller by propeller.

Conventions. The database is in propeller coefficients, C_T = T/(rho n^2 D^4)
and C_P = P/(rho n^3 D^5). His formula is in rotor coefficients, normalised by
rho A V_tip^2 and rho A V_tip^3, so C_T,rotor = C_T 4/pi^3 and
C_P,rotor = C_P 4/pi^4. Profile power with a uniform Cd0 and the inflow
neglected is (Nb/2) rho Omega^3 Cd0 integral(c r^3 dr); over a constant-chord
blade from the hub to the tip that is exactly sigma Cd0 / 8 in rotor
coefficients. These blades start at r/R 0.30 and are not quite constant chord,
so the integral is done on the measured chord, and sigma/8 is printed beside
it to show the difference is small.

    python profile_power_check.py

Inputs  : data/uiuc_propdb/, data/bemt_validation.csv (run validate_bemt.py first)
Outputs : data/profile_power_points.csv    every 9 in point in the window, parts 3 and 4
          data/profile_power_summary.csv   per propeller
          data/bemt_error_split.csv        part 1
          data/rpm_sweep_change.csv        part 2
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import aerosandbox as asb
import rotor_uncertainty as ru
import validate_bemt as vb

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")

CT_ROT = 4 / np.pi ** 3        # propeller C_T to rotor C_T
CP_ROT = 4 / np.pi ** 4        # propeller C_P to rotor C_P
KAPPA_TYPICAL = 1.15           # the usual hover induced-power factor, for context only


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def blade_integrals(geom, nb, root=vb.ROOT_CUT):
    """sigma over the blade, and the exact profile-power integral
    Nb/(2 pi) * integral of (c/R) x^3 dx, both from the root cut to the tip."""
    x = np.linspace(root, 1.0, 401)
    c = np.interp(x, geom["r/R"], geom["c/R"])
    sigma = nb / np.pi * np.trapz(c, x)
    i3 = nb / (2 * np.pi) * np.trapz(c * x ** 3, x)
    return sigma, i3


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: the error split
# ─────────────────────────────────────────────────────────────────────────────
def smooth_noise(g, col):
    """Point-to-point scatter of a measured coefficient about a quadratic in
    log RPM through its own sweep, as a fraction of the local value. This is
    everything the balance sees that a smooth curve does not: measurement
    noise and any genuine unsteadiness, together. It is an upper bound on
    both, because a quadratic will not follow every real bend."""
    x = np.log(g.rpm.values)
    y = g[col].values
    fit = np.polyval(np.polyfit(x, y, 2), x)
    return (y / fit - 1), fit


def error_split(pts):
    rows = []
    w = pts[pts.in_window & (pts.D_in == 9) & ~pts.drawn]
    full = pts[(pts.D_in == 9) & ~pts.drawn]
    for c, name in [("CT", "thrust"), ("CP", "power")]:
        e = w[f"err_{c}"]
        prop_mean = w.groupby("label")[f"err_{c}"].transform("mean")
        within = e - prop_mean
        noise = []
        for _, g in full.groupby("label"):
            r, _ = smooth_noise(g.sort_values("rpm"), f"{c}_meas")
            noise.append(r)
        noise = np.concatenate(noise)
        rows.append(dict(
            quantity=name, n=len(w), n_propellers=w.label.nunique(),
            bias=e.mean(), mae=e.abs().mean(),
            mae_of_propeller_offsets=prop_mean.abs().mean(),
            mae_within_propeller=within.abs().mean(),
            propeller_offset_min=w.groupby("label")[f"err_{c}"].mean().min(),
            propeller_offset_max=w.groupby("label")[f"err_{c}"].mean().max(),
            share_of_variance_between=prop_mean.var(ddof=0) / e.var(ddof=0),
            measured_scatter_rms=np.sqrt(np.mean(noise ** 2)),
            measured_scatter_mae=np.mean(np.abs(noise))))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: what the RPM sweep does to the coefficients
# ─────────────────────────────────────────────────────────────────────────────
def sweep_change(pts):
    rows = []
    for lab, g in pts[(pts.D_in == 9) & ~pts.drawn].groupby("label", sort=False):
        g = g.sort_values("rpm")
        r = dict(label=lab, rpm_lo=g.rpm.iloc[0], rpm_hi=g.rpm.iloc[-1],
                 Re75_lo=g.Re_75.iloc[0], Re75_hi=g.Re_75.iloc[-1])
        for c in ["CT", "CP"]:
            _, fit = smooth_noise(g, f"{c}_meas")
            m = g[f"{c}_meas"]
            r[f"{c}_lo"], r[f"{c}_hi"] = fit[0], fit[-1]
            # lowest RPM to highest, on the smooth curve, as a share of the low-RPM value
            r[f"{c}_change_pct"] = 100 * (fit[-1] / fit[0] - 1)
            # what the email quoted: largest raw point over smallest, minus one
            r[f"{c}_raw_max_over_min_pct"] = 100 * (m.max() / m.min() - 1)
        rows.append(r)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 needs the zero-lift drag of the section
# ─────────────────────────────────────────────────────────────────────────────
def zero_lift(alphas, cl, cd):
    """Angle and drag where the lift curve crosses zero, by linear
    interpolation between the two bracketing points of the first crossing."""
    s = np.sign(cl)
    k = np.flatnonzero(s[:-1] * s[1:] <= 0)
    if len(k) == 0:
        return np.nan, np.nan
    k = k[0]
    t = cl[k] / (cl[k] - cl[k + 1])
    return alphas[k] + t * (alphas[k + 1] - alphas[k]), cd[k] + t * (cd[k + 1] - cd[k])


def zero_lift_tables(Res):
    af = asb.Airfoil(name="SDA1075", coordinates=vb.sda1075())
    alphas = np.arange(-8.0, 2.01, 0.1)
    nf = []
    for Re in Res:
        a = af.get_aero_from_neuralfoil(alpha=alphas, Re=np.full_like(alphas, Re),
                                        model_size=ru.MODEL)
        nf.append(zero_lift(alphas, np.asarray(a["CL"]), np.asarray(a["CD"])))

    # XFoil on the same Kulfan geometry NeuralFoil sees, as in validate_bemt.py
    from multiprocessing import Pool
    import xfoil_decomposition as xd
    coords = xd.panel_coords(af.to_kulfan_airfoil())
    xa = np.arange(-8.0, 2.01, 0.25)
    jobs = [(f"Re{int(Re)}", coords, float(Re), xa) for Re in Res]
    with Pool(xd.N_PROC) as pool:
        res = dict(pool.imap_unordered(xd.run_polar, jobs))
    xf = []
    for Re in Res:
        pol = res[f"Re{int(Re)}"]
        a = np.array(sorted(pol))
        if len(a) < 5:
            xf.append((np.nan, np.nan))
            continue
        xf.append(zero_lift(a, np.array([pol[k][0] for k in a]), np.array([pol[k][1] for k in a])))
    return pd.DataFrame(dict(Re=Res,
                             alpha0_nf=[z[0] for z in nf], cd0_nf=[z[1] for z in nf],
                             alpha0_xf=[z[0] for z in xf], cd0_xf=[z[1] for z in xf]))


# ─────────────────────────────────────────────────────────────────────────────
# Parts 3 and 4, point by point
# ─────────────────────────────────────────────────────────────────────────────
def profile_points(pts, cases, sec, z):
    geoms = {c["label"]: c for c in cases}
    rows = []
    w = pts[pts.in_window & (pts.D_in == 9) & ~pts.drawn]
    for _, p in w.iterrows():
        c = geoms[p.label]
        sigma, i3 = blade_integrals(c["geom"], c["blades"])
        ct, cp = p.CT_meas * CT_ROT, p.CP_meas * CP_ROT
        cp_ideal = ct ** 1.5 / np.sqrt(2)                 # momentum theory, the least induced power can be
        # the solver's own split at this point, and its power-weighted blade drag
        D = c["D_in"] * 0.0254
        rot = ru.Rotor(R=D / 2, n_blades=c["blades"], rpm=p.rpm, root_cut=vb.ROOT_CUT,
                       n_elem=vb.N_ELEM, section=sec, blade=c["geom"])
        out, _ = rot.solve(0.0)
        n = p.rpm / 60
        cp_prof_solver = out["P_profile"] / (ru.RHO * n ** 3 * D ** 5) * CP_ROT
        cp_ind_solver = out["P_induced"] / (ru.RHO * n ** 3 * D ** 5) * CP_ROT
        ct_solver = p.CT_pred * CT_ROT
        cd0_nf = float(np.interp(p.Re_75, z.Re, z.cd0_nf))
        cd0_xf = float(np.interp(p.Re_75, z.Re.values[z.cd0_xf.notna()], z.cd0_xf.dropna()))
        rows.append(dict(
            label=p.label, pitch_in=p.pitch_in, blades=p.blades, rpm=p.rpm, Re_75=p.Re_75,
            sigma=sigma, sigma_over_8=sigma / 8, profile_integral=i3,
            CT_rotor=ct, CP_rotor=cp, CT_over_sigma=ct / sigma,
            FM_measured=cp_ideal / cp,
            # part 3, his first suggestion: all of the power is profile
            cd0_all_profile=cp / i3,
            # the same with the ideal induced power removed: an upper bound on Cd0
            # that assumes nothing but momentum theory
            cd0_upper_bound=(cp - cp_ideal) / i3,
            cd0_kappa_typical=(cp - KAPPA_TYPICAL * cp_ideal) / i3,
            # the solver at the same point
            solver_profile_share=cp_prof_solver / (cp_prof_solver + cp_ind_solver),
            cd_eff_solver=cp_prof_solver / i3,
            kappa_solver=cp_ind_solver / (ct_solver ** 1.5 / np.sqrt(2)),
            # part 4, his fallback: zero-lift drag as Cd0
            cd0_zero_lift_nf=cd0_nf, cd0_zero_lift_xf=cd0_xf,
            kappa_implied_nf=(cp - cd0_nf * i3) / cp_ideal,
            kappa_implied_xf=(cp - cd0_xf * i3) / cp_ideal,
            profile_share_implied_nf=cd0_nf * i3 / cp,
        ))
    return pd.DataFrame(rows)


def main():
    pts = pd.read_csv(os.path.join(DATA, "bemt_validation.csv"))
    cases = vb.load_cases()

    print("1. Where the 9 and 13 percent comes from (9 in propellers, Re 40,000 to 98,000)\n")
    es = error_split(pts)
    es.to_csv(os.path.join(DATA, "bemt_error_split.csv"), index=False)
    for _, r in es.iterrows():
        print(f"  {r.quantity:6}  bias {100 * r.bias:+5.1f}%   mean abs error {100 * r.mae:4.1f}%   "
              f"of which propeller offsets {100 * r.mae_of_propeller_offsets:4.1f}% "
              f"(range {100 * r.propeller_offset_min:+.0f} to {100 * r.propeller_offset_max:+.0f}), "
              f"scatter inside a propeller {100 * r.mae_within_propeller:3.1f}%, "
              f"{100 * r.share_of_variance_between:.0f}% of the variance between propellers")
        print(f"          measured points off a smooth curve through their own sweep: "
              f"{100 * r.measured_scatter_mae:.1f}% mean abs, {100 * r.measured_scatter_rms:.1f}% rms")

    print("\n2. The RPM sweep, lowest RPM to highest, 9 in propellers\n")
    sw = sweep_change(pts)
    sw.to_csv(os.path.join(DATA, "rpm_sweep_change.csv"), index=False)
    print(sw[["label", "Re75_lo", "Re75_hi", "CT_change_pct", "CP_change_pct",
              "CT_raw_max_over_min_pct", "CP_raw_max_over_min_pct"]].round(1).to_string(index=False))

    print("\n3 and 4. Building the sections ...", flush=True)
    sec = vb.neuralfoil_section()
    Res = np.array([30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3, 100e3])
    z = zero_lift_tables(Res)
    print("\n  SDA1075 zero-lift angle and drag:")
    print(z.round(4).to_string(index=False))

    pp = profile_points(pts, cases, sec, z)
    pp.to_csv(os.path.join(DATA, "profile_power_points.csv"), index=False)
    agg = {c: "mean" for c in pp.columns if c not in ("label", "pitch_in", "blades", "rpm")}
    s = pp.groupby("label", sort=False).agg(dict(agg, rpm="size")).rename(columns={"rpm": "n"})
    s.insert(0, "pitch_in", pp.groupby("label", sort=False).pitch_in.first())
    s.insert(1, "blades", pp.groupby("label", sort=False).blades.first())
    s.to_csv(os.path.join(DATA, "profile_power_summary.csv"))

    print("\n3. His first suggestion: read Cd0 off the lowest-thrust points (means over each window)\n")
    print(s[["pitch_in", "blades", "n", "CT_rotor", "CT_over_sigma", "FM_measured",
             "cd0_all_profile", "cd0_upper_bound", "cd0_kappa_typical",
             "cd_eff_solver", "solver_profile_share"]].round(4).to_string())
    sig = pp.sigma_over_8 / pp.profile_integral - 1
    print(f"\n  sigma/8 against the exact integral on the measured chord: "
          f"{100 * sig.min():+.1f} to {100 * sig.max():+.1f}%")

    print("\n4. His fallback: zero-lift drag as Cd0, and the induced-power factor it leaves\n")
    print(s[["pitch_in", "blades", "Re_75", "cd0_zero_lift_nf", "cd0_zero_lift_xf",
             "profile_share_implied_nf", "kappa_implied_nf", "kappa_implied_xf",
             "kappa_solver"]].round(4).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
