"""
What an 11 percent drag error does to a rotor, and then to an aircraft.

The benchmark in this repository measures how far NeuralFoil's section drag is
from a wind tunnel below Re = 500,000. That number is only interesting if it
survives the rest of a design calculation, which is the question Bharath
Govindarajan (IIT Madras) put to us: does the error build up over the course of
the calculation, or wash out?

This script answers it in three steps.

  1. A blade-element momentum theory rotor in hover, with section lift and drag
     coming from NeuralFoil, trimmed to a fixed thrust. Blade sections are held
     inside the Reynolds range the benchmark actually covers, and the induced
     and profile shares of power are reported separately, because the profile
     share is the only part the drag error can touch.

  2. The measured drag error, drawn from this repository's fitted Gamma model,
     pushed through that rotor by Monte Carlo. The answer depends entirely on
     an assumption nobody usually states: whether the error is the same at
     every blade element (it is a property of the airfoil, so mostly yes) or
     independent between them (so it partly cancels). Both are run and both
     are reported, correlated first, because it is the honest headline.

  3. A weight closure loop. Hover power sets battery energy, battery energy
     sets battery mass, battery mass sets weight, and weight sets thrust,
     which sets hover power again. Iterated to a fixed point, this says
     whether a 1 percent section drag error arrives at the aircraft as more
     or less than 1 percent of takeoff weight. That ratio is the
     amplification factor, and it is the number the paper was missing.

What this is not: a prediction of real rotor power. BEMT is itself a model with
its own error, which is not quantified here, and the Gamma error model was
fitted to two-dimensional wind-tunnel residuals, not to a rotating blade.
This propagates a measured aerodynamic uncertainty through a simplified rotor.

Inputs  : data/error_model_fit.json          the fitted Gamma drag-error model
          data/xfoil_decomposition_by_Re.csv the measured signed bias by Re
Outputs : data/rotor_design.csv              blade stations at the trim point
          data/rotor_propagation.csv         Monte Carlo, four error cases
          data/rotor_weight_closure.csv      the closure loop and amplification
          figures/33_rotor_power_uncertainty.png
          figures/34_weight_amplification.png

Usage   : python rotor_uncertainty.py              the full study, about 17 minutes
          python rotor_uncertainty.py --explore     rotor sizing sweep only
          python rotor_uncertainty.py --from-cache  reuse the saved Monte Carlo
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.stats import gamma as gamma_dist

import aerosandbox as asb

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
FIG = os.path.join(OUT, "figures")

RHO = 1.225          # kg/m^3, sea level
NU = 1.46e-5         # m^2/s, kinematic viscosity at 15 C
A_SOUND = 340.3      # m/s
G = 9.80665

SECTION = "e387"     # in the benchmark set, measured in both tunnels
MODEL = "large"

# The Reynolds range the two tunnels actually cover. Blade stations outside it
# are flagged, because the measured error does not apply there.
RE_VALID = (60e3, 500e3)


# ═════════════════════════════════════════════════════════════════════════
# Section aerodynamics: one NeuralFoil table, interpolated
# ═════════════════════════════════════════════════════════════════════════
class Section:
    """NeuralFoil lift, drag and confidence for one airfoil, on a grid.

    The rotor solver and the Monte Carlo both hit the section thousands of
    times. Building the table once and interpolating keeps the physics
    identical between the nominal run and every perturbed run, so any
    difference between them is the perturbation and nothing else.
    """

    def __init__(self, name=SECTION, n_alpha=141, n_Re=45):
        self.name = name
        self.af = asb.Airfoil(name)
        self.camber_pct = float(self.af.max_camber()) * 100
        self.thickness_pct = float(self.af.max_thickness()) * 100
        self.alphas = np.linspace(-10, 20, n_alpha)
        self.Res = np.geomspace(30e3, 900e3, n_Re)
        A, R = np.meshgrid(self.alphas, self.Res, indexing="ij")
        aero = self.af.get_aero_from_neuralfoil(
            alpha=A.ravel(), Re=R.ravel(), model_size=MODEL)
        shape = A.shape
        grid = (self.alphas, self.Res)
        kw = dict(bounds_error=False, fill_value=None)      # linear extrapolation
        self._CL = RegularGridInterpolator(grid, aero["CL"].reshape(shape), **kw)
        self._CD = RegularGridInterpolator(grid, np.maximum(aero["CD"], 1e-4).reshape(shape), **kw)
        self._conf = RegularGridInterpolator(
            grid, np.clip(aero["analysis_confidence"], 0, 1).reshape(shape), **kw)

    def __call__(self, alpha, Re):
        pts = np.column_stack([np.atleast_1d(alpha), np.atleast_1d(Re)])
        return self._CL(pts), self._CD(pts)

    def confidence(self, alpha, Re):
        return self._conf(np.column_stack([np.atleast_1d(alpha), np.atleast_1d(Re)]))


# ═════════════════════════════════════════════════════════════════════════
# The rotor
# ═════════════════════════════════════════════════════════════════════════
class Rotor:
    """A hovering rotor, discretised into blade elements.

    Geometry is a linearly tapered blade with linear washout, which is what a
    lift rotor of this class actually looks like. Collective is the pitch at
    75 percent span, the usual reference, and the twist rate is quoted per
    unit r/R, so twist_rate_deg = 15 puts about 13 degrees of washout between
    the root cut and the tip.
    """

    def __init__(self, R=0.30, n_blades=2, chord_root=0.060, taper=0.7,
                 rpm=4200, twist_rate_deg=15.0, root_cut=0.15, n_elem=24,
                 section=None):
        self.R, self.Nb, self.rpm = R, n_blades, rpm
        self.omega = rpm * 2 * np.pi / 60
        self.section = section if section is not None else Section()
        self.root_cut = root_cut
        x = np.linspace(root_cut, 1.0, n_elem + 1)
        self.x = 0.5 * (x[1:] + x[:-1])                      # element centres, r/R
        self.dx = np.diff(x)
        self.r = self.x * R
        self.dr = self.dx * R
        self.chord = chord_root * (1 - (1 - taper) * (self.x - root_cut) / (1 - root_cut))
        self.twist_rate_deg = twist_rate_deg
        self.twist = np.deg2rad(twist_rate_deg) * (0.75 - self.x)      # zero at 0.75R
        self.area = np.pi * R ** 2
        self.solidity = self.Nb * np.trapz(self.chord, self.r) / self.area

    # ── one element, one inflow ──────────────────────────────────────────
    def _element(self, vi, i, collective, cd_scale):
        """Thrust and torque per unit span at element i for an induced velocity vi."""
        u_t = self.omega * self.r[i]
        u = np.hypot(u_t, vi)
        phi = np.arctan2(vi, u_t)
        alpha = np.rad2deg(self.twist[i] + collective - phi)
        Re = u * self.chord[i] / NU
        cl, cd = self.section(alpha, Re)
        cl, cd = float(cl[0]), float(cd[0]) * float(cd_scale)
        q = 0.5 * RHO * u ** 2 * self.chord[i] * self.Nb
        dT = q * (cl * np.cos(phi) - cd * np.sin(phi))
        dFx = q * (cl * np.sin(phi) + cd * np.cos(phi))       # in-plane force per span
        dFx_i = q * cl * np.sin(phi)                          # induced part
        dFx_p = q * cd * np.cos(phi)                          # profile part
        return dT, dFx, dFx_i, dFx_p, alpha, Re, cl, cd, phi

    def _tip_loss(self, i, phi):
        """Prandtl tip and root loss."""
        s = max(abs(np.sin(phi)), 1e-3)
        f_t = self.Nb * (1 - self.x[i]) / (2 * s)
        f_r = self.Nb * (self.x[i] - self.root_cut) / (2 * s)
        F = ((2 / np.pi) * np.arccos(np.clip(np.exp(-f_t), 0, 1)) *
             (2 / np.pi) * np.arccos(np.clip(np.exp(-f_r), 0, 1)))
        return max(F, 1e-3)

    def solve(self, collective, cd_scale=1.0):
        """Run the rotor at a given collective. Returns per-element and totals.

        At each element the blade-element thrust is matched to the momentum
        thrust 4 pi rho vi^2 F r dr by solving for the induced velocity.
        """
        cd_scale = np.broadcast_to(np.atleast_1d(cd_scale), (len(self.x),))
        rows = []
        for i in range(len(self.x)):
            def residual(vi):
                dT = self._element(vi, i, collective, cd_scale[i])[0]
                phi = np.arctan2(vi, self.omega * self.r[i])
                F = self._tip_loss(i, phi)
                return dT - 4 * np.pi * RHO * vi ** 2 * F * self.r[i]

            vi_hi = 0.25 * self.omega * self.r[i] + 1e-3
            try:
                if residual(1e-4) <= 0:
                    vi = 1e-4                                  # element makes no thrust
                else:
                    vi = brentq(residual, 1e-4, vi_hi, xtol=1e-8, rtol=1e-10)
            except ValueError:
                vi = 1e-4
            dT, dFx, dFx_i, dFx_p, alpha, Re, cl, cd, phi = self._element(
                vi, i, collective, cd_scale[i])
            rows.append(dict(x=self.x[i], r=self.r[i], dr=self.dr[i], chord=self.chord[i],
                             vi=vi, phi_deg=np.rad2deg(phi), alpha_deg=alpha, Re=Re,
                             CL=cl, CD=cd, dT=dT * self.dr[i],
                             dQ=dFx * self.r[i] * self.dr[i],
                             dQ_i=dFx_i * self.r[i] * self.dr[i],
                             dQ_p=dFx_p * self.r[i] * self.dr[i]))
        d = pd.DataFrame(rows)
        T = d.dT.sum()
        Q, Qi, Qp = d.dQ.sum(), d.dQ_i.sum(), d.dQ_p.sum()
        P, Pi, Pp = self.omega * Q, self.omega * Qi, self.omega * Qp
        v_tip = self.omega * self.R
        out = dict(collective_deg=np.rad2deg(collective), T=T, Q=Q, P=P,
                   P_induced=Pi, P_profile=Pp,
                   induced_frac=Pi / P if P > 0 else np.nan,
                   CT=T / (RHO * self.area * v_tip ** 2),
                   v_tip=v_tip, tip_mach=v_tip / A_SOUND,
                   Re_75=float(np.interp(0.75, d.x, d.Re)),
                   Re_min=d.Re.min(), Re_max=d.Re.max(),
                   FM=(T ** 1.5 / np.sqrt(2 * RHO * self.area)) / P if P > 0 else np.nan)
        out["CT_sigma"] = out["CT"] / self.solidity
        out["disk_loading"] = T / self.area
        return out, d

    def trim(self, thrust, cd_scale=1.0, lo=-12.0, hi=24.0):
        """Collective at 75 percent span, in degrees, that gives the target thrust."""
        f = lambda c_deg: self.solve(np.deg2rad(c_deg), cd_scale)[0]["T"] - thrust
        c = brentq(f, lo, hi, xtol=1e-6)
        out, d = self.solve(np.deg2rad(c), cd_scale)
        return out, d


# ═════════════════════════════════════════════════════════════════════════
# The measured drag error, as a sampler
# ═════════════════════════════════════════════════════════════════════════
class DragError:
    """Draws relative drag errors from this repository's fitted Gamma model.

    The model gives the expected magnitude E|dCD/CD| as a function of
    NeuralFoil's confidence, Reynolds number, angle of attack, camber and
    thickness, with a Gamma spread around it. The magnitude carries no sign,
    so the sign is supplied separately from the measured bias by Reynolds
    number: the probability of a positive draw is set so that the mean signed
    error reproduces the bias the tunnels actually show.

    Sign convention follows the benchmark: e = (CD_NeuralFoil - CD_tunnel) /
    CD_tunnel. A positive e means the model over-predicts drag, so the true
    drag is CD_model / (1 + e).
    """

    def __init__(self):
        fit = json.load(open(os.path.join(DATA, "error_model_fit.json")))
        self.beta = fit["beta"]
        self.phi = fit["dispersion_phi"]
        by_re = pd.read_csv(os.path.join(DATA, "xfoil_decomposition_by_Re.csv"))
        dec = pd.read_csv(os.path.join(DATA, "xfoil_decomposition.csv"))
        mid = dec.groupby(["tunnel", "Re_bin"]).Re.mean()
        by_re["Re"] = [mid[(t, b)] for t, b in zip(by_re.tunnel, by_re.Re_bin)]
        by_re = by_re.sort_values("Re")
        # pooled signed bias against Re, weighted by points, both tunnels
        g = (by_re.assign(w=by_re.n)
                  .groupby(pd.cut(by_re.Re, np.geomspace(50e3, 550e3, 8)), observed=True)
                  .apply(lambda d: pd.Series(dict(
                      Re=np.average(d.Re, weights=d.w),
                      bias=np.average(d.bias_CD_NF_WT_all, weights=d.w)))))
        self.bias_Re = g.Re.to_numpy()
        self.bias_val = g.bias.to_numpy()

    def expected_magnitude(self, conf, Re, alpha_deg, camber_pct, thickness_pct):
        b = self.beta
        eta = (b["one"]
               + b["log_unconf"] * np.log10(1.001 - np.clip(conf, 0, 1))
               + b["log_Re"] * np.log10(np.asarray(Re) / 1e5)
               + b["alpha"] * np.asarray(alpha_deg)
               + b["alpha_sq"] * np.asarray(alpha_deg) ** 2
               + b["camber"] * camber_pct
               + b["thickness"] * thickness_pct)
        return np.exp(np.clip(eta, -12, 6))

    def bias(self, Re):
        return np.interp(Re, self.bias_Re, self.bias_val)

    def draw(self, mu, Re, size, rng, signed=True, correlated=True):
        """Relative drag errors, shape (size, n_elements).

        correlated : one draw shared by every blade element, because the error
                     is a property of the airfoil and every element is the
                     same airfoil.
        signed     : the sign follows the measured bias at that Reynolds
                     number. Otherwise it is a coin flip, which is the
                     symmetric-band reading of the same magnitude.
        """
        mu = np.atleast_1d(mu)
        n = len(mu)
        shape_k = 1.0 / self.phi
        if correlated:
            # one standardised Gamma variate per draw, scaled by each element's mean
            z = rng.gamma(shape_k, self.phi, size=(size, 1))
            mag = z * mu[None, :]
            s_ref = self._signs(self.bias(np.mean(Re)), np.mean(mu), (size, 1), rng, signed)
            sign = np.broadcast_to(s_ref, (size, n))
        else:
            mag = rng.gamma(shape_k, self.phi, size=(size, n)) * mu[None, :]
            sign = self._signs(self.bias(np.asarray(Re))[None, :], mu[None, :],
                               (size, n), rng, signed)
        return mag * sign

    @staticmethod
    def _signs(bias, mu, shape, rng, signed):
        if not signed:
            return rng.choice([-1.0, 1.0], size=shape)
        p_pos = np.clip(0.5 * (1 + np.asarray(bias) / np.asarray(mu)), 0.0, 1.0)
        return np.where(rng.random(shape) < p_pos, 1.0, -1.0)


# ═════════════════════════════════════════════════════════════════════════
# The design point
# ═════════════════════════════════════════════════════════════════════════
# Chosen against three constraints at once, and the sizing sweep below shows
# why they leave very little room:
#
#   * every blade station inside the benchmark's Reynolds range, so the
#     measured error actually applies to this blade,
#   * 75 percent span near the top of that range, which is where a lift rotor
#     of this class wants to be,
#   * an induced share of hover power near the 60 percent Govindarajan
#     quoted, which is the check that the rotor is representative at all.
#
# These fight each other. Reynolds number at a station is set by chord times
# speed, and blade loading is set by chord divided into thrust, so raising one
# lowers the other. A rotor that sits at Re = 400,000 with a well-loaded blade
# belongs on a 20 to 35 kg aircraft, not on a 2 kg quadcopter.
DESIGN = dict(R=0.50, n_blades=2, chord_root=0.100, taper=0.7, rpm=1910,
              twist_rate_deg=15.0, root_cut=0.15, n_elem=24)
N_ROTORS = 4
# The hover thrust the weight closure below settles on, so the rotor the study
# analyses and the aircraft the study closes are the same aircraft: 15.7 kg
# all-up over four rotors, a disk loading of 49 N/m^2.
THRUST_PER_ROTOR = 38.51

# Weight closure inputs. A long-endurance survey or delivery aircraft.
CLOSURE = dict(payload_kg=2.5, struct_frac=0.40, endurance_h=0.75,
               pack_Wh_per_kg=180.0, usable_frac=0.85, drivetrain_eff=0.85)

N_DRAWS = 1000
SEED = 20260910


def design_point(section):
    r = Rotor(section=section, **DESIGN)
    out, d = r.trim(THRUST_PER_ROTOR)
    return r, out, d


def sizing_sweep(section):
    """Why this rotor and not another: Reynolds number against blade loading."""
    rows = []
    for R in [0.30, 0.40, 0.50, 0.60]:
        for v_tip in [70, 85, 100, 115]:
            for c_over_R in [0.12, 0.16, 0.20]:
                rpm = v_tip / R * 60 / (2 * np.pi)
                rot = Rotor(R=R, chord_root=c_over_R * R, taper=0.7, rpm=rpm,
                            twist_rate_deg=15.0, section=section)
                T = THRUST_PER_ROTOR * (R / DESIGN["R"]) ** 2      # constant disk loading
                try:
                    o, d = rot.trim(T)
                except Exception:
                    continue
                # what a fixed, fully correlated 11.7 percent drag error does here,
                # so the sweep says how the answer moves with the rotor, not just
                # what one rotor gives
                o_hi, _ = rot.trim(T, cd_scale=1.0 / 1.117)
                rows.append(dict(R=R, v_tip=v_tip, c_over_R=c_over_R, rpm=round(rpm),
                                 dP_pct_at_11p7=100 * (o["P"] - o_hi["P"]) / o_hi["P"],
                                 solidity=rot.solidity, thrust_N=T,
                                 aircraft_kg=N_ROTORS * T / G, power_W=o["P"],
                                 induced_frac=o["induced_frac"], FM=o["FM"],
                                 CT_sigma=o["CT_sigma"], tip_mach=o["tip_mach"],
                                 Re_75=o["Re_75"], Re_min=o["Re_min"], Re_max=o["Re_max"],
                                 all_inside_envelope=bool(o["Re_min"] >= RE_VALID[0]
                                                          and o["Re_max"] <= RE_VALID[1]),
                                 alpha_min=d.alpha_deg.min(), alpha_max=d.alpha_deg.max()))
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# Monte Carlo: the measured drag error, through the rotor
# ═════════════════════════════════════════════════════════════════════════
def monte_carlo(rotor, nominal, stations, err, n=N_DRAWS, seed=SEED):
    """Four cases: correlated or independent across the blade, signed or not.

    The design used NeuralFoil, so the drag it assumed is CD_model. The truth
    is CD_model / (1 + e) with e the measured relative error, and the rotor is
    re-trimmed to the same thrust on that truth. What the designer got wrong
    is then (P_model - P_true) / P_true.
    """
    conf = err_conf = rotor.section.confidence(stations.alpha_deg.values, stations.Re.values)
    mu = err.expected_magnitude(err_conf, stations.Re.values, stations.alpha_deg.values,
                                rotor.section.camber_pct, rotor.section.thickness_pct)
    rows = []
    for correlated in (True, False):
        for signed in (True, False):
            rng = np.random.default_rng(seed)
            e = err.draw(mu, stations.Re.values, n, rng, signed=signed, correlated=correlated)
            for k in range(n):
                cd_scale = 1.0 / (1.0 + e[k])
                o, _ = rotor.trim(THRUST_PER_ROTOR, cd_scale=cd_scale)
                rows.append(dict(case=f"{'correlated' if correlated else 'independent'}"
                                      f"_{'signed' if signed else 'symmetric'}",
                                 draw=k, e_mean=float(np.mean(e[k])),
                                 P_true=o["P"], P_model=nominal["P"],
                                 dP_pct=100 * (nominal["P"] - o["P"]) / o["P"],
                                 profile_frac_true=1 - o["induced_frac"],
                                 FM_true=o["FM"]))
    return pd.DataFrame(rows), mu, conf


# ═════════════════════════════════════════════════════════════════════════
# Weight closure: power to battery to weight, iterated to a fixed point
# ═════════════════════════════════════════════════════════════════════════
def closure(rotor, cd_scale=1.0, cfg=None, tol=1e-6, max_iter=60):
    """Take-off mass that closes on itself for a given section drag.

    Heavier battery needs more thrust, which needs more power, which needs a
    heavier battery. Iterated until the mass stops moving.
    """
    c = dict(CLOSURE if cfg is None else cfg)
    m = c["payload_kg"] / (1 - c["struct_frac"]) * 2.0          # a starting guess
    for it in range(max_iter):
        T = m * G / N_ROTORS
        o, _ = rotor.trim(T, cd_scale=cd_scale)
        p_elec = N_ROTORS * o["P"] / c["drivetrain_eff"]
        m_batt = p_elec * c["endurance_h"] / (c["pack_Wh_per_kg"] * c["usable_frac"])
        m_new = (c["payload_kg"] + m_batt) / (1 - c["struct_frac"])
        if abs(m_new - m) < tol:
            m = m_new
            break
        m = m + 0.6 * (m_new - m)                                # damped, it converges quickly
    T = m * G / N_ROTORS
    o, _ = rotor.trim(T, cd_scale=cd_scale)
    p_elec = N_ROTORS * o["P"] / c["drivetrain_eff"]
    m_batt = p_elec * c["endurance_h"] / (c["pack_Wh_per_kg"] * c["usable_frac"])
    return dict(iterations=it + 1, m_total_kg=m, m_battery_kg=m_batt,
                m_structure_kg=c["struct_frac"] * m, payload_kg=c["payload_kg"],
                thrust_per_rotor_N=T, power_rotor_W=o["P"], power_elec_W=p_elec,
                energy_Wh=p_elec * c["endurance_h"], induced_frac=o["induced_frac"],
                Re_75=o["Re_75"], FM=o["FM"])


# ═════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════
def figures(rotor, nominal, stations, mc, closure_tbl, mu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.9, "axes.grid": True,
        "grid.color": "#dddddd", "grid.linewidth": 0.7, "grid.alpha": 0.7,
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "#cccccc",
        "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.axisbelow": True})
    C_COR, C_IND, C_RE = "#b2182b", "#2166ac", "#e08214"

    def clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ── 33: where the blade sits, and what the error does to hover power ──
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    ax.plot(stations.x, stations.Re / 1e3, "o-", color=C_RE, lw=2, ms=5)
    ax.axhspan(RE_VALID[0] / 1e3, RE_VALID[1] / 1e3, color="#1a9850", alpha=0.10)
    for y in RE_VALID:
        ax.axhline(y / 1e3, color="#1a9850", lw=1, ls="--")
    ax.annotate("the benchmark's range", xy=(0.19, RE_VALID[1] / 1e3 - 42), fontsize=9,
                color="#1a7a3c")
    ax2 = ax.twinx()
    ax2.plot(stations.x, 100 * mu, color="#777777", lw=1.6, ls=":")
    ax2.set_ylabel("expected drag error (%)", color="#777777", fontsize=10)
    ax2.tick_params(axis="y", colors="#777777"); ax2.grid(False)
    ax2.set_ylim(0, max(20, 100 * mu.max() * 1.3))
    ax.annotate(f"the blade averages {100 * mu.mean():.1f}% expected drag error,\n"
                f"against 11.7% pooled over the whole benchmark",
                xy=(0.03, 0.02), xycoords="axes fraction", va="bottom",
                fontsize=9, color="#555555")
    ax.set_xlabel("blade station, r / R")
    ax.set_ylabel("chord Reynolds number (thousands)")
    ax.set_title("The blade sits inside the benchmark")
    ax.set_ylim(0, 560); clean(ax)

    ax = axes[1]
    ax.bar(["induced", "profile"], [nominal["P_induced"], nominal["P_profile"]],
           color=["#2166ac", "#b2182b"], width=0.55)
    for i, v in enumerate([nominal["P_induced"], nominal["P_profile"]]):
        ax.annotate(f"{v:.0f} W\n{100 * v / nominal['P']:.0f}%", (i, v), ha="center",
                    va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("hover power, one rotor (W)")
    ax.set_ylim(0, nominal["P"] * 0.95)
    ax.set_title("Only the profile share can move")
    ax.annotate("a drag error cannot touch the induced part,\n"
                "so it arrives divided by roughly three",
                xy=(0.43, 0.82), xycoords="axes fraction", va="bottom",
                fontsize=9, color="#555555")
    clean(ax)

    ax = axes[2]
    for case, color in [("correlated_signed", C_COR), ("independent_signed", C_IND)]:
        d = mc[mc.case == case]
        ax.hist(d.dP_pct, bins=45, histtype="stepfilled", alpha=0.45, color=color,
                label=f"{case.split('_')[0]} across the blade\n"
                      f"(sd {d.dP_pct.std():.1f}%)")
        ax.axvline(d.dP_pct.mean(), color=color, lw=1.6, ls="--")
    ax.axvline(0, color="#444444", lw=1)
    ax.set_xlabel("error in predicted hover power (%)")
    ax.set_ylabel(f"draws (of {N_DRAWS})")
    ax.set_title("What the measured drag error does")
    ax.legend(loc="upper left", fontsize=9); clean(ax)

    fig.suptitle("The measured section drag error, carried into rotor hover power",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "33_rotor_power_uncertainty.png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/33_rotor_power_uncertainty.png")

    # ── 34: the closure loop, and where the error grows and where it shrinks ──
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    t = closure_tbl.sort_values("e_pct")
    base = t[t.label == "model"].m_total_kg.iloc[0]
    ax.plot(t.e_pct, t.m_total_kg, "o-", color=C_COR, lw=2, ms=7)
    ax.axhline(base, color="#444444", lw=1, ls="--")
    ax.annotate(f"what the model predicts, {base:.2f} kg", xy=(0.03, base),
                xycoords=("axes fraction", "data"), xytext=(0, 7),
                textcoords="offset points", fontsize=9, color="#444444")
    for _, r in t.iterrows():
        if r.label == "model":
            continue
        up = r.m_total_kg > base
        ax.annotate(f"{r.label}\n{1000 * (r.m_total_kg - base):+,.0f} g",
                    (r.e_pct, r.m_total_kg), textcoords="offset points",
                    xytext=(14 if r.e_pct < 0 else -14, 0),
                    ha="left" if r.e_pct < 0 else "right", va="center", fontsize=8.5)
    pad = 0.16 * (t.e_pct.max() - t.e_pct.min())
    ax.set_xlim(t.e_pct.min() - pad, t.e_pct.max() + pad)
    ax.set_ylim(t.m_total_kg.min() - 0.35, t.m_total_kg.max() + 0.35)
    ax.set_xlabel("section drag error the design carried in (%)")
    ax.set_ylabel("take-off mass that closes (kg)")
    ax.set_title("The closure loop moves the whole aircraft")
    clean(ax)

    ax = axes[1]
    q = closure_tbl[closure_tbl.label != "model"].sort_values("e_pct")
    labels = list(q.label)
    idx = np.arange(len(labels))
    w = 0.26
    ax.bar(idx - w, q.e_pct.abs(), w, color="#999999", label="section drag error in")
    ax.bar(idx, q.d_power_pct_fixed_mass.abs(), w, color=C_RE,
           label="hover power error, mass held fixed")
    ax.bar(idx + w, q.d_power_pct.abs(), w, color=C_COR,
           label="hover power error, after closure")
    ax.bar(idx + 2 * w, q.d_mass_pct.abs(), w, color=C_IND,
           label="take-off weight error")
    for i, r in enumerate(q.itertuples()):
        for off, v in [(-w, abs(r.e_pct)), (0, abs(r.d_power_pct_fixed_mass)),
                       (w, abs(r.d_power_pct)), (2 * w, abs(r.d_mass_pct))]:
            ax.annotate(f"{v:.1f}", (i + off, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(idx + w / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("error (%)")
    ax.set_ylim(0, q.e_pct.abs().max() * 1.30)
    amps = ", ".join(f"{v:.2f}" for v in q.amplification)
    gains = ", ".join(f"{v:.1f}x" for v in q.loop_gain)
    ax.set_title("Where it grows, and where it shrinks")
    ax.legend(loc="upper right", fontsize=8.5)
    clean(ax)
    fig.text(0.52, -0.04,
             f"The closure loop multiplies the hover-power error by {gains}. "
             f"Take-off weight still moves only {amps} percent per percent of section drag, "
             f"because payload and structure do not care about drag.",
             ha="center", fontsize=9.5, color="#555555")

    fig.suptitle("From section drag to hover power to battery to take-off weight",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "34_weight_amplification.png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/34_weight_amplification.png")


# ═════════════════════════════════════════════════════════════════════════
def main(explore=False, from_cache=False):
    print("Building the section table from NeuralFoil ...")
    sec = Section()
    print(f"  {sec.name}: {sec.camber_pct:.2f}% camber, {sec.thickness_pct:.2f}% thick")

    if explore:
        sw = sizing_sweep(sec)
        sw.to_csv(os.path.join(DATA, "rotor_sizing_sweep.csv"), index=False)
        keep = ["R", "v_tip", "c_over_R", "aircraft_kg", "CT_sigma", "FM",
                "induced_frac", "Re_75", "Re_min", "Re_max", "all_inside_envelope",
                "dP_pct_at_11p7"]
        print(sw[keep].round(3).to_string(index=False))
        return

    rotor, nom, st = design_point(sec)
    print("\nDesign point, hover:")
    print(f"  {N_ROTORS} rotors, R = {rotor.R} m, {rotor.Nb} blades, {rotor.rpm} rpm, "
          f"tip speed {nom['v_tip']:.0f} m/s (M {nom['tip_mach']:.2f})")
    print(f"  thrust {nom['T']:.2f} N per rotor, {N_ROTORS * nom['T'] / G:.2f} kg all-up, "
          f"disk loading {nom['disk_loading']:.0f} N/m^2, solidity {rotor.solidity:.4f}, "
          f"CT/sigma {nom['CT_sigma']:.3f}")
    print(f"  power {nom['P']:.1f} W per rotor: {nom['P_induced']:.1f} W induced "
          f"({100 * nom['induced_frac']:.1f}%), {nom['P_profile']:.1f} W profile "
          f"({100 * (1 - nom['induced_frac']):.1f}%), figure of merit {nom['FM']:.3f}")
    print(f"  Reynolds number {nom['Re_min']:,.0f} at the root cut to {nom['Re_max']:,.0f} "
          f"at the tip, {nom['Re_75']:,.0f} at 75% span")
    inside = ((st.Re >= RE_VALID[0]) & (st.Re <= RE_VALID[1])).sum()
    print(f"  {inside} of {len(st)} stations inside the benchmark's "
          f"{RE_VALID[0]:,.0f} to {RE_VALID[1]:,.0f}")
    if inside < len(st):
        print("  WARNING: stations outside the validated range, the measured error "
              "does not apply there")

    # the gate: is this rotor representative at all?
    gate = "PASS" if 0.5 <= nom["induced_frac"] <= 0.75 else "CHECK"
    print(f"\n  Verification gate, induced share near the 60% Govindarajan quoted: "
          f"{100 * nom['induced_frac']:.1f}%  [{gate}]")

    err = DragError()
    st_out = st.copy()
    st_out["confidence"] = sec.confidence(st.alpha_deg.values, st.Re.values)
    st_out["inside_envelope"] = (st.Re >= RE_VALID[0]) & (st.Re <= RE_VALID[1])
    # what the fitted error model expects at each station's own conditions
    st_out["expected_errCD"] = err.expected_magnitude(
        st_out.confidence.values, st.Re.values, st.alpha_deg.values,
        sec.camber_pct, sec.thickness_pct)
    st_out.to_csv(os.path.join(DATA, "rotor_design.csv"), index=False)
    print(f"  expected drag error along the blade: {100 * st_out.expected_errCD.min():.1f}% "
          f"to {100 * st_out.expected_errCD.max():.1f}%, "
          f"blade average {100 * st_out.expected_errCD.mean():.1f}%")

    cache = os.path.join(DATA, "rotor_propagation.csv")
    conf = sec.confidence(st.alpha_deg.values, st.Re.values)
    mu = err.expected_magnitude(conf, st.Re.values, st.alpha_deg.values,
                                sec.camber_pct, sec.thickness_pct)
    if from_cache and os.path.exists(cache):
        print("\nMonte Carlo: reusing data/rotor_propagation.csv")
        mc = pd.read_csv(cache)
    else:
        print("\nMonte Carlo, four cases, "
              f"{N_DRAWS} draws each ({4 * N_DRAWS} rotor trims) ...")
        mc, mu, conf = monte_carlo(rotor, nom, st, err)
        mc.to_csv(cache, index=False)

    # the analytic cross-check the plan demanded
    pf = 1 - nom["induced_frac"]
    e_bar = float(np.mean(mu))
    print(f"\n  Analytic check: profile share {100 * pf:.1f}% x mean drag error "
          f"{100 * e_bar:.1f}% = {100 * pf * e_bar:.2f}% power error if fully correlated")
    summ = (mc.groupby("case")
              .agg(mean_dP=("dP_pct", "mean"), sd_dP=("dP_pct", "std"),
                   p05=("dP_pct", lambda s: s.quantile(0.05)),
                   p50=("dP_pct", "median"),
                   p95=("dP_pct", lambda s: s.quantile(0.95)),
                   mean_abs=("dP_pct", lambda s: s.abs().mean()))
              .reset_index())
    print("\nPower error by case (percent of true hover power):")
    print(summ.round(3).to_string(index=False))
    summ.to_csv(os.path.join(DATA, "rotor_propagation_summary.csv"), index=False)

    # ── weight closure at the correlated, signed percentiles ──────────────
    print("\nWeight closure ...")
    cs = mc[mc.case == "correlated_signed"].e_mean
    rows = [dict(label="model", e_pct=0.0, d_power_pct_fixed_mass=0.0,
                 **closure(rotor, 1.0))]
    for lab, q in [("5th percentile", 0.05), ("median", 0.50), ("95th percentile", 0.95)]:
        e = float(cs.quantile(q))
        # the same drag error at the design mass, so the closure loop's own
        # contribution can be separated from the direct profile-power effect
        o_fix, _ = rotor.trim(THRUST_PER_ROTOR, cd_scale=1.0 / (1.0 + e))
        rows.append(dict(label=lab, e_pct=100 * e,
                         d_power_pct_fixed_mass=100 * (nom["P"] - o_fix["P"]) / o_fix["P"],
                         **closure(rotor, 1.0 / (1.0 + e))))
    tbl = pd.DataFrame(rows)
    base = tbl[tbl.label == "model"].iloc[0]
    tbl["d_mass_g"] = 1000 * (base.m_total_kg - tbl.m_total_kg)
    tbl["d_mass_pct"] = 100 * (base.m_total_kg - tbl.m_total_kg) / tbl.m_total_kg
    tbl["d_battery_g"] = 1000 * (base.m_battery_kg - tbl.m_battery_kg)
    tbl["d_power_pct"] = 100 * (base.power_rotor_W - tbl.power_rotor_W) / tbl.power_rotor_W
    with np.errstate(divide="ignore", invalid="ignore"):
        tbl["amplification"] = tbl.d_mass_pct / tbl.e_pct
    tbl["loop_gain"] = tbl.d_power_pct / tbl.d_power_pct_fixed_mass
    tbl.to_csv(os.path.join(DATA, "rotor_weight_closure.csv"), index=False)
    show = ["label", "e_pct", "m_total_kg", "m_battery_kg", "power_rotor_W", "d_mass_g",
            "d_mass_pct", "d_power_pct_fixed_mass", "d_power_pct", "loop_gain",
            "amplification"]
    print(tbl[show].round(3).to_string(index=False))

    amp = tbl[tbl.label == "95th percentile"].amplification.iloc[0]
    print(f"\n  Amplification factor at the 95th percentile: {amp:.2f} percent of take-off "
          f"weight per percent of section drag")
    print("  " + ("the error compounds" if amp > 1 else "the error is diluted, not amplified"))

    figures(rotor, nom, st, mc, tbl, mu)
    print("\nDone.")


if __name__ == "__main__":
    main(explore="--explore" in sys.argv, from_cache="--from-cache" in sys.argv)
