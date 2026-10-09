"""
A 3D-printed E387 rotor for the thrust stand, and what it should measure,
written down before anything is printed.

Section 3.5 leaves two gaps open. The rotor solver has only been checked
against propellers below Re 98,000 at 75 percent span, while the rotor in the
paper runs at 117,500 to 475,500. And profile power could only be bracketed,
not measured, because nothing in the UIUC database was run near zero thrust.
Bharath Govindarajan (IIT Madras) pointed at exactly that condition: at zero
thrust there is no induced power, so all of the torque on a static stand is
section drag.

This script designs rotors that a hobby stand can test, writes the STL files
for the Bambu A1, checks the blades will hold together, and predicts what the
stand should read.

The blade. Every rotor has the same one: the E387, which is in the benchmark
and was measured in both tunnels, with the trailing edge at 0.25 percent of
the chord as in the paper's own designs. 40 mm chord, 240 mm diameter (the
most the A1 bed takes in one piece), two blades, no taper and no twist, so
every rotor is the same blade at a different pitch. Inboard of 28 mm the
section thickens toward the hub for strength; that region carries well under
1 percent of the torque, and the solver treats it as plain E387.

The test articles.

  Z-4.6, Z-3.6, Z-2.6   three pitches around zero lift. NeuralFoil puts the
                        E387's zero-lift angle near -3.6 degrees at these
                        Reynolds numbers; the tunnel models put it anywhere
                        from -2.7 to -4.0. Three rotors a degree apart bracket
                        that, so the torque at zero thrust can be read off the
                        measurements by interpolation whatever the real angle
                        is. Near zero thrust the induced inflow grows as the
                        square root of thrust and soaks up most of a pitch
                        change, so thrust stays small and torque moves only
                        about 2 percent per degree: the as-built pitch does
                        not have to be exact.
  L+4.0                 a lifting rotor, to check thrust and power above the
                        Reynolds numbers the UIUC propellers reach. At 4
                        degrees about 60 percent of its power is induced, the
                        split of the rotor in section 3.5.
  HUB                   the hub alone, to measure and subtract the windage
                        of the hub and the motor bell.

The predictions use the solver of section 3.5 (rotor_uncertainty.Rotor), 24
blade elements, Prandtl tip and root loss, with one extension: a blade
element that makes negative thrust is balanced against momentum theory with
the flow reversed, rather than held at zero inflow, because the Z rotors sit
on both sides of zero thrust. Section lift and drag come from

  neuralfoil           NeuralFoil large, free transition, n_crit 9
  neuralfoil_tripped   NeuralFoil with transition forced at 5 percent chord
                       on both surfaces, the drag a print rough enough to trip
                       the boundary layer at the nose would show
  xfoil                XFoil on the same geometry
  uiuc, princeton      the measured E387 clean polars from each archive, every
                       model at a Reynolds number averaged into one table
                       (zero-thrust prediction only, see below)

Two kinds of prediction are written.

  1. Each rotor at fixed pitch, across RPM: thrust, torque, power.
  2. The torque at zero thrust, Q0, against RPM: the pitch is trimmed until
     thrust is zero and the torque read off. This is the quantity the stand
     can measure without any rotor model, by interpolating the three Z rotors
     to zero thrust, and it is where the tunnel polars come in, since a
     section at zero lift is inside both archives' measured range.

The drag error band. For the lifting rotor, the paper's fitted Gamma model
(section 3.4) is pushed through the rotor by the same correlated, signed
Monte Carlo as section 3.5. It is not applied to the Z rotors: the model was
fitted to points with |CL| > 0.1, and at zero lift it would be an
extrapolation. There the two archives' own E387 polars stand in for it.

Air. Predictions are made at 28 C and 99.2 kPa, a typical October afternoon
at 174 m elevation. Compare in coefficient form (C_T, C_Q against Re at 75
percent span), which does not depend on the day; the dimensional columns are
for sizing the load cells.

The strength check. Centrifugal stress plus flap bending from the predicted
thrust, at every station, against a printed PETG strength of 30 MPa in the
plane of the layers with a safety factor of 5. Each rotor's maximum RPM is
the highest multiple of 250 that passes, capped where shaft power would pass
100 W, which keeps the A2212 1000KV motor inside its rating on 12 V.

    python printed_rotor.py                full run; the first one solves the XFoil grid and caches it
    python printed_rotor.py --refresh-xfoil  solve the XFoil grid again
    python printed_rotor.py --bore 5.0     hub bore in mm; measure the motor's
                                           prop adapter with calipers first
    python printed_rotor.py --stl-only     rewrite the STL files and stop

Inputs  : data/uiuc_experimental.csv, data/soartech8_experimental.csv
          data/error_model_fit.json, data/xfoil_decomposition*.csv (the error model)
          data/bemt_validation_summary.csv (the solver's own measured error)
Outputs : hardware/printed_rotor/*.stl
          data/printed_rotor_xfoil_polars.csv   every converged XFoil point (the cache)
          data/printed_rotor_design.json        every input, max RPM, mass, STL hashes
          data/printed_rotor_prediction.csv     each rotor at fixed pitch
          data/printed_rotor_zero_thrust.csv    torque at zero thrust, every source
          figures/37_printed_rotor_prediction.png
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

import aerosandbox as asb
import rotor_uncertainty as ru

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
FIG = os.path.join(OUT, "figures")
HW = os.path.join(OUT, "hardware", "printed_rotor")

# ─────────────────────────────────────────────────────────────────────────────
# Design
# ─────────────────────────────────────────────────────────────────────────────
R = 0.120              # m, tip radius: 240 mm fits the A1's 256 mm bed in one piece
CHORD = 0.040          # m
NB = 2
TE = 0.0025            # trailing edge, fraction of chord, as in the paper's designs
R_HUB = 0.014          # m, hub radius; the solver's blade starts here
R_FULL = 0.028         # m, plain E387 from here out; thicker inboard
ROOT_THICKEN = 2.2     # thickness multiple at the hub face
HUB_H_MIN = 12.0       # mm, hub height, raised if the root section needs more
EMBED = 4.0            # mm, how far each blade runs into the hub
N_ELEM = 24

ARTICLES = {           # name: pitch in degrees, nose up positive, from the plane of rotation
    "Z-4.6": -4.6,
    "Z-3.6": -3.6,
    "Z-2.6": -2.6,
    "L+4.0": 4.0,
}
THETA0 = -3.6          # the middle of the Z family, NeuralFoil's zero-lift pitch

# Air on the day the predictions are referenced to
T_REF_C, P_REF_KPA = 28.0, 99.2

# PETG and the strength rule
RHO_PETG = 1270.0      # kg/m^3, solid (print at 100 percent infill)
STRENGTH_MPA = 30.0    # printed, in the plane of the layers, conservative
SAFETY = 5.0
P_SHAFT_MAX = 100.0    # W, the A2212 1000KV on 12 V

N_DRAWS = 1000


def air(T_C, p_kPa):
    """Density and kinematic viscosity of dry air (Sutherland's law)."""
    T = T_C + 273.15
    rho = p_kPa * 1e3 / (287.05 * T)
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    return rho, mu / rho


RHO, NU = air(T_REF_C, P_REF_KPA)
ru.RHO, ru.NU = RHO, NU          # the solver reads these module constants


# ─────────────────────────────────────────────────────────────────────────────
# The section
# ─────────────────────────────────────────────────────────────────────────────
def e387(te=TE):
    kf = asb.Airfoil("e387").to_kulfan_airfoil()
    return asb.KulfanAirfoil(name="e387", lower_weights=kf.lower_weights,
                             upper_weights=kf.upper_weights,
                             leading_edge_weight=kf.leading_edge_weight, TE_thickness=te)


def surfaces(kf, n=100):
    """x, upper y, lower y at the same cosine-spaced x, unit chord."""
    c = kf.to_airfoil(n_coordinates_per_side=n).coordinates
    i = int(np.argmin(c[:, 0]))
    up, lo = c[:i + 1][::-1], c[i:]
    return up[:, 0], up[:, 1], lo[:, 1]


def thicken(x, yu, yl, s):
    """Scale thickness about the camber line by s."""
    cam, half = (yu + yl) / 2, (yu - yl) / 2
    return cam + s * half, cam - s * half


def polygon_props(x, yu, yl):
    """Area, centroid and the second moment for flap bending (about the axis
    through the centroid parallel to the chord), plus the farthest fibre."""
    px = np.concatenate([x[::-1], x[1:]])
    py = np.concatenate([yu[::-1], yl[1:]])
    x0, y0, x1, y1 = px, py, np.roll(px, -1), np.roll(py, -1)
    cr = x0 * y1 - x1 * y0
    A = cr.sum() / 2
    cx = ((x0 + x1) * cr).sum() / (6 * A)
    cy = ((y0 + y1) * cr).sum() / (6 * A)
    Ixx = ((y0 ** 2 + y0 * y1 + y1 ** 2) * cr).sum() / 12 - A * cy ** 2
    A, Ixx = abs(A), abs(Ixx)
    return A, cx, cy, Ixx, float(np.max(np.abs(py - cy)))


def thickness_multiple(r):
    """ROOT_THICKEN at the hub face, 1 from R_FULL out, smooth in between."""
    t = np.clip((R_FULL - np.asarray(r, float)) / (R_FULL - R_HUB), 0, 1)
    return 1 + (ROOT_THICKEN - 1) * t * t * (3 - 2 * t)


# ─────────────────────────────────────────────────────────────────────────────
# Section tables for the solver
# ─────────────────────────────────────────────────────────────────────────────
ALPHAS = np.round(np.arange(-10.0, 16.01, 0.2), 3)
RES = np.geomspace(15e3, 400e3, 40)


class KulfanSection(ru.Section):
    """ru.Section, but straight from the Kulfan parameters (no refit), and
    with forced transition available."""

    def __init__(self, kf, name, xtr=1.0):
        self.name, self.af = name, kf
        self.camber_pct = float(kf.max_camber()) * 100
        self.thickness_pct = float(kf.max_thickness()) * 100
        self.alphas, self.Res = ALPHAS, RES
        A, Rg = np.meshgrid(ALPHAS, RES, indexing="ij")
        aero = kf.get_aero_from_neuralfoil(alpha=A.ravel(), Re=Rg.ravel(), model_size=ru.MODEL,
                                           xtr_upper=xtr, xtr_lower=xtr)
        kw = dict(bounds_error=False, fill_value=None)
        g = (ALPHAS, RES)
        self._CL = RegularGridInterpolator(g, aero["CL"].reshape(A.shape), **kw)
        self._CD = RegularGridInterpolator(g, np.maximum(aero["CD"], 1e-4).reshape(A.shape), **kw)
        self._conf = RegularGridInterpolator(
            g, np.clip(aero["analysis_confidence"], 0, 1).reshape(A.shape), **kw)


class TableSection(ru.Section):
    """A section from a filled (alpha, Re) table. Reynolds number is held at
    the edges of the table rather than extrapolated."""

    def __init__(self, name, alphas, Res, CL, CD, camber_pct, thickness_pct):
        self.name, self.af = name, None
        self.camber_pct, self.thickness_pct = camber_pct, thickness_pct
        self.alphas, self.Res = np.asarray(alphas), np.asarray(Res)
        kw = dict(bounds_error=False, fill_value=None)
        g = (self.alphas, np.log(self.Res))
        self._CL = RegularGridInterpolator(g, CL, **kw)
        self._CD = RegularGridInterpolator(g, np.maximum(CD, 1e-4), **kw)

    def _pts(self, alpha, Re):
        Re = np.clip(np.atleast_1d(Re), self.Res.min(), self.Res.max())
        return np.column_stack([np.atleast_1d(alpha), np.log(Re)])

    def __call__(self, alpha, Re):
        p = self._pts(alpha, Re)
        return self._CL(p), self._CD(p)

    def confidence(self, alpha, Re):
        return np.ones(len(np.atleast_1d(alpha)))


def fill_rows(alphas, CL, CD):
    """Each Reynolds row filled outside its measured or converged range: lift
    carried on at the slope of its last two degrees, drag held."""
    CL, CD = CL.copy(), CD.copy()
    for j in range(CL.shape[1]):
        ok = np.isfinite(CL[:, j]) & np.isfinite(CD[:, j])
        a, cl, cd = alphas[ok], CL[ok, j], CD[ok, j]
        lo, hi = a.min(), a.max()
        s_lo = np.polyfit(a[a <= lo + 2.0], cl[a <= lo + 2.0], 1)[0]
        s_hi = np.polyfit(a[a >= hi - 2.0], cl[a >= hi - 2.0], 1)[0]
        CL[:, j] = np.interp(alphas, a, cl)
        CD[:, j] = np.interp(alphas, a, cd)
        CL[alphas < lo, j] = cl[0] + s_lo * (alphas[alphas < lo] - lo)
        CL[alphas > hi, j] = cl[-1] + s_hi * (alphas[alphas > hi] - hi)
    return CL, CD


def tunnel_section(archive, kf):
    """The archive's E387 clean drag polars as one table. Polars within 5
    percent of the same Reynolds number are averaged across models; a
    Reynolds row is kept only if its lift reaches zero, since the table is
    only used at zero thrust."""
    if archive == "uiuc":
        d = pd.read_csv(os.path.join(DATA, "uiuc_experimental.csv"))
        key = "file"
    else:
        d = pd.read_csv(os.path.join(DATA, "soartech8_experimental.csv"))
        key = "model"
    d = d[d.asb_name.astype(str).str.lower().str.startswith("e387") & (d.config == "clean")]
    a_grid = np.round(np.arange(-7.0, 12.01, 0.25), 3)
    polars = []
    for (m, Re), g in d.groupby([key, "Re"]):
        g = g.sort_values("alpha")
        cl = np.interp(a_grid, g.alpha, g.CL, left=np.nan, right=np.nan)
        cd = np.interp(a_grid, g.alpha, g.CD, left=np.nan, right=np.nan)
        polars.append((float(Re), str(m), cl, cd))
    polars.sort()
    groups, used = [], []
    for Re, m, cl, cd in polars:
        if groups and Re <= groups[-1][0][0] * 1.05:
            groups[-1].append((Re, m, cl, cd))
        else:
            groups.append([(Re, m, cl, cd)])
    rows_Re, CLs, CDs = [], [], []
    for grp in groups:
        cl = np.nanmean([p[2] for p in grp], axis=0)
        cd = np.nanmean([p[3] for p in grp], axis=0)
        if not np.nanmin(cl) <= 0:
            continue
        rows_Re.append(np.mean([p[0] for p in grp]))
        CLs.append(cl)
        CDs.append(cd)
        used.append(dict(Re=round(rows_Re[-1]), models=sorted({p[1] for p in grp})))
    CL, CD = fill_rows(a_grid, np.array(CLs).T, np.array(CDs).T)
    sec = TableSection(archive, a_grid, np.array(rows_Re), CL, CD,
                       float(kf.max_camber()) * 100, float(kf.max_thickness()) * 100)
    sec.rows = used
    return sec


XFOIL_CACHE = os.path.join(DATA, "printed_rotor_xfoil_polars.csv")


def xfoil_section(kf, refresh=False):
    """The same grid filled by XFoil, as validate_bemt.xfoil_section does.

    The grid covers what these blades see: -7 to 10 degrees, Re 40,000 to
    320,000. Below that XFoil stalls for many minutes per sweep on this
    section, and the inboard stations that run there carry little torque;
    the table holds its lowest row instead. Converged points are cached, so
    the solve runs once."""
    alphas = np.round(np.arange(-7.0, 10.01, 0.5), 3)
    Res = np.geomspace(40e3, 320e3, 14)
    if os.path.exists(XFOIL_CACHE) and not refresh:
        pts = pd.read_csv(XFOIL_CACHE)
    else:
        from multiprocessing import Pool
        import xfoil_decomposition as xd

        if not os.path.exists(xd.XFOIL):
            raise SystemExit(f"XFoil not found at {xd.XFOIL}; build it per THIRD_PARTY_XFOIL.md")
        coords = xd.panel_coords(kf)
        jobs = [(f"{Re:.0f}", coords, float(Re), alphas) for Re in Res]
        with Pool(xd.N_PROC) as pool:
            results = dict(pool.imap_unordered(xd.run_polar, jobs))
        pts = pd.DataFrame([dict(Re=float(key), alpha=a, CL=v[0], CD=v[1])
                            for key, pol in results.items() for a, v in pol.items()])
        pts.sort_values(["Re", "alpha"]).to_csv(XFOIL_CACHE, index=False, float_format="%.6g")
    CL = np.full((len(alphas), len(Res)), np.nan)
    CD = np.full_like(CL, np.nan)
    keep, n_conv = [], 0
    for j, Re in enumerate(Res):
        pol = pts[np.isclose(pts.Re, round(Re))]
        n_conv += len(pol)
        if len(pol) < 0.4 * len(alphas):
            continue
        for _, p in pol.iterrows():
            k = int(np.argmin(np.abs(alphas - p.alpha)))
            CL[k, j], CD[k, j] = p.CL, p.CD
        keep.append(j)
    CL, CD = fill_rows(alphas, CL[:, keep], CD[:, keep])
    sec = TableSection("xfoil", alphas, Res[keep], CL, CD,
                       float(kf.max_camber()) * 100, float(kf.max_thickness()) * 100)
    sec.coverage = n_conv / (len(alphas) * len(Res))
    sec.Re_kept = (float(Res[keep].min()), float(Res[keep].max()), len(keep), len(Res))
    return sec


# ─────────────────────────────────────────────────────────────────────────────
# The rotor, at zero airspeed, with thrust of either sign
# ─────────────────────────────────────────────────────────────────────────────
def make_rotor(section, pitch_deg, rpm):
    x0 = R_HUB / R
    blade = pd.DataFrame({"r/R": [x0, 1.0], "c/R": [CHORD / R] * 2, "beta": [pitch_deg] * 2})
    return ru.Rotor(R=R, n_blades=NB, rpm=rpm, root_cut=x0, n_elem=N_ELEM,
                    section=section, blade=blade)


def static_solve(rot, collective_deg=0.0, cd_scale=1.0):
    """rot.solve, plus elements that make negative thrust. Those the parent
    holds at zero inflow; here the flow through that annulus reverses and
    the element is balanced against momentum thrust -4 pi rho vi^2 F r."""
    coll = np.deg2rad(collective_deg)
    _, d = rot.solve(coll, cd_scale)
    cds = np.broadcast_to(np.atleast_1d(cd_scale), (len(rot.x),))
    rows = d.to_dict("records")
    for i, row in enumerate(rows):
        if row["phi_deg"] > 1e-3 or row["dT"] >= 0:
            continue

        def res(phi):
            u_t = rot.omega * rot.r[i]
            vi = u_t * np.tan(phi)
            u = np.hypot(u_t, vi)
            alpha = np.rad2deg(rot.twist[i] + coll - phi)
            cl, cd = rot.section(alpha, u * rot.chord[i] / ru.NU)
            dT = 0.5 * ru.RHO * u ** 2 * rot.chord[i] * rot.Nb * (
                cl[0] * np.cos(phi) - cd[0] * cds[i] * np.sin(phi))
            return dT + 4 * np.pi * ru.RHO * vi ** 2 * rot._tip_loss(i, phi) * rot.r[i]

        hi = -1e-7
        lo = -np.deg2rad(0.5)
        while res(lo) < 0 and lo > -np.deg2rad(45):
            hi, lo = lo, lo - np.deg2rad(0.5)
        phi = brentq(res, lo, hi, xtol=1e-12)
        vi = rot.omega * rot.r[i] * np.tan(phi)
        dT, dFx, dFx_i, dFx_p, alpha, Re, cl, cd, phi = rot._element(vi, i, coll, cds[i])
        rows[i] = dict(x=rot.x[i], r=rot.r[i], dr=rot.dr[i], chord=rot.chord[i], vi=vi,
                       phi_deg=np.rad2deg(phi), alpha_deg=alpha, Re=Re, CL=cl, CD=cd,
                       dT=dT * rot.dr[i], dQ=dFx * rot.r[i] * rot.dr[i],
                       dQ_i=dFx_i * rot.r[i] * rot.dr[i], dQ_p=dFx_p * rot.r[i] * rot.dr[i])
    d = pd.DataFrame(rows)
    T, Q, Qi, Qp = d.dT.sum(), d.dQ.sum(), d.dQ_i.sum(), d.dQ_p.sum()
    n, D = rot.rpm / 60, 2 * R
    out = dict(T=T, Q=Q, P=rot.omega * Q, P_induced=rot.omega * Qi, P_profile=rot.omega * Qp,
               induced_frac=Qi / Q if Q > 0 else np.nan,
               CT=T / (ru.RHO * n ** 2 * D ** 4), CQ=Q / (ru.RHO * n ** 2 * D ** 5),
               Re_75=float(np.interp(0.75, d.x, d.Re)), Re_tip=float(d.Re.iloc[-1]),
               Re_root=float(d.Re.iloc[0]), tip_mach=rot.omega * R / ru.A_SOUND,
               alpha_75=float(np.interp(0.75, d.x, d.alpha_deg)))
    return out, d


def trim_zero_thrust(section, rpm, cd_scale=1.0):
    """Pitch for zero thrust, and the torque there."""
    rot = make_rotor(section, THETA0, rpm)
    f = lambda c: static_solve(rot, c, cd_scale)[0]["T"]
    lo, hi = -3.0, 3.0
    while f(lo) > 0:
        lo -= 1.0
    while f(hi) < 0:
        hi += 1.0
    c = brentq(f, lo, hi, xtol=1e-5)
    out, d = static_solve(rot, c, cd_scale)
    out["pitch_deg"] = THETA0 + c
    return out, d


# ─────────────────────────────────────────────────────────────────────────────
# Strength, and the RPM each rotor may run to
# ─────────────────────────────────────────────────────────────────────────────
def blade_properties(kf):
    """Area, flap second moment and farthest fibre along the span, in SI."""
    x, yu, yl = surfaces(kf)
    r = np.linspace(R_HUB, R, 300)
    props = []
    for s in thickness_multiple(r):
        u, l = thicken(x, yu, yl, s)
        A, _, _, I, y = polygon_props(x, u, l)
        props.append((A * CHORD ** 2, I * CHORD ** 4, y * CHORD))
    A, I, y = np.array(props).T
    return r, A, I, y


def peak_stress(props, rpm, stations):
    """Largest centrifugal plus flap-bending stress along one blade, MPa.
    Bending is taken without centrifugal relief, which overstates it."""
    r, A, I, y = props
    om = rpm * 2 * np.pi / 60
    dr = np.gradient(r)
    cf = np.cumsum((RHO_PETG * A * om ** 2 * r * dr)[::-1])[::-1]
    tpu = np.interp(r, stations.r, stations.dT / stations.dr) / NB      # thrust per span, one blade
    M = np.array([np.sum(tpu[k:] * (r[k:] - r[k]) * dr[k:]) for k in range(len(r))])
    return float(np.max(cf / A + np.abs(M) * y / I)) / 1e6


def max_rpm(section, props, pitch):
    allow = STRENGTH_MPA / SAFETY
    best = None
    for rpm in np.arange(1000, 12001, 250):
        out, d = static_solve(make_rotor(section, pitch, rpm))
        s = peak_stress(props, rpm, d)
        if s > allow or out["P"] > P_SHAFT_MAX:
            break
        best = (int(rpm), s, out["P"])
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: STL files
# ─────────────────────────────────────────────────────────────────────────────
def blade_mesh(kf, pitch_deg, z0):
    """One blade along +X, leading edge toward +Y, so the rotor turns
    counter-clockwise seen from the top (+Z, the suction side), and thrust
    is toward +Z. Sections are stacked on their centroid, so spinning puts
    no bending into the blade."""
    import manifold3d as m

    x, yu, yl = surfaces(kf)
    n = len(x)
    _, xc, yc, _, _ = polygon_props(x, yu, yl)
    th = np.deg2rad(pitch_deg)
    rs = np.concatenate([np.linspace(R_HUB * 1e3 - EMBED, R_FULL * 1e3, 30),
                         np.linspace(R_FULL * 1e3 + 4.0, R * 1e3, 24)])   # constant section: few stations needed
    rings = []
    for r in rs:
        u, l = thicken(x, yu, yl, thickness_multiple(r / 1e3))
        px = np.concatenate([x[::-1], x[1:]])             # TE, upper to LE, lower to TE
        py = np.concatenate([u[::-1], l[1:]])
        s, t = (px - xc) * CHORD * 1e3, (py - yc) * CHORD * 1e3
        Y = -s * np.cos(th) - t * np.sin(th)
        Z = -s * np.sin(th) + t * np.cos(th) + z0
        rings.append(np.column_stack([np.full_like(Y, r), Y, Z]))
    V = np.vstack(rings)
    k = 2 * n - 1
    F = []
    for j in range(len(rs) - 1):
        a, b = j * k, (j + 1) * k
        for i in range(k):
            i2 = (i + 1) % k
            F += [(a + i, a + i2, b + i2), (a + i, b + i2, b + i)]
    for base in (0, (len(rs) - 1) * k):             # end caps, strips between the surfaces
        U = lambda q: base + (n - 1 - q)            # upper surface at x[q]
        L = lambda q: base + (n - 1 + q) if q else base + n - 1
        F.append((U(0), U(1), L(1)))
        for q in range(1, n - 1):
            F += [(U(q), U(q + 1), L(q + 1)), (U(q), L(q + 1), L(q))]
    F = np.array(F, dtype=np.int64)
    nside = 2 * k * (len(rs) - 1)

    def normal(f):
        p = V[f]
        return np.cross(p[1] - p[0], p[2] - p[0])

    # orient the sides outward, the root cap toward -X and the tip cap toward +X
    q = int(np.argmin(np.abs(x - 0.3)))             # upper surface near max thickness,
    probe = F[2 * (n - 2 - q)]                      # where outward is unambiguously up
    cen = rings[0].mean(axis=0)
    if np.dot(normal(probe), V[probe].mean(axis=0) - cen) < 0:
        F[:nside] = F[:nside, ::-1]
    ncap = (len(F) - nside) // 2
    root, tip = slice(nside, nside + ncap), slice(nside + ncap, None)
    if normal(F[root][len(F[root]) // 2])[0] > 0:
        F[root] = F[root][:, ::-1]
    if normal(F[tip][len(F[tip]) // 2])[0] < 0:
        F[tip] = F[tip][:, ::-1]
    man = m.Manifold(m.Mesh(vert_properties=V.astype(np.float32),
                            tri_verts=F.astype(np.uint32)))
    if man.status() != m.Error.NoError:
        raise RuntimeError(f"blade mesh is not a closed solid: {man.status()}")
    return man


def hub_height(kf):
    """Tall enough to hold the thickest root section at any article's pitch."""
    x, yu, yl = surfaces(kf)
    u, l = thicken(x, yu, yl, ROOT_THICKEN)
    _, xc, yc, _, _ = polygon_props(x, yu, yl)
    ext = 0.0
    for p in ARTICLES.values():
        th = np.deg2rad(p)
        s = (np.concatenate([x, x]) - xc) * CHORD * 1e3
        t = (np.concatenate([u, l]) - yc) * CHORD * 1e3
        Z = -s * np.sin(th) + t * np.cos(th)
        ext = max(ext, 2 * np.max(np.abs(Z)))
    return max(HUB_H_MIN, np.ceil(ext + 1.0))


def hub(h, bore_mm):
    import manifold3d as m
    body = m.Manifold.cylinder(h, R_HUB * 1e3, -1.0, 128)
    hole = m.Manifold.cylinder(h + 2, bore_mm / 2, -1.0, 64).translate((0, 0, -1))
    mark = m.Manifold.sphere(1.5, 32).translate((0, 9.0, h))       # top face, suction side
    return body - hole - mark


def write_stl(man, path):
    mesh = man.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    F = np.asarray(mesh.tri_verts).astype(np.int64)
    p = V[F]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"printed_rotor.py, E387 test rotor, mm".ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        rec["n"], rec["v"] = nrm, p
        f.write(rec.tobytes())
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_stl(kf, bore_mm):
    import manifold3d as m
    os.makedirs(HW, exist_ok=True)
    h = hub_height(kf)
    files = {}
    for name, pitch in ARTICLES.items():
        b1 = blade_mesh(kf, pitch, h / 2)
        b2 = b1.rotate((0, 0, 180))
        man = (b1 + b2 + m.Manifold.cylinder(h, R_HUB * 1e3, -1.0, 128)) - (
            m.Manifold.cylinder(h + 2, bore_mm / 2, -1.0, 64).translate((0, 0, -1))
            + m.Manifold.sphere(1.5, 32).translate((0, 9.0, h)))
        files[name] = man
    files["HUB"] = hub(h, bore_mm)
    info = {}
    for name, man in files.items():
        if man.status() != m.Error.NoError:
            raise RuntimeError(f"{name}: {man.status()}")
        lo = np.array(man.bounding_box()[:3]); hi = np.array(man.bounding_box()[3:])
        if lo[2] < -1e-3:
            raise RuntimeError(f"{name}: part dips {-lo[2]:.2f} mm below the hub base")
        path = os.path.join(HW, f"rotor_{name}.stl" if name != "HUB" else "hub_tare.stl")
        sha = write_stl(man, path)
        vol = man.volume() / 1e3                                     # cm^3
        info[name] = dict(file=os.path.relpath(path, OUT), sha256=sha,
                          volume_cm3=round(vol, 2), mass_g=round(vol * RHO_PETG / 1e3, 1),
                          size_mm=[round(float(v), 1) for v in hi - lo],
                          genus=int(man.genus()), triangles=int(man.num_tri()))
        print(f"  {info[name]['file']}: {info[name]['mass_g']} g solid, "
              f"{info[name]['size_mm'][0]} x {info[name]['size_mm'][1]} x "
              f"{info[name]['size_mm'][2]} mm")
    return h, info


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────
def rpm_list(rpm_max, start=3000, step=1000):
    pts = list(range(start, rpm_max + 1, step))
    if not pts or pts[-1] != rpm_max:
        pts.append(rpm_max)
    return pts


def band(section, pitch, rpms, err, n=N_DRAWS):
    """Percentiles of thrust and torque when the true drag is the NeuralFoil
    drag divided by (1 + e), with e drawn from the paper's Gamma model, one
    draw shared by every element and signed by the measured bias, as in the
    headline case of section 3.5."""
    rows = []
    for rpm in rpms:
        rot = make_rotor(section, pitch, rpm)
        nom, st = static_solve(rot)
        mu = err.expected_magnitude(section.confidence(st.alpha_deg.values, st.Re.values),
                                    st.Re.values, st.alpha_deg.values,
                                    section.camber_pct, section.thickness_pct)
        e = err.draw(mu, st.Re.values, n, np.random.default_rng(ru.SEED),
                     signed=True, correlated=True)
        T, Q = [], []
        for k in range(n):
            o, _ = static_solve(rot, 0.0, 1.0 / (1.0 + e[k]))
            T.append(o["T"]); Q.append(o["Q"])
        T, Q = np.array(T), np.array(Q)
        rows.append(dict(rpm=rpm, mu_blade=float(np.average(mu, weights=st.dQ.clip(0))),
                         **{f"Q_Nmm_p{p}": float(np.percentile(Q, q)) * 1e3
                            for p, q in [("2_5", 2.5), ("10", 10), ("50", 50), ("90", 90), ("97_5", 97.5)]},
                         T_N_p10=float(np.percentile(T, 10)), T_N_p90=float(np.percentile(T, 90))))
    return pd.DataFrame(rows)


def figure(pred, zero, design):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"neuralfoil": ("#1f77b4", "-", "NeuralFoil"),
             "neuralfoil_tripped": ("#1f77b4", ":", "NeuralFoil, tripped at 5% chord"),
             "xfoil": ("#ff7f0e", "--", "XFoil"),
             "uiuc": ("#2ca02c", "-", "UIUC tunnel E387 polars"),
             "princeton": ("#d62728", "-", "Princeton tunnel E387 polars")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for src, g in zero.groupby("source"):
        c, ls, lab = style[src]
        ax.plot(g.Re_75 / 1e3, g.CQ * 1e4, ls, color=c, lw=2, marker="o", ms=4, label=lab)
    ax.set_xlabel("Reynolds number at 75% span (thousands)")
    ax.set_ylabel(r"Torque coefficient at zero thrust, $C_Q \times 10^4$")
    ax.set_title("Z rotors, read at zero thrust: all of this torque is section drag",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)

    ax = axes[1]
    L = pred[pred.article == "L+4.0"]
    for src in ["neuralfoil", "xfoil", "neuralfoil_tripped"]:
        g = L[L.source == src]
        c, ls, lab = style[src]
        ax.plot(g.Re_75 / 1e3, g.CQ * 1e4, ls, color=c, lw=2, marker="o", ms=4, label=lab)
    g = L[L.source == "neuralfoil"]
    D, n = 2 * R, g.rpm / 60
    k = 1e-3 / (RHO * n ** 2 * D ** 5) * 1e4
    ax.fill_between(g.Re_75 / 1e3, g.Q_Nmm_p10 * k, g.Q_Nmm_p90 * k, color="#1f77b4", alpha=0.18,
                    label="paper's drag error model, 10th to 90th percentile")
    ax.set_xlabel("Reynolds number at 75% span (thousands)")
    ax.set_ylabel(r"Torque coefficient, $C_Q \times 10^4$")
    ax.set_title(f"L+4.0, lifting rotor (max {design['articles']['L+4.0']['max_rpm']} rpm)",
                 fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.suptitle("Printed E387 rotor, 240 mm, predicted before printing", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG, "37_printed_rotor_prediction.png")
    fig.savefig(path, dpi=160)
    print(f"  wrote {os.path.relpath(path, OUT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bore", type=float, default=6.0, help="hub bore, mm")
    ap.add_argument("--stl-only", action="store_true")
    ap.add_argument("--refresh-xfoil", action="store_true", help="rerun XFoil, ignore the cache")
    args = ap.parse_args()

    kf = e387()
    print(f"E387, trailing edge {TE * 100:.2f}% chord: thickness "
          f"{kf.max_thickness() * 100:.2f}%, camber {kf.max_camber() * 100:.2f}%")
    print("Writing STL files ...")
    h, stl = build_stl(kf, args.bore)
    if args.stl_only:
        return

    print(f"Air: {T_REF_C} C, {P_REF_KPA} kPa, rho {RHO:.4f} kg/m^3, nu {NU:.3e} m^2/s")
    print("Building section tables ...", flush=True)
    secs = {"neuralfoil": KulfanSection(kf, "neuralfoil"),
            "neuralfoil_tripped": KulfanSection(kf, "neuralfoil_tripped", xtr=0.05)}
    secs["xfoil"] = xfoil_section(kf, refresh=args.refresh_xfoil)
    xs = secs["xfoil"]
    print(f"  XFoil converged at {100 * xs.coverage:.0f}% of the grid; kept "
          f"{xs.Re_kept[2]} of {xs.Re_kept[3]} Reynolds rows, {xs.Re_kept[0]:.0f} to {xs.Re_kept[1]:.0f}")
    tun = {a: tunnel_section(a, kf) for a in ("uiuc", "princeton")}
    for a, s in tun.items():
        print(f"  {a}: {len(s.rows)} Reynolds rows, " +
              ", ".join(f"{r['Re'] / 1e3:.0f}k ({len(r['models'])})" for r in s.rows))

    print("Strength and maximum RPM ...")
    props = blade_properties(kf)
    articles = {}
    for name, pitch in ARTICLES.items():
        rpm, s, P = max_rpm(secs["neuralfoil"], props, pitch)
        articles[name] = dict(pitch_deg=pitch, max_rpm=rpm, peak_stress_MPa=round(s, 2),
                              shaft_power_W_at_max=round(P, 1), **stl[name])
        print(f"  {name}: max {rpm} rpm, peak stress {s:.2f} MPa "
              f"(allowed {STRENGTH_MPA / SAFETY:.1f}), shaft power {P:.1f} W")
    articles["HUB"] = dict(stl["HUB"])

    print("Fixed-pitch predictions ...", flush=True)
    rows = []
    for name, pitch in ARTICLES.items():
        for src in ("neuralfoil", "neuralfoil_tripped", "xfoil"):
            for rpm in rpm_list(articles[name]["max_rpm"]):
                o, d = static_solve(make_rotor(secs[src], pitch, rpm))
                conf = (float(np.interp(0.75, d.x, secs[src].confidence(d.alpha_deg.values, d.Re.values)))
                        if src == "neuralfoil" else np.nan)
                rows.append(dict(article=name, pitch_deg=pitch, source=src, rpm=rpm,
                                 Re_75=o["Re_75"], Re_tip=o["Re_tip"], alpha_75=o["alpha_75"],
                                 conf_75=conf, T_N=o["T"], Q_Nmm=o["Q"] * 1e3, P_W=o["P"],
                                 induced_frac=o["induced_frac"], CT=o["CT"], CQ=o["CQ"]))
    pred = pd.DataFrame(rows)

    print(f"Drag error band on L+4.0, {N_DRAWS} draws per RPM ...", flush=True)
    err = ru.DragError()
    b = band(secs["neuralfoil"], ARTICLES["L+4.0"], rpm_list(articles["L+4.0"]["max_rpm"]), err)
    b["article"], b["source"] = "L+4.0", "neuralfoil"
    pred = pred.merge(b, on=["article", "source", "rpm"], how="left")

    print("Torque at zero thrust ...", flush=True)
    zrows = []
    rpm_z = min(articles[k]["max_rpm"] for k in ARTICLES if k.startswith("Z"))
    for src, sec in list(secs.items()) + list(tun.items()):
        for rpm in rpm_list(rpm_z):
            o, d = trim_zero_thrust(sec, rpm)
            w = (d.dQ * (d.Re < 60e3)).sum() / d.dQ.sum()
            zrows.append(dict(source=src, rpm=rpm, pitch_deg=o["pitch_deg"], Re_75=o["Re_75"],
                              Re_tip=o["Re_tip"], Q_Nmm=o["Q"] * 1e3, P_W=o["P"], CQ=o["CQ"],
                              induced_frac=o["induced_frac"],
                              torque_share_below_Re60k=float(w)))
    zero = pd.DataFrame(zrows)

    vb = pd.read_csv(os.path.join(DATA, "bemt_validation_summary.csv"))
    design = dict(
        purpose="Predictions for printed E387 test rotors, made before printing.",
        section=dict(name="E387", kulfan="asb.Airfoil('e387').to_kulfan_airfoil()",
                     TE_thickness_frac=TE, thickness_pct=round(kf.max_thickness() * 100, 3),
                     camber_pct=round(kf.max_camber() * 100, 3)),
        blade=dict(R_m=R, chord_m=CHORD, n_blades=NB, twist_deg=0.0, taper=1.0,
                   hub_radius_m=R_HUB, plain_E387_from_m=R_FULL, root_thicken=ROOT_THICKEN,
                   hub_height_mm=h, bore_mm=args.bore, rotation="counter-clockwise seen "
                   "from the marked face; leading edge (the rounded edge) leads"),
        air=dict(T_C=T_REF_C, p_kPa=P_REF_KPA, rho=RHO, nu=NU),
        solver=dict(model="rotor_uncertainty.Rotor, 24 elements, Prandtl tip and root loss, "
                          "negative-thrust elements balanced with reversed momentum",
                    root_cut=R_HUB / R, neuralfoil_model=ru.MODEL),
        strength=dict(petg_density=RHO_PETG, strength_MPa=STRENGTH_MPA, safety_factor=SAFETY,
                      shaft_power_cap_W=P_SHAFT_MAX),
        xfoil=dict(coverage=xs.coverage, Re_rows_kept=xs.Re_kept),
        tunnel_rows={a: s.rows for a, s in tun.items()},
        bemt_validation=vb[vb.label == "all 9 in"].to_dict("records"),
        articles=articles,
    )
    with open(os.path.join(DATA, "printed_rotor_design.json"), "w") as f:
        json.dump(design, f, indent=2, default=float)
    pred.to_csv(os.path.join(DATA, "printed_rotor_prediction.csv"), index=False, float_format="%.6g")
    zero.to_csv(os.path.join(DATA, "printed_rotor_zero_thrust.csv"), index=False, float_format="%.6g")
    print("  wrote data/printed_rotor_design.json, data/printed_rotor_prediction.csv, "
          "data/printed_rotor_zero_thrust.csv")

    print("\nTorque at zero thrust, N mm (and pitch for zero thrust):")
    print(zero.pivot(index="rpm", columns="source", values="Q_Nmm").round(2).to_string())
    print(zero.pivot(index="rpm", columns="source", values="pitch_deg").round(2).to_string())
    print("\nFixed pitch, NeuralFoil:")
    print(pred[pred.source == "neuralfoil"][["article", "rpm", "Re_75", "T_N", "Q_Nmm", "P_W",
                                             "induced_frac"]].round(3).to_string(index=False))
    figure(pred, zero, design)


if __name__ == "__main__":
    main()
