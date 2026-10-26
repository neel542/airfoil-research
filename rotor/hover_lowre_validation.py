"""
The rotor solver against measured hover rotors inside the benchmark's own
Reynolds range, and the zero-thrust profile-power check done without a stand.

Section 3.5 says the solver was checked below its operating point, not at it:
the UIUC propellers stop at Re 98,000 at 75 percent span, and six of the seven
are mostly induced power. Harris (2020), NASA CR-2020-5001147, Appendix A,
tabulates four sets of measured hover rotors that sit where the design rotor
sits. All are untwisted and constant chord, with one symmetric section from
root to tip:

    Knight and Hefner 1937   NACA 0015, 2 to 5 blades, tip Re 267,825
    Landgrebe 1971           NACA 0012, 2 to 8 blades, tip Re 411,205 to 548,274
    Ramasamy 2015 ("Mani")   NACA 0012, 3 and 6 blades, tip Re 219,000 to 331,000
    Bhagwat and Ramasamy     NACA 0012, 2 to 6 blades, tip Re 139,528 to 334,866
    2018 ("Mahendra & Mani")

Four parts, each printed and written to data/:

  1. Parse. Appendix A is read from the report's own text layer
     (data/harris2020/appendix_a.txt, made with pdftotext from the NTRS PDF
     whose SHA-256 is in data/harris2020/source.json), never typed by hand.
     Every row is checked twice: CT/sigma and CP/sigma must equal CT and CP
     divided by sigma to the printed precision, and the Knight and Hefner and
     Landgrebe rows must match the University of Liege rotor database's copy.
     That copy carries Harris's own row notes, so it checks this parse, not
     Harris: NACA TN-626 itself gives its rotor data only as plotted curves.
  2. Zero thrust. An untwisted symmetric blade at zero pitch makes no lift, so
     there is no inflow, every station runs at zero incidence, and all of the
     shaft power is section drag:
         CP/sigma = 1/2 * integral from rc to 1 of Cd(0, Re_tip x, M_tip x) x^3 dx.
     That is the near-zero-thrust reading Govindarajan suggested, on rotors
     someone else built and measured. Cd comes from NeuralFoil in free
     transition at the benchmark's n_crit of 9, at lower n_crit, with
     transition forced at 5 percent chord, and from XFoil in free transition.
  3. The whole sweep. Every measured point with thrust, at tip Mach 0.3 or
     less (which drops Landgrebe), is trimmed to its measured thrust in
     rotor_uncertainty.Rotor and the predicted power is set against the
     measured power. Collective is compared only for Knight and Hefner;
     Harris says the collective settings in Ramasamy's data are inaccurate.
  4. Swirl. The solver has none. A first-order estimate of what it would add,
     from the angular momentum the solver's own torque leaves in the wake, set
     against the part 3 shortfall. See swirl().

What this does not settle. The report gives no hub tare for any of the four
tests, so measured power may include hub and blade-root drag the solver does
not have. Harris himself warns that torque at zero thrust was hard to measure
and scattered. Appendix A gives Knight and Hefner's root cutout as 0.150 and
Table 2 gives 16.7 percent; at zero thrust that changes the integral by less
than 0.1 percent. Compressibility is in part 2 (NeuralFoil's Mach input) and
not in part 3, which only runs to tip Mach 0.25.

    python hover_lowre_validation.py            # parts 1 to 4
    python hover_lowre_validation.py --zero     # parts 1 and 2 only

Inputs  : data/harris2020/appendix_a.txt, data/harris2020/source.json,
          data/harris2020/liege_rotor_db/*.csv
Outputs : data/hover_lowre_rotors.csv        every row of Appendix A, parsed
          data/hover_zero_thrust.csv         part 2
          data/hover_lowre_validation.csv    part 3, per point
          data/hover_lowre_summary.csv       part 3, per experiment
          data/hover_swirl.csv               part 4, per point
          figures/38_hover_lowre.png
"""

import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import aerosandbox as asb
import rotor_uncertainty as ru

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this script lives one folder down)
DATA = os.path.join(OUT, "data")
FIG = os.path.join(OUT, "figures")
SRC = os.path.join(DATA, "harris2020")

IN2M = 0.0254
FPS2MS = 0.3048
MODEL = ru.MODEL                 # the benchmark's NeuralFoil model, "large"
MACH_MAX = 0.30                  # part 3 runs incompressible; Landgrebe is 0.47 to 0.63

# Harris (2020) Table 2: chord and airfoil per experiment. Radius comes from
# each row's own solidity, R = Nb c / (pi sigma), which also catches the
# cut-down blades in the 2018 test.
EXPERIMENTS = {
    "Knight1937":   dict(title="Knight &", chord_in=2.000, airfoil="naca0015"),
    "Landgrebe1971": dict(title="Landgrebe", chord_in=1.470, airfoil="naca0012"),
    "Ramasamy2015": dict(title="Mani's Test Data", chord_in=2.290, airfoil="naca0012"),
    "Bhagwat2018":  dict(title="Mahendra &", chord_in=2.297, airfoil="naca0012"),
}
ROW = re.compile(
    r"^\s*(\d)\s+(0\.\d+)\s+([\d.]+)\s+([\d,]+)\s+([\d.]+)\s+(-?[\d.]+)\s+"
    r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s*(.*?)\s*$")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: parse and check
# ─────────────────────────────────────────────────────────────────────────────
def parse():
    rows, exp = [], None
    with open(os.path.join(SRC, "appendix_a.txt"), encoding="utf-8") as f:
        for line in f:
            for key, e in EXPERIMENTS.items():
                if line.strip().startswith(e["title"]):
                    exp = key
            m = ROW.match(re.sub(r"-\s+(\d)", r"-\1", line))     # one row prints "- 0.00133"
            if not m:
                continue
            g = m.groups()
            rows.append(dict(experiment=exp, n_blades=int(g[0]), sigma=float(g[1]),
                             v_tip_fps=float(g[2]), Re_tip=float(g[3].replace(",", "")),
                             M_tip=float(g[4]), collective_deg=float(g[5]),
                             CT=float(g[6]), CP=float(g[7]), CT_sigma_printed=float(g[8]),
                             CP_sigma_printed=float(g[9]), root_cut=float(g[10]), note=g[11]))
    d = pd.DataFrame(rows)
    d["chord_m"] = d.experiment.map({k: e["chord_in"] * IN2M for k, e in EXPERIMENTS.items()})
    d["airfoil"] = d.experiment.map({k: e["airfoil"] for k, e in EXPERIMENTS.items()})
    d["R_m"] = d.n_blades * d.chord_m / (np.pi * d.sigma)
    d["nu"] = d.v_tip_fps * FPS2MS * d.chord_m / d.Re_tip
    # the ratios are recomputed from the coefficients; the printed ones are
    # kept only to check the parse, and four of them are misprinted
    d["CT_sigma"] = d.CT / d.sigma
    d["CP_sigma"] = d.CP / d.sigma
    return d


# Misprints in Appendix A, each found by check() and confirmed against the
# rows around it. The CT and CP columns are right in every case.
KNOWN_MISPRINTS = {
    ("Landgrebe1971", 6, 411205, 0.0, "CT"): "CT/sigma printed +0.00019, CT is -0.000020",
    ("Bhagwat2018", 3, 139528, 3.6, "CP"): "CP/sigma printed 0.000005, CP/sigma is 0.002596",
    ("Bhagwat2018", 3, 139528, 8.3, "CP"): "CP/sigma printed 0.000005, CP/sigma is 0.005157",
    ("Bhagwat2018", 3, 139528, 9.3, "CP"): "CP/sigma printed 0.000005, CP/sigma is 0.006482",
}


def check(d):
    """The two checks on the parse. Returns a list of problems, empty if none."""
    bad, seen = [], set()
    for col in ("CT", "CP"):
        printed = d[f"{col}_sigma_printed"]
        diff = (d[col] / d.sigma - printed).abs()
        # the printed ratios carry 5 or 6 decimals, so near zero thrust only
        # the absolute difference means anything
        off = (diff / np.maximum(printed.abs(), 1e-6) > 0.01) & (diff > 2e-5)
        for i in d.index[off]:
            r = d.loc[i]
            key = (r.experiment, int(r.n_blades), int(r.Re_tip), float(r.collective_deg), col)
            if key in KNOWN_MISPRINTS:
                seen.add(key)
            else:
                bad.append(f"{col}/sigma mismatch, row {i}: {key}")
    bad += [f"listed misprint not found: {k}" for k in KNOWN_MISPRINTS if k not in seen]
    liege = os.path.join(SRC, "liege_rotor_db")
    for key, fn in (("Knight1937", "Knight1937_data.csv"), ("Landgrebe1971", "Landgrebe1971_data.csv")):
        # split on commas and spaces both: two Landgrebe rows in that file have
        # a space where a comma should be, and one has "- 0.00133"
        theirs = []
        with open(os.path.join(liege, fn), encoding="utf-8") as f:
            for line in f:
                tok = re.sub(r"-\s+(\d)", r"-\1", line).replace(",", " ").split()
                if line.startswith("#") or not tok or not re.match(r"^-?\d", tok[0]):
                    continue
                v = [float(t) for t in tok[:10]]
                theirs.append([v[0], v[5], v[6], v[7]])
        theirs = np.array(theirs)
        mine = d[d.experiment == key][["n_blades", "collective_deg", "CT", "CP"]].to_numpy()
        if len(mine) != len(theirs) or not np.allclose(mine, theirs, rtol=0, atol=1e-7):
            bad.append(f"{key}: {len(mine)} rows here, {len(theirs)} in the Liege database, "
                       f"or the values differ")
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: zero thrust
# ─────────────────────────────────────────────────────────────────────────────
NF_CASES = [("NF free, n_crit 9", dict(n_crit=9.0)),
            ("NF free, n_crit 7", dict(n_crit=7.0)),
            ("NF free, n_crit 5", dict(n_crit=5.0)),
            ("NF free, n_crit 3", dict(n_crit=3.0)),
            ("NF tripped at 5% chord", dict(xtr_upper=0.05, xtr_lower=0.05))]


def profile_integral(cd, x):
    """CP/sigma = 1/2 integral Cd x^3 dx over the blade."""
    return 0.5 * np.trapz(cd * x ** 3, x)


def xfoil_cd0(airfoil, Res):
    """XFoil zero-incidence drag, free transition at n_crit 9, on a Re grid.
    Points that do not converge come back NaN."""
    from multiprocessing import Pool
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))
    import xfoil_decomposition as xd

    if not os.path.exists(xd.XFOIL):
        return np.full(len(Res), np.nan)
    coords = asb.Airfoil(airfoil).repanel(n_points_per_side=80).coordinates
    jobs = [(f"{Re:.0f}", coords, float(Re), np.array([0.0])) for Re in Res]
    with Pool(xd.N_PROC) as pool:
        res = dict(pool.imap_unordered(xd.run_polar, jobs))
    return np.array([res.get(f"{Re:.0f}", {}).get(0.0, (np.nan, np.nan))[1] for Re in Res])


def zero_thrust(d):
    z = d[(d.collective_deg.abs() < 1e-9)].copy()
    z["measured"] = ~z.note.str.contains("Extrapolated|Questionable", case=False)
    # momentum theory's least induced power at the small residual thrust
    z["CP_sigma_induced_min"] = np.sign(z.CT) * np.abs(z.CT) ** 1.5 / np.sqrt(2) / z.sigma
    x = np.linspace(0.0, 1.0, 401)
    xf_grid = np.geomspace(15e3, 600e3, 24)
    xf = {a: xfoil_cd0(a, xf_grid) for a in z.airfoil.unique()}
    out = []
    for _, r in z.iterrows():
        xs = x[x >= r.root_cut]
        af = asb.Airfoil(r.airfoil)
        row = r.to_dict()
        for label, kw in NF_CASES:
            a = af.get_aero_from_neuralfoil(alpha=np.zeros_like(xs), Re=r.Re_tip * xs,
                                            mach=r.M_tip * xs, model_size=MODEL, **kw)
            row[label] = profile_integral(np.asarray(a["CD"]), xs)
        cdx = xf[r.airfoil]
        ok = np.isfinite(cdx)
        row["XFoil free, n_crit 9"] = (profile_integral(np.interp(np.log(r.Re_tip * xs),
                                                                  np.log(xf_grid[ok]), cdx[ok]), xs)
                                       if ok.sum() >= 4 else np.nan)
        row["xfoil_converged"] = int(ok.sum())
        out.append(row)
    z = pd.DataFrame(out)
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: the sweep
# ─────────────────────────────────────────────────────────────────────────────
class Section(ru.Section):
    """ru.Section with NeuralFoil's transition inputs passed through."""

    def __init__(self, name, **nf_kw):
        self.name = name
        self.af = asb.Airfoil(name)
        self.alphas = np.linspace(-10, 20, 121)
        self.Res = np.geomspace(15e3, 700e3, 40)
        A, Rg = np.meshgrid(self.alphas, self.Res, indexing="ij")
        aero = self.af.get_aero_from_neuralfoil(alpha=A.ravel(), Re=Rg.ravel(),
                                                model_size=MODEL, **nf_kw)
        from scipy.interpolate import RegularGridInterpolator as RGI
        grid, kw = (self.alphas, self.Res), dict(bounds_error=False, fill_value=None)
        self._CL = RGI(grid, aero["CL"].reshape(A.shape), **kw)
        self._CD = RGI(grid, np.maximum(aero["CD"], 1e-4).reshape(A.shape), **kw)
        self._conf = RGI(grid, np.clip(aero["analysis_confidence"], 0, 1).reshape(A.shape), **kw)


SWEEP_SECTIONS = [("free", {}), ("tripped", dict(xtr_upper=0.05, xtr_lower=0.05))]


def _rotor(r, section):
    """One measured point as a solver rotor: untwisted, constant chord, the
    point's own viscosity. Returns the rotor, the measured thrust and tip speed."""
    ru.NU = r.nu
    v_tip = r.v_tip_fps * FPS2MS
    rpm = v_tip / r.R_m * 60 / (2 * np.pi)
    blade = pd.DataFrame({"r/R": [r.root_cut, 1.0], "c/R": [r.chord_m / r.R_m] * 2,
                          "beta": [0.0, 0.0]})
    T = r.CT * ru.RHO * np.pi * r.R_m ** 2 * v_tip ** 2
    rot = ru.Rotor(R=r.R_m, n_blades=int(r.n_blades), rpm=rpm, root_cut=r.root_cut,
                   n_elem=30, section=section, blade=blade)
    return rot, T, v_tip


def sweep(d):
    pts = d[(d.CT > 1e-5) & (d.M_tip <= MACH_MAX)
            & ~d.note.str.contains("Extrapolated|Questionable", case=False)].copy()
    sections = {(a, lab): Section(a, **kw) for a in pts.airfoil.unique() for lab, kw in SWEEP_SECTIONS}
    out = []
    for i, r in pts.iterrows():
        row = r.to_dict()
        for lab, _ in SWEEP_SECTIONS:
            rot, T, v_tip = _rotor(r, sections[(r.airfoil, lab)])
            try:
                o, _ = rot.trim(T, lo=-6.0, hi=30.0)
            except (ValueError, RuntimeError) as e:
                row[f"CP_{lab}"] = np.nan
                row[f"err_{lab}"] = f"{type(e).__name__}"
                continue
            cp = o["P"] / (ru.RHO * np.pi * r.R_m ** 2 * v_tip ** 3)
            row[f"CP_{lab}"] = cp
            row[f"CP_err_{lab}"] = cp / r.CP - 1
            row[f"collective_{lab}"] = o["collective_deg"]
            row[f"profile_frac_{lab}"] = o["P_profile"] / o["P"]
            row[f"n_multiroot_{lab}"] = o["n_multiroot"]
        out.append(row)
    ru.NU = 1.46e-5
    return pd.DataFrame(out)


CT_SIGMA_MAX = 0.10     # above this the blades start to stall and the section
                        # polar, not the rotor model, decides the answer


def summarise(p):
    """Power error at the measured thrust, per experiment and loading band.
    Points past CT/sigma 0.10 are counted and left out of the averages."""
    rows = []
    groups = [(k, g) for k, g in p.groupby("experiment")] + [("all", p)]
    for k, g in groups:
        kept = g[g.CT_sigma <= CT_SIGMA_MAX]
        for band, gb in (("CT/sigma <= 0.10", kept),
                         ("CT/sigma < 0.04", kept[kept.CT_sigma < 0.04]),
                         ("0.04 to 0.10", kept[kept.CT_sigma >= 0.04])):
            row = dict(experiment=k, band=band, n=len(gb),
                       n_left_out=len(g) - len(kept) if band.startswith("CT/sigma <=") else 0,
                       Re_tip_min=int(gb.Re_tip.min()) if len(gb) else np.nan,
                       Re_tip_max=int(gb.Re_tip.max()) if len(gb) else np.nan)
            for lab, _ in SWEEP_SECTIONS:
                e = gb[f"CP_err_{lab}"].dropna()
                row[f"n_{lab}"] = len(e)
                row[f"mean_err_{lab}"] = e.mean()
                row[f"median_err_{lab}"] = e.median()
                row[f"mae_{lab}"] = e.abs().mean()
                row[f"profile_frac_{lab}"] = gb[f"profile_frac_{lab}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: how much could swirl explain
# ─────────────────────────────────────────────────────────────────────────────
SWIRL_CT_SIGMA_MIN = 0.02   # below this there is too little through-flow for the
                            # momentum estimate to mean anything; it diverges as vi -> 0
CONTRACTION = 1 / np.sqrt(2)  # ideal hover wake: half the disk area far downstream


def swirl(p):
    """Abhishek (IIT Kanpur, 19 October 2026): 2D drag and BEMT cannot give the
    whole profile torque, a lot of it is 3D effects and swirl losses. The 3D
    part needs a 3D solver. The swirl part can be sized from what is here.

    The solver has no swirl: it takes the in-plane velocity as omega r. The
    torque it does carry leaves angular momentum in the wake, dQ = dm r v_theta,
    with dm = 2 pi rho r dr vi F, the same mass flow its momentum balance uses.
    The kinetic energy of that swirl, sum 1/2 dm v_theta^2, is a first-order
    estimate of the power swirl would add. Three versions, each as a fraction of
    power: at the disk radius; in a wake contracted to r/sqrt(2), which keeps
    r v_theta and so doubles the energy; and that again with the torque raised
    to the measured power, since the solver's torque is low and the energy goes
    as its square. Free-transition sections, every swept point from CT/sigma
    0.02 to 0.10."""
    q = p[(p.CT_sigma >= SWIRL_CT_SIGMA_MIN) & (p.CT_sigma <= CT_SIGMA_MAX)
          & p.CP_free.notna()]
    sections = {a: Section(a) for a in q.airfoil.unique()}
    out = []
    for _, r in q.iterrows():
        rot, T, v_tip = _rotor(r, sections[r.airfoil])
        o, e = rot.trim(T, lo=-6.0, hi=30.0)
        F = np.array([rot._tip_loss(i, np.deg2rad(phi)) for i, phi in enumerate(e.phi_deg)])
        dm = 2 * np.pi * ru.RHO * e.r * e.dr * e.vi * F
        v_theta = e.dQ / (dm * e.r)
        frac = (0.5 * dm * v_theta ** 2).sum() / o["P"]
        out.append(dict(experiment=r.experiment, n_blades=r.n_blades, Re_tip=r.Re_tip,
                        CT_sigma=r.CT_sigma, CP_err_free=r.CP_err_free,
                        swirl_disk=frac, swirl_contracted=frac / CONTRACTION ** 2,
                        swirl_at_measured_power=frac / CONTRACTION ** 2 * r.CP / r.CP_free,
                        v_theta_max_over_v_tip=float((v_theta / v_tip).max())))
    ru.NU = 1.46e-5
    return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
def figure(z, p=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 2 if p is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.2 * ncol, 4.6))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    marks = {"Knight1937": "s", "Landgrebe1971": "^", "Ramasamy2015": "o", "Bhagwat2018": "D"}
    names = {"Knight1937": "Knight & Hefner 1937 (0015)", "Landgrebe1971": "Landgrebe 1971",
             "Ramasamy2015": "Ramasamy 2015", "Bhagwat2018": "Bhagwat & Ramasamy 2018"}
    zm = z[z.measured]
    for k, g in zm.groupby("experiment"):
        ax.scatter(g.Re_tip / 1e3, g.CP_sigma * 1e3, marker=marks[k], s=36, color="0.15",
                   facecolors="none" if k == "Knight1937" else "0.15", label=names[k], zorder=3)
    z12 = z[z.airfoil == "naca0012"].sort_values("Re_tip").drop_duplicates("Re_tip")
    for col, style in (("NF free, n_crit 9", dict(color="#1f5fa8", lw=2)),
                       ("XFoil free, n_crit 9", dict(color="#1f5fa8", lw=1, ls="--")),
                       ("NF free, n_crit 5", dict(color="#c9822b", lw=1.5)),
                       ("NF tripped at 5% chord", dict(color="#b23a3a", lw=2))):
        ax.plot(z12.Re_tip / 1e3, z12[col] * 1e3, label=f"NACA 0012, {col}", **style)
    ax.set_xlabel("tip Reynolds number (thousands)")
    ax.set_ylabel("$C_P/\\sigma$ at zero thrust  (x1000)")
    ax.set_title("Zero thrust: all of the power is section drag")
    ax.grid(alpha=0.25)
    k15 = z[(z.airfoil == "naca0015") & z.measured]
    if len(k15):
        ax.scatter(k15.Re_tip / 1e3, k15["NF free, n_crit 9"] * 1e3, marker="_", s=160,
                   color="#1f5fa8", label="NACA 0015 at 267,825, NF free, n_crit 9", zorder=4)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    if p is not None:
        ax = axes[1]
        q = p[p.CT_sigma <= CT_SIGMA_MAX]
        for lab, color in (("free", "#1f5fa8"), ("tripped", "#b23a3a")):
            for k, g in q.groupby("experiment"):
                ax.scatter(g.CT_sigma, g[f"CP_err_{lab}"] * 100, marker=marks[k], s=20,
                           color=color, alpha=0.75,
                           label=f"{names[k]}" if lab == "free" else None)
            m = q[f"CP_err_{lab}"].mean() * 100
            ax.axhline(m, color=color, lw=1.2, ls="--")
            ax.text(CT_SIGMA_MAX, m + (1.2 if lab == "tripped" else -4.5), f"{lab} transition, mean {m:+.0f}%", color=color,
                    ha="right", fontsize=8)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xlabel("$C_T/\\sigma$ (measured)")
        ax.set_ylabel("BEMT power at measured thrust, error (%)")
        ax.set_title(f"Whole sweep, {len(q)} points to $C_T/\\sigma$ {CT_SIGMA_MAX}")
        ax.legend(fontsize=7, frameon=False, loc="lower right", title="blue free, red tripped",
                  title_fontsize=7)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, "38_hover_lowre.png")
    fig.savefig(path, dpi=160)
    return path


def main(zero_only=False):
    d = parse()
    bad = check(d)
    print(f"Part 1. {len(d)} rows parsed: " +
          ", ".join(f"{k} {n}" for k, n in d.experiment.value_counts().sort_index().items()))
    if bad:
        raise SystemExit("parse check failed:\n  " + "\n  ".join(bad))
    print(f"  CT/sigma and CP/sigma consistent except the {len(KNOWN_MISPRINTS)} listed misprints; "
          "Knight & Hefner and Landgrebe match the Liege copy")
    d.to_csv(os.path.join(DATA, "hover_lowre_rotors.csv"), index=False, float_format="%.7g")

    z = zero_thrust(d)
    z.to_csv(os.path.join(DATA, "hover_zero_thrust.csv"), index=False, float_format="%.6g")
    cols = [c for c, _ in NF_CASES] + ["XFoil free, n_crit 9"]
    print("\nPart 2. Zero thrust, measured CP/sigma over predicted, measured points only")
    zm = z[z.measured]
    show = zm.groupby(["experiment", "Re_tip"]).agg(
        n=("CP_sigma", "size"), M_tip=("M_tip", "first"), measured=("CP_sigma", "mean"),
        **{c: (c, "mean") for c in cols}).reset_index()
    for _, r in show.iterrows():
        ratios = "  ".join(f"{r.measured / r[c]:.2f}" for c in cols)
        print(f"  {r.experiment:13s} tip Re {r.Re_tip:8,.0f}  M {r.M_tip:.2f}  n {r.n}  "
              f"meas {r.measured:.5f}  ratio [{ratios}]")
    print("  ratio columns: " + " | ".join(cols))

    p = None
    if not zero_only:
        p = sweep(d)
        p.to_csv(os.path.join(DATA, "hover_lowre_validation.csv"), index=False, float_format="%.6g")
        s = summarise(p)
        s.to_csv(os.path.join(DATA, "hover_lowre_summary.csv"), index=False, float_format="%.4g")
        print(f"\nPart 3. {len(p)} points trimmed to measured thrust, tip Mach <= {MACH_MAX}; "
              f"averages over CT/sigma <= {CT_SIGMA_MAX}")
        print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        w = swirl(p)
        w.to_csv(os.path.join(DATA, "hover_swirl.csv"), index=False, float_format="%.6g")
        print(f"\nPart 4. Swirl, as a fraction of power, {len(w)} points at CT/sigma "
              f"{SWIRL_CT_SIGMA_MIN} to {CT_SIGMA_MAX}, against the free-transition shortfall")
        bands = pd.cut(w.CT_sigma, [SWIRL_CT_SIGMA_MIN, 0.04, 0.07, CT_SIGMA_MAX + 1e-9], right=False)
        for (band, g) in list(w.groupby(bands, observed=True)) + [("all", w)]:
            print(f"  CT/sigma {str(band):16s} n {len(g):3d}  disk {g.swirl_disk.mean():.2%}  "
                  f"contracted {g.swirl_contracted.mean():.2%}  "
                  f"at measured power {g.swirl_at_measured_power.mean():.2%}  "
                  f"(max {g.swirl_at_measured_power.max():.2%})  "
                  f"shortfall {-g.CP_err_free.mean():.1%}")
    print("\nFigure:", figure(z, p))


if __name__ == "__main__":
    main(zero_only="--zero" in sys.argv)
