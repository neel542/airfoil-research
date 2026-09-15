"""
Validate the blade-element rotor model against measured propellers.

Bharath Govindarajan (IIT Madras): "I would focus on first validating your
tools before putting it inside the sizing loop." Section 3.5 pushes the
measured airfoil error through a blade-element momentum solver, and until
this script existed the solver itself had been checked against nothing.

The UIUC Propeller Database holds static thrust and power for propellers
that Deters designed and 3D-printed, so their airfoil section is known
exactly: SDA1075 over the whole blade, coordinates published in the 2014
paper. Two blade families, DA4002 (c/R 0.18) at four pitches and DA4022
(c/R 0.23) with two, three and four blades, each at 5 and 9 inch diameter.
Same laboratory as the two airfoil archives the benchmark is built on.

The test: build each propeller in the solver from its measured chord and
pitch distribution, look the section up in NeuralFoil exactly as the rotor
study does, run it at every measured RPM, and compare thrust and power
coefficients. Nothing is tuned to the data.

Two limits, both carried into the paper. These propellers reach Re 98,000
at 75 percent span; the rotor in section 3.5 sits at 118,000 to 475,000, so
this validates the solver below its operating point, not at it. And below
about Re 40,000 the comparison is dominated by NeuralFoil's section polar,
which stalls early there and which the benchmark does not cover, so the
clean window is Re 40,000 to 98,000 and the headline is reported inside it.

    python validate_bemt.py            NeuralFoil section, the paper's numbers
    python validate_bemt.py --xfoil    XFoil section on the same Kulfan geometry,
                                       to split solver error from section error

Inputs  : data/uiuc_propdb/
Outputs : data/bemt_validation.csv           every measured point, predicted and measured
          data/bemt_validation_summary.csv   per propeller and pooled, in the window and overall
          data/bemt_validation_blades.csv    the blade-count series at one matched RPM
          figures/36_bemt_validation.png
          (the --xfoil run writes the same CSVs with an _xfoil suffix and no figure)
"""

import glob
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import aerosandbox as asb
import rotor_uncertainty as ru

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
PROP = os.path.join(DATA, "uiuc_propdb")
FIG = os.path.join(OUT, "figures")

ROOT_CUT = 0.30        # the printed hub sits inboard of this; the blade starts here
N_ELEM = 24            # same as the rotor study
WINDOW_RE = 40e3       # below this NeuralFoil's SDA1075 polar stalls early and the
                       # benchmark has no measurement to say whether that is right
RPM_MATCH = 4900.0     # the blade-count comparison, an RPM every DA4022 sweep reaches

# grid for a propeller section: the root runs well past 20 degrees at zero
# airspeed and the 5 inch blades sit below Re 30,000
ALPHA_RANGE, N_ALPHA = (-10.0, 32.0), 197          # 0.214 degree spacing, as the rotor study
RE_RANGE, N_RE = (6e3, 300e3), 45


# ─────────────────────────────────────────────────────────────────────────────
# The propellers
# ─────────────────────────────────────────────────────────────────────────────
STATIC = re.compile(r"^(da40\d\d)_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:_(\d)b)?_static_\w+\.txt$")


def load_cases():
    """One case per static file: geometry, measurements, and what it is."""
    cases = []
    for path in sorted(glob.glob(os.path.join(PROP, "*_static_*.txt"))):
        m = STATIC.match(os.path.basename(path))
        if not m:
            continue
        fam, D, P, nb = m.group(1), m.group(2), m.group(3), int(m.group(4) or 2)
        geom = pd.read_csv(os.path.join(PROP, f"{fam}_{D}x{P}_geom.txt"), sep=r"\s+")
        static = pd.read_csv(path, sep=r"\s+")
        cases.append(dict(label=f"{fam.upper()} {D}x{P} {nb}b", family=fam.upper(),
                          D_in=float(D), pitch_in=float(P), blades=nb, geom=geom,
                          static=static, file=os.path.basename(path), drawn=False))
    # the DA4002 9x6.75 once more on its drawing rather than the built article,
    # to show how much of the error is geometry
    base = next(c for c in cases if c["file"].startswith("da4002_9x6.75"))
    drawn = dict(base, label=base["label"] + " (drawn)", drawn=True,
                 geom=pd.read_csv(os.path.join(PROP, "da4002_geom.txt"), sep=r"\s+"))
    cases.append(drawn)
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# The section, two ways
# ─────────────────────────────────────────────────────────────────────────────
def sda1075():
    return np.loadtxt(os.path.join(PROP, "sda1075.dat"), skiprows=1)


def kulfan_rms(af):
    """RMS distance, in chords, between the coordinates and their 17-number
    Kulfan fit, the same gate the benchmark applies to its own airfoils."""
    kf = af.to_kulfan_airfoil()
    raw, fit = af.coordinates, kf.coordinates
    i_raw, i_fit = np.argmin(raw[:, 0]), np.argmin(fit[:, 0])
    err = []
    for r, f in [(raw[:i_raw + 1], fit[:i_fit + 1]), (raw[i_raw:], fit[i_fit:])]:
        order = np.argsort(f[:, 0])
        err.append(r[:, 1] - np.interp(r[:, 0], f[order, 0], f[order, 1]))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def neuralfoil_section():
    print("Building the SDA1075 table from NeuralFoil ...", flush=True)
    return ru.Section(name="SDA1075", coordinates=sda1075(), n_alpha=N_ALPHA, n_Re=N_RE,
                      alpha_range=ALPHA_RANGE, Re_range=RE_RANGE)


def xfoil_section():
    """The same grid filled by XFoil on the Kulfan geometry, which is what
    NeuralFoil was trained to reproduce. XFoil will not converge everywhere
    on a 12 percent section at these Reynolds numbers, so each Re row is
    filled along alpha from its converged points, the edges are held, and a
    row with too few converged points is dropped. Coverage is reported."""
    from multiprocessing import Pool
    import xfoil_decomposition as xd

    print("Building the SDA1075 table from XFoil ...", flush=True)
    sec = ru.Section.__new__(ru.Section)
    sec.name = "SDA1075 (XFoil)"
    sec.af = asb.Airfoil(name="SDA1075", coordinates=sda1075())
    sec.camber_pct = float(sec.af.max_camber()) * 100
    sec.thickness_pct = float(sec.af.max_thickness()) * 100
    # 0.5 degree spacing, which is the step run_polar warm-starts along anyway,
    # so asking for anything finer just doubles the solve count for nothing
    alphas = np.linspace(-10.0, 24.0, 69)             # XFoil has no business past 24
    Res = np.geomspace(15e3, 300e3, 36)               # nor below 15,000 on this section
    coords = xd.panel_coords(sec.af.to_kulfan_airfoil())
    jobs = [(f"Re{int(Re)}", coords, float(Re), alphas) for Re in Res]
    with Pool(xd.N_PROC) as pool:
        results = dict(pool.imap_unordered(xd.run_polar, jobs))
    CL, CD, keep, n_conv = [], [], [], 0
    for Re in Res:
        pol = results[f"Re{int(Re)}"]
        a = np.array(sorted(pol)); n_conv += len(a)
        if len(a) < 0.4 * len(alphas):
            continue
        cl = np.array([pol[k][0] for k in a]); cd = np.array([pol[k][1] for k in a])
        CL.append(np.interp(alphas, a, cl)); CD.append(np.interp(alphas, a, cd))
        keep.append(Re)
    CL, CD, keep = np.array(CL).T, np.array(CD).T, np.array(keep)
    sec.alphas, sec.Res = alphas, keep
    sec.coverage = n_conv / (len(alphas) * len(Res))
    kw = dict(bounds_error=False, fill_value=None)
    sec._CL = RegularGridInterpolator((alphas, keep), CL, **kw)
    sec._CD = RegularGridInterpolator((alphas, keep), np.maximum(CD, 1e-4), **kw)
    sec._conf = RegularGridInterpolator((alphas, keep), np.ones_like(CL), **kw)
    print(f"  XFoil converged at {100 * sec.coverage:.0f}% of the grid; "
          f"{len(keep)} of {len(Res)} Reynolds rows kept "
          f"({keep.min():.0f} to {keep.max():.0f})")
    return sec


# ─────────────────────────────────────────────────────────────────────────────
# Run every measured point through the solver
# ─────────────────────────────────────────────────────────────────────────────
def run_case(case, sec):
    D = case["D_in"] * 0.0254
    rows = []
    for _, m in case["static"].iterrows():
        rot = ru.Rotor(R=D / 2, n_blades=case["blades"], rpm=float(m.RPM), root_cut=ROOT_CUT,
                       n_elem=N_ELEM, section=sec, blade=case["geom"])
        out, d = rot.solve(0.0)
        n = m.RPM / 60
        CT, CP = out["T"] / (ru.RHO * n ** 2 * D ** 4), out["P"] / (ru.RHO * n ** 3 * D ** 5)
        rows.append(dict(label=case["label"], family=case["family"], D_in=case["D_in"],
                         pitch_in=case["pitch_in"], blades=case["blades"], drawn=case["drawn"],
                         rpm=float(m.RPM), Re_75=out["Re_75"], Re_tip=d.Re.iloc[-2],
                         CT_meas=float(m.CT), CP_meas=float(m.CP), CT_pred=CT, CP_pred=CP,
                         err_CT=CT / m.CT - 1, err_CP=CP / m.CP - 1,
                         induced_frac=out["induced_frac"], FM=out["FM"],
                         alpha_root=float(d.alpha_deg.iloc[0]),
                         alpha_75=float(np.interp(0.75, d.x, d.alpha_deg)),
                         n_fallback=int(out["n_fallback"]),
                         in_window=bool(out["Re_75"] >= WINDOW_RE)))
    return rows


def summarise(pts):
    """Mean signed and absolute error, per propeller and pooled, inside the
    clean Reynolds window and over everything."""
    def block(d, label):
        w = d[d.in_window]
        r = dict(label=label, n=len(d), n_window=len(w),
                 Re75_min=d.Re_75.min(), Re75_max=d.Re_75.max())
        for tag, sub in [("window", w), ("all", d)]:
            for c in ["err_CT", "err_CP"]:
                r[f"{c}_mean_{tag}"] = sub[c].mean() if len(sub) else np.nan
                r[f"{c}_abs_{tag}"] = sub[c].abs().mean() if len(sub) else np.nan
        r["n_fallback"] = int(d.n_fallback.sum())
        return r
    rows = [block(g, lab) for lab, g in pts.groupby("label", sort=False)]
    real = pts[~pts.drawn]
    rows.append(block(real[real.D_in == 9], "all 9 in"))
    rows.append(block(real[real.D_in == 5], "all 5 in"))
    rows.append(block(real, "all"))
    return pd.DataFrame(rows)


def blade_series(pts):
    """DA4022 9x6.75 with 2, 3 and 4 blades at one RPM: the section and the
    pitch are identical across the three, so what changes is the inflow and
    tip-loss model, and this is the one comparison the section error cannot
    contaminate."""
    rows = []
    for nb in [2, 3, 4]:
        d = pts[(pts.family == "DA4022") & (pts.D_in == 9) & (pts.blades == nb)].sort_values("rpm")
        if d.rpm.min() > RPM_MATCH or d.rpm.max() < RPM_MATCH:
            continue
        r = dict(blades=nb, rpm=RPM_MATCH, Re_75=float(np.interp(RPM_MATCH, d.rpm, d.Re_75)))
        for c in ["CT_meas", "CT_pred", "CP_meas", "CP_pred"]:
            r[c] = float(np.interp(RPM_MATCH, d.rpm, d[c]))
        rows.append(r)
    b = pd.DataFrame(rows).set_index("blades")
    for c in ["CT", "CP"]:
        for src in ["meas", "pred"]:
            b[f"{c}_{src}_ratio"] = b[f"{c}_{src}"] / b[f"{c}_{src}"].shift(1)
    return b.reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# Figure 36
# ─────────────────────────────────────────────────────────────────────────────
def figure(pts):
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
        "legend.fontsize": 8.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "figure.facecolor": "white", "savefig.facecolor": "white", "axes.axisbelow": True})

    def clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    blues = ["#6baed6", "#3182bd", "#08519c", "#08306b"]
    reds = ["#fc9272", "#de2d26", "#a50f15"]
    real = pts[~pts.drawn]
    nine = real[real.D_in == 9]
    colour = {}
    for i, lab in enumerate(sorted(nine[nine.family == "DA4002"].label.unique())):
        colour[lab] = blues[i % 4]
    for i, lab in enumerate(sorted(nine[nine.family == "DA4022"].label.unique())):
        colour[lab] = reds[i % 3]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, c, title in [(axes[0], "CT", "Thrust coefficient"), (axes[1], "CP", "Power coefficient")]:
        for lab, g in real[real.D_in == 5].groupby("label"):
            g = g.sort_values("Re_75")
            ax.plot(g.Re_75 / 1e3, g[f"{c}_meas"], "o", color="#bbbbbb", ms=3.5, mew=0)
            ax.plot(g.Re_75 / 1e3, g[f"{c}_pred"], "-", color="#bbbbbb", lw=1.2)
        for lab, g in nine.groupby("label"):
            g = g.sort_values("Re_75")
            ax.plot(g.Re_75 / 1e3, g[f"{c}_meas"], "o", color=colour[lab], ms=4, mew=0, label=lab)
            ax.plot(g.Re_75 / 1e3, g[f"{c}_pred"], "-", color=colour[lab], lw=1.8)
        ax.axvspan(WINDOW_RE / 1e3, 100, color="#1a9850", alpha=0.07)
        ax.axvline(WINDOW_RE / 1e3, color="#1a9850", lw=1, ls="--")
        ax.set_xlabel("Reynolds number at 75% span (thousands)")
        ax.set_ylabel(f"$C_{{{c[1]}}}$, propeller convention")
        ax.set_title(f"{title}: measured (points), predicted (lines)")
        ax.set_xlim(0, 100); clean(ax)
    axes[0].plot([], [], "o-", color="#bbbbbb", label="5 in, below the window")
    axes[0].legend(loc="lower right", ncol=2)

    ax = axes[2]
    for lab, g in nine.groupby("label"):
        g = g.sort_values("Re_75")
        ax.plot(g.Re_75 / 1e3, 100 * g.err_CT, "-", color=colour[lab], lw=1.6)
        ax.plot(g.Re_75 / 1e3, 100 * g.err_CP, ":", color=colour[lab], lw=1.6)
    ax.plot([], [], "-", color="#444444", label="thrust")
    ax.plot([], [], ":", color="#444444", label="power")
    ax.axhline(0, color="#444444", lw=1)
    ax.axvspan(WINDOW_RE / 1e3, 100, color="#1a9850", alpha=0.07)
    ax.axvline(WINDOW_RE / 1e3, color="#1a9850", lw=1, ls="--")
    w = nine[nine.in_window]
    ax.annotate(f"inside the window, 9 in:\nthrust {100 * w.err_CT.abs().mean():.0f}% mean abs error, "
                f"bias {100 * w.err_CT.mean():+.0f}%\npower {100 * w.err_CP.abs().mean():.0f}% mean abs error, "
                f"bias {100 * w.err_CP.mean():+.0f}%",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=9, color="#555555")
    ax.set_xlabel("Reynolds number at 75% span (thousands)")
    ax.set_ylabel("predicted / measured, minus one (%)")
    ax.set_title("The error, 9 in propellers")
    ax.set_xlim(0, 100); ax.set_ylim(-45, 30); ax.legend(loc="lower right"); clean(ax)

    fig.suptitle("The rotor solver against measured propellers with a known section (UIUC, SDA1075)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG, "36_bemt_validation.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {os.path.relpath(path, OUT)}")


# ─────────────────────────────────────────────────────────────────────────────
def main(xfoil=False):
    cases = load_cases()
    af = asb.Airfoil(name="SDA1075", coordinates=sda1075())
    print(f"{len(cases)} cases, {sum(len(c['static']) for c in cases)} measured points. "
          f"SDA1075: {100 * float(af.max_camber()):.2f}% camber, "
          f"{100 * float(af.max_thickness()):.1f}% thick, Kulfan fit RMS {100 * kulfan_rms(af):.3f}% chord")
    sec = xfoil_section() if xfoil else neuralfoil_section()

    pts = pd.DataFrame(sum((run_case(c, sec) for c in cases), []))
    pts["section"] = "xfoil" if xfoil else "neuralfoil"
    sfx = "_xfoil" if xfoil else ""
    pts.to_csv(os.path.join(DATA, f"bemt_validation{sfx}.csv"), index=False)

    print(f"\nEvery case, at four RPMs across its sweep ({'XFoil' if xfoil else 'NeuralFoil'} section):")
    print(f"{'propeller':30} {'RPM':>6} {'Re75':>7} {'CT meas':>8} {'CT pred':>8} {'err%':>6} "
          f"{'CP meas':>8} {'CP pred':>8} {'err%':>6} {'ind%':>5} {'a_root':>6}")
    for lab, g in pts.groupby("label", sort=False):
        g = g.sort_values("rpm")
        for _, r in g.iloc[[1, len(g) // 3, 2 * len(g) // 3, -1]].iterrows():
            print(f"{lab:30} {r.rpm:6.0f} {r.Re_75:7.0f} {r.CT_meas:8.4f} {r.CT_pred:8.4f} "
                  f"{100 * r.err_CT:6.1f} {r.CP_meas:8.4f} {r.CP_pred:8.4f} {100 * r.err_CP:6.1f} "
                  f"{100 * r.induced_frac:5.1f} {r.alpha_root:6.1f}")

    s = summarise(pts)
    s.to_csv(os.path.join(DATA, f"bemt_validation_summary{sfx}.csv"), index=False)
    print(f"\nSummary. 'window' is Re at 75% span >= {WINDOW_RE:,.0f}; errors are predicted over "
          f"measured minus one, in percent.\n")
    show = s.copy()
    for c in [c for c in show.columns if c.startswith("err_")]:
        show[c] = (100 * show[c]).round(1)
    print(show[["label", "n", "n_window", "Re75_max", "err_CT_mean_window", "err_CT_abs_window",
                "err_CP_mean_window", "err_CP_abs_window", "err_CT_mean_all", "err_CP_mean_all",
                "n_fallback"]].round(0).to_string(index=False))

    b = blade_series(pts)
    b.to_csv(os.path.join(DATA, f"bemt_validation_blades{sfx}.csv"), index=False)
    print(f"\nDA4022 9x6.75 at {RPM_MATCH:.0f} RPM, two to four blades (same section, same pitch):\n")
    print(b.round(4).to_string(index=False))

    assert int(pts.n_fallback.sum()) == 0, "an element fell back to zero inflow"
    if not xfoil:
        figure(pts)
    print("\nDone.")


if __name__ == "__main__":
    main(xfoil="--xfoil" in sys.argv)
