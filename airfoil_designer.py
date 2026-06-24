"""
airfoil_designer.py  --  config-driven low-Re airfoil design tool
=================================================================

One YAML mission spec -> a robust, multi-objective airfoil + data + figures
(+ optional true-XFoil validation). Wraps the whole research pipeline so a new
vehicle/mission is a config change, not a code change.

    python airfoil_designer.py configs/delivery_drone.yaml

Everything that was hard-coded across the research scripts is now a knob:
  * operating envelope (Re, AoA ranges and grid density)
  * objective weights: efficiency (worst-case L/D), safety (stall margin),
    structure (spar depth), noise (trailing-edge delta*)
  * manufacturing-tolerance robustness (on/off, error magnitude, ensemble size)
  * geometry constraints, NeuralFoil fidelity, multi-start seeds

See the docstrings of the research scripts for the physics behind each metric.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import yaml
import numpy as np
import pandas as pd
import aerosandbox as asb


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _f(x):
    """YAML sometimes parses 50e3 as a string; coerce numbers robustly."""
    return float(x)


def load_config(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    c.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    c.setdefault("model", "large")
    c.setdefault("seeds", ["naca4412", "naca6412"])
    env = c["envelope"]
    env["Re"] = [_f(v) for v in env["Re"]]
    env["AoA"] = [_f(v) for v in env["AoA"]]
    env.setdefault("n_Re", 5); env.setdefault("n_AoA", 5)
    dp = c.setdefault("design_point", {})
    dp.setdefault("Re", 200e3); dp.setdefault("AoA", 4.0)
    dp["Re"], dp["AoA"] = _f(dp["Re"]), _f(dp["AoA"])
    o = c.setdefault("objectives", {})
    for k, d in [("efficiency", 1.0), ("safety", 0.0), ("structure", 0.0), ("noise", 0.0)]:
        o[k] = _f(o.get(k, d))
    con = c.setdefault("constraints", {})
    con["thickness"] = [_f(v) for v in con.get("thickness", [0.08, 0.16])]
    con["TE_thickness"] = _f(con.get("TE_thickness", 0.0025))
    m = c.setdefault("manufacturing", {})
    m.setdefault("enabled", False)
    m["sigma_weight"] = _f(m.get("sigma_weight", 0.04))
    m["n_ensemble"] = int(m.get("n_ensemble", 6))
    c.setdefault("validate_xfoil", True)
    s = c.setdefault("safety_check", {})
    s["AoA"], s["Re"] = _f(s.get("AoA", 12.0)), _f(s.get("Re", env["Re"][0]))
    return c


# Reference scales make objective weights unit-free and comparable.
S_LD, S_SAFE, S_STRUCT, S_NOISE = 40.0, 1.4, 0.12, 0.014


def grids(c, fine=False):
    env = c["envelope"]
    nR = 11 if fine else env["n_Re"]
    nA = 9 if fine else env["n_AoA"]
    return (np.geomspace(env["Re"][0], env["Re"][1], nR),
            np.linspace(env["AoA"][0], env["AoA"][1], nA))


def te_deltastar(aero):
    return (aero["upper_bl_theta_31"] * aero["upper_bl_H_31"]
            + aero["lower_bl_theta_31"] * aero["lower_bl_H_31"])


def make_perturbations(c, seed):
    m = c["manufacturing"]
    if not m["enabled"]:
        return [dict(up=np.zeros(8), lo=np.zeros(8), le=0.0)]
    rng = np.random.default_rng(seed)
    sw = m["sigma_weight"]
    base = [dict(up=np.zeros(8), lo=np.zeros(8), le=0.0)]
    return base + [dict(up=rng.normal(0, sw, 8), lo=rng.normal(0, sw, 8),
                        le=rng.normal(0, sw * 0.5)) for _ in range(m["n_ensemble"])]


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def solve_one(c, seed, perts):
    af0 = asb.KulfanAirfoil(seed) if isinstance(seed, str) else seed
    opti = asb.Opti()
    uw = opti.variable(init_guess=af0.upper_weights, lower_bound=-0.2, upper_bound=0.5)
    lw = opti.variable(init_guess=af0.lower_weights, lower_bound=-0.5, upper_bound=0.4)
    le = opti.variable(init_guess=af0.leading_edge_weight, lower_bound=-0.2, upper_bound=0.2)
    te = c["constraints"]["TE_thickness"]
    nominal = asb.KulfanAirfoil(upper_weights=uw, lower_weights=lw,
                                leading_edge_weight=le, TE_thickness=te)

    tlo, thi = c["constraints"]["thickness"]
    opti.subject_to(nominal.local_thickness(x_over_c=np.linspace(0.02, 0.98, 20)) > 0.004)
    opti.subject_to(nominal.max_thickness() < thi)
    opti.subject_to(nominal.max_thickness() > tlo)

    re_opt, aoa_opt = grids(c)
    model = c["model"]
    o = c["objectives"]

    # efficiency: worst-case L/D over (manufacturing) x (Re, AoA)
    g = opti.variable(init_guess=20.0)
    for p in perts:
        af = asb.KulfanAirfoil(upper_weights=uw + p["up"], lower_weights=lw + p["lo"],
                               leading_edge_weight=le + p["le"], TE_thickness=te)
        for Re in re_opt:
            aero = af.get_aero_from_neuralfoil(alpha=aoa_opt, Re=Re, model_size=model)
            opti.subject_to(g <= aero["CL"] / aero["CD"])

    obj = o["efficiency"] * g / S_LD
    if o["safety"]:
        cl_hi = nominal.get_aero_from_neuralfoil(
            alpha=c["safety_check"]["AoA"], Re=c["safety_check"]["Re"], model_size=model)["CL"]
        obj = obj + o["safety"] * cl_hi / S_SAFE
    if o["structure"]:
        obj = obj + o["structure"] * nominal.max_thickness() / S_STRUCT
    if o["noise"]:
        cruise = nominal.get_aero_from_neuralfoil(
            alpha=c["design_point"]["AoA"], Re=c["design_point"]["Re"], model_size=model)
        obj = obj - o["noise"] * te_deltastar(cruise) / S_NOISE

    opti.maximize(obj)
    sol = opti.solve(verbose=False, max_iter=300)
    return sol(nominal), float(sol(obj))


def design(c):
    perts = make_perturbations(c, seed=1)
    best_af, best = None, -np.inf
    seeds = list(c["seeds"])
    for s in seeds:
        try:
            af, obj = solve_one(c, s, perts)
        except Exception as e:
            print(f"    seed {s}: solve failed ({type(e).__name__})"); continue
        if obj > best:
            best_af, best = af, obj
    if best_af is None:
        raise RuntimeError("all seeds failed")
    return best_af


# --------------------------------------------------------------------------- #
# Evaluation / validation
# --------------------------------------------------------------------------- #
def evaluate(af, c):
    re_f, aoa_f = grids(c, fine=True)
    model = c["model"]
    rows = []
    for Re in re_f:
        aero = af.get_aero_from_neuralfoil(alpha=aoa_f, Re=Re, model_size=model)
        cl, cd = np.atleast_1d(aero["CL"]), np.atleast_1d(aero["CD"])
        cf = np.atleast_1d(aero["analysis_confidence"])
        for k, a in enumerate(aoa_f):
            rows.append(dict(Re=float(Re), AoA=float(a), CL=float(cl[k]), CD=float(cd[k]),
                             L_over_D=float(cl[k] / cd[k]), analysis_confidence=float(cf[k])))
    return pd.DataFrame(rows)


def xfoil_polar(coords, Re, alphas, xfoil):
    af = asb.Airfoil(coordinates=coords).repanel(n_points_per_side=80)
    # Sweep internally in fine 0.5-deg steps from 0 up: each viscous solve
    # warm-starts from a nearby converged point, which is essential for thick /
    # high-camber low-Re sections. The caller matches its requested alphas
    # against the accumulated polar.
    fine = np.arange(0.0, float(max(alphas)) + 0.25, 0.5)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "af.dat"), "w") as f:
            f.write("design\n")
            for x, y in af.coordinates:
                f.write(f"{x:.6f} {y:.6f}\n")
        cmds = (["PLOP", "G F", "", "LOAD af.dat", "PANE", "OPER",
                 f"VISC {Re:.0f}", "ITER 300", "PACC", "polar.txt", ""]
                + [f"ALFA {a:.3f}" for a in fine] + ["PACC", "", "QUIT"])
        try:
            subprocess.run([xfoil], input="\n".join(cmds) + "\n",
                           capture_output=True, text=True, timeout=180, cwd=d)
        except Exception:
            return pd.DataFrame()
        pol = os.path.join(d, "polar.txt")
        if not os.path.exists(pol):
            return pd.DataFrame()
        rows = []
        for line in open(pol):
            p = line.split()
            if len(p) >= 5:
                try:
                    rows.append((float(p[0]), float(p[1]), float(p[2])))
                except ValueError:
                    continue
        return pd.DataFrame(rows, columns=["alpha", "CL", "CD"])


# --------------------------------------------------------------------------- #
def main(config_path):
    c = load_config(config_path)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", c["name"])
    os.makedirs(outdir, exist_ok=True)
    print(f"=== Designing '{c['name']}' ===")
    print(f"  envelope Re={c['envelope']['Re']}  AoA={c['envelope']['AoA']}  "
          f"objectives={c['objectives']}  mfg={c['manufacturing']['enabled']}")

    af = design(c)

    # geometry
    co = af.coordinates
    pd.DataFrame({"x": co[:, 0], "y": co[:, 1]}).to_csv(
        os.path.join(outdir, "coords.csv"), index=False)

    # evaluation
    df = evaluate(af, c)
    df.to_csv(os.path.join(outdir, "evaluation.csv"), index=False)
    a_sweep = np.linspace(c["envelope"]["AoA"][0], 16, 33)
    clmax = float(np.atleast_1d(af.get_aero_from_neuralfoil(
        alpha=a_sweep, Re=c["safety_check"]["Re"], model_size=c["model"])["CL"]).max())
    print(f"  worst-case L/D={df.L_over_D.min():.1f}  mean L/D={df.L_over_D.mean():.1f}  "
          f"peak L/D={df.L_over_D.max():.1f}  CLmax={clmax:.2f}  "
          f"max-thk={af.max_thickness()*100:.1f}%")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(co[:, 0], co[:, 1], color="#1f77b4", lw=2)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(f"{c['name']}  (max thk {af.max_thickness()*100:.1f}%)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "shape.png"), dpi=150); plt.close(fig)

    piv = df.pivot_table(index="AoA", columns="Re", values="L_over_D")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, len(piv.columns), 0, len(piv.index)])
    ax.set_xticks(np.arange(len(piv.columns)) + 0.5)
    ax.set_xticklabels([f"{x/1e3:.0f}k" for x in piv.columns], rotation=45, fontsize=8)
    ax.set_yticks(np.arange(len(piv.index)) + 0.5)
    ax.set_yticklabels([f"{y:.0f}" for y in piv.index], fontsize=8)
    ax.set_xlabel("Re"); ax.set_ylabel("AoA [deg]"); ax.set_title(f"{c['name']}: L/D envelope")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "LD_heatmap.png"), dpi=150); plt.close(fig)

    # optional XFoil validation at the design point Re
    xf = shutil.which("xfoil") or os.path.join(os.path.dirname(__file__), ".venv", "bin", "xfoil")
    if c["validate_xfoil"] and os.path.exists(xf):
        Re = c["design_point"]["Re"]
        # Always start the sweep at 0 deg: viscous XFoil cold-starts reliably
        # near zero lift and warm-steps up, even if the envelope starts higher.
        alphas = np.arange(0.0, c["envelope"]["AoA"][1] + 0.5, 1.0)
        pol = xfoil_polar(co, Re, alphas, xf)
        if pol.empty:
            print(f"  XFoil validation @Re={Re/1e3:.0f}k: no converged points (skipped)")
        if not pol.empty:
            nf = af.get_aero_from_neuralfoil(alpha=alphas, Re=Re, model_size=c["model"])
            nfld = np.atleast_1d(nf["CL"]) / np.atleast_1d(nf["CD"])
            merged = []
            for k, a in enumerate(alphas):
                m = pol[np.isclose(pol.alpha, a, atol=0.05)]
                merged.append(dict(alpha=float(a), NF_LD=float(nfld[k]),
                                   XFoil_LD=float(m.CL.iloc[0] / m.CD.iloc[0]) if len(m) else np.nan))
            vdf = pd.DataFrame(merged)
            vdf.to_csv(os.path.join(outdir, "xfoil_validation.csv"), index=False)
            gap = (vdf.NF_LD - vdf.XFoil_LD).abs().mean()
            print(f"  XFoil validation @Re={Re/1e3:.0f}k: converged {vdf.XFoil_LD.notna().sum()}/{len(vdf)}, "
                  f"mean |NF-XFoil| L/D = {gap:.1f}")
    print(f"  -> outputs in runs/{c['name']}/")
    return af


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python airfoil_designer.py <config.yaml>"); sys.exit(1)
    main(sys.argv[1])
