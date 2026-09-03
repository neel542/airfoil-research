"""
A fitted, cross-validated error model for NeuralFoil's drag prediction.

The paper's confidence table gives the expected drag error in six confidence
bins. This script replaces the table with a small fitted model that a designer
can evaluate for any prediction: expected |dCD/CD| as a function of NeuralFoil's
confidence score, Reynolds number, angle of attack, and the airfoil's camber
and thickness. The model is a Gamma generalized linear model with a log link
(positive, heavy-tailed target; multiplicative coefficients), fitted by
iteratively reweighted least squares in plain numpy so it has no dependency
beyond the project's requirements.

Every number reported is out of sample:
  * 10-fold cross-validation with whole airfoils held out (an airfoil measured
    in both tunnels is held out from both at once), and
  * train on one tunnel, test on the other, both directions.
Baselines: a constant (the overall mean error) and the six-bin confidence
table from the paper, both fitted on the same training folds.

Inputs  : data/uiuc_neuralfoil_validation.csv, data/uiuc_kulfan_fit.csv,
          data/soartech8_neuralfoil_validation.csv, data/soartech8_kulfan_fit.csv
Outputs : data/error_model_fit.json          coefficients, transforms, CV metrics, worked example
          data/error_model_cv.csv            per-fold / per-direction metrics for every model
          data/error_model_calibration.csv   predicted vs actual error by decile of prediction
          data/error_model.json              (updated) points to the fitted model
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(OUT, "data")
CONF_BINS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0000001]
Y_FLOOR = 1e-4
rng = np.random.default_rng(0)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load():
    u = pd.read_csv(os.path.join(DATA, "uiuc_neuralfoil_validation.csv"))
    u = u[(u.config == "clean") & (u.NF_mode == "free") & (u.model == "large") & u.fit_ok].copy()
    uf = pd.read_csv(os.path.join(DATA, "uiuc_kulfan_fit.csv")).set_index("asb_name")
    u["camber"] = u.asb_name.map(uf.max_camber)
    u["thickness"] = u.asb_name.map(uf.max_thickness)
    u["tunnel"] = "UIUC"
    u["airfoil"] = u.asb_name

    s = pd.read_csv(os.path.join(DATA, "soartech8_neuralfoil_validation.csv"))
    s = s[(s.config == "clean") & (s.NF_mode == "free") & (s.nf == "large") & s.primary & s.fit_ok].copy()
    sf = pd.read_csv(os.path.join(DATA, "soartech8_kulfan_fit.csv")).set_index("model")
    cam = np.where(sf.primary_source == "measured", sf.meas_camber, sf.design_camber)
    thk = np.where(sf.primary_source == "measured", sf.meas_thickness, sf.design_thickness)
    s["camber"] = s.model.map(pd.Series(cam, index=sf.index))
    s["thickness"] = s.model.map(pd.Series(thk, index=sf.index))
    s["tunnel"] = "Princeton"
    # cluster key: the AeroSandbox name where the section exists in both tunnels,
    # otherwise the Princeton family name, so a shared airfoil is one cluster
    s["airfoil"] = s.asb_name.where(s.asb_name.notna() & (s.asb_name != ""), s.family)

    cols = ["tunnel", "airfoil", "Re", "alpha", "WT_CL", "WT_CD", "NF_CD", "NF_conf", "err_CD", "camber", "thickness"]
    d = pd.concat([u[cols], s[cols]], ignore_index=True)
    d = d[(d.WT_CL.abs() > 0.1) & (d.WT_CD > 0)].reset_index(drop=True)   # attached flow, as in the paper's table
    d["y"] = np.maximum(d.err_CD.abs(), Y_FLOOR)
    return d


# Feature transforms. Written out here and in the JSON so anyone can re-implement.
def features(d, spec):
    cols = {"one": np.ones(len(d))}
    cols["log_unconf"] = np.log10(1.001 - np.clip(d.NF_conf.to_numpy(float), 0, 1))
    cols["log_Re"] = np.log10(d.Re.to_numpy(float) / 1e5)
    cols["alpha"] = d.alpha.to_numpy(float)
    cols["alpha_sq"] = d.alpha.to_numpy(float) ** 2
    cols["camber"] = d.camber.to_numpy(float) * 100        # % chord
    cols["thickness"] = d.thickness.to_numpy(float) * 100  # % chord
    return np.column_stack([cols[c] for c in spec]), list(spec)


SPECS = {
    "constant": ["one"],
    "confidence_only": ["one", "log_unconf"],
    "confidence_Re": ["one", "log_unconf", "log_Re"],
    "full": ["one", "log_unconf", "log_Re", "alpha", "alpha_sq", "camber", "thickness"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Gamma GLM with log link, IRLS
# ─────────────────────────────────────────────────────────────────────────────
def fit_gamma(X, y, n_iter=50, tol=1e-8):
    beta = np.zeros(X.shape[1]); beta[0] = np.log(y.mean())
    for _ in range(n_iter):
        eta = np.clip(X @ beta, -12, 6)     # keep exp() finite while IRLS settles
        mu = np.exp(eta)
        z = eta + (y - mu) / mu                      # working response; weights are 1 for Gamma/log
        beta_new, *_ = np.linalg.lstsq(X, z, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    mu = predict(X, beta)
    dev = 2 * np.sum((y - mu) / mu - np.log(y / mu))
    phi = np.sum(((y - mu) / mu) ** 2) / max(len(y) - X.shape[1], 1)   # Pearson dispersion
    return beta, phi, dev


def predict(X, beta):
    return np.exp(np.clip(X @ beta, -12, 6))


def gamma_quantile(mu, phi, q):
    """q-quantile of a Gamma with mean mu and dispersion phi (shape 1/phi)."""
    return stats.gamma.ppf(q, a=1 / phi, scale=mu * phi)


def bin_table_predict(train, test):
    t = pd.cut(train.NF_conf, CONF_BINS)
    means = train.groupby(t, observed=False).y.mean()
    means = means.fillna(train.y.mean())
    return pd.cut(test.NF_conf, CONF_BINS).map(means).astype(float).fillna(train.y.mean()).to_numpy()


def metrics(y, pred, phi=None):
    m = dict(mae=np.mean(np.abs(y - pred)),
             mean_deviance=np.mean(2 * ((y - pred) / pred - np.log(y / pred))),
             spearman=stats.spearmanr(pred, y).correlation,
             mean_actual=y.mean(), mean_pred=pred.mean())
    if phi is not None:
        m["coverage_80"] = np.mean(y <= gamma_quantile(pred, phi, 0.8))
        m["coverage_95"] = np.mean(y <= gamma_quantile(pred, phi, 0.95))
    return m


def evaluate(train, test, spec):
    Xtr, _ = features(train, SPECS[spec]); Xte, _ = features(test, SPECS[spec])
    beta, phi, _ = fit_gamma(Xtr, train.y.to_numpy())
    return metrics(test.y.to_numpy(), predict(Xte, beta), phi), beta, phi


# ─────────────────────────────────────────────────────────────────────────────
def main():
    d = load()
    print(f"{len(d)} attached-flow clean points, {d.airfoil.nunique()} distinct airfoils, "
          f"{(d.tunnel == 'UIUC').sum()} UIUC + {(d.tunnel == 'Princeton').sum()} Princeton")
    rows = []

    # (a) 10-fold grouped CV by airfoil on the pooled data
    airfoils = np.array(sorted(d.airfoil.unique()))
    folds = rng.permutation(len(airfoils)) % 10
    fold_of = dict(zip(airfoils, folds))
    d["fold"] = d.airfoil.map(fold_of)
    for k in range(10):
        tr, te = d[d.fold != k], d[d.fold == k]
        for spec in SPECS:
            m, _, _ = evaluate(tr, te, spec)
            rows.append(dict(scheme="airfoil_10fold", fold=k, model=spec, n_test=len(te), **m))
        rows.append(dict(scheme="airfoil_10fold", fold=k, model="six_bin_table", n_test=len(te),
                         **metrics(te.y.to_numpy(), bin_table_predict(tr, te))))

    # (b) cross-tunnel transfer
    for a, b in [("UIUC", "Princeton"), ("Princeton", "UIUC")]:
        tr, te = d[d.tunnel == a], d[d.tunnel == b]
        for spec in SPECS:
            m, _, _ = evaluate(tr, te, spec)
            rows.append(dict(scheme=f"train_{a}_test_{b}", fold=0, model=spec, n_test=len(te), **m))
        rows.append(dict(scheme=f"train_{a}_test_{b}", fold=0, model="six_bin_table", n_test=len(te),
                         **metrics(te.y.to_numpy(), bin_table_predict(tr, te))))

    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(DATA, "error_model_cv.csv"), index=False)
    summary = (cv.groupby(["scheme", "model"])[["mae", "mean_deviance", "spearman", "coverage_80", "coverage_95"]]
               .mean().reset_index())
    pd.set_option("display.width", 200)
    print("\nOut-of-sample performance (mean over folds):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # (c) final fit on everything, for the JSON
    X, names = features(d, SPECS["full"])
    beta, phi, dev = fit_gamma(X, d.y.to_numpy())
    pred = predict(X, beta)
    print("\nFull model, fitted on all data (multiplicative effect per unit of each feature):")
    for n, b in zip(names, beta):
        print(f"  {n:>12}: beta = {b:+.4f}   x{np.exp(b):.3f} per unit")
    print(f"  dispersion phi = {phi:.3f}  (Gamma shape {1 / phi:.2f})")

    # calibration by decile of predicted error (out of sample: use the CV predictions)
    oos = np.full(len(d), np.nan)
    for k in range(10):
        tr, te = d[d.fold != k], d[d.fold == k]
        Xtr, _ = features(tr, SPECS["full"]); Xte, _ = features(te, SPECS["full"])
        b_k, _, _ = fit_gamma(Xtr, tr.y.to_numpy())
        oos[te.index] = predict(Xte, b_k)
    d["pred_oos"] = oos
    d["decile"] = pd.qcut(d.pred_oos, 10, labels=False)
    cal = d.groupby("decile").agg(n=("y", "size"), mean_pred=("pred_oos", "mean"), mean_actual=("y", "mean"),
                                  median_actual=("y", "median"), mean_conf=("NF_conf", "mean"),
                                  mean_logRe=("Re", lambda r: np.log10(r).mean())).reset_index()
    cal.to_csv(os.path.join(DATA, "error_model_calibration.csv"), index=False)
    print("\nCalibration (held-out predictions, by decile of predicted error):")
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # worked examples
    ex = pd.DataFrame(dict(NF_conf=[0.98, 0.98, 0.90, 0.50, 0.16], Re=[200e3, 60e3, 100e3, 100e3, 200e3],
                           alpha=[4, 4, 4, 4, 4], camber=[0.03, 0.03, 0.03, 0.03, 0.03],
                           thickness=[0.10, 0.10, 0.10, 0.10, 0.10]))
    Xe, _ = features(ex, SPECS["full"])
    ex["expected_abs_errCD"] = predict(Xe, beta)
    ex["p80_abs_errCD"] = gamma_quantile(ex.expected_abs_errCD, phi, 0.8)
    print("\nWorked examples (3% camber, 10% thick, alpha 4 deg):")
    print(ex.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    full_cv = summary[(summary.scheme == "airfoil_10fold")].set_index("model")
    xt = summary[summary.scheme.str.startswith("train_")].set_index(["scheme", "model"])
    fit_json = dict(
        target="expected |dCD/CD| of NeuralFoil large, free transition, n_crit 9, clean airfoil, attached flow (|CL| > 0.1)",
        model="Gamma GLM, log link: E[y] = exp(sum_i beta_i * x_i); 80th percentile = Gamma(shape 1/phi, scale mu*phi).ppf(0.8)",
        features={
            "one": "1",
            "log_unconf": "log10(1.001 - analysis_confidence)",
            "log_Re": "log10(Re / 1e5)",
            "alpha": "angle of attack, deg",
            "alpha_sq": "alpha^2",
            "camber": "max camber, % chord",
            "thickness": "max thickness, % chord",
        },
        beta={n: float(b) for n, b in zip(names, beta)},
        dispersion_phi=float(phi),
        n_points=int(len(d)), n_airfoils=int(d.airfoil.nunique()),
        cross_validation={
            "scheme": "10-fold with whole airfoils held out, pooled UIUC + Princeton",
            "mae": {m: float(full_cv.loc[m, "mae"]) for m in full_cv.index},
            "spearman": {m: float(full_cv.loc[m, "spearman"]) for m in full_cv.index},
            "coverage_80_full": float(full_cv.loc["full", "coverage_80"]),
            "coverage_95_full": float(full_cv.loc["full", "coverage_95"]),
        },
        cross_tunnel={
            f"{s}:{m}": {"mae": float(xt.loc[(s, m), "mae"]), "spearman": float(xt.loc[(s, m), "spearman"])}
            for s, m in xt.index
        },
        worked_examples=ex.round(4).to_dict(orient="records"),
        source="fit_error_model.py",
    )
    with open(os.path.join(DATA, "error_model_fit.json"), "w") as f:
        json.dump(fit_json, f, indent=2)

    em_path = os.path.join(DATA, "error_model.json")
    em = json.load(open(em_path))
    em["fitted_error_model"] = "data/error_model_fit.json"
    em["fitted_error_model_note"] = (
        f"Gamma GLM of expected |dCD/CD| on confidence, Re, alpha, camber, thickness; "
        f"held-out-airfoil MAE {full_cv.loc['full', 'mae']:.3f} vs {full_cv.loc['constant', 'mae']:.3f} for a constant "
        f"and {full_cv.loc['six_bin_table', 'mae']:.3f} for the six-bin confidence table; "
        f"80th-percentile band covers {100 * full_cv.loc['full', 'coverage_80']:.0f}% of held-out points.")
    with open(em_path, "w") as f:
        json.dump(em, f, indent=2)
    print("\nwrote data/error_model_fit.json, data/error_model_cv.csv, data/error_model_calibration.csv; "
          "updated data/error_model.json")


if __name__ == "__main__":
    main()
