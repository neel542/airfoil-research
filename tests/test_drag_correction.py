"""The drag-correction result recorded in plan/improve-neuralfoil.md must reproduce."""
import os
import sys
import warnings

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmark"))
import testbench as tb
import drag_correction as dc


def test_correction_A_matches_record():
    warnings.simplefilter("ignore")
    res = tb.run({"correction_A": dc.candidate(dc.FEATURES["A"])}).set_index("split")
    saved = pd.read_csv(os.path.join(ROOT, "data", "drag_correction_bench.csv"))
    saved = saved[saved.candidate == "correction_A"].set_index("split")
    for split in saved.index:
        assert abs(res.loc[split, "mean_abs_errCD"] - saved.loc[split, "mean_abs_errCD"]) < 1e-6
    for split in ["UIUC->Princeton", "Princeton->UIUC"]:
        assert res.loc[split, "vs_nf_ci_hi"] < 0          # better than NeuralFoil, whole interval


def test_gated_correction_matches_record():
    """Entry 2 of the plan: correction A at Re <= 150,000 only. It was a recorded failure."""
    warnings.simplefilter("ignore")
    res = tb.run({"g": dc.candidate(dc.FEATURES["A"], dc.RE_GATE)}).set_index("split")
    saved = pd.read_csv(os.path.join(ROOT, "data", "drag_correction_gated_bench.csv")).set_index("split")
    for split in saved.index:
        assert abs(res.loc[split, "mean_abs_errCD"] - saved.loc[split, "mean_abs_errCD"]) < 1e-6


def test_residual_scatter_matches_record():
    """Plan entry 3: what the error is made of after correction A."""
    import residual_scatter as rs
    rs.main.__globals__["print"] = lambda *a, **k: None
    rs.main()
    t = pd.read_csv(os.path.join(ROOT, "data", "residual_scatter.csv")).set_index("item")
    assert abs(t.loc["within_polar", "value"] - 0.78) < 0.01
    assert abs(t.loc["between_airfoil", "value"] - 0.083) < 0.005
    assert abs(t.loc["thickness", "value"] - 0.413) < 0.005


def test_boundary_layer_state_matches_record():
    """Plan entry 4: F and G reproduce, and the features cover every point."""
    import bl_state_experiment as bl
    bl_state = pd.read_csv(os.path.join(ROOT, "data", "nf_bl_state.csv"))
    assert len(bl_state) == 9130 and bl_state.notna().all().all()
    tb.load = bl.load
    res = tb.run({"G": bl.cand(bl.G_FEATS)}).set_index("split")
    saved = pd.read_csv(os.path.join(ROOT, "data", "bl_state_bench.csv"))
    saved = saved[saved.candidate == "correction_G"].set_index("split")
    for split in saved.index:
        assert abs(res.loc[split, "mean_abs_errCD"] - saved.loc[split, "mean_abs_errCD"]) < 1e-6


def test_error_bar_experiment_matches_record():
    """Plan entry 5: the error-bar models reproduce, and the recorded success rule stays failed."""
    import numpy as np
    import bl_error_bar_experiment as eb
    d = eb.load()
    assert len(d) == 8211 and d.airfoil.nunique() == 94
    saved = pd.read_csv(os.path.join(ROOT, "data", "bl_error_bar_bench.csv"))
    for m in ["confidence_Re", "confidence_Re_BL"]:
        te, mu, _ = eb.predictions(d, m)["airfoil-CV"]
        row = saved[(saved.model == m) & (saved.split == "airfoil-CV")].iloc[0]
        assert abs(eb.deviance(te.y.to_numpy(), mu).mean() - row.mean_deviance) < 1e-5
    rule = pd.read_csv(os.path.join(ROOT, "data", "bl_error_bar_experiment.csv"))
    assert rule[rule.analysis == "2 success rule"].value.iloc[0] == 0
    p2u = saved[(saved.model == "confidence_Re_BL") & (saved.split == "Princeton->UIUC")].iloc[0]
    assert p2u.vs_bar_hi > 0      # the interval includes zero, which is why the rule fails


def test_paper_section_3_10_numbers_match_the_data():
    """Section 3.10 of PAPER.md quotes the bench files. Every number it gives must still come from them,
    and the same section must be in PAPER.html."""
    import re
    import numpy as np
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_claims as tc

    def csv(name):
        return pd.read_csv(os.path.join(ROOT, "data", name))

    prose = tc._prose("PAPER.md")
    sec = prose[prose.index("3.10 Can a correction learned"):prose.index("4. Discussion")]
    html = tc._prose("PAPER.html")
    assert "3.10 Can a correction learned in one tunnel fix the other?" in html and "39_drag_correction" in open(
        os.path.join(ROOT, "PAPER.html"), encoding="utf-8").read()

    bench = pd.concat([csv(f) for f in ["drag_correction_bench.csv", "drag_correction_gated_bench.csv",
                                        "drag_correction_E_bench.csv", "bl_state_bench.csv"]]).drop_duplicates(
        ["candidate", "split"]).set_index(["candidate", "split"])
    # the table: every cell
    rows = {"A: Re and lift": "correction_A", "B: A plus camber, thickness": "correction_B",
            "C: B plus confidence": "correction_C", "A below Re 150,000 only": "correction_A_Re<=150k",
            "E: more flexible Re and lift": "correction_E", "F: A plus boundary-layer state": "correction_F",
            "G: F plus transition movement": "correction_G"}
    splits = ["UIUC->Princeton", "Princeton->UIUC", "airfoil-CV"]
    for label, cand in rows.items():
        m = re.search(re.escape(label) + r" \| (−[\d.]+) \| (−[\d.]+) \| (−[\d.]+) \|", sec)
        assert m, label
        for cell, split in zip(m.groups(), splits):
            quoted = -float(cell[1:])
            assert abs(quoted - bench.loc[(cand, split), "vs_neuralfoil"] * 100) < 0.006, (label, split)

    def quoted(text):
        assert text in sec, text

    # first test
    a = bench.loc["correction_A"]
    for text in ["11.10 to 9.92", "12.32 to 11.46", "from 11.69 to 10.48", "0.80 to 1.59", "0.11 to 1.56",
                 "a change of 1.19 points", "a change of 0.86 points"]:
        quoted(text)
    assert abs(a.loc["UIUC->Princeton", "mean_abs_errCD"] * 100 - 9.92) < 0.006
    assert abs(a.loc["Princeton->UIUC", "mean_abs_errCD"] * 100 - 11.46) < 0.006
    assert abs(a.loc["airfoil-CV", "mean_abs_errCD"] * 100 - 10.48) < 0.006
    assert abs(a.loc["UIUC->Princeton", "vs_nf_ci_lo"] * 100 + 1.59) < 0.006
    assert abs(a.loc["UIUC->Princeton", "vs_nf_ci_hi"] * 100 + 0.80) < 0.006
    assert abs(a.loc["Princeton->UIUC", "vs_nf_ci_lo"] * 100 + 1.56) < 0.006
    assert abs(a.loc["Princeton->UIUC", "vs_nf_ci_hi"] * 100 + 0.11) < 0.006
    assert 1 - 0.86 > 0.139 and 1 - 0.86 < 0.141       # "missed it by 0.14"
    b, c = bench.loc["correction_B", "Princeton->UIUC"], bench.loc["correction_C", "Princeton->UIUC"]
    assert abs(b.vs_nf_ci_lo * 100 + 1.28) < 0.006 and abs(b.vs_nf_ci_hi * 100 - 0.63) < 0.006
    assert abs(c.vs_nf_ci_lo * 100 + 1.34) < 0.006 and abs(c.vs_nf_ci_hi * 100 - 0.55) < 0.006
    assert bench.loc[("correction_A_Re<=150k", "UIUC->Princeton"), "vs_neuralfoil"] * 100 \
        - a.loc["UIUC->Princeton", "vs_neuralfoil"] * 100 > 0.36             # "0.37 points worse than A"
    for k, g in [("F", 0.28), ("G", 0.46)]:
        assert abs((a.loc["airfoil-CV", "vs_neuralfoil"] - bench.loc[(f"correction_{k}", "airfoil-CV"), "vs_neuralfoil"]) * 100 - g) < 0.006
    assert abs((bench.loc[("correction_E", "airfoil-CV"), "vs_neuralfoil"] - a.loc["airfoil-CV", "vs_neuralfoil"]) * 100 + 0.10) < 0.006
    assert abs(bench.loc[("correction_F", "UIUC->Princeton"), "mean_abs_errCD"] * 100 - 10.23) < 0.006
    assert abs(bench.loc[("correction_G", "UIUC->Princeton"), "mean_abs_errCD"] * 100 - 10.31) < 0.006
    for text in ["10.23 and 10.31 percent against 9.92", "0.28 and 0.46 points over A", "added 0.10 points"]:
        quoted(text)

    # by Reynolds band
    r = csv("drag_correction_by_Re.csv").set_index(["tunnel", "Re_bin"])
    for key, (lo, hi) in {("Princeton", "60k"): (16.8, 13.2), ("UIUC", "40-60k"): (21.7, 17.9),
                          ("UIUC", "300-400k"): (9.8, 10.2), ("UIUC", "400-500k"): (9.2, 10.3)}.items():
        assert abs(r.loc[key, "neuralfoil"] * 100 - lo) < 0.06 and abs(r.loc[key, "corrected"] * 100 - hi) < 0.06, key
    for text in ["from 16.8 to 13.2 percent", "from 21.7 to 17.9", "9.8 to 10.2", "9.2 to 10.3"]:
        quoted(text)

    # locked tunnel
    lt = csv("drag_correction_ltpt.csv").iloc[0]
    assert lt.n_points == 234 and abs(lt.neuralfoil * 100 - 13.0) < 0.05 and abs(lt.corrected * 100 - 12.6) < 0.05
    assert abs(lt.neuralfoil_median * 100 - 11.2) < 0.05 and abs(lt.corrected_median * 100 - 10.2) < 0.05
    assert abs(lt.share_improved * 100 - 53) < 0.5
    for text in ["from 13.0 to 12.6 percent", "median from 11.2 to 10.2", "better on 53 percent of the 234 points"]:
        quoted(text)

    # what is left
    t = csv("residual_scatter.csv").set_index("item").value
    assert [round(t[k] * 100) for k in ["between_airfoil", "between_polar", "within_polar"]] == [8, 14, 78]
    assert round(float(t["points per polar"])) == 14
    assert abs(t["thickness"] - 0.41) < 0.005 and abs(t["camber"] - 0.23) < 0.005
    dec = csv("xfoil_decomposition.csv")
    assert dec.polar.nunique() == 667
    for text in ["across 667 polars of about 14 points each", "0.41 against thickness and 0.23 against camber",
                 "8 percent of the squared error", "is 14 percent", "is 78 percent"]:
        quoted(text)
    d = dc.with_shape(tb.load())
    p = tb.cv_predict(d, dc.candidate(dc.FEATURES["A"]))
    wt = d.WT_CD.to_numpy(float)
    off = pd.Series(np.log(wt / p)).groupby(d.airfoil.to_numpy()).transform("mean").to_numpy()
    assert abs(np.mean(np.abs(p - wt) / wt) * 100 - 10.48) < 0.006
    assert abs(np.mean(np.abs(p * np.exp(off) - wt) / wt) * 100 - 10.22) < 0.006
    quoted("from 10.48 to 10.22 percent")

    # boundary-layer state, inside a polar, and noise
    e = csv("bl_state_experiment.csv")
    sp = e[e.analysis == "4 within-polar spearman of |residual| with"].set_index("item").value
    assert abs(sp["xs"] + 0.22) < 0.005 and abs(sp["jump_s"] + 0.17) < 0.005
    assert abs(e[e.analysis.str.startswith("5 share")].value.iloc[0] - 0.05) < 0.01
    for text in ["correlation of −0.22", "moving fast, −0.17", "about 5 percent of the scatter inside a polar", "3.7 percent in drag"]:
        quoted(text)

    # error bar
    eb = csv("bl_error_bar_bench.csv").set_index(["model", "split"])
    rel = lambda s: -eb.loc[("confidence_Re_BL", s), "vs_bar_rel"] * 100
    assert abs(rel("airfoil-CV") - 7.4) < 0.05 and abs(rel("UIUC->Princeton") - 7.9) < 0.05
    assert abs(rel("Princeton->UIUC") - 1.3) < 0.05
    assert eb.loc[("confidence_Re_BL", "Princeton->UIUC"), "vs_bar_hi"] > 0
    # post hoc: full_BL minus full, interval above zero in both cross-tunnel splits
    d5 = csv("bl_error_bar_experiment.csv")
    ph = d5[d5.analysis.str.contains("full_BL minus full")].set_index("item").value
    for s in ["UIUC->Princeton", "Princeton->UIUC"]:
        assert ph[f"{s} interval low"] > 0
    for text in ["improved by 7.4 percent", "learning on UIUC by 7.9 percent", "only 1.3 percent"]:
        quoted(text)

    # the bias that motivated it
    dec["se"] = (dec.NF_CD - dec.WT_CD) / dec.WT_CD
    by = dec.groupby(["tunnel", "Re_bin"]).se.mean() * 100
    assert 11.5 < by[("Princeton", "60k")] < 13.5 and 11.5 < by[("UIUC", "40-60k")] < 13.5
    assert -7 < by[("UIUC", "400-500k")] < -4 and -5 < by[("Princeton", "300k")] < -3.5
    low = dec[dec.WT_CL <= 0].groupby("tunnel").se.mean() * 100
    assert 9.5 < low.min() and low.max() < 14.5
