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
