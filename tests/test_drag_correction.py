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
