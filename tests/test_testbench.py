"""The test bench must reproduce the paper's NeuralFoil baseline exactly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))
import testbench as tb


def test_baseline_matches_paper():
    res = tb.run({"neuralfoil": tb.neuralfoil})
    tb.check_baseline(res)
    cv = res[res.split == "airfoil-CV"].iloc[0]
    assert cv.n_points == 9130 and cv.n_airfoils == 94
