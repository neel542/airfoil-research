# Robust multi-point airfoil optimization at low Reynolds number

Gradient-based shape optimization with **AeroSandbox** (`Opti`/IPOPT) and the
differentiable **NeuralFoil** surrogate (a neural emulator of XFoil).

## What it produces

| Airfoil | Objective | Peak L/D | Worst-case L/D | Mean L/D | Max thickness |
|---------|-----------|---------:|---------------:|---------:|--------------:|
| **A** single-point | max L/D at Re=200k, AoA=4° | **232.9** † | 7.1 | 86.6 | 13% |
| **B** robust       | max **worst-case** L/D over the envelope | 105.1 | **38.2** | 64.2 | 9% |

† Every L/D here is a surrogate estimate. Against 1,400 wind-tunnel points on
20 airfoils (below), NeuralFoil's L/D is typically **16% off** and **14% too
high on average**, so true L/D ≤ shown. Airfoil A's peak sits where NeuralFoil
reports **confidence ≈ 0**, below the range where its drag was validated, so
232.9 is an *optimistic upper bound, not a value*. See the
[wind-tunnel benchmark](#wind-tunnel-benchmark-20-airfoils) section and the
uncertainty-annotated figures `18_LD_vs_AoA_uncertainty.png` /
`19_tradeoff_uncertainty.png`.

Operating envelope for the robust design: **Re ∈ [50k, 500k]**, **AoA ∈ [0°, 8°]**.

**Headline result:** B trades ~55% of A's peak L/D for a **5.4× higher worst-case
L/D**. A is a razor-sharp peak that collapses at the corners of the envelope;
B is a flat plateau that performs decently everywhere.

## Design choices

1. **Geometry - Kulfan/CST (8 weights per surface + LE weight).** Smooth,
   low-dimensional, manufacturable, and analytically differentiable, so the
   shape feeds clean gradients to the optimizer.
2. **Aero - NeuralFoil (`large`).** A differentiable XFoil surrogate. The
   analytic ∂(CL,CD)/∂shape is what makes optimizing over a 25-point envelope
   tractable in seconds. Same fidelity used for optimization and evaluation so
   the numbers are self-consistent.
3. **Optimizer - AeroSandbox `Opti` → IPOPT.** Gradient-based NLP with exact
   auto-diff derivatives.
4. **Robust formulation - epigraph max-min.** Introduce a scalar `g`, constrain
   `g ≤ L/D` at every (Re, AoA) grid point, and maximize `g`. This turns the
   non-smooth worst-case objective into a smooth NLP.
5. **Re grid is log-spaced** (it spans a full decade); the optimizer uses a
   coarse 5×5 grid, evaluation uses a finer 11×9 grid to test generalization to
   unseen operating points.
6. **Thickness band 9-13%** keeps both airfoils structurally viable and
   comparable. (A drove to the 13% ceiling, B to the 9% floor - peak wants
   thick, robust wants thin.)
7. **Multi-start + continuation.** The problem is non-convex. Each design is
   solved from several NACA seeds; the tradeoff family is solved by continuation
   (warm-starting each λ from the previous), which keeps the Pareto front
   monotonic and stops the peak design from sticking in a weak local optimum.
8. **Tradeoff family.** Blended objective `λ·(L/D at design pt) + (1−λ)·g`.
   λ=1 → A, λ=0 → B; intermediate λ trace the peak-vs-robustness Pareto front.

## Caveat

NeuralFoil's `analysis_confidence` is low for these aggressive high-L/D low-Re
sections (0.00 for A, ~0.16 for B); they sit outside its validated range. The
benchmark below shows that confidence is a reliable warning about **drag**
error (10% when confidence > 0.95, 37% when < 0.5) but not about lift. Treat
absolute L/D values (especially A's ~233 peak) as optimistic surrogate
ceilings; the robust **A-vs-B comparison** is the reliable conclusion.

## Wind-tunnel benchmark: 20 airfoils

(`uiuc_lsat_parse.py`, `uiuc_neuralfoil_validation.py`, `e387_neuralfoil_validation.py`)

The whole pipeline rests on NeuralFoil, yet NeuralFoil had only been checked
against experiment at Re = 1.8×10⁶ (its paper's one validation case). This
benchmark checks it against the official plain-text polars of the UIUC
Low-Speed Airfoil Tests archive (Selig et al. 1995, Vol. 1; Selig & McGranahan
2004, NREL/SR-500-34515): every airfoil in those volumes with geometry in the
AeroSandbox database, **22 airfoils, 30 files, 1,763 measured points,
Re 40k-500k**, clean and boundary-layer-tripped runs kept separate. Two
airfoils (A18, BE50) are excluded from the statistics because the 17-parameter
Kulfan basis cannot reproduce them to within 0.5% chord; the other 20 fit to
0.01-0.18% (E387: 0.15%).

**Clean runs, NeuralFoil `large`, 20 airfoils, 1,408 points:**

| Re | n | mean ΔCL | drag error, mean / median | drag bias | confidence |
|---:|---:|---:|---:|---:|---:|
| 60k | 174 | 0.066 | 22% / 16% | +20% | 0.93 |
| 100k | 329 | 0.093 | 15% / 11% | +1% | 0.92 |
| 200k | 468 | 0.080 | 10% / 7% | +1% | 0.92 |
| 300-350k | 357 | 0.075 | 11% / 7% | −2% | 0.92 |
| 460-500k | 80 | 0.043 | 12% / 7% | −8% | 0.89 |
| **all** | **1,408** | **0.078** | **13% / 8%** | **+2%** | **0.92** |

- **Drag** is good from Re = 100k up (median 7-11%) and degrades below it
  (over-predicted by ~20% at 60k). `xxlarge` is no better than `large`.
- **Lift** error is concentrated in high-camber shapes: 0.058 for the 16
  airfoils under 5% camber, 0.143 for FX 63-137, NACA 6409, S1210, S1223.
- **L/D** (CL > 0.2, 1,033 points): typically 16% off, 22% on average, and
  **over-predicted by 14%** (24% at Re ≈ 100k). This is the band drawn on every
  L/D figure (`data/error_model.json`).
- **Trip strips.** Against the 247 tripped runs (E387, SD2030, FX 63-137;
  zigzag trip at 2%/5% chord), NeuralFoil with free transition under-predicts
  drag by 19%; with transition forced at the trip location (`xtr_upper`,
  `xtr_lower`) the bias is 2% and lift error halves. The transition inputs
  work, which had not been checked against experiment before.
- **What `analysis_confidence` tracks: drag, not lift.** Drag error falls
  monotonically from 37% (confidence < 0.5) to 10% (> 0.95); r = −0.46, same
  sign in every Re band. Lift error is flat across confidence (r = +0.07). An
  earlier version of this project reported r ≈ −0.48 between confidence and
  *relative* lift error on the E387 alone; that was driven by near-zero-lift
  points and does not hold in absolute terms or on the wider benchmark.
- **E387 in detail** (`e387_neuralfoil_validation.py`, with headless XFoil at
  the same conditions): lift within ~5% from Re = 200k up and 11% at 100k;
  drag ~12% from 200k up and 22% at 100k, where both NeuralFoil and XFoil miss
  the laminar-separation-bubble drag (figure 13).

Figures: `12`-`17`, `21`-`23`; data in `data/uiuc_experimental.csv`,
`data/uiuc_neuralfoil_validation.csv` and the `data/uiuc_validation_*.csv`
summaries; raw source files in `data/uiuc_lsat/`.

## Uncertainty-aware optimizer (`uncertainty_aware_design.py`)

The benchmark above showed that NeuralFoil's confidence is a reliable warning
about its drag error. This module feeds that back into the optimizer: alongside
worst-case L/D, it rewards designs that live where NeuralFoil is confident (one
dial, `w_conf`), on the same 5×5 grid, seeds and continuation as the pipeline.

| `w_conf` | worst-case L/D | mean confidence | drag error expected at that confidence |
|---:|---:|---:|---:|
| 0 (blind) | 38.5 | 0.16 | ~37% |
| 0.5 | 37.7 | 0.96 | ~10% |
| 1 | 37.7 | 0.96 | ~10% |
| 2 | 37.5 | 0.96 | ~10% |
| 4 | 36.7 | 0.97 | ~10% |
| 8 | 35.2 | 0.98 | ~10% |

With the dial off the optimizer lands, like airfoil B, at confidence 0.16: a
predicted 38.5 that the surrogate's own drag record says is wrong by a third on
average. There is a family of near-equal designs inside and outside the
validated region and nothing breaks the tie. `w_conf = 0.5` breaks it: the
design moves to confidence 0.96 for a 2% drop in predicted worst-case L/D.
(An earlier version of this sweep, on a coarser 3×3 grid, found a poorer
baseline at 32 and made the confidence term look like a free performance gain;
it is not, it costs ~2%.) No prior airfoil-optimization work appears to feed a
surrogate's own measured reliability back into the design loop this way.
Figure: `20_trust_vs_performance.png`; data in `data/uncertainty_aware_sweep.csv`.

## Multi-objective extension (`multiobjective.py`)

L/D alone is a thin objective for an aircraft that flies over people, gets built
in the field, and must not stall. This module designs across **four physically
grounded, differentiable objectives** and traces the dial from pure efficiency
to "community-friendly" (quiet + gentle-stall):

| Objective | Metric | Direction |
|-----------|--------|-----------|
| Efficiency | worst-case L/D over the envelope | maximize |
| Safety | CLmax / stall AoA at Re=50k (lift retained at 12°) | maximize |
| Structure | max thickness = spar depth (stiffness ∝ t³) | maximize |
| Noise | trailing-edge δ*=θ·H (Brooks-Pope-Marcolini self-noise) | minimize |

Results (three weight profiles):

| Profile | Worst-case L/D | Stall AoA | Max thickness | TE noise δ* |
|---------|---------------:|----------:|--------------:|------------:|
| efficiency | **38.5** | 9.0° | 8.0% | 15.1e-3 |
| balanced | 33.9 | 10.0° | 10.4% | 11.9e-3 |
| community | 17.8 | **11.5°** | **13.7%** | **11.1e-3** |

**Finding:** structure and low-Re efficiency are in strong tension - forcing a
13.7% spar costs ~54% of worst-case L/D. The "community" design buys a 3.5°
stall margin (vs 1° for efficiency) and ~27% quieter TE for that price. The
noise proxy uses NeuralFoil's boundary-layer output directly, so it is a real
acoustic signal, not a hand-wave. New figures: `5_multiobj_shapes.png`,
`6_multiobj_radar.png`, `7_multiobj_stall.png`; metrics in
`data/multiobjective_metrics.csv`.

## Manufacturing robustness + validation (`manufacturing_robust.py`, `xfoil_validate.py`)

A field-built or 3D-printed wing never matches the CAD shape. This stage makes
the design robust to *build error* as well as operating conditions, then
validates it two ways.

**Manufacturing-robust optimization.** Build error is modeled as bounded random
perturbations of the Kulfan weights (~0.5% chord RMS surface error, realistic
for foam-cut / printed wings). `B_mfg` maximizes worst-case L/D over both the
operating envelope *and* a fixed 8-member error ensemble (sample-based robust
optimization). `B_nominal` is robust to operating conditions only.

**Out-of-sample validation** against 40 *fresh* build-error realizations the
optimizer never saw:

| Design | As-designed worst-case L/D | Built: mean | Built: 5th-pct | Built: min |
|--------|---------------------------:|------------:|---------------:|-----------:|
| B_nominal | 34.3 | 20.3 | 9.3 | 8.5 |
| B_mfg | 33.7 | 26.5 | **19.5** | 10.3 |

**Finding:** the two designs are near-identical on paper, but under realistic
build error the nominal design's reliable (5th-percentile) worst-case L/D
collapses to 9.3, while the manufacturing-robust design holds 19.5 - **more than
2× more reliable when actually built**, for negligible as-designed cost.

**Fidelity validation against true XFoil.** A headless XFoil 6.99 binary
(compiled from source with a no-op X11 stub; see `THIRD_PARTY_XFOIL.md`) gives
independent ground truth. NeuralFoil `large` (the optimization model) agrees
with true XFoil to **~5-6%** of L/D, `xxlarge` to ~3-4%. That surrogate error is
far smaller than the manufacturing effect above, so the robustness conclusion is
safe. (Note: the aggressive `B_nominal` shape only converged in XFoil for 5/10
angles - an honest signal that the surrogate is optimistic where real viscous
flow separates.) Figures: `8_manufacturing_robustness.png`,
`9_fidelity_check.png`; data in `data/manufacturing_validation.csv`,
`data/xfoil_validation.csv`, `data/fidelity_check.csv`.

## Envelope-wide XFoil trust map (`xfoil_validate_envelope.py`)

The 200k check was one slice. This sweeps true XFoil across the full Re envelope
(50k-500k) and maps the NeuralFoil-vs-XFoil L/D error over every (Re, AoA) cell.

Two findings worth their own line:

1. **Manufacturing robustness ⇒ numerical robustness.** `B_mfg` converges in
   XFoil at **all 45 (Re, AoA) points (9/9 per Re)**; the aggressive `B_nominal`
   converges only **5/9 per Re** - its 5-8° band is separated flow XFoil can't
   solve at all (gray cells in `10_trust_map.png`). A design robust to build
   error is also one with attached, well-behaved flow.
2. **The surrogate is trustworthy in the meat of the envelope.** NeuralFoil L/D
   is within ~3-8% of true XFoil for α≳2° across all Re. Error grows at **low
   lift (α≤1°)** - up to ~18-23%, because L/D is hypersensitive to small CL
   errors there - and peaks around **Re=100k**. NeuralFoil's own
   `analysis_confidence` tracks this (lower for `B_nominal`, 0.2-0.3, than
   `B_mfg`, 0.5-0.65), so the model honestly flags its weak spots.

Practical upshot: trust the surrogate for cruise/climb design points; re-check
near-zero-lift and ~100k conditions, and any finalist as aggressive as
`B_nominal`, in XFoil or wind tunnel. Figures: `10_trust_map.png`,
`11_envelope_polars.png`; data in `data/xfoil_validation_envelope.csv`.

## The reusable tool (`airfoil_designer.py`)

Everything above is wrapped into one config-driven tool so a new vehicle/mission
is a **YAML change, not a code change** - the "useful to others" direction. A
student or NGO engineer points it at their own envelope and gets a robust,
multi-objective, XFoil-validated section.

```bash
python airfoil_designer.py configs/delivery_drone.yaml
python airfoil_designer.py configs/small_turbine.yaml
```

The config exposes every knob: operating envelope + grid density, design point,
objective weights (efficiency / safety / structure / noise), manufacturing-
tolerance robustness (on/off, error magnitude, ensemble size), geometry
constraints, NeuralFoil fidelity, and multi-start seeds. The tool optimizes
(multi-start), evaluates on a fine grid, writes CSV + figures, and runs an
optional best-effort XFoil validation - all into `runs/<name>/`.

Two example missions show the dial producing genuinely different airfoils:

| Mission | Objectives (priority) | Mfg-robust | Result |
|---------|-----------------------|:----------:|--------|
| `delivery_drone` | efficiency + **safety + quiet** (overflies people) | yes (~0.5% chord) | 10.3% thick, CLmax 1.98, worst-case L/D 41; **XFoil 9/9, gap 1.6 L/D** |
| `small_turbine` | efficiency + **structure** (thick blade) | no (molded) | **13.9% thick**, worst-case L/D 59; XFoil convergence poor past 1.5° (aggressive section - honestly reported) |

Same code, opposite designs: the drone is thin/quiet/forgiving, the turbine is
thick/efficient. The structure weight visibly drives section depth (10% vs 14%).
The tool also reports XFoil validation *coverage*, so an aggressive design that
real viscous flow can't sustain is flagged rather than trusted blindly.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Easiest start: the guided notebook
jupyter notebook tutorial.ipynb

# Or design from a mission config
python airfoil_designer.py configs/delivery_drone.yaml

# Or reproduce the research studies
python airfoil_pipeline.py          # single-point vs robust + tradeoff
python multiobjective.py            # efficiency / safety / structure / noise
python manufacturing_robust.py      # build-tolerance robustness
python xfoil_validate_envelope.py   # surrogate trust map vs true XFoil
python uiuc_lsat_parse.py           # official UIUC wind-tunnel files -> one table
python uiuc_neuralfoil_validation.py  # 20-airfoil benchmark of NeuralFoil
python e387_neuralfoil_validation.py  # E387 in detail, with XFoil
python uncertainty_aware_design.py  # confidence-aware optimizer sweep
python rebuild_all_figures.py       # every figure, from the saved CSVs
```

New to the project? Start with [`tutorial.ipynb`](tutorial.ipynb) - it designs an
airfoil for a mission you specify in a few seconds and explains each step.
True-XFoil validation is optional and needs an `xfoil` binary (see
[THIRD_PARTY_XFOIL.md](THIRD_PARTY_XFOIL.md)); everything else runs with
`requirements.txt` alone. Licensed MIT (see [LICENSE](LICENSE)).

## Outputs

```
data/
  airfoil_A_kulfan.csv / airfoil_A_coords.csv   # geometry (params + coordinates)
  airfoil_B_kulfan.csv / airfoil_B_coords.csv
  grid_evaluation.csv      # CL, CD, CM, L/D, confidence for A & B over 11×9 grid
  tradeoff_family.csv      # peak/worst/mean L/D + Pareto flag for each λ
  uiuc_lsat/               # raw UIUC wind-tunnel files (Vol 1, Vol 4), as downloaded
  uiuc_experimental.csv    # all 1,763 measured points in one table
  uiuc_neuralfoil_validation.csv + uiuc_validation_*.csv   # benchmark, point-by-point + summaries
  e387_experimental_NREL.csv / e387_neuralfoil_validation.csv   # E387 detail (+ XFoil)
  uncertainty_aware_sweep.csv
figures/
  1_shapes.png             # overlaid optimized shapes
  2_LD_vs_AoA.png          # L/D vs AoA at Re = 50k / 200k / 500k
  3_LD_heatmap.png         # L/D over the (Re, AoA) envelope: A, B, and B−A
  4_tradeoff.png           # peak-vs-robustness Pareto frontier
  12-16_e387_*.png         # E387: NeuralFoil vs XFoil vs experiment
  17_multifoil_parity.png  # 20-airfoil lift / drag parity
  20_trust_vs_performance.png
  21-23_uiuc_*.png         # error by airfoil, confidence calibration, tripped runs
```
