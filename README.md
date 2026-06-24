# Robust multi-point airfoil optimization at low Reynolds number

Gradient-based shape optimization with **AeroSandbox** (`Opti`/IPOPT) and the
differentiable **NeuralFoil** surrogate (a neural emulator of XFoil).

## What it produces

| Airfoil | Objective | Peak L/D | Worst-case L/D | Mean L/D | Max thickness |
|---------|-----------|---------:|---------------:|---------:|--------------:|
| **A** single-point | max L/D at Re=200k, AoA=4° | **232.9** | 7.1 | 86.6 | 13% |
| **B** robust       | max **worst-case** L/D over the envelope | 105.1 | **38.2** | 64.2 | 9% |

Operating envelope for the robust design: **Re ∈ [50k, 500k]**, **AoA ∈ [0°, 8°]**.

**Headline result:** B trades ~55% of A's peak L/D for a **5.4× higher worst-case
L/D**. A is a razor-sharp peak that collapses at the corners of the envelope;
B is a flat plateau that performs decently everywhere.

## Design choices

1. **Geometry — Kulfan/CST (8 weights per surface + LE weight).** Smooth,
   low-dimensional, manufacturable, and analytically differentiable, so the
   shape feeds clean gradients to the optimizer.
2. **Aero — NeuralFoil (`large`).** A differentiable XFoil surrogate. The
   analytic ∂(CL,CD)/∂shape is what makes optimizing over a 25-point envelope
   tractable in seconds. Same fidelity used for optimization and evaluation so
   the numbers are self-consistent.
3. **Optimizer — AeroSandbox `Opti` → IPOPT.** Gradient-based NLP with exact
   auto-diff derivatives.
4. **Robust formulation — epigraph max-min.** Introduce a scalar `g`, constrain
   `g ≤ L/D` at every (Re, AoA) grid point, and maximize `g`. This turns the
   non-smooth worst-case objective into a smooth NLP.
5. **Re grid is log-spaced** (it spans a full decade); the optimizer uses a
   coarse 5×5 grid, evaluation uses a finer 11×9 grid to test generalization to
   unseen operating points.
6. **Thickness band 9–13%** keeps both airfoils structurally viable and
   comparable. (A drove to the 13% ceiling, B to the 9% floor — peak wants
   thick, robust wants thin.)
7. **Multi-start + continuation.** The problem is non-convex. Each design is
   solved from several NACA seeds; the tradeoff family is solved by continuation
   (warm-starting each λ from the previous), which keeps the Pareto front
   monotonic and stops the peak design from sticking in a weak local optimum.
8. **Tradeoff family.** Blended objective `λ·(L/D at design pt) + (1−λ)·g`.
   λ=1 → A, λ=0 → B; intermediate λ trace the peak-vs-robustness Pareto front.

## Caveat

NeuralFoil's `analysis_confidence` is modest (~0.05–0.12) for these aggressive
high-L/D low-Re sections — they sit near the edge of its training distribution.
Treat absolute L/D values (especially A's ~233 peak) as surrogate estimates;
the robust **A-vs-B comparison** is the reliable conclusion. Validate finalists
in XFoil/RANS before committing.

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
| Noise | trailing-edge δ*=θ·H (Brooks–Pope–Marcolini self-noise) | minimize |

Results (three weight profiles):

| Profile | Worst-case L/D | Stall AoA | Max thickness | TE noise δ* |
|---------|---------------:|----------:|--------------:|------------:|
| efficiency | **38.5** | 9.0° | 8.0% | 15.1e-3 |
| balanced | 33.9 | 10.0° | 10.4% | 11.9e-3 |
| community | 17.8 | **11.5°** | **13.7%** | **11.1e-3** |

**Finding:** structure and low-Re efficiency are in strong tension — forcing a
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
collapses to 9.3, while the manufacturing-robust design holds 19.5 — **more than
2× more reliable when actually built**, for negligible as-designed cost.

**Fidelity validation against true XFoil.** A headless XFoil 6.99 binary
(compiled from source with a no-op X11 stub; see `THIRD_PARTY_XFOIL.md`) gives
independent ground truth. NeuralFoil `large` (the optimization model) agrees
with true XFoil to **~5–6%** of L/D, `xxlarge` to ~3–4%. That surrogate error is
far smaller than the manufacturing effect above, so the robustness conclusion is
safe. (Note: the aggressive `B_nominal` shape only converged in XFoil for 5/10
angles — an honest signal that the surrogate is optimistic where real viscous
flow separates.) Figures: `8_manufacturing_robustness.png`,
`9_fidelity_check.png`; data in `data/manufacturing_validation.csv`,
`data/xfoil_validation.csv`, `data/fidelity_check.csv`.

## Envelope-wide XFoil trust map (`xfoil_validate_envelope.py`)

The 200k check was one slice. This sweeps true XFoil across the full Re envelope
(50k–500k) and maps the NeuralFoil-vs-XFoil L/D error over every (Re, AoA) cell.

Two findings worth their own line:

1. **Manufacturing robustness ⇒ numerical robustness.** `B_mfg` converges in
   XFoil at **all 45 (Re, AoA) points (9/9 per Re)**; the aggressive `B_nominal`
   converges only **5/9 per Re** — its 5–8° band is separated flow XFoil can't
   solve at all (gray cells in `10_trust_map.png`). A design robust to build
   error is also one with attached, well-behaved flow.
2. **The surrogate is trustworthy in the meat of the envelope.** NeuralFoil L/D
   is within ~3–8% of true XFoil for α≳2° across all Re. Error grows at **low
   lift (α≤1°)** — up to ~18–23%, because L/D is hypersensitive to small CL
   errors there — and peaks around **Re=100k**. NeuralFoil's own
   `analysis_confidence` tracks this (lower for `B_nominal`, 0.2–0.3, than
   `B_mfg`, 0.5–0.65), so the model honestly flags its weak spots.

Practical upshot: trust the surrogate for cruise/climb design points; re-check
near-zero-lift and ~100k conditions, and any finalist as aggressive as
`B_nominal`, in XFoil or wind tunnel. Figures: `10_trust_map.png`,
`11_envelope_polars.png`; data in `data/xfoil_validation_envelope.csv`.

## The reusable tool (`airfoil_designer.py`)

Everything above is wrapped into one config-driven tool so a new vehicle/mission
is a **YAML change, not a code change** — the "useful to others" direction. A
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
optional best-effort XFoil validation — all into `runs/<name>/`.

Two example missions show the dial producing genuinely different airfoils:

| Mission | Objectives (priority) | Mfg-robust | Result |
|---------|-----------------------|:----------:|--------|
| `delivery_drone` | efficiency + **safety + quiet** (overflies people) | yes (~0.5% chord) | 10.3% thick, CLmax 1.98, worst-case L/D 41; **XFoil 9/9, gap 1.6 L/D** |
| `small_turbine` | efficiency + **structure** (thick blade) | no (molded) | **13.9% thick**, worst-case L/D 59; XFoil convergence poor past 1.5° (aggressive section — honestly reported) |

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
```

New to the project? Start with [`tutorial.ipynb`](tutorial.ipynb) — it designs an
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
figures/
  1_shapes.png             # overlaid optimized shapes
  2_LD_vs_AoA.png          # L/D vs AoA at Re = 50k / 200k / 500k
  3_LD_heatmap.png         # L/D over the (Re, AoA) envelope: A, B, and B−A
  4_tradeoff.png           # peak-vs-robustness Pareto frontier
```
