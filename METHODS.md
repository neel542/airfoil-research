# Methods

A reproducible methodology for **robust, multi-objective airfoil design at low
Reynolds number** using differentiable surrogate aerodynamics. Written to be
lifted into a technical report or paper methods section.

## 1. Problem setting

Small UAVs, distributed wind turbines, and similar low-Reynolds-number lifting
surfaces operate over a *wide, uncontrolled* range of conditions
(Re ∈ [50k, 500k], angle of attack α ∈ [0°, 8°]) and are frequently built with
non-trivial manufacturing error. Designing for a single operating point yields
sections that are fragile off-design. We instead optimize for robust,
multi-criteria performance across the whole envelope.

## 2. Geometry parameterization

Airfoils use the **Kulfan / Class-Shape-Transformation (CST)** parameterization
(AeroSandbox `KulfanAirfoil`): 8 upper-surface weights, 8 lower-surface weights,
a leading-edge modification weight, and a fixed trailing-edge thickness (0.25%
chord). This basis is smooth, low-dimensional (17 design variables),
manufacturable, and analytically differentiable — essential for gradient-based
optimization.

## 3. Aerodynamic model

Aerodynamic coefficients come from **NeuralFoil** (`large` model), a neural
surrogate trained to emulate XFoil. It returns CL, CD, CM, a self-reported
`analysis_confidence`, and boundary-layer state (momentum thickness θ and shape
factor H along each surface). Its key property is **differentiability**: exact
∂(CL, CD)/∂(shape) gradients make optimization over a 25-point operating grid
tractable in seconds. The same fidelity is used for optimization and evaluation
so results are self-consistent.

## 4. Optimization

All problems are solved with **AeroSandbox `Opti`** (an IPOPT interior-point NLP
with automatic differentiation). Geometry constraints: local thickness > 0.4%
chord at 20 stations (no self-intersection) and a max-thickness band.

**Robust efficiency (max-min).** The worst-case L/D over the envelope is made
smooth with an *epigraph* reformulation: introduce a scalar g, constrain
g ≤ L/D(Re_i, α_j) for every grid point, and maximize g. The Re grid is
log-spaced (it spans a decade).

**Non-convexity.** The problem is non-convex, so each design is solved from
multiple NACA seeds (multi-start) and the best objective is kept. Families of
designs are traced by *continuation* — warm-starting each case from the previous
solution — which keeps the Pareto front monotonic.

## 5. Multi-objective formulation

Four physically grounded, differentiable objectives, combined as a weighted sum
of metrics normalized by reference scales (so weights are unit-free):

| Objective | Metric | Rationale |
|-----------|--------|-----------|
| Efficiency | worst-case L/D (g) | range / endurance across the envelope |
| Safety | CL at α = 12°, Re = 50k | lift retained near stall ⇒ later, gentler stall |
| Structure | max thickness | spar depth; bending stiffness ∝ t³ |
| Noise | TE displacement thickness δ\* = θ·H | Brooks–Pope–Marcolini TE self-noise ∝ δ\*_TE |

Sweeping the weights traces a design "dial" from pure efficiency to
community-friendly (quiet, gentle-stall, structurally deep).

## 6. Manufacturing-tolerance robustness

Build error is modeled as bounded Gaussian perturbations of the Kulfan weights
(σ calibrated to ~0.5% chord RMS surface error). Using **sample-based robust
optimization**, a fixed ensemble of error realizations is drawn once; the
epigraph worst-case is taken over both the operating grid and the ensemble, so
the inner problem stays deterministic and differentiable. Designs are then
validated **out-of-sample** against a fresh, larger ensemble of perturbations the
optimizer never saw.

## 7. Validation

**Independent ground truth.** Finalists are re-analyzed in true **XFoil 6.99**
(a headless binary built from source; see `THIRD_PARTY_XFOIL.md`). NeuralFoil
agrees with XFoil to ~3–8% of L/D across most of the envelope. A *trust map* over
(Re, α) localizes where the surrogate is reliable and where XFoil itself fails to
converge (separated flow), flagging where finalists need higher-fidelity CFD or
wind-tunnel testing. NeuralFoil's `analysis_confidence` is reported throughout
and tracks these weak spots.

## 8. Reproducibility

Pinned dependencies in `requirements.txt`; fixed RNG seeds for all perturbation
ensembles; every study writes its data to CSV before plotting. The config-driven
tool (`airfoil_designer.py` + `configs/*.yaml`) reproduces any mission design
from a single spec.

## Key limitations

- Absolute L/D values are surrogate estimates; the *relative* (A-vs-B,
  robust-vs-nominal) conclusions are the reliable outputs.
- Aggressive high-lift sections sit near NeuralFoil's training-distribution edge
  and near XFoil's convergence limit — treat their absolute numbers with caution.
- 2-D sectional analysis only: no 3-D, rotational, or unsteady effects.
- XFoil validation is steady, fully-turbulent-transition-modeled RANS-free panel
  + integral-BL; it is ground truth *relative to NeuralFoil*, not flight test.
