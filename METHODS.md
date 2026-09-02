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
manufacturable, and analytically differentiable - essential for gradient-based
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
designs are traced by *continuation* - warm-starting each case from the previous
solution - which keeps the Pareto front monotonic.

**Confidence-aware objective.** For the uncertainty-aware sweep the objective
is g / S_LD + w_conf · mean(analysis_confidence) over the optimization grid,
with the same grid, seeds and continuation as the base problem, so w_conf = 0
reproduces the robust design exactly.

## 5. Multi-objective formulation

Four physically grounded, differentiable objectives, combined as a weighted sum
of metrics normalized by reference scales (so weights are unit-free):

| Objective | Metric | Rationale |
|-----------|--------|-----------|
| Efficiency | worst-case L/D (g) | range / endurance across the envelope |
| Safety | CL at α = 12°, Re = 50k | lift retained near stall ⇒ later, gentler stall |
| Structure | max thickness | spar depth; bending stiffness ∝ t³ |
| Noise | TE displacement thickness δ\* = θ·H | Brooks-Pope-Marcolini TE self-noise ∝ δ\*_TE |

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
agrees with XFoil to ~3-8% of L/D across most of the envelope. A *trust map* over
(Re, α) localizes where the surrogate is reliable and where XFoil itself fails to
converge (separated flow), flagging where finalists need higher-fidelity CFD or
wind-tunnel testing. NeuralFoil's `analysis_confidence` is reported throughout
and tracks these weak spots.

**Experimental benchmark against wind-tunnel data.** XFoil is not ground truth;
it is itself a model. We therefore benchmark NeuralFoil against *measured*
low-Re polars from the UIUC Low-Speed Airfoil Tests archive (Selig et al. 1995,
1996; Lyon et al. 1998; Selig & McGranahan 2004), using the archive's official
plain-text tables rather than the report PDFs. Every airfoil in Vols 1-4 with
design geometry in the AeroSandbox database is included in its plain
configuration: 57 airfoils, 5,441 drag-run and 21,462 lift-run points,
Re = 30k-500k, with clean and boundary-layer-tripped runs kept separate and
files that mix several trip heights excluded (`uiuc_lsat_parse.py`). This
addresses a real gap: the NeuralFoil paper (Sharpe & Hansman 2025) validates
against experiment only at Re = 1.8×10⁶. Each airfoil is refit in the
17-parameter Kulfan basis; the two whose fit error exceeds the 0.5%-chord
manufacturing sigma (A18, BE50) are excluded from the statistics. Clean runs
are compared with free-transition NeuralFoil (n_crit = 9); tripped runs with
transition forced at the header trip locations (`xtr_upper`, `xtr_lower`) and,
for contrast, free. The E387 is also run in headless XFoil at the same
conditions.

*Result (55 airfoils, 4,428 clean drag-run points, NeuralFoil `large`):* mean
absolute lift error 0.072 (0.066 below 5% camber, 0.113 above); drag error 12%
mean, 8% median, degrading to 22% / 15% at Re = 60k where drag is
over-predicted by ~13%; L/D typically 15% off and over-predicted by 15% on
average. With transition forced at the trip, the drag under-prediction on 885
tripped runs falls from 15% to 6% (19% to 2% on the precisely documented 2004
runs). `analysis_confidence` is a calibrated indicator of **drag** error (32%
below 0.5, 9% above 0.95; r = −0.43, same sign in every Re band; airfoil-level
rank correlation −0.65) and carries no information about lift error (r = 0.00).
For the E387 alone, lift is within ~5% from Re = 200k up (11% at 100k) and
drag within ~12% (22% at 100k, the laminar-bubble regime that XFoil also
misses). On the 298 clean lift sweeps that pass through stall, CLmax is
over-predicted by 0.06 (~6%) and the stall angle placed within ~1.5°;
post-stall CL error nearly doubles while confidence halves
(`uiuc_stall_validation.py`). Scripts: `uiuc_neuralfoil_validation.py`,
`e387_neuralfoil_validation.py`; data `data/uiuc_*.csv`; `figures/12-17, 21-25`.

## 8. Reproducibility

Pinned dependencies in `requirements.txt`; fixed RNG seeds for all perturbation
ensembles; every study writes its data to CSV before plotting. The config-driven
tool (`airfoil_designer.py` + `configs/*.yaml`) reproduces any mission design
from a single spec.

## Key limitations

- Absolute L/D values are surrogate estimates with an empirically measured error
  band (§7: 15% typical, 21% mean, over-predicted by 15% on average across 55
  airfoils); the *relative* (A-vs-B, robust-vs-nominal) conclusions are the
  more reliable outputs. Very high quoted L/D (e.g. airfoil A's ~233, at
  confidence ≈ 0) should be read as an optimistic surrogate ceiling, not a
  measured value.
- Aggressive high-lift sections sit near NeuralFoil's training-distribution edge
  and near XFoil's convergence limit - treat their absolute numbers with caution.
- 2-D sectional analysis only: no 3-D, rotational, or unsteady effects.
- XFoil validation is steady, fully-turbulent-transition-modeled RANS-free panel
  + integral-BL; it is ground truth *relative to NeuralFoil*, not flight test.
