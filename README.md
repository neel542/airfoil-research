# How far can NeuralFoil be trusted below Re = 500,000?

**NeuralFoil** is a neural network that copies the airfoil solver XFoil and
answers in a fraction of a second. People use it to design slow-flying wings.
Nobody had checked it against a real wind tunnel at those speeds.

So this repo checks it. 9,100 wind-tunnel measurements, 94 airfoils, two
separate tunnels (UIUC and Princeton), Re 60k-500k. XFoil itself gets run at
every one of those points too, which splits the error into XFoil's share and
the network's share. Fifteen airfoils appear in both tunnels, so the tunnels
can be checked against each other. Every number gets an error bar built by
resampling whole airfoils, and a fitted model turns any prediction into an
expected drag error.

The other half is the design pipeline this was built for: robust,
multi-objective, build-tolerant airfoil optimization with **AeroSandbox**
(`Opti`/IPOPT).

**What came out.** NeuralFoil's drag is off by 11-12% on average and 8% in a
typical case, and its lift by 0.07-0.09. Both tunnels agree. At Re = 60k the
drag error grows to 17-22%. But almost none of that is the network's fault.
XFoil misses the same points by 12%, and the network differs from XFoil by
only 2.8%. It even lands closer to the tunnel than XFoil does, on 55% of
points. The two tunnels disagree with each other by 12% in drag, so from
Re = 200k up the model is already as close as the experiments can resolve.
The confidence score warns about drag, and about XFoil's physics, but not
about lift.

Full paper: `PAPER.md` or `PAPER.html`. Short version: `SUMMARY.md`. Methods:
`METHODS.md`.

## What it produces

| Airfoil | Objective | Peak L/D | Worst-case L/D | Mean L/D | Max thickness |
|---------|-----------|---------:|---------------:|---------:|--------------:|
| **A** single-point | max L/D at Re=200k, AoA=4° | **232.9** † | 7.1 | 86.6 | 13% |
| **B** robust       | max **worst-case** L/D over the envelope | 105.1 | **38.2** | 64.2 | 9% |

† Every L/D here is a model estimate, not a measurement. Against the 9,100
wind-tunnel points below, NeuralFoil's L/D is typically **15% off** and **15%
too high on average**, so the real value is at or under what's shown. Airfoil
A's peak is worse than that. It sits where NeuralFoil reports **confidence
near 0**, outside the range where its drag was ever checked, so 232.9 is an
*optimistic ceiling, not a number*. See the
[wind-tunnel benchmark](#wind-tunnel-benchmark-55-airfoils), the XFoil split,
and figures `18_LD_vs_AoA_uncertainty.png` and
`19_tradeoff_uncertainty.png`.

Operating envelope for the robust design: **Re ∈ [50k, 500k]**, **AoA ∈ [0°, 8°]**.

**The short version:** B gives up about 55% of A's peak and gets back a
**5.4× higher worst case**. A is a spike that collapses at the corners of the
envelope. B is a flat plateau that works nearly everywhere.

## Design choices

1. **Shape: Kulfan/CST**, 8 numbers per surface plus one for the leading
   edge. It's smooth, it uses few numbers, it always gives something you
   could build, and it works with calculus. That last part is what feeds
   clean gradients to the optimizer.
2. **Aerodynamics: NeuralFoil (`large`).** A differentiable XFoil stand-in.
   Because the exact effect of each shape number on lift and drag comes free,
   optimizing over a 25-point envelope takes seconds. The same model does
   both the optimizing and the scoring, so the numbers stay consistent.
3. **Optimizer: AeroSandbox `Opti` into IPOPT.** Gradient-based, with exact
   derivatives.
4. **Worst case, made smooth.** Add one number `g`, force `g ≤ L/D` at every
   grid point, then push `g` up. Since `g` can't rise above the worst point,
   pushing it lifts the worst case with it. A jagged objective becomes a
   smooth one.
5. **The Re grid is log-spaced**, because it spans a full decade. The
   optimizer uses a coarse 5×5 grid. Scoring uses a finer 11×9 grid, to see
   whether the design holds at conditions it was never tuned on.
6. **Thickness stays between 9 and 13%** so both airfoils are buildable and
   comparable. A ran to the 13% ceiling and B to the 9% floor. Peak wants
   thick, robust wants thin.
7. **Many starts, then continuation.** The problem has more than one hilltop.
   Each design is solved from several NACA starting shapes. The tradeoff
   family warm-starts each λ from the previous answer, which keeps the front
   clean and stops the peak design getting stuck somewhere weak.
8. **The tradeoff family** blends the two goals: `λ·(L/D at design point) +
   (1−λ)·g`. λ=1 gives A, λ=0 gives B, and everything between traces the
   curve from peak to robust.

## Caveat

NeuralFoil reports low confidence for these aggressive slow-speed sections:
0.00 for A and about 0.16 for B. Both sit outside the range where it was
checked. The benchmark below shows confidence is a reliable warning about
**drag** error, 9% when it's above 0.95 and 32% when it's below 0.5, but it
says nothing about lift. So read the absolute L/D numbers, especially A's
233, as optimistic ceilings. The **A-against-B comparison** is the part to
trust.

## Wind-tunnel benchmark: 55 airfoils

(`uiuc_lsat_parse.py`, `uiuc_neuralfoil_validation.py`, `uiuc_stall_validation.py`, `e387_neuralfoil_validation.py`)

The whole pipeline rests on NeuralFoil. But NeuralFoil had only ever been
checked against an experiment once, at Re = 1.8×10⁶, in its own paper. So
this benchmark checks it against the UIUC Low-Speed Airfoil Tests archive
(Selig et al. 1995, 1996; Lyon et al. 1998; Selig & McGranahan 2004), which
publishes its tables as plain text.

Every airfoil in Volumes 1-4 with a shape in the AeroSandbox database is
included, in its plain form with no flaps or gurney flaps. That's **57
airfoils, 91 files, 5,441 drag points and 21,462 lift points, Re 30k-500k**.
Clean runs and trip-strip runs stay separate.

Two airfoils, A18 and BE50, are dropped from the statistics because the
17-number Kulfan shape can't reproduce them to within 0.5% of chord. The
other 55 fit to 0.01-0.2%, with the E387 at 0.15%. Even 0.15% matters. Run
XFoil on the true E387 and on its Kulfan fit and you get ~3% difference in CL
and 4-7% in CD, which is about half the E387 lift error and a third of the
drag error. So the numbers below are pipeline errors, meaning shape fit plus
network, and an upper bound on the network by itself.

**Clean runs, NeuralFoil `large`, 55 airfoils, 4,428 points:**

| Re | n | mean ΔCL | drag error, mean / median | drag bias | confidence |
|---:|---:|---:|---:|---:|---:|
| 60k | 565 | 0.084 | 22% / 15% | +13% | 0.92 |
| 100k | 1,105 | 0.085 | 14% / 10% | +2% | 0.91 |
| 200k | 1,183 | 0.071 | 10% / 7% | 0% | 0.91 |
| 300-400k | 1,187 | 0.061 | 10% / 7% | −4% | 0.91 |
| 400-500k | 388 | 0.049 | 9% / 7% | −7% | 0.95 |
| **all** | **4,428** | **0.072** | **12% / 8%** | **0%** | **0.91** |

- **Drag** is good from Re = 100k up (median 7-10%) and degrades below it
  (over-predicted by ~13% at 60k). `xxlarge` is no better than `large`.
- **Lift** error is concentrated in high-camber shapes: 0.066 for the 45
  airfoils under 5% camber, 0.113 for the ten above (S1223, S1210, FX 63-137,
  NACA 6409, E423, GOE 417a, BW3, LRN1007, SG6043, USNPS4).
- **L/D** (CL > 0.2, 3,151 points): typically 15% off, 21% on average, and
  **over-predicted by 15%** (20% at Re ≈ 100k). This is the band drawn on every
  L/D figure (`data/error_model.json`).
- **Trip strips.** Against 885 tripped runs on 15 airfoils (zigzag trip at
  2%/5% chord, both surfaces), NeuralFoil with free transition under-predicts
  drag by 15%; with transition forced at the trip location (`xtr_upper`,
  `xtr_lower`) the bias is 6% (on the precisely documented 2004 runs: 19% → 2%)
  and lift error falls. The transition inputs work, which had not been checked
  against experiment before.
- **Through stall** (`uiuc_stall_validation.py`, 298 clean lift sweeps past
  CLmax, Re 30k-500k): CLmax over-predicted by 0.06 (~6%); stall angle within
  ~1.5° (NeuralFoil ~1° early); past stall the CL error nearly doubles while
  confidence drops 0.89 → 0.52; CM off by 0.018 (~20% of typical). Measured
  hysteresis is 0.05 typical but >0.4 in the worst tenth. Figures `24`-`25`;
  data `data/uiuc_stall_validation.csv`.
- **What `analysis_confidence` tracks: drag, not lift.** Drag error falls
  monotonically from 32% (confidence < 0.5) to 9% (> 0.95); r = −0.43, same
  sign in every Re band, and rank correlation −0.65 at the airfoil level. Lift
  error is flat across confidence (r = 0.00). An earlier version of this
  project reported r ≈ −0.48 between confidence and *relative* lift error on
  the E387 alone; that was driven by near-zero-lift points and does not hold in
  absolute terms or on the wider benchmark.
- **E387 in detail** (`e387_neuralfoil_validation.py`, with headless XFoil at
  the same conditions): lift within ~5% from Re = 200k up and 11% at 100k;
  drag ~12% from 200k up and 22% at 100k, where both NeuralFoil and XFoil miss
  the laminar-separation-bubble drag (figure 13).

Figures: `12`-`17`, `21`-`25`; data in `data/uiuc_experimental.csv`,
`data/uiuc_experimental_lift.csv`, `data/uiuc_neuralfoil_validation.csv`,
`data/uiuc_stall_validation.csv` and the `data/uiuc_validation_*.csv`
summaries; raw source files in `data/uiuc_lsat/`.

## Second tunnel: the Princeton "Airfoils at Low Speeds" data set

(`soartech8_parse.py`, `soartech8_neuralfoil_validation.py`)

The whole benchmark then runs again on a separate, older data set: SoarTech
8, *Airfoils at Low Speeds* (Selig, Donovan & Fraser 1989). Those runs
happened in the Princeton 3×4 ft smoke tunnel between 1986 and 1989, and the
same archive shares them (`Stec8.zip`, unpacked in `data/soartech8/`).

That gives **127 airfoil and setup blocks, 7,762 drag points and 15,565 lift
points on 54 airfoils (68 models), Re 60k-300k**. It also gives something
UIUC doesn't: somebody measured **the actual models with a profiler** and
published those shapes next to the design ones. Clean and single-trip blocks
are kept. Flaps, gurney flaps, blowing and mixed setups are dropped. All 68
models fit the Kulfan shape to under 0.07% of chord.

**Clean runs, NeuralFoil `large` on the measured model shape, 4,702 points:**

| Re | n | mean ΔCL | drag error, mean / median | drag bias | confidence |
|---:|---:|---:|---:|---:|---:|
| 60k | 759 | 0.101 | 17% / 13% | +12% | 0.96 |
| 100k | 1,018 | 0.092 | 12% / 8% | +3% | 0.95 |
| 150k | 810 | 0.082 | 10% / 7% | +1% | 0.95 |
| 200k | 1,157 | 0.081 | 9% / 6% | −2% | 0.95 |
| 300k | 958 | 0.077 | 9% / 7% | −4% | 0.96 |
| **all** | **4,702** | **0.086** | **11% / 8%** | **+1%** | **0.96** |

- **It replicates.** Drag 11% / 8% (UIUC 12% / 8%), the same Re trend and the
  same sign change of the bias; L/D typically 15% off, 21% mean, over-predicted
  by 15% (UIUC: identical). Lift slightly worse (0.086 vs 0.072), same +0.04 to
  +0.05 bias in both tunnels. `xxlarge` again no better. An `n_crit` sweep
  (5, 7, 9, 11) on this tunnel gives 13.0 / 11.3 / **11.1** / 20.8% drag error:
  the default 9 is the best setting (figure `26`).
- **Tunnel vs tunnel.** 15 airfoils were measured in both tunnels: 131 polar
  pairs, 2,139 matched (Re, α) points. The two experiments differ by **12% in
  drag** (UIUC ~6% higher) and 0.048 in CL; NeuralFoil differs from UIUC by
  11% and from Princeton by 10% on the same points, and from Re = 200k up all
  three are 7-9%. NeuralFoil's drag error at Re ≥ 200k is at the
  reproducibility limit of the experiments. Lift is not: the tunnels agree
  (0.048) better than NeuralFoil agrees with either (0.066 / 0.080). Figure
  `27`; `data/cross_tunnel_*.csv`.
- **Measured vs design geometry.** The 56 models with both coordinate sets
  deviate from their design by 0.22% chord RMS typically (max 0.68%, E387B),
  i.e. the 0.5% build-error scale assumed in the manufacturing study is
  realistic. Running NeuralFoil on the measured shape lowers drag error from
  12.8% to 11.1% and lift error from 0.091 to 0.081 (46 of 56 models improve);
  the shape change alone moves NeuralFoil's drag by 5% and lift by 0.03.
  Figure `28`; `data/soartech8_geometry_effect.csv`.
- **Trips** (34 runs, 18 models, 1,744 points, mostly a single upper-surface
  strip at 20-70% chord): no free-transition bias here, because a mid-chord
  strip sits near natural transition; forcing transition still cuts drag error
  13.5% → 11.8% and lift error 0.084 → 0.071.
- **Stall** (114 clean lift sweeps past CLmax at Re ≥ 55k, 35 airfoils): CLmax
  over-predicted by 0.11 (UIUC 0.06), stall angle within ~1.4°. At Re 20-50k
  (MB253515) NeuralFoil finds no stall before 20°.
- **Confidence.** Point level replicates at the extremes (30% drag error below
  0.5, 10% above 0.95; lift r = −0.03) but is weaker in between (r = −0.26 vs
  −0.43, flat middle bins) and the **airfoil-level rank correlation does not
  replicate** (−0.08, n.s., vs −0.65): 89% of Princeton points sit above 0.95.
  The score separates trusted from untrusted drag; it is not a fine-grained
  error estimate.

Data: `data/soartech8_experimental*.csv`, `data/soartech8_*_coords.csv`,
`data/soartech8_neuralfoil_validation.csv` and `data/soartech8_*.csv`
summaries; raw files in `data/soartech8/`.

## XFoil decomposition, honest statistics, fitted error model

(`xfoil_decomposition.py`, `clustered_statistics.py`, `fit_error_model.py`)

- **Where the error comes from.** Headless XFoil at all 9,130 clean conditions
  of both tunnels on the same Kulfan geometry NeuralFoil saw (converged at
  8,814, 96.5%). Mean |ΔCD/CD|: **NeuralFoil-vs-tunnel 11.2%, XFoil-vs-tunnel
  12.1%, NeuralFoil-vs-XFoil 2.8%** (median 1.7%). Signed errors correlate at
  r = 0.95; XFoil explains 86% of the variance of NeuralFoil's error; the
  network's own part is < 4% in every Re band of both tunnels, including 60k.
  NeuralFoil is *closer* to experiment than XFoil on 55% of points and has the
  lower mean error in every band (it smooths XFoil's scatter). Lift: 0.079 /
  0.082 / 0.011. The confidence score correlates with |XFoil − tunnel|
  (r = −0.40) at least as strongly as with |NeuralFoil − XFoil| (−0.33): it
  flags where XFoil's solution is ill-conditioned, which is also where XFoil's
  physics is wrong. XFoil converges at only 56% of conditions with confidence
  < 0.5 vs 99% above 0.95. Figure `29`; `data/xfoil_decomposition*.csv`.
- **Clustered intervals.** Points within a polar are one α sweep and are not
  independent; the independent unit is the airfoil. An airfoil-cluster
  bootstrap (B = 2,000) gives 95% intervals **2-5× wider** than the naive point
  bootstrap. Everything survives: drag error 12.3% [11.3, 13.4] UIUC, 11.1%
  [10.3, 12.0] Princeton; L/D bias +15% [12, 18] in both; r(conf, |ΔCD/CD|)
  −0.43 [−0.50, −0.35] and −0.27 [−0.33, −0.20]; lift r includes zero in both.
  The one non-replicating statistic is the airfoil-level rank correlation
  (−0.65 [−0.80, −0.46] vs −0.10 [−0.32, 0.15]), and the correlation anatomy
  shows why: 26% of the confidence variance is between airfoils in the UIUC
  set vs 7% in Princeton, so there the correlation lives entirely along the α
  sweep. Figure `30`; `data/clustered_statistics.csv`,
  `data/confidence_correlation_decomposition.csv`.
- **Fitted error model.** Gamma GLM (log link) of expected |ΔCD/CD| on
  log10(1.001 − confidence), log10(Re/1e5), α, α², camber, thickness; 8,211
  attached-flow points. Effects: ×1.87 per decade of (1 − confidence), ×0.43
  per decade of Re, ×1.035 per % camber, no thickness effect. Held-out-airfoil
  MAE 0.074 vs 0.087 constant vs 0.079 six-bin table; Spearman 0.39; 80th/95th
  percentile bands cover 80%/95%; transfers UIUC→Princeton (0.071 vs 0.085)
  and back (0.077 vs 0.089). Coefficients + worked examples in
  `data/error_model_fit.json`. Figure `31`.

## Uncertainty-aware optimizer (`uncertainty_aware_design.py`)

The benchmark above showed that NeuralFoil's confidence is a reliable warning
about its drag error. This module feeds that back into the optimizer: alongside
worst-case L/D, it rewards designs that live where NeuralFoil is confident (one
dial, `w_conf`), on the same 5×5 grid, seeds and continuation as the pipeline.

| `w_conf` | worst-case L/D | mean confidence | drag error expected at that confidence |
|---:|---:|---:|---:|
| 0 (blind) | 38.5 | 0.16 | ~32% |
| 0.5 | 37.7 | 0.96 | ~9% |
| 1 | 37.7 | 0.96 | ~9% |
| 2 | 37.5 | 0.96 | ~9% |
| 4 | 36.7 | 0.97 | ~9% |
| 8 | 35.2 | 0.98 | ~9% |

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
python uiuc_neuralfoil_validation.py  # 55-airfoil benchmark of NeuralFoil
python uiuc_stall_validation.py     # lift through stall: CLmax, stall angle, CM
python soartech8_parse.py           # Princeton "Airfoils at Low Speeds" files -> tables
python soartech8_neuralfoil_validation.py  # second tunnel, tunnel-vs-tunnel, measured geometry
python e387_neuralfoil_validation.py  # E387 in detail, with XFoil
python xfoil_decomposition.py       # XFoil at all 9,130 clean points: physics vs network error (~25 min, 8 procs)
python clustered_statistics.py      # airfoil-cluster bootstrap intervals + correlation anatomy
python fit_error_model.py           # cross-validated Gamma GLM error model -> data/error_model_fit.json
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
  uiuc_lsat/               # raw UIUC wind-tunnel files (Vols 1-4), as downloaded
  uiuc_experimental.csv    # all 1,763 measured points in one table
  uiuc_neuralfoil_validation.csv + uiuc_validation_*.csv   # benchmark, point-by-point + summaries
  uiuc_experimental_lift.csv / uiuc_stall_validation.csv   # lift runs through stall
  e387_experimental_NREL.csv / e387_neuralfoil_validation.csv   # E387 detail (+ XFoil)
  soartech8/               # raw Princeton SoarTech 8 files (Stec8.zip), as downloaded
  soartech8_experimental*.csv / soartech8_*_coords.csv   # Princeton polars, lift, measured + design coords
  soartech8_neuralfoil_validation.csv + soartech8_*.csv  # second-tunnel benchmark + summaries
  cross_tunnel_*.csv       # UIUC vs Princeton vs NeuralFoil on the 15 shared airfoils
  xfoil_decomposition*.csv # XFoil at every clean point: NF-vs-tunnel, XFoil-vs-tunnel, NF-vs-XFoil
  clustered_statistics.csv / confidence_correlation_decomposition.csv   # cluster-bootstrap CIs, correlation anatomy
  error_model_fit.json / error_model_cv.csv / error_model_calibration.csv  # fitted error model + CV
  uncertainty_aware_sweep.csv
figures/
  1_shapes.png             # overlaid optimized shapes
  2_LD_vs_AoA.png          # L/D vs AoA at Re = 50k / 200k / 500k
  3_LD_heatmap.png         # L/D over the (Re, AoA) envelope: A, B, and B−A
  4_tradeoff.png           # peak-vs-robustness Pareto frontier
  12-16_e387_*.png         # E387: NeuralFoil vs XFoil vs experiment
  17_multifoil_parity.png  # 55-airfoil lift / drag parity
  20_trust_vs_performance.png
  21-23_uiuc_*.png         # error by airfoil, confidence calibration, tripped runs
  24-25_uiuc_*.png         # CLmax / stall-angle parity, lift curves through stall
  26-28_*.png              # two tunnels + n_crit, tunnel vs tunnel, measured vs design geometry
  29_xfoil_decomposition.png / 30_clustered_statistics.png / 31_error_model.png
```
