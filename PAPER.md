# How far can NeuralFoil be trusted below Re = 500,000? A two-tunnel, 94-airfoil wind-tunnel benchmark with an XFoil error decomposition, and its use in robust airfoil design

**Author:** Neel Madhav
**Date:** 2026

---

## Abstract

Small unmanned aircraft and small wind turbines operate at chord Reynolds
numbers between about 50,000 and 500,000, where laminar separation bubbles
dominate the drag and single-point designs are fragile. Design work in this
regime increasingly relies on NeuralFoil, a differentiable neural surrogate
of XFoil that returns a self-reported confidence score. Its published
validation is against XFoil itself and against one experiment at
Re = 1,800,000. Nothing had established how well it matches wind-tunnel
measurements where it is most used, or what its confidence score means in
experimental terms.

This paper benchmarks NeuralFoil against **9,100 clean drag-polar points on
94 airfoils in two independent wind tunnels**: the UIUC Low-Speed Airfoil
Tests archive (55 airfoils, Re = 60,000 to 500,000) and the Princeton
*Airfoils at Low Speeds* data set (54 airfoils, Re = 60,000 to 300,000),
plus 2,600 boundary-layer-tripped points and 400 lift sweeps through stall.
A headless XFoil is run at every one of the 9,100 conditions, so the error
is split into XFoil's physics error and the network's emulation error. The
15 airfoils measured in both tunnels give the reproducibility floor of the
experiments themselves. Every statistic carries an airfoil-cluster bootstrap
interval, and a cross-validated error model is fitted so that a designer can
convert a NeuralFoil prediction into an expected drag error.

Results. NeuralFoil's drag is off by 11 to 12 percent on average (8 percent
median) and its lift by 0.07 to 0.09 in lift coefficient, the same in both
tunnels, with drag error rising to 17 to 22 percent at Re = 60,000. Almost
all of that is inherited: XFoil at the same points is off by 12 percent, the
network differs from XFoil by only 2.8 percent, the two signed errors
correlate at 0.95, and NeuralFoil is closer to the tunnel than XFoil on 55
percent of points. The two tunnels disagree with each other by 12 percent
in drag on shared airfoils, so from Re = 200,000 up the model's drag error
is at the reproducibility limit of the experiments. The confidence score
tracks drag error but not lift error, and tracks XFoil's own error more
strongly than the network's. Airfoil-cluster intervals are two to five
times wider than naive ones, but every headline result survives; the
airfoil-level confidence correlation seen in the UIUC data does not
replicate in the Princeton data, and the correlation anatomy shows why. A
Gamma regression on confidence, Reynolds number, angle of attack and camber
predicts held-out drag error with 14 percent lower absolute error than a
constant and transfers between tunnels. Running the model on the measured
rather than the design shape of each Princeton model removes about a sixth
of its drag error. Finally, a robust, multi-objective, manufacturing-
tolerant design pipeline is presented, and the measured confidence
calibration is used inside it: a small confidence reward moves the design
out of the surrogate's least reliable region for about 2 percent of
predicted worst-case L/D.

---

## 1. Introduction

### 1.1 The low-Reynolds-number problem

A delivery drone, a hand-launched aircraft and a rooftop wind turbine share
a flow regime. At chord Reynolds numbers below about 500,000 the laminar
boundary layer separates before it transitions, reattaches as a turbulent
layer, and encloses a laminar separation bubble. The bubble adds pressure
drag, moves with angle of attack and Reynolds number, and can burst into a
leading-edge stall with little warning. It is also the regime in which
integral boundary-layer methods such as XFoil are least reliable, and in
which wind-tunnel measurements are hardest to reproduce.

### 1.2 One operating point is not enough

An airfoil optimized at a single Reynolds number and angle of attack can
lose most of its performance a short distance away in either variable.
Small aircraft and turbines never hold one condition, so the useful
objective is robustness: acceptable performance everywhere in the operating
envelope, not a sharp optimum at one point.

### 1.3 A fast surrogate whose experimental error was unknown

Optimizing a shape requires thousands of aerodynamic evaluations, and a
viscous panel solver is too slow for that inside a gradient loop. The field
has therefore adopted **NeuralFoil** (Sharpe and Hansman, 2025), a neural
network trained on a very large set of XFoil solutions. It is fast, it is
differentiable, so the optimizer receives exact shape gradients, and it
returns a confidence score with every prediction.

NeuralFoil's published experimental comparison is a single NACA 0012 case at
Re = 1,800,000, an order of magnitude above where small aircraft fly. Its
main validation is against XFoil on about 400,000 held-out cases. That
establishes how faithfully it copies XFoil. It does not establish how well
either XFoil or NeuralFoil matches a wind tunnel below Re = 500,000, whether
the confidence score means anything experimentally, or how much of any
disagreement belongs to the network rather than to the physics model it
emulates. Those are the questions this paper answers.

### 1.4 Related work

**Low-Reynolds-number airfoil aerodynamics and its measurement.** The
laminar separation bubble and its effect on drag below Re of about 500,000
are well documented (Drela, 1989; Selig, 2003), and the UIUC group has
described both the wind-tunnel methods and their limits (Selig, Deters and
Williamson, 2011). Two points from that literature matter here. First,
XFoil itself is known to be least reliable in exactly this regime; a recent
multi-model study at Re = 68,000 to 159,000 found XFoil unreliable at the
lowest of those Reynolds numbers and improving at the highest (Demie, Ancha
and Kahsay, 2026). Second, the experiments have their own reproducibility
limit: low-Re drag measured by the wake-rake method is sensitive to
turbulence level, model accuracy and rake position (Selig, Deters and
Williamson, 2011). Neither point had previously been turned into a number
that a NeuralFoil user can act on, which is what sections 3.2 and 3.3 do.

**Neural surrogates for airfoil analysis.** Machine-learned surrogates of
airfoil aerodynamics are now common in shape optimization (Li, Du and
Martins, 2022). NeuralFoil is distinctive in the size of its XFoil training
set, in being differentiable, and in returning a self-reported confidence
score. Its published validation is against XFoil itself and against one
experiment at Re = 1,800,000. No experimental benchmark of it at low
Reynolds number had been published.

**Robust and multi-objective airfoil design.** The fragility of single-point
optimization and the need to optimize over a range of conditions are old
observations (Drela, 1998; Li, Huyse and Padula, 2002). The worst-case
(max-min) formulation, weighted multi-objective scalarization and
sample-based robustness to geometric perturbation used in sections 2.3 to
2.5 are standard methods. This paper applies them and does not claim them
as new.

**Using a surrogate's uncertainty inside the optimizer.** Managing an
approximation's trustworthiness inside a design loop is an established
field. Trust-region model management (Alexandrov, Dennis, Lewis and
Torczon, 1998) restricts steps to where the approximation has been checked;
Kriging-based methods use the surrogate's own predictive variance in the
acquisition function (Jones, Schonlau and Welch, 1998; Queipo et al., 2005;
Forrester, Sobester and Keane, 2008); and recent work penalizes a neural
surrogate's predicted uncertainty directly in the objective (Yang, Li,
Zhang and Chen, 2026). What distinguishes section 3.8 from those methods is
narrower than "no one has done this before". Those methods use a
statistical uncertainty the surrogate computes about itself. This paper
first measures what NeuralFoil's confidence score corresponds to in
wind-tunnel error, then uses the score with that measured meaning attached.
The optimizer is standard; the penalty has an experimentally calibrated
scale.

### 1.5 Contributions

1. **A two-tunnel, 94-airfoil experimental benchmark of NeuralFoil below
   Re = 500,000**: 9,100 clean measurements, 2,600 with a boundary-layer
   trip and 400 lift sweeps through stall, from the UIUC archive and the
   Princeton data set. It gives a measured error for lift, drag, L/D and
   maximum lift by Reynolds number, shows the error is the same in two
   independent laboratories, and shows what the confidence score does and
   does not predict.
2. **A decomposition of that error into XFoil's part and the network's
   part**, by running headless XFoil at every one of the 9,100 conditions.
3. **The experimental noise floor**, from the 15 airfoils measured in both
   tunnels, and a separation of the surrogate's error from the wind-tunnel
   models' build error using the Princeton profiler measurements.
4. **Honest statistics**: airfoil-cluster bootstrap intervals for every
   headline number, an anatomy of the confidence-versus-error correlation,
   and a cross-validated error model that converts a prediction into an
   expected drag error.
5. **A reproducible design pipeline** for worst-case, multi-objective and
   manufacturing-tolerant airfoil optimization, and a confidence-aware
   variant that uses the measured calibration inside the design loop.

---

## 2. Methods

### 2.1 Geometry

Every airfoil is represented in the Kulfan class-shape-transformation (CST)
basis (Kulfan, 2008) with 8 upper-surface weights, 8 lower-surface weights
and one leading-edge modification weight, 17 parameters in all, with a fixed
trailing-edge thickness of 0.25 percent of chord. The basis is smooth,
low-dimensional, manufacturable and analytically differentiable, which is
what a gradient-based optimizer requires.

### 2.2 Aerodynamic model

Lift, drag and moment coefficients come from NeuralFoil's `large` network
(the `xxlarge` network is run as a check), which returns CL, CD, CM, a
confidence score in [0, 1] and boundary-layer state. Because it is
differentiable, exact gradients of CL and CD with respect to the 17 shape
parameters are available, and an optimization over a 25-point operating
grid completes in seconds.

### 2.3 Worst-case performance

The design objective is the worst-case lift-to-drag ratio over the
operating envelope, made smooth by an epigraph reformulation: a scalar *g*
is constrained to lie at or below L/D at every grid point (five Reynolds
numbers, log-spaced over a decade, by five angles of attack) and *g* is
maximized. The problem is non-convex, so each design is solved from several
NACA seeds and the best result kept, and families of designs are traced by
continuation, each solve warm-started from the previous one. All problems
are solved with the IPOPT interior-point method through AeroSandbox.

### 2.4 Four objectives

Four physically grounded objectives are combined as a weighted sum of
metrics normalized by reference scales:

| Objective | Metric | Direction |
|-----------|--------|-----------|
| Efficiency | worst-case L/D over the envelope | maximize |
| Safety | CL at 12 degrees, Re = 50,000 (lift retained near stall) | maximize |
| Structure | maximum thickness (spar depth) | maximize |
| Noise | trailing-edge displacement thickness, a proxy for self-noise | minimize |

### 2.5 Manufacturing tolerance

Build error is modelled as bounded Gaussian perturbations of the Kulfan
weights with a scale of about 0.5 percent of chord in RMS surface error. A
fixed ensemble of perturbations is drawn once; the epigraph worst case is
taken over the operating grid and the ensemble together, so the inner
problem stays deterministic. Designs are then evaluated against a fresh,
larger ensemble the optimizer never saw.

### 2.6 Wind-tunnel data

**UIUC.** Measured lift and drag polars come from the UIUC Low-Speed Airfoil
Tests archive (Selig et al., 1995, 1996; Lyon et al., 1998; Selig and
McGranahan, 2004), which publishes its wind-tunnel tables as plain-text
files. Every airfoil in those four volumes whose design coordinates exist in
the AeroSandbox airfoil database is included in its plain configuration
(flapped and gurney-flap variants cannot be represented by the surrogate):
57 airfoils, 91 data files, 5,441 drag-run points at Re = 40,000 to 500,000,
plus the companion lift runs (21,000 points) that sweep through stall and
back. Clean runs and runs with a boundary-layer trip are separate files in
the archive and are kept separate. A parser turns the files into one table,
so every number is traceable to its source file and run.

For 55 of the 57 airfoils the 17-parameter description reproduces the
published coordinates to better than 0.5 percent of chord, the
manufacturing-error scale used elsewhere in this paper; the two that do not
(A18 and BE50, at 0.9 to 1.3 percent) are reported but excluded from the
statistics, because their disagreement with experiment cannot be separated
from the shape error. The E387 is reproduced to 0.15 percent. Even that
matters: XFoil on the true E387 coordinates and on the 17-parameter version
gives lift coefficients about 3 percent apart and drag coefficients 4 to 7
percent apart, roughly half of the E387 lift error and a third of the drag
error reported in section 3.1. The errors reported here are therefore what
a designer using this pipeline experiences, and an upper bound on the
network's own error.

NeuralFoil was run at every measured condition, with free transition for
clean runs (n_crit = 9) and, for tripped runs, both free and with transition
forced at the trip locations given in the file header.

**Princeton.** The whole test is repeated on *Airfoils at Low Speeds*
(Selig, Donovan and Fraser, 1989), measured in the Princeton University 3
by 4 foot smoke tunnel in 1986 to 1989 and distributed by the same archive
as plain text. It holds 127 airfoil-and-configuration blocks (7,762
drag-polar points) and 135 lift files (15,565 points) on 54 airfoils at
Re = 60,000 to 300,000. Clean and tripped blocks were kept; flapped,
gurney-flap, blowing, clay-leading-edge and mixed-configuration blocks were
excluded. Uniquely, this data set publishes profiler-measured coordinates
of the actual wind-tunnel models next to the design coordinates, so
NeuralFoil was run on both shapes for every model. All 68 models fit the
17-parameter description to better than 0.07 percent of chord. Because the
tunnel's turbulence level is not documented, n_crit was swept from 5 to 11
on this data set; 9 was best.

**What the sample really is.** The headline count of 94 airfoils is a
union: 55 UIUC airfoils plus 54 Princeton airfoils minus the 15 measured in
both. The 68 Princeton models include several sections built twice by
different builders (the E387, E374, E205, E214, S2091, S3021, S4061, HQ 2/9
and FX 63-137) and three repeat runs of the same model. The 9,130 clean
points sit in 667 polars (356 UIUC, 311 Princeton), and the points within a
polar are one angle-of-attack sweep on one model at one Reynolds number.
They are not independent observations. The independent unit is the
airfoil, and the statistics in section 3.4 treat it as such.

### 2.7 Separating XFoil's error from the network's

NeuralFoil is trained to reproduce XFoil, so its disagreement with a wind
tunnel is the sum of XFoil's disagreement with the tunnel and the network's
disagreement with XFoil. Only the second part belongs to the network. A
headless XFoil 6.99 (built from source with the plotting stubbed out) was
run at every one of the 9,130 clean conditions on exactly the 17-parameter
geometry NeuralFoil saw, viscous, n_crit = 9, free transition, at the
measured Reynolds number and angles. Each polar is swept outward from 0
degrees in 0.5 degree steps with the measured angles inserted, so every
point is warm-started from its neighbour, which XFoil's viscous solver
needs at these Reynolds numbers. Points where XFoil failed to converge are
kept as missing and counted. The three pairwise errors, NeuralFoil versus
tunnel, XFoil versus tunnel and NeuralFoil versus XFoil, are then compared
on the converged points.

### 2.8 Statistics

**Clustered uncertainty.** Every headline statistic is given a 95 percent
interval from a cluster bootstrap that resamples airfoils with replacement,
each with all of its points, 2,000 times. The naive point bootstrap is
computed alongside so the two can be compared. Airfoils shared by the two
tunnels are treated as separate clusters in the pooled analysis, since they
are separate physical models in separate facilities.

**Anatomy of a correlation.** The correlation between confidence and
absolute drag error is split into the correlation of airfoil means (between
airfoils), the correlation after subtracting airfoil means from both
variables (within airfoils), and the same split at the level of individual
polars.

**A fitted error model.** Expected absolute drag error is modelled as a
Gamma generalized linear model with a log link, fitted by iteratively
reweighted least squares, on log10(1.001 minus confidence), log10(Re /
100,000), angle of attack and its square, maximum camber and maximum
thickness. Every number reported for it is out of sample: 10-fold
cross-validation with whole airfoils held out (an airfoil in both tunnels
is held out from both at once), and training on one tunnel to test on the
other in both directions. Baselines are a constant and the six-bin
confidence table of section 3.1, refitted on each training fold.

---

## 3. Results

### 3.1 The benchmark: NeuralFoil against the UIUC tunnel

**The reference airfoil first.** The Eppler E387 has been measured in more
wind tunnels than any other low-Re section. Against its 112 clean
measurements at six Reynolds numbers (errors are absolute values, averaged;
bias is the signed average):

| Re | Lift error (ΔCL) | Lift error, relative | Drag error (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.057 | 11% | 22 | −5 | 0.88 |
| 200,000 | 0.030 | 5.4% | 12 | 0 | 0.82 |
| 300,000 | 0.025 | 4.9% | 12 | −1 | 0.81 |
| 350,000 | 0.024 | 4.7% | 12 | −3 | 0.81 |
| 460,000 | 0.023 | 5.1% | 12 | −4 | 0.81 |
| 500,000 | 0.023 | 5.1% | 12 | −5 | 0.81 |

Lift is within about 5 percent from Re = 200,000 up and 11 percent at
Re = 100,000. Drag is about 12 percent from Re = 200,000 up and 22 percent
at Re = 100,000, where the laminar separation bubble sits in the middle of
the low-drag range. There is no large systematic drag bias for the E387: 3
percent low overall. The drag polars show both NeuralFoil and XFoil
missing that bubble drag.

**Fifty-five airfoils.** Across the benchmark set (4,428 clean
measurements):

| Re | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 565 | 0.084 | 22 | 15 | +13 | 0.92 |
| 100,000 | 1,105 | 0.085 | 14 | 10 | +2 | 0.91 |
| 200,000 | 1,183 | 0.071 | 10 | 7 | 0 | 0.91 |
| 300,000 to 400,000 | 1,187 | 0.061 | 10 | 7 | −4 | 0.91 |
| 400,000 to 500,000 | 388 | 0.049 | 9 | 7 | −7 | 0.95 |
| **All** | **4,428** | **0.072** | **12** | **8** | **0** | **0.91** |

The three coefficients behave differently.

1. **Drag is good from Re = 100,000 up and degrades below it.** The median
   drag error is 7 percent from Re = 200,000 up, 10 percent at Re = 100,000
   and 15 percent at Re = 60,000, where NeuralFoil over-predicts drag by
   about 13 percent on average. The bias changes sign with Reynolds number,
   high at the low end and about 7 percent low at the high end, and is
   close to zero overall. The `xxlarge` network is no more accurate than
   `large`.
2. **Lift error is concentrated in highly cambered shapes.** The mean lift
   error of 0.072 splits into 0.066 for the 45 airfoils with less than 5
   percent camber and 0.113 for the ten with more (among them the S1223,
   S1210, FX 63-137 and NACA 6409), the high-lift shapes near the edge of
   what XFoil handles.
3. **L/D error is larger than either component.** Over the 3,151 clean
   points with CL above 0.2, NeuralFoil's L/D is off by 15 percent in the
   median and 21 percent in the mean, and the error is one-sided:
   NeuralFoil over-predicts L/D by 15 percent on average, most strongly at
   Re = 100,000 (20 percent). Every L/D figure in this paper carries that
   band, and every quoted L/D should be read as optimistic.

**Trip strips.** Fifteen of the benchmark airfoils were also measured with
a zigzag trip near the leading edge on both surfaces. Against those 885
tripped measurements, NeuralFoil with free transition under-predicts drag
by 15 percent on average, as it should, since nothing told it about the
trip. With transition forced at the trip locations, the bias falls to 6
percent and the lift error drops by a quarter; on the three airfoils of the
2004 report, where the trip is documented most precisely, it falls from 19
percent to 2 percent and the lift error halves. The transition inputs do
what they claim, which had not been checked against experiment either.

**Through stall.** The archive's lift runs sweep to 15 to 24 degrees, past
maximum lift, stepping up and then back down. Against the 298 clean runs on
the 55 benchmark airfoils where the sweep passed the stall (Re = 30,000 to
500,000), NeuralFoil over-predicts maximum lift by 0.06 on average (about 6
percent) and places the stall about 1 degree early, within 1.5 degrees in
the median. Past the stall its lift error nearly doubles (0.08 to 0.15) and
its confidence drops from 0.89 to 0.52. Its pitching-moment coefficient is
rougher, off by 0.018 against typical values near 0.08. The outliers are
the S1223, USNPS4, SD7080 and PT40, each over-predicted by 0.15 to 0.17 in
maximum lift. Stall hysteresis, the gap between the lift on the way up and
on the way down, is small in the median run (0.05) but exceeds 0.4 in the
worst tenth, and no steady model can reproduce it. The lift curves in the
figure show how far the two branches can differ.

**What the confidence score tracks.** Binning the 3,995 attached-flow
benchmark points by NeuralFoil's confidence score:

| Confidence | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) |
|---|---:|---:|---:|---:|
| below 0.5 | 222 | 0.069 | 32 | 27 |
| 0.5 to 0.7 | 102 | 0.085 | 31 | 27 |
| 0.7 to 0.8 | 62 | 0.096 | 26 | 20 |
| 0.8 to 0.9 | 157 | 0.074 | 21 | 17 |
| 0.9 to 0.95 | 387 | 0.082 | 15 | 11 |
| above 0.95 | 3,065 | 0.071 | 9 | 7 |

The score is a drag-error indicator and nothing else. Drag error falls from
32 percent to 9 percent as confidence rises (r = −0.43 across points, with
the same sign in every Reynolds-number band and at both positive and
negative angles), and at the airfoil level the shapes NeuralFoil is least
sure about are the ones it gets most wrong (rank correlation −0.65). Lift
error is flat across the bins and its correlation with confidence is zero
(r = 0.00). When NeuralFoil reports low confidence, the warning applies to
drag, and therefore to L/D, but not to lift. Section 3.4 gives these
correlations honest intervals and section 3.3 shows what the score is
actually detecting. Section 3.8 uses the score inside the optimizer.

### 3.2 The same test in a second wind tunnel

A benchmark against one laboratory cannot tell the model's error apart from
the laboratory's. Everything in section 3.1 was therefore repeated on the
Princeton data set: 54 airfoils, 68 wind-tunnel models, 4,702 clean
measurements at Re = 60,000 to 300,000, taken a decade earlier, in a
different tunnel, with different instruments, by different model builders.
Against those measurements, with NeuralFoil run on the measured shape of
each model:

| Re | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 759 | 0.101 | 17 | 13 | +12 | 0.96 |
| 100,000 | 1,018 | 0.092 | 12 | 8 | +3 | 0.95 |
| 150,000 | 810 | 0.082 | 10 | 7 | +1 | 0.95 |
| 200,000 | 1,157 | 0.081 | 9 | 6 | −2 | 0.95 |
| 300,000 | 958 | 0.077 | 9 | 7 | −4 | 0.96 |
| **All** | **4,702** | **0.086** | **11** | **8** | **+1** | **0.96** |

**The UIUC numbers replicate.** Drag is off by 11 percent on average and 8
percent in the median (UIUC: 12 and 8), degrades toward Re = 60,000 in the
same way, and its bias changes sign with Reynolds number in the same way
(from 12 percent high at Re = 60,000 to 4 percent low at 300,000; UIUC: 13
percent high to 7 percent low). L/D is over-predicted by 15 percent with a
median error of 15 percent and a mean error of 21 percent, the same three
numbers as the UIUC set. Lift is slightly worse (0.086 against 0.072) with
the same sign: NeuralFoil reads about 0.04 to 0.05 high in both tunnels.
The `xxlarge` network is again no better. Lowering n_crit to 7 removes the
over-prediction at Re = 60,000 but adds a 5 percent under-prediction
everywhere else, and 11 doubles the error.

**Two tunnels disagree with each other by about as much as NeuralFoil
disagrees with either.** Fifteen airfoils were measured in both tunnels.
Pairing their clean runs at matched Reynolds number and angle of attack
gives 131 polar pairs and 2,139 matched points:

| Re | Points | Airfoils | UIUC vs Princeton, drag (%) | NeuralFoil vs UIUC (%) | NeuralFoil vs Princeton (%) | UIUC vs Princeton, lift (ΔCL) |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 340 | 11 | 18 | 18 | 17 | 0.049 |
| 100,000 | 519 | 15 | 16 | 14 | 12 | 0.048 |
| 200,000 | 636 | 15 | 9 | 9 | 7 | 0.048 |
| 300,000 | 483 | 13 | 7 | 7 | 7 | 0.051 |
| **All** | **2,139** | **15** | **12** | **11** | **10** | **0.048** |

The two experiments differ in drag by 12 percent on average, with the UIUC
tunnel reading about 6 percent higher, and NeuralFoil differs from each of
them by 10 to 11 percent on the very same points; on 53 percent of the
points it is closer to the Princeton value than the UIUC measurement is.
From Re = 200,000 up, all three numbers are 7 to 9 percent. NeuralFoil's
drag error at those Reynolds numbers is therefore at the reproducibility
limit of low-Reynolds-number wind-tunnel testing, and no model can be shown
to do better without better experiments. Lift is different: the two tunnels
agree with each other (0.048) better than NeuralFoil agrees with either
(0.066 and 0.080), so the lift error is the model's.

**Build error is part of the "model error".** For the 56 models with both
sets of coordinates, the as-built shape differs from the design by 0.22
percent of chord in the median and by up to 0.68 percent (the E387B),
exactly the range assumed for the manufacturing study in section 2.5.
Running NeuralFoil on the measured shape instead of the design lowers the
drag error from 12.8 to 11.1 percent and the lift error from 0.091 to
0.081, and helps 46 of the 56 models. The shape change alone moves
NeuralFoil's drag by 5 percent and its lift by 0.03, roughly half the size
of the remaining error. For the worst-built model, the E387B, the drag
error falls from 13 to 8 percent. Some of what looks like surrogate error
in every benchmark of this kind, including section 3.1, is the wind-tunnel
model not being the airfoil it was meant to be.

**Trip strips, again.** Thirty-four tripped runs on 18 models (1,744
points) were measured, most with a single strip on the upper surface at 20
to 70 percent of chord, a different regime from the leading-edge trips of
section 3.1. NeuralFoil run normally has no drag bias here, because a strip
that far back sits near where the flow would have transitioned anyway;
forcing transition at the strip still lowers the drag error from 13.5 to
11.8 percent and the lift error from 0.084 to 0.071. For the strips nearest
the leading edge (30 percent of chord or less) the drag error falls from 15
to 13 percent.

**Through stall.** Against the 114 clean lift sweeps at Re = 55,000 and
above that passed the stall (35 airfoils), NeuralFoil over-predicts maximum
lift by 0.11 on average, more than the 0.06 of the UIUC set, and places the
stall within 1.4 degrees in the median. At the lowest Reynolds numbers,
20,000 to 50,000 on the MB253515, it finds no stall before 20 degrees while
the measurement stalls at 12 to 15.

**The confidence score, again.** The point-level result replicates at the
extremes: drag error is 30 percent below a confidence of 0.5 and 10 percent
above 0.95, and lift error is flat (r = −0.03). Two things do not replicate.
The correlation with drag error is weaker (r = −0.26 against −0.43), the
middle bins are flat at 22 to 25 percent rather than falling steadily, and
at the airfoil level the rank correlation between an airfoil's mean
confidence and its mean drag error is −0.08 and not significant, against
−0.65 in the UIUC set. Section 3.4 explains why.

### 3.3 Where the error comes from: XFoil or the network

XFoil converged at 8,814 of the 9,130 clean conditions (96.5 percent). On
those points the three pairwise drag errors are:

| Tunnel | Re | Points | NeuralFoil vs tunnel (%) | XFoil vs tunnel (%) | NeuralFoil vs XFoil (%) |
|---|---:|---:|---:|---:|---:|
| UIUC | 60,000 | 565 | 20.5 | 22.0 | 3.3 |
| UIUC | 100,000 | 1,105 | 12.5 | 14.9 | 3.9 |
| UIUC | 200,000 | 1,183 | 9.4 | 10.8 | 3.6 |
| UIUC | 300,000 to 400,000 | 1,187 | 9.3 | 9.9 | 2.5 |
| UIUC | 400,000 to 500,000 | 388 | 8.9 | 9.2 | 1.7 |
| Princeton | 60,000 | 759 | 16.7 | 17.3 | 2.9 |
| Princeton | 100,000 | 1,018 | 11.7 | 12.3 | 2.6 |
| Princeton | 150,000 | 810 | 9.7 | 10.1 | 2.3 |
| Princeton | 200,000 | 1,157 | 9.1 | 9.4 | 2.3 |
| Princeton | 300,000 | 958 | 8.8 | 9.2 | 2.3 |
| **Both** | **all** | **9,130** | **11.2** | **12.1** | **2.8** |

**The error is XFoil's.** The network differs from the solver it emulates
by 2.8 percent in drag on average (1.7 percent in the median) and by 0.011
in lift coefficient, about a quarter of its disagreement with the wind
tunnels. The signed NeuralFoil and XFoil errors correlate at r = 0.95, and
XFoil's error alone accounts for 86 percent of the variance of NeuralFoil's.
In every Reynolds-number band of both tunnels the network's own
contribution is below 4 percent, including at Re = 60,000, where the total
error is 17 to 22 percent. The Re = 60,000 over-prediction, the sign change
of the bias with Reynolds number and the high-camber lift error of section
3.1 are all XFoil's behaviour, reproduced faithfully. XFoil's own lift
error against the tunnels is 0.082, against NeuralFoil's 0.079.

**NeuralFoil is slightly closer to experiment than XFoil is.** On 55
percent of points the network is nearer the measurement than XFoil, and its
mean drag error is lower in every band (11.2 against 12.1 percent overall,
12.5 against 14.9 at Re = 100,000 in the UIUC tunnel). A network fitted to
hundreds of thousands of XFoil polars smooths XFoil's point-to-point
scatter, and some of that scatter is error. The 3.5 percent of conditions
where XFoil did not converge are also informative: NeuralFoil's mean drag
error there is 25 percent and its mean confidence 0.57, against 11 percent
and 0.95 where XFoil converged, and XFoil converges at only 56 percent of
the conditions where NeuralFoil's confidence is below 0.5, against 99
percent where it is above 0.95.

**What the confidence score detects.** Binning the attached-flow converged
points by confidence, with all three errors:

| Confidence | Points | NeuralFoil vs tunnel (%) | XFoil vs tunnel (%) | NeuralFoil vs XFoil (%) | Lift, NeuralFoil vs XFoil (ΔCL) |
|---|---:|---:|---:|---:|---:|
| below 0.5 | 158 | 32 | 41 | 10 | 0.034 |
| 0.5 to 0.7 | 98 | 28 | 34 | 10 | 0.029 |
| 0.7 to 0.8 | 79 | 26 | 32 | 10 | 0.030 |
| 0.8 to 0.9 | 213 | 21 | 27 | 8 | 0.021 |
| 0.9 to 0.95 | 631 | 15 | 17 | 5 | 0.017 |
| above 0.95 | 6,742 | 9 | 10 | 2 | 0.009 |

The score was trained to report the network's uncertainty about XFoil, and
it does: the emulation error rises from 2 to 10 percent as confidence
falls. But it correlates more strongly with XFoil's error against the
tunnel (r = −0.40) than with the network's error against XFoil (r = −0.33)
or against the tunnel (r = −0.33). Low confidence marks the conditions,
separation bubbles and post-stall flow, where XFoil's solution is
ill-conditioned or absent. Those are also the conditions where XFoil's
physics is wrong. The confidence score is therefore a usable warning about
the physics as well as about the emulation, which is more than it was
designed to be, and the reason it works as a proxy for experimental drag
error. Section 3.8 relies on that.

### 3.4 How sure are these numbers?

**Clustered intervals.** With airfoils resampled as clusters, the 95 percent
intervals are two to five times wider than the naive point-bootstrap
intervals, and every headline result survives:

| Statistic | UIUC | Princeton | Both tunnels |
|---|---|---|---|
| Mean drag error (%) | 12.3 [11.3, 13.4] | 11.1 [10.3, 12.0] | 11.7 [11.0, 12.4] |
| Median drag error (%) | 8.1 [7.4, 8.8] | 7.5 [6.8, 8.2] | 7.8 [7.3, 8.3] |
| Drag bias (%) | +0.4 [−1.4, +2.1] | +1.3 [0.0, +2.6] | +0.9 [−0.2, +2.0] |
| Mean lift error (ΔCL) | 0.072 [0.064, 0.081] | 0.086 [0.080, 0.092] | 0.079 [0.073, 0.085] |
| L/D bias (%) | +15.1 [12.4, 18.0] | +15.0 [12.2, 18.2] | +15.1 [13.0, 17.2] |
| r(confidence, drag error) | −0.43 [−0.50, −0.35] | −0.27 [−0.33, −0.20] | −0.37 [−0.43, −0.31] |
| r(confidence, lift error) | 0.00 [−0.07, 0.07] | −0.03 [−0.07, 0.02] | 0.01 [−0.04, 0.05] |
| Airfoil-level rank correlation | −0.65 [−0.80, −0.46] | −0.10 [−0.32, 0.15] | −0.39 [−0.54, −0.23] |
| Drag error, confidence below 0.5 (%) | 32 [26, 41] | 30 [24, 39] | 32 [27, 38] |
| Drag error, confidence above 0.95 (%) | 9.1 [8.3, 10.0] | 9.6 [8.9, 10.4] | 9.4 [8.9, 9.9] |

Brackets are airfoil-cluster 95 percent intervals. The overall drag bias
is not distinguishable from zero in either tunnel; the L/D over-prediction
is; the drag correlation with confidence is clearly negative in both
tunnels; the lift correlation includes zero in both. The one statistic that
genuinely fails to replicate is the airfoil-level rank correlation, whose
Princeton interval spans zero.

**Why the airfoil-level result does not replicate.** Splitting the
confidence-versus-drag-error correlation by level:

| Level | UIUC | Princeton |
|---|---:|---:|
| All points | −0.43 | −0.27 |
| Between airfoils (airfoil means) | −0.69 | −0.07 |
| Within airfoils | −0.40 | −0.28 |
| Between polars (polar means) | −0.37 | −0.07 |
| Within a polar (along the alpha sweep) | −0.45 | −0.31 |
| Share of confidence variance between airfoils | 26% | 7% |

In the UIUC set the confidence score varies both between airfoils and along
each polar, and both variations predict drag error. In the Princeton set
93 percent of the score's variance is within airfoils, because almost every
Princeton model sits at a confidence above 0.95, so there is nothing
between airfoils to correlate; the correlation that remains lives entirely
along the alpha sweep, where confidence drops toward stall and drag error
rises with it. The point-level correlation is real in both sets. The
airfoil-level correlation is a property of the UIUC sample, which happens
to include shapes NeuralFoil is unsure about, and is
not a general property of the score itself.

**A fitted error model.** A Gamma regression of expected absolute drag
error on the confidence score, Reynolds number, angle of attack, camber and
thickness, fitted on the 8,211 attached-flow points of both tunnels, gives
multiplicative effects of ×1.87 per decade of (1 minus confidence), ×0.43
per decade of Reynolds number, ×1.035 per percent of camber, and no
thickness effect. Out of sample, with whole airfoils held out, its mean
absolute error in predicting the drag error is 0.074, against 0.087 for a
constant and 0.079 for the six-bin confidence table (a 14 percent
improvement on the constant and 7 percent on the table), and its rank
correlation with the realized error is 0.39. Its 80th and 95th percentile
bands cover 80 and 95 percent of held-out points. Trained on the UIUC data
alone and tested on Princeton, its error is 0.071 against 0.085 for a
constant; in the other direction, 0.077 against 0.089. The held-out
calibration is close to the diagonal from a predicted 6 percent (actual 7)
to a predicted 29 percent (actual 29). For a representative 3 percent
camber section at 4 degrees, the model expects a drag error of 7 percent at
confidence 0.98 and Re = 200,000 (80th percentile 11 percent), 10 percent
at Re = 60,000, 13 percent at confidence 0.90 and Re = 100,000, and 18
percent at confidence 0.16 and Re = 200,000, the region where the
unconstrained optimizer of section 3.8 lands. The coefficients are
published with the code so the model can be evaluated without this paper.

### 3.5 Peak performance versus robustness

| Airfoil | Objective | Peak L/D | Worst-case L/D | Max thickness |
|---------|-----------|---------:|---------------:|--------------:|
| A (single-point) | best L/D at one condition | 232.9 † | 7.1 | 13% |
| B (robust) | best worst-case L/D | 105.1 | **38.2** | 9% |

Airfoil B gives up about 55 percent of A's peak L/D and gains a **5.4 times
higher worst-case** L/D. Airfoil A is a sharp peak that collapses as soon
as conditions move away from its design point; B is a broad plateau.

† Airfoil A's peak is produced at a condition where NeuralFoil reports
almost no confidence, so 233 is an optimistic ceiling, not a trustworthy
value; the fitted model of section 3.4 puts the expected drag error there
near 18 percent, and section 3.1 shows L/D is over-predicted by 15 percent
even where confidence is high. Airfoil A sits on a razor-thin peak.

### 3.6 Four objectives

| Setting | Worst-case L/D | Stall angle | Max thickness | Noise proxy |
|---------|---------------:|------------:|--------------:|------------:|
| Efficiency-first | **38.5** | 9.0 degrees | 8.0% | 15.1 × 10⁻³ |
| Balanced | 33.9 | 10.0 degrees | 10.4% | 11.9 × 10⁻³ |
| Community-first | 17.8 | **11.5 degrees** | **13.7%** | **11.1 × 10⁻³** |

Structural depth and low-speed efficiency pull against each other. Forcing
the section thick enough for a deep spar (13.7 percent) costs about 54
percent of its worst-case L/D, and buys a 2.5 degree larger stall margin
and a thinner trailing-edge boundary layer.

### 3.7 Does the robust airfoil survive being built badly?

| Design | Nominal worst-case L/D | As built, mean | As built, worst 5% |
|--------|---------------------------:|------------:|----------------------:|
| B, robust to conditions only | 34.3 | 20.3 | 9.3 |
| B, robust to conditions and build error | 33.7 | 26.5 | **19.5** |

Nominally the two designs are almost identical. Under realistic build
error, evaluated on perturbations the optimizer never saw, the design that
expected the error keeps more than double the reliable worst-case
performance at almost no nominal cost.

### 3.8 Using the calibrated confidence inside the optimizer

Section 3.1 showed that NeuralFoil's confidence score is a reliable warning
about its drag error, section 3.3 showed that the warning is largely about
XFoil's physics, and drag is what carries the L/D error. So put the warning
inside the optimizer. Alongside the worst-case L/D, the
optimizer is given a second reward for landing where NeuralFoil is
confident, controlled by one weight, `w_conf`. The optimization grid (5 by
5) and the three starting shapes are the same as for the robust airfoil B
in section 3.5, the thickness range is slightly wider (8 to 16 percent
instead of 9 to 13), and each weight is also started from the previous one.

| `w_conf` | Worst-case L/D | Mean confidence | Lowest confidence | Drag error expected at that confidence (section 3.1) |
|---:|---:|---:|---:|---:|
| 0 (trust the model blindly) | 38.5 | 0.16 | 0.01 | about 32% |
| 0.5 | 37.7 | 0.96 | 0.89 | about 9% |
| 1 | 37.7 | 0.96 | 0.90 | about 9% |
| 2 | 37.5 | 0.96 | 0.92 | about 9% |
| 4 | 36.7 | 0.97 | 0.93 | about 9% |
| 8 | 35.2 | 0.98 | 0.94 | about 9% |

With the weight at zero, the optimizer finds airfoil B again: a
predicted worst-case L/D of 38.5 (B: 38.2) at a mean confidence of 0.16 (B:
0.16). At that confidence the benchmark says NeuralFoil's drag is wrong by
about a third on average, so the 38.5 is a number the model itself does not
stand behind. The optimizer went there because, as far as the surrogate can
tell, there is a family of shapes with almost the same worst-case L/D, some
inside the validated region and some outside it, and with nothing to break
the tie it picked one outside.

A small weight (`w_conf` = 0.5) breaks the tie. The design moves to a mean
confidence of 0.96, where the measured drag error is about 9 percent, and
the predicted worst-case L/D drops by 2 percent, from 38.5 to 37.7.
Increasing the weight buys a little more confidence for a steadily larger
price: at `w_conf` = 8 the predicted L/D is 35.2. The shape changes are
modest; the two extremes are shown in the figure.

Two cautions. The "expected drag error" column is a population statistic
from 94 other airfoils, not a measurement of these designs, which have not
been tested. And, as section 1.4 sets out, penalizing a surrogate's
uncertainty inside an optimizer is not new; trust-region management and
Kriging-based acquisition functions have done it for decades. What is
specific to this section is that the penalty's scale is measured: the
confidence score has been calibrated against 9,100 wind-tunnel points,
including a decomposition showing that it flags XFoil's physics error and
not only the network's. The experiment establishes that trusting the
surrogate blindly lands, by default, in the region where it is known to be
least reliable, and that leaving that region is nearly free.

---

## 4. Discussion

Know when to trust the surrogate, and know what to trust it about. From Re = 100,000 up, NeuralFoil's drag is
typically within 7 to 10 percent of measurement and its lift within a few
hundredths of the lift coefficient for ordinary shapes, in two independent
wind tunnels, and from Re = 200,000 up the drag error is no larger than the
disagreement between the two tunnels. At those Reynolds numbers the
surrogate has reached the limit of what the experiments can check. Below
Re = 100,000, for highly cambered high-lift shapes, and for L/D in general,
it is optimistic.

The decomposition changes what that means. NeuralFoil is a faithful copy of
XFoil, to within 3 percent in drag and 0.01 in lift, and it is marginally
closer to experiment than XFoil itself. The 11 to 12 percent drag error, the
Re = 60,000 over-prediction and the high-camber lift error are XFoil's
physics, inherited intact. A user who wants better low-Re drag than this
will not get it from a bigger network trained on more XFoil; they will need
a better physics model to train on, or experimental correction. The
confidence score, meanwhile, is more useful than it was designed to be: it flags the conditions where XFoil's solution is
ill-conditioned, which are the conditions where XFoil's physics is also
wrong, so it works as a proxy for experimental drag error. A designer can
watch it as a live warning for drag and L/D, or better, evaluate the fitted
error model of section 3.4, and send only the most doubtful, highest-value
designs for higher-fidelity analysis or a wind tunnel. If the blade will fly
rough or dirty, the transition inputs should be used; they work. When a
design is built, the as-built shape should be measured and re-analysed: a
few tenths of a percent of chord in build error moves the drag prediction
by about 5 percent.

Two lessons here generalize. Points within a polar are not independent, and treating them as such understates the
uncertainty by a factor of two to five. And a correlation measured at the
point level can be produced by very different structures: the UIUC and
Princeton sets give the same conclusion about the confidence score at the
point level for different reasons, and only the anatomy of section 3.4
shows that the airfoil-level version was a property of one sample.

The design results point the same direction from a different angle.
Designs chosen to be robust, to changing conditions or to build error, tend
to keep the flow attached rather than separated, and the confidence-aware
optimizer shows that staying inside NeuralFoil's validated region costs
almost nothing in predicted performance.

---

## 5. Limitations

- **Most of this paper is a validation and framework study.** The shape
  parameterization, the surrogate and the base optimization methods were
  built by others, and, as section 1.4 sets out, robust and
  uncertainty-penalized optimization are established fields. The new
  contributions are the two-tunnel experimental benchmark, the XFoil
  decomposition, the experimental noise floor, the build-error separation,
  the honest statistics and error model, and the experimentally calibrated
  use of the confidence score. Any specific optimized shape still needs
  higher-fidelity or physical confirmation.
- **Every L/D number is an estimate**, with a median measured error of 15
  percent and a mean over-prediction of 15 percent. The conclusions worth
  trusting most are relative ones, such as robust beating peak, not the raw
  values.
- **The sample is 94 distinct airfoils, not 9,100 independent points**
  (section 2.6), and the benchmark covers their plain configuration only.
  Flapped and gurney-flap variants and UIUC Volume 5 are not included. Two
  UIUC airfoils were excluded because the 17-parameter description could
  not reproduce them. The Princeton data are older, the tunnel's turbulence
  level is undocumented, and the two tunnels disagree with each other by 12
  percent in drag, which bounds every number here. Where only design
  coordinates exist (all of the UIUC set), part of every reported error is
  the model builder's; the Princeton comparison puts that part at roughly 2
  percentage points of drag error. Part is also the shape description
  itself. The numbers are pipeline errors, not pure network errors.
- **The XFoil decomposition is on converged points only** (96.5 percent).
  XFoil's failures cluster where NeuralFoil's confidence is low, so the
  decomposition is least complete exactly where the errors are largest.
- **The fitted error model is descriptive.** It explains a modest share of
  the point-to-point variation (rank correlation 0.39 out of sample); its
  value is in the calibrated mean and percentile bands, not in predicting
  individual points.
- **The "expected drag error" attached to the optimized designs is a
  population statistic**, not a measurement of those designs.
- **This is two-dimensional, steady analysis only.** No three-dimensional
  wing, rotating blade or unsteady gust is modelled.
- **None of the optimized airfoils have been physically tested.**

---

## 6. Conclusion

This paper benchmarks NeuralFoil against 9,100 wind-tunnel measurements on
94 airfoils in two independent tunnels below Re = 500,000, runs XFoil at
every one of those conditions, and gives every number an honest interval.
The surrogate is accurate on drag from Re = 100,000 up, optimistic below
that and for high-lift shapes, and the same in both laboratories; from
Re = 200,000 up its drag error is no larger than the disagreement between
the laboratories. Almost all of its error is XFoil's, inherited faithfully;
the network itself adds about 3 percent in drag and 0.01 in lift, and is
slightly closer to experiment than the solver it copies. Its confidence
score is a usable warning about drag error, and about XFoil's physics
error, but not about lift; the airfoil-level version of that result belongs
to one sample. A fitted model converts confidence, Reynolds number, angle
of attack and camber into an expected drag error that transfers between
tunnels. The transition inputs reproduce trip-strip measurements, and
running the model on the measured rather than the designed shape removes
about a sixth of its drag error. Fed back into the optimizer with its
measured meaning, the confidence score moves designs out of the surrogate's
least reliable region for about 2 percent of predicted performance.

The most valuable next steps are (1) a third tunnel with documented
turbulence and flapped configurations, (2) a training set for the surrogate
that includes experimental or higher-fidelity low-Re data, since the
decomposition shows that more XFoil will not help, and (3) a physical
wind-tunnel test of 3D-printed robust and peak-tuned airfoils with their
as-built shapes measured, which would turn this validation study into an
original experimental result.

---

## Author Contributions

The author designed and directed this study: choosing the question, setting
its goals and scope, and reviewing and interpreting every result. AI tools
were used for parts of the code implementation, figure generation and
drafting, under the author's direction.

---

## Acknowledgments

All experimental data used in this work are from public sources and are
cited below.

---

## References

1. Sharpe, P. D., and Hansman, R. J. (2025). *NeuralFoil: An airfoil
   aerodynamics analysis tool using physics-informed machine learning.*
   arXiv:2503.16323.
2. Selig, M. S., and McGranahan, B. D. (2004). *Wind Tunnel Aerodynamic
   Tests of Six Airfoils for Use on Small Wind Turbines.* NREL/SR-500-34515.
3. McGhee, R. J., Walker, B. S., and Millard, B. F. (1988). *Experimental
   Results for the Eppler 387 Airfoil at Low Reynolds Numbers in the
   Langley Low-Turbulence Pressure Tunnel.* NASA TM-4062.
4. Kulfan, B. M. (2008). Universal parametric geometry representation
   method. *Journal of Aircraft*, 45(1), 142-158.
5. Drela, M. (1989). XFOIL: An analysis and design system for low Reynolds
   number airfoils. In *Low Reynolds Number Aerodynamics* (pp. 1-12).
   Springer.
6. Sharpe, P. D. (2021). *AeroSandbox: A differentiable framework for
   aircraft design optimization.* MIT master's thesis and software.
7. Selig, M. S., Guglielmo, J. J., Broeren, A. P., and Giguère, P. (1995).
   *Summary of Low-Speed Airfoil Data, Volume 1.* SoarTech Publications.
8. Selig, M. S., Lyon, C. A., Giguère, P., Ninham, C. P., and Guglielmo,
   J. J. (1996). *Summary of Low-Speed Airfoil Data, Volume 2.* SoarTech
   Publications.
9. Lyon, C. A., Broeren, A. P., Giguère, P., Gopalarathnam, A., and Selig,
   M. S. (1998). *Summary of Low-Speed Airfoil Data, Volume 3.* SoarTech
   Publications. Tabulated data for Volumes 1 to 4 from the UIUC Low-Speed
   Airfoil Tests archive, https://m-selig.ae.illinois.edu/pd.html
10. Selig, M. S., Donovan, J. F., and Fraser, D. B. (1989). *Airfoils at
    Low Speeds.* Soartech 8, SoarTech Publications, Virginia Beach.
    Tabulated data from https://m-selig.ae.illinois.edu/uiuc_lsat.html
11. Selig, M. S. (2003). *Low Reynolds Number Airfoil Design Lecture Notes.*
    VKI Lecture Series, Low Reynolds Number Aerodynamics on Aircraft
    Including Applications in Emerging UAV Technology, von Karman Institute
    for Fluid Dynamics, RTO/AVT-VKI-104.
12. Selig, M. S., Deters, R. W., and Williamson, G. A. (2011). Wind tunnel
    testing airfoils at low Reynolds numbers. *49th AIAA Aerospace Sciences
    Meeting*, AIAA 2011-875.
13. Demie, A. B., Ancha, V. R., and Kahsay, M. B. (2026). Multi-model
    assessment and experimental validation of a custom high-camber airfoil
    for wind-lens technology application. *Wind*, 6(2), 28.
14. Li, J., Du, X., and Martins, J. R. R. A. (2022). Machine learning in
    aerodynamic shape optimization. *Progress in Aerospace Sciences*, 134,
    100849.
15. Drela, M. (1998). Pros and cons of airfoil optimization. In *Frontiers
    of Computational Fluid Dynamics 1998* (pp. 363-381). World Scientific.
16. Li, W., Huyse, L., and Padula, S. (2002). Robust airfoil optimization to
    achieve drag reduction over a range of Mach numbers. *Structural and
    Multidisciplinary Optimization*, 24(1), 38-50.
17. Alexandrov, N. M., Dennis, J. E., Lewis, R. M., and Torczon, V. (1998).
    A trust-region framework for managing the use of approximation models
    in optimization. *Structural Optimization*, 15(1), 16-23.
18. Jones, D. R., Schonlau, M., and Welch, W. J. (1998). Efficient global
    optimization of expensive black-box functions. *Journal of Global
    Optimization*, 13(4), 455-492.
19. Queipo, N. V., Haftka, R. T., Shyy, W., Goel, T., Vaidyanathan, R., and
    Tucker, P. K. (2005). Surrogate-based analysis and optimization.
    *Progress in Aerospace Sciences*, 41(1), 1-28.
20. Forrester, A. I. J., Sobester, A., and Keane, A. J. (2008). *Engineering
    Design via Surrogate Modelling: A Practical Guide.* Wiley.
21. Yang, Y., Li, R., Zhang, Y., and Chen, H. (2026). *Uncertainty-aware
    data-based method for fast and reliable shape optimization.*
    arXiv:2601.21956.

---

*Reproducibility: all code, configuration files, extracted data (as CSV
files), the fitted error model (as JSON) and figures are available in the
project repository. Software versions are pinned and random seeds are
fixed. Every study saves its data to disk before making any plots, and a
test suite pins every number quoted in this paper to those data files.*
