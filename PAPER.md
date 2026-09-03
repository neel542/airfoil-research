# How far can you trust NeuralFoil below Re = 500,000? A two-tunnel test on 94 airfoils, with XFoil run at every point

**Author:** Neel Madhav
**Date:** 2026

---

## Abstract

Small drones and small wind turbines fly slowly. That puts them in a range
engineers call low Reynolds number, roughly 50,000 to 500,000. Down there a
bubble of stalled air forms on the wing and makes most of the drag, and a
wing tuned for one exact condition falls apart as soon as things change.

To design wings fast, the field now uses NeuralFoil. It's a neural network
that copies an older solver called XFoil, and it answers in a fraction of a
second. It also reports a confidence score with every answer. But its only
published wind-tunnel check sits at Re = 1,800,000, about ten times faster
than a drone flies. Nobody knew how far off it is where people actually use
it, or whether that confidence score means anything real.

This paper tests it against **9,100 wind-tunnel measurements on 94 airfoils
in two separate tunnels**. One set comes from the UIUC low-speed archive (55
airfoils, Re = 60,000 to 500,000). The other comes from an older Princeton
data set, *Airfoils at Low Speeds* (54 airfoils, Re = 60,000 to 300,000). On
top of that there are 2,600 points with a trip strip and 400 lift sweeps
that go past the stall. XFoil itself was run at all 9,100 conditions, which
splits the error into XFoil's share and the network's share. Fifteen
airfoils were tested in both tunnels, so the tunnels can be checked against
each other. Every number carries an error bar built by resampling whole
airfoils, and a fitted model turns any prediction into an expected drag
error.

What came out. NeuralFoil's drag is off by 11 to 12 percent on average, 8
percent in a typical case, and its lift by 0.07 to 0.09. Both tunnels agree.
At Re = 60,000 the drag error grows to 17 to 22 percent. Almost none of that
belongs to the network. XFoil misses the same points by 12 percent, and
NeuralFoil differs from XFoil by only 2.8 percent. The two errors track each
other at r = 0.95. NeuralFoil even lands closer to the tunnel than XFoil
does, on 55 percent of points. The two tunnels disagree with each other by
12 percent in drag, so from Re = 200,000 up the model is already as close as
the
experiments can resolve. The confidence score warns about drag but not lift,
and it tracks XFoil's error more strongly than the network's. Honest error
bars run two to five times wider than naive ones, and every headline result
survives them. One result doesn't: a link between an airfoil's average
confidence and its average error shows up in the UIUC data and vanishes in
the Princeton data, and this paper shows why. A fitted model reads
confidence, Reynolds number, angle and camber and predicts held-out drag
error 14 percent better than a flat guess, in either tunnel. Running the
model on the measured shape of each Princeton wing, instead of its drawing,
removes about a sixth of the drag error. Finally, a design pipeline uses all
of this: a small reward for staying where NeuralFoil is confident moves a
design out of its worst region for about 2 percent of predicted worst-case
L/D.

---

## 1. Introduction

### 1.1 Why slow air is hard

A delivery drone, a hand-launched plane and a rooftop wind turbine all share
one problem. They move slowly, and slow air behaves differently.

Below a Reynolds number of about 500,000, the thin layer of air hugging the
wing lifts off the surface before it goes turbulent. It then reattaches
further back, trapping a pocket of stalled air underneath. That pocket is
called a laminar separation bubble. It adds drag, it slides around as speed
and angle change, and it can burst without warning and stall the wing. This
is also the range where solvers like XFoil are least reliable, and where
wind-tunnel results are hardest to repeat.

### 1.2 One operating point isn't enough

Tune an airfoil for a single speed and a single angle and it can lose most
of its performance a short step away in either direction. Real drones and
turbines never hold one condition. So the goal isn't a sharp peak. It's
decent performance across the whole range the wing will actually see.

### 1.3 A fast stand-in nobody had checked

Finding a good shape takes thousands of aerodynamic calculations, and a real
viscous solver is far too slow to sit inside that loop. So the field turned
to **NeuralFoil** (Sharpe and Hansman, 2025), a neural network trained on a
huge pile of XFoil results. It's fast. It's also differentiable, which means
the optimizer can see exactly how a small shape change moves lift and drag.
And it hands back a confidence score every time.

Here's the gap. NeuralFoil's one published comparison against a real
experiment is a single NACA 0012 case at Re = 1,800,000. Its main check is
against XFoil itself, on about 400,000 held-out cases. That tells you how
well it copies XFoil. It doesn't tell you whether XFoil or NeuralFoil match
a wind tunnel below Re = 500,000. It doesn't say what the confidence score
means. And it doesn't say how much of any error belongs to the network
rather than to the solver it was built to imitate. This paper answers all
three.

### 1.4 Related work

**Slow-speed airfoils and how they get measured.** The separation bubble and
its effect on drag are well documented (Drela, 1989; Selig, 2003), and the
UIUC group has written up both the tunnel methods and their limits (Selig,
Deters and Williamson, 2011). Two points from that work matter here. First,
XFoil is known to struggle in exactly this range. A recent study at Re =
68,000 to 159,000 found it unreliable at the low end and better at the high
end (Demie, Ancha and Kahsay, 2026). Second, the experiments have limits of
their own. Slow-speed drag measured with a wake rake shifts with the
tunnel's turbulence, the model's accuracy and where the rake sits (Selig,
Deters and Williamson, 2011). Neither point had been turned into a number a
NeuralFoil user could act on. Sections 3.2 and 3.3 do that.

**Neural stand-ins for airfoil analysis.** Machine-learned surrogates are
common in shape optimization now (Li, Du and Martins, 2022). NeuralFoil
stands out for the size of its XFoil training set, for being
differentiable, and for reporting a confidence score. Its published checks
are against XFoil and against one experiment at Re = 1,800,000. No
slow-speed experimental benchmark of it had been published.

**Robust and multi-objective design.** People have known for a long time
that single-point optimization is fragile and that you should design across
a range (Drela, 1998; Li, Huyse and Padula, 2002). The worst-case setup,
the weighted multi-objective score and the sample-based build-error
robustness in sections 2.3 to 2.5 are all standard. This paper uses them. It
doesn't claim them.

**Feeding a model's uncertainty back into the optimizer.** This is an
established field too. Trust-region methods keep each step inside the
region where the approximation has been checked (Alexandrov, Dennis, Lewis
and Torczon, 1998). Kriging methods put the surrogate's own predicted
variance into the search rule (Jones, Schonlau and Welch, 1998; Queipo et
al., 2005; Forrester, Sobester and Keane, 2008). And recent work penalizes a
neural surrogate's predicted uncertainty directly (Yang, Li, Zhang and Chen,
2026). So what's different in section 3.8 is narrower than "nobody has done
this". Those methods use an uncertainty the model computes about itself.
This paper first measures what NeuralFoil's confidence score is worth in
real wind-tunnel error, then uses the score with that measured meaning
attached. The optimizer is ordinary. The scale on the penalty is the part
that was measured.

### 1.5 What this paper adds

1. **A two-tunnel, 94-airfoil test of NeuralFoil below Re = 500,000.** 9,100
   clean measurements, 2,600 with a trip strip, 400 lift sweeps through
   stall. It gives a measured error for lift, drag, L/D and maximum lift at
   each Reynolds number. It shows the error is the same in two separate
   labs, and shows what the confidence score can and can't predict.
2. **A split of that error into XFoil's part and the network's part**, by
   running XFoil at every one of the 9,100 conditions.
3. **The noise floor of the experiments themselves**, from the 15 airfoils
   measured in both tunnels, plus a way to pull the wings' build error out
   of the model's error using Princeton's measured coordinates.
4. **Honest statistics.** Error bars that resample whole airfoils, a
   breakdown of where the confidence-versus-error link actually lives, and a
   cross-checked model that turns a prediction into an expected error.
5. **A working design pipeline** for worst-case, multi-objective and
   build-tolerant airfoils, plus a version that uses the measured confidence
   calibration inside the design loop.

---

## 2. Methods

### 2.1 Describing the shape

Every airfoil here is written as 17 numbers: 8 for the top surface, 8 for
the bottom, and 1 for the leading edge. This is the Kulfan, or CST, basis
(Kulfan, 2008), with the trailing edge fixed at 0.25 percent of the chord.
It's smooth, it uses few numbers, it always gives a shape you could build,
and it plays well with calculus. That last part is what a gradient-based
optimizer needs.

### 2.2 Getting lift and drag

Lift, drag and moment all come from NeuralFoil's `large` network, with the
bigger `xxlarge` run as a check. It returns CL, CD, CM, a confidence score
between 0 and 1, and the state of the boundary layer. Because it's
differentiable, the exact effect of each of the 17 shape numbers on lift and
drag is available for free. That's what turns an optimization over a
25-point grid of conditions into a few seconds of work.

### 2.3 Chasing the worst case

The design goal is the worst lift-to-drag ratio anywhere in the operating
range, not the average. That sounds awkward to optimize, because "the worst"
isn't a smooth function. The fix is a standard trick. Add one extra number,
call it *g*, and force it to sit at or below the L/D at every grid point.
Then push *g* as high as it will go. Since *g* can never rise above the
worst point, pushing it up drags the worst case up with it, and the whole
thing stays smooth enough to solve quickly.

The grid is five Reynolds numbers, spaced evenly on a log scale across a
decade, by five angles of attack. The problem has more than one hilltop, so
each design is solved from several NACA starting shapes and only the best is
kept. Families of designs are traced by starting each solve from the
previous answer. Everything runs through AeroSandbox and IPOPT.

### 2.4 Four goals at once

Four real goals are combined into one weighted score. Each is scaled first,
so the weights can be compared fairly.

| Goal | What gets measured | Direction |
|-----------|--------|-----------|
| Efficiency | worst-case L/D across the range | higher is better |
| Safety | lift held at 12 degrees, Re = 50,000 | higher is better |
| Structure | maximum thickness, meaning spar depth | higher is better |
| Noise | trailing-edge boundary-layer thickness | lower is better |

### 2.5 Designing for a wing built badly

No real wing matches its drawing, especially a 3D-printed or hand-built one.
So small random bumps are added to the 17 shape numbers, sized to about 0.5
percent of the chord in surface error. One fixed set of these errors is
drawn once, and the worst case is taken across both the operating grid and
that set. Then the finished design is tested against a fresh, larger batch
of errors it never saw during optimization.

### 2.6 The wind-tunnel data

**UIUC.** The measurements come from the UIUC Low-Speed Airfoil Tests
archive (Selig et al., 1995, 1996; Lyon et al., 1998; Selig and McGranahan,
2004), which publishes its tables as plain text. Every airfoil in those four
volumes with coordinates in the AeroSandbox database is included, in its
plain form, since the surrogate can't represent flaps or gurney flaps. That
gives 57 airfoils, 91 files and 5,441 drag-run points at Re = 40,000 to
500,000, plus 21,000 lift-run points that sweep through stall and back.
Clean runs and trip-strip runs live in separate files and are kept separate
here. A parser turns it all into one table, so every number traces back to
its source file.

For 55 of the 57 airfoils, the 17-number description reproduces the
published shape to better than 0.5 percent of the chord, which is the same
build-error scale used elsewhere in this paper. The two that don't, A18 and
BE50 at 0.9 to 1.3 percent, are reported but left out of the statistics.
Their disagreement with experiment can't be separated from their shape
error. The E387 comes out at 0.15 percent. Even that little matters: run
XFoil on the true E387 and on its 17-number version and lift shifts by about
3 percent, drag by 4 to 7 percent. That's roughly half the E387 lift error
and a third of the drag error in section 3.1. So the errors here are what a
designer using this pipeline actually gets, and an upper bound on the
network alone.

NeuralFoil was run at every measured condition. Clean runs got free
transition with n_crit = 9. Trip-strip runs got run twice, once free and
once with transition forced at the trip positions in the file header.

**Princeton.** The whole test then repeats on *Airfoils at Low Speeds*
(Selig, Donovan and Fraser, 1989). Those runs happened in Princeton's 3 by 4
foot smoke tunnel between 1986 and 1989, and the same archive shares them as
plain text.
It holds 127 airfoil-and-setup blocks (7,762 drag points) and 135 lift files
(15,565 points) on 54 airfoils at Re = 60,000 to 300,000. Clean and
trip-strip blocks were kept. Flaps, gurney flaps, blowing, clay leading
edges and mixed setups were dropped. This set has one thing UIUC doesn't:
somebody measured the actual wind-tunnel models with a profiler and
published those coordinates next to the design ones. So NeuralFoil was run
twice on every model, on the drawing and on the real thing. All 68 models
fit the 17-number description to better than 0.07 percent of chord. The
tunnel's turbulence level isn't recorded anywhere, so n_crit was swept from
5 to 11 on this data; 9 came out best.

**What the sample really is.** That headline of 94 airfoils is a union: 55
UIUC plus 54 Princeton minus the 15 that appear in both. The 68 Princeton
models include several sections built twice by different builders, among
them the E387, E374, E205, E214, S2091, S3021, S4061, HQ 2/9 and FX 63-137,
plus three repeat runs of the same model. The 9,130 clean points sit inside
667 polars, 356 from UIUC and 311 from Princeton, and every point in one
polar is a single angle sweep on a single model at a single speed. Those
points are not 9,130 independent facts. The independent unit is the airfoil,
and section 3.4 treats it that way.

### 2.7 Splitting XFoil's error from the network's

NeuralFoil copies XFoil. So when it disagrees with a wind tunnel, two
separate errors are stacked: XFoil's disagreement with the tunnel, plus the
network's disagreement with XFoil. Only the second one is really the
network's.

To separate them, a headless XFoil 6.99 was run at all 9,130 clean
conditions. It got exactly the 17-number shape NeuralFoil saw, run viscous
with n_crit = 9 and free transition, at the measured speed and angles. Each polar
sweeps outward from 0 degrees in half-degree steps with the measured angles
slotted in, so every point starts from its neighbour's answer. XFoil's
viscous solver needs that warm start at these speeds. Points where it failed
to converge are recorded as missing and counted. The three comparisons,
NeuralFoil against tunnel, XFoil against tunnel, and NeuralFoil against
XFoil, are then made on the points that did converge.

### 2.8 Statistics

**Honest error bars.** Every headline number gets a 95 percent interval from
a bootstrap that resamples airfoils, not points. Each airfoil comes with all
of its points attached, 2,000 times over. The naive point bootstrap runs
alongside so the two can be compared. In the pooled version, an airfoil
tested in both tunnels counts as two clusters, because those really are two
different physical models in two different labs.

**Where a correlation lives.** The link between confidence and drag error
gets split three ways. First, the correlation between airfoil averages.
Second, what's left after subtracting each airfoil's average from both
sides. Third, the same split done at the level of single polars.

**A fitted error model.** Expected drag error is fitted as a Gamma model
with a log link, solved by iteratively reweighted least squares. Its inputs
are log10(1.001 minus confidence), log10(Re / 100,000), angle of attack and
its square, maximum camber and maximum thickness. Every number reported for
it is out of sample. That means 10-fold cross-validation with whole airfoils
held out, where an airfoil in both tunnels is held out of both at once, plus
training on one tunnel and testing on the other in both directions. The
baselines are a flat constant and the six-bin confidence table from section
3.1, both refitted on each training fold.

---

## 3. Results

### 3.1 The test against the UIUC tunnel

**Start with the reference airfoil.** The Eppler E387 has been measured in
more tunnels than any other slow-speed section, so it's the natural place to
begin. Against its 112 clean measurements at six speeds, with errors given
as absolute values averaged, and bias as the signed average:

| Re | Lift error (ΔCL) | Lift error, relative | Drag error (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.057 | 11% | 22 | −5 | 0.88 |
| 200,000 | 0.030 | 5.4% | 12 | 0 | 0.82 |
| 300,000 | 0.025 | 4.9% | 12 | −1 | 0.81 |
| 350,000 | 0.024 | 4.7% | 12 | −3 | 0.81 |
| 460,000 | 0.023 | 5.1% | 12 | −4 | 0.81 |
| 500,000 | 0.023 | 5.1% | 12 | −5 | 0.81 |

Lift is good to about 5 percent from Re = 200,000 up, and 11 percent at Re =
100,000. Drag is the harder one. It sits near 12 percent from Re = 200,000
up and jumps to 22 percent at Re = 100,000, where the separation bubble
parks itself right in the middle of the low-drag range. There's no big
one-way drag bias for the E387, just 3 percent low overall. Look at the drag
curves and you'll see both NeuralFoil and XFoil missing that bubble drag.

**Now all 55 airfoils.** The E387 is a well-behaved shape. Across the full
benchmark set, 4,428 clean measurements, the picture widens:

| Re | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 565 | 0.084 | 22 | 15 | +13 | 0.92 |
| 100,000 | 1,105 | 0.085 | 14 | 10 | +2 | 0.91 |
| 200,000 | 1,183 | 0.071 | 10 | 7 | 0 | 0.91 |
| 300,000 to 400,000 | 1,187 | 0.061 | 10 | 7 | −4 | 0.91 |
| 400,000 to 500,000 | 388 | 0.049 | 9 | 7 | −7 | 0.95 |
| **All** | **4,428** | **0.072** | **12** | **8** | **0** | **0.91** |

The three coefficients behave differently.

1. **Drag is fine from Re = 100,000 up, and gets worse below that.** The
   typical drag error is 7 percent from Re = 200,000 up, 10 percent at
   Re = 100,000, and 15 percent at Re = 60,000, where NeuralFoil reads about
   13 percent high. The bias flips sign along the way: high at the slow end,
   about 7 percent low at the fast end, near zero overall. The bigger
   `xxlarge` network is no better than `large`.
2. **The lift error hides in the very cambered shapes.** That average of
   0.072 splits in two. It's 0.066 for the 45 airfoils under 5 percent
   camber and 0.113 for the ten above it, including the S1223, S1210, FX
   63-137 and NACA 6409. Those are the high-lift shapes sitting near the
   edge of what XFoil handles well.
3. **L/D is worse than either piece, because it divides one by the other.**
   Across the 3,151 clean points with CL above 0.2, NeuralFoil's L/D is off
   by 15 percent in a typical case and 21 percent on average. And the error
   leans one way. NeuralFoil reads 15 percent high on average, worst at
   Re = 100,000 where it reads 20 percent high. Every L/D chart in this
   paper is drawn with that band on it, and every L/D number quoted anywhere
   here should be read as optimistic.

**Trip strips.** Fifteen of the benchmark airfoils were also run with a
zigzag trip near the leading edge on both surfaces. A trip forces the
boundary layer turbulent early, the way dirt or insects would on a real
blade. Against those 885 tripped measurements, NeuralFoil run normally reads
15 percent low on drag, which is exactly what it should do, since nothing
told it a trip was there. Run it again with transition forced at the trip
positions and the bias drops to 6 percent while the lift error falls by a
quarter. On the three airfoils from the 2004 report, where the trip is
documented most carefully, it falls from 19 percent to 2 percent and the
lift error halves. So NeuralFoil's transition inputs do what they say, which
had not been checked against experiment either.

**Through the stall.** The archive's lift runs sweep out to 15 to 24
degrees, past maximum lift, stepping up and then back down again. That
gives 298 clean runs on the 55 benchmark airfoils, at Re = 30,000 to
500,000. Against them, NeuralFoil reads maximum lift 0.06 high on average,
about 6 percent. It puts the stall about 1 degree early, and within 1.5
degrees in a typical run. Past the stall its lift error nearly doubles,
from 0.08 to 0.15, while its confidence falls from 0.89 to 0.52. So the
warning light works there too. Pitching moment is rougher, off by 0.018
against typical values near 0.08. The worst offenders are the S1223,
USNPS4, SD7080 and PT40, each read 0.15 to 0.17 high on maximum lift. One
thing no steady model can reproduce is stall hysteresis, the gap between the
lift going up and the lift coming back down. It's small in a typical run at
0.05, but it passes 0.4 in the worst tenth, and the lift curves show how far
the two branches can differ.

**What the confidence score actually tells you.** Sort the 3,995
attached-flow benchmark points into bins by confidence:

| Confidence | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) |
|---|---:|---:|---:|---:|
| below 0.5 | 222 | 0.069 | 32 | 27 |
| 0.5 to 0.7 | 102 | 0.085 | 31 | 27 |
| 0.7 to 0.8 | 62 | 0.096 | 26 | 20 |
| 0.8 to 0.9 | 157 | 0.074 | 21 | 17 |
| 0.9 to 0.95 | 387 | 0.082 | 15 | 11 |
| above 0.95 | 3,065 | 0.071 | 9 | 7 |

It's a drag warning and nothing else. Drag error slides steadily from 32
percent down to 9 percent as confidence rises, with r = −0.43 across all
points, the same sign in every speed band and at positive and negative
angles alike. At the airfoil level, the shapes NeuralFoil is least sure
about are the ones it gets most wrong, with a rank correlation of −0.65.
Lift error, meanwhile, is flat across every bin, and its correlation with
confidence is exactly zero. So when NeuralFoil says it isn't sure, believe
it about drag, and therefore about L/D, but not about lift. Section 3.4 puts
proper error bars on these correlations and section 3.3 works out what the
score is really detecting. Section 3.8 uses the score inside the optimizer.

### 3.2 The same test in a second wind tunnel

Testing against one lab can't separate the model's error from that lab's
error. So everything in section 3.1 was run again on the Princeton data: 54
airfoils, 68 wind-tunnel models, 4,702 clean measurements at Re = 60,000 to
300,000. Different decade, different tunnel, different instruments,
different people building the models. Here's what came back, with NeuralFoil
run on each model's measured shape:

| Re | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 759 | 0.101 | 17 | 13 | +12 | 0.96 |
| 100,000 | 1,018 | 0.092 | 12 | 8 | +3 | 0.95 |
| 150,000 | 810 | 0.082 | 10 | 7 | +1 | 0.95 |
| 200,000 | 1,157 | 0.081 | 9 | 6 | −2 | 0.95 |
| 300,000 | 958 | 0.077 | 9 | 7 | −4 | 0.96 |
| **All** | **4,702** | **0.086** | **11** | **8** | **+1** | **0.96** |

**The UIUC numbers hold up.** Drag is off by 11 percent on average and 8
percent in a typical case, against 12 and 8 at UIUC. It degrades toward
Re = 60,000 the same way, and the bias flips sign the same way. That flip
runs from 12 percent high at Re = 60,000 to 4 percent low at 300,000, next
to 13 percent high and 7 percent low at UIUC. L/D is read 15 percent high with a
typical error of 15 percent and an average error of 21 percent, which is the
same three numbers as the UIUC set. Lift is a bit worse, 0.086 against
0.072, and leans the same way, about 0.04 to 0.05 high in both tunnels.
`xxlarge` is again no better. Dropping n_crit to 7 kills the over-prediction
at Re = 60,000 but adds a 5 percent under-prediction everywhere else, and 11
doubles the error.

**Two tunnels disagree with each other by about as much as NeuralFoil
disagrees with either one.** Fifteen airfoils were measured in both. Pairing
their clean runs at matching speeds and angles gives 131 polar pairs and
2,139 matched points:

| Re | Points | Airfoils | UIUC vs Princeton, drag (%) | NeuralFoil vs UIUC (%) | NeuralFoil vs Princeton (%) | UIUC vs Princeton, lift (ΔCL) |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 340 | 11 | 18 | 18 | 17 | 0.049 |
| 100,000 | 519 | 15 | 16 | 14 | 12 | 0.048 |
| 200,000 | 636 | 15 | 9 | 9 | 7 | 0.048 |
| 300,000 | 483 | 13 | 7 | 7 | 7 | 0.051 |
| **All** | **2,139** | **15** | **12** | **11** | **10** | **0.048** |

The two experiments differ on drag by 12 percent, with UIUC reading about 6
percent higher. NeuralFoil differs from each of them by 10 to 11 percent on
those very same points. On 53 percent of them it lands closer to the
Princeton value than the UIUC measurement does. And from Re = 200,000 up,
all three numbers collapse to 7 to 9 percent. At those speeds NeuralFoil's
drag error has hit the repeatability limit of slow-speed wind-tunnel testing
itself, and no model can be shown to beat it without better experiments.
Lift is a different story. The two tunnels agree with each other at 0.048,
better than NeuralFoil agrees with either at 0.066 and 0.080, so the lift
error is the model's.

**Build error hides inside the model error.** For the 56 models with both
sets of coordinates, the built shape differs from the drawing by 0.22
percent of chord in a typical case and up to 0.68 percent for the E387B.
That's exactly the range assumed for the build study in section 2.5. Running
NeuralFoil on the measured shape instead of the drawing drops the drag error
from 12.8 to 11.1 percent and the lift error from 0.091 to 0.081, and it
helps 46 of the 56 models. The shape change on its own moves NeuralFoil's
drag by 5 percent and its lift by 0.03, which is about half the size of what
error remains. For the worst-built model, the E387B, drag error falls from
13 to 8 percent. So part of what looks like model error in every benchmark
of this kind, section 3.1 included, is really the wind-tunnel model not
being the airfoil it was meant to be.

**Trip strips again.** Thirty-four tripped runs on 18 models, 1,744 points,
mostly a single strip on the upper surface somewhere between 20 and 70
percent of chord. That's a different situation from the leading-edge trips
in section 3.1. NeuralFoil run normally shows no drag bias here, because a
strip that far back sits near where the flow would have gone turbulent
anyway. Forcing transition at the strip still helps, dropping drag error
from 13.5 to 11.8 percent and lift error from 0.084 to 0.071. For the strips
closest to the leading edge, at 30 percent of chord or less, drag error
falls from 15 to 13 percent.

**Through the stall again.** Against the 114 clean lift sweeps at Re =
55,000 and up that passed the stall, on 35 airfoils, NeuralFoil reads
maximum lift 0.11 high on average. That's worse than the 0.06 of the UIUC
set. It places the stall within 1.4 degrees in a typical run. Down at the
very slowest speeds, 20,000 to 50,000 on the MB253515, it finds no stall at
all before 20 degrees while the measurement stalls at 12 to 15.

**The confidence score again.** At the extremes it repeats: drag error is 30
percent below a confidence of 0.5 and 10 percent above 0.95, and lift error
stays flat at r = −0.03. Two things don't repeat. The correlation with drag
error is weaker, r = −0.26 against −0.43, and the middle bins sit flat at 22
to 25 percent instead of stepping down. And at the airfoil level, the rank
correlation between an airfoil's average confidence and its average drag
error is −0.08 and not significant, against −0.65 in the UIUC set. Section
3.4 explains why.

### 3.3 So whose error is it, XFoil's or the network's?

XFoil converged at 8,814 of the 9,130 clean conditions, or 96.5 percent. On
those points the three drag comparisons look like this:

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

**It's XFoil's.** The network differs from the solver it copies by 2.8
percent in drag on average, 1.7 percent in a typical case, and by 0.011 in
lift. That's about a quarter of its disagreement with the tunnels. The two
signed errors track each other at r = 0.95, and XFoil's error alone accounts
for 86 percent of the variation in NeuralFoil's. In every speed band of both
tunnels the network's own share stays under 4 percent, including at Re =
60,000 where the total error runs 17 to 22 percent. So the Re = 60,000
over-prediction, the bias flipping sign with speed, and the high-camber lift
error in section 3.1 are all XFoil's behaviour, copied faithfully. XFoil's
own lift error against the tunnels is 0.082, next to NeuralFoil's 0.079.

**And NeuralFoil is slightly closer to reality than XFoil is.** On 55
percent of points the network lands nearer the measurement. Its average drag
error is lower in every band too: 11.2 against 12.1 percent overall, and
12.5 against 14.9 at Re = 100,000 in the UIUC tunnel. That makes sense. Fit a
network to hundreds of thousands of XFoil polars and you smooth out XFoil's
point-to-point scatter, and some of that scatter is error. The 3.5 percent
of conditions where XFoil wouldn't converge are worth a look too.
NeuralFoil's average drag error there is 25 percent and its average
confidence 0.57, against 11 percent and 0.95 where XFoil did converge. XFoil
converges at only 56 percent of the conditions where NeuralFoil's confidence
is below 0.5, and 99 percent where it's above 0.95.

**What the confidence score is really detecting.** Sort the converged
attached-flow points by confidence and show all three errors at once:

| Confidence | Points | NeuralFoil vs tunnel (%) | XFoil vs tunnel (%) | NeuralFoil vs XFoil (%) | Lift, NeuralFoil vs XFoil (ΔCL) |
|---|---:|---:|---:|---:|---:|
| below 0.5 | 158 | 32 | 41 | 10 | 0.034 |
| 0.5 to 0.7 | 98 | 28 | 34 | 10 | 0.029 |
| 0.7 to 0.8 | 79 | 26 | 32 | 10 | 0.030 |
| 0.8 to 0.9 | 213 | 21 | 27 | 8 | 0.021 |
| 0.9 to 0.95 | 631 | 15 | 17 | 5 | 0.017 |
| above 0.95 | 6,742 | 9 | 10 | 2 | 0.009 |

The score was trained to report how unsure the network is about XFoil, and
it does that job: the copying error climbs from 2 to 10 percent as
confidence drops. But look closer. It tracks XFoil's error against the
tunnel more tightly, at r = −0.40, than it tracks the network's error
against XFoil at r = −0.33, or against the tunnel at r = −0.33. Low
confidence marks the conditions where XFoil's answer is shaky or missing
altogether, meaning separation bubbles and post-stall flow. Those are the
same conditions where XFoil's physics is wrong. So the score warns about the
physics as well as the copying, which is more than it was built to do, and
it's the reason it works as a stand-in for real drag error. Section 3.8
relies on that.

### 3.4 How sure are these numbers?

**Honest error bars.** Resample airfoils instead of points and the 95
percent intervals get two to five times wider. Every headline result still
survives:

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

Brackets are the airfoil-resampled 95 percent intervals. Read them and a few
things become clear. The overall drag bias can't be told apart from zero in
either tunnel. The L/D over-prediction can. The drag correlation with
confidence is firmly negative in both tunnels, and the lift correlation
includes zero in both. Only one statistic really fails to repeat, and that's
the airfoil-level rank correlation, whose Princeton interval straddles zero.

**Why that one doesn't repeat.** Split the confidence-versus-drag-error link
by level and the answer shows up:

| Level | UIUC | Princeton |
|---|---:|---:|
| All points | −0.43 | −0.27 |
| Between airfoils (airfoil averages) | −0.69 | −0.07 |
| Within airfoils | −0.40 | −0.28 |
| Between polars (polar averages) | −0.37 | −0.07 |
| Within a polar (along the angle sweep) | −0.45 | −0.31 |
| Share of confidence spread between airfoils | 26% | 7% |

In the UIUC set the confidence score moves both from airfoil to airfoil and
along each polar, and both movements predict drag error. In the Princeton
set, 93 percent of the score's spread sits inside airfoils rather than
between them, because nearly every Princeton model sits above 0.95. There's
simply nothing left between airfoils to correlate. What remains lives
entirely along the angle sweep, where confidence sags toward the stall while
drag error climbs. So the point-level result is real in both sets. The
airfoil-level one is a feature of the UIUC sample, which happens to include
shapes NeuralFoil is unsure about, and is not a general property of the
score itself.

**A fitted error model.** Take the 8,211 attached-flow points from both
tunnels and fit expected drag error against confidence, Reynolds number,
angle of attack, camber and thickness. Each decade of (1 minus confidence)
multiplies the expected error by 1.87. Each decade of Reynolds number
multiplies it by 0.43. Each percent of camber multiplies it by 1.035.
Thickness does nothing at all.

Out of sample, with whole airfoils held out, the model predicts the drag
error with a mean absolute error of 0.074. A flat constant gets 0.087 and
the six-bin confidence table gets 0.079, so the fit beats the constant by 14
percent and the table by 7. Its rank correlation with what actually happened
is 0.39. Its 80th and 95th percentile bands cover 80 and 95 percent of
held-out points, which is what they're supposed to do. Train it on UIUC
alone and test on Princeton and it scores 0.071 against 0.085 for a
constant; go the other way and it's 0.077 against 0.089. Its held-out
calibration tracks the diagonal from a predicted 6 percent, where the truth
was 7, up to a predicted 29 percent, where the truth was 29.

Here's what that means in practice. Take an ordinary 3 percent camber
section at 4 degrees. The model expects 7 percent drag error at confidence
0.98 and Re = 200,000, with an 80th percentile of 11 percent. Slow it to
Re = 60,000 and that becomes 10 percent. Drop confidence to 0.90 at Re =
100,000 and it's 13 percent. Drop it to 0.16 at Re = 200,000, which is
exactly where the unconstrained optimizer of section 3.8 wants to go, and
it's 18 percent. The coefficients ship with the code, so anyone can run the
model without this paper.

### 3.5 Peak performance against a wing that holds up everywhere

| Airfoil | Goal | Peak L/D | Worst-case L/D | Max thickness |
|---------|-----------|---------:|---------------:|--------------:|
| A (single-point) | best L/D at one condition | 232.9 † | 7.1 | 13% |
| B (robust) | best worst-case L/D | 105.1 | **38.2** | 9% |

Airfoil B gives up about 55 percent of A's headline peak and gets back a
**5.4 times higher worst case**. A is a spike that collapses the moment
conditions move. B is a broad plateau that works nearly everywhere.

† A's peak comes from a condition where NeuralFoil reports almost no
confidence, so 233 is an optimistic ceiling and not a real number. The
fitted model in section 3.4 puts the expected drag error there near 18
percent, and section 3.1 shows L/D reads 15 percent high even where
confidence is good. Airfoil A sits on a razor-thin peak.

### 3.6 Four goals at once

| Setting | Worst-case L/D | Stall angle | Max thickness | Noise proxy |
|---------|---------------:|------------:|--------------:|------------:|
| Efficiency-first | **38.5** | 9.0 degrees | 8.0% | 15.1 × 10⁻³ |
| Balanced | 33.9 | 10.0 degrees | 10.4% | 11.9 × 10⁻³ |
| Community-first | 17.8 | **11.5 degrees** | **13.7%** | **11.1 × 10⁻³** |

Structural depth and slow-speed efficiency pull hard against each other.
Making the section thick enough for a deep spar, 13.7 percent, costs about
54 percent of its worst-case L/D. What it buys is a 2.5 degree bigger margin
before the stall and a thinner boundary layer at the trailing edge.

### 3.7 Does the robust wing survive being built badly?

| Design | On paper, worst-case L/D | As built, average | As built, worst 5% |
|--------|---------------------------:|------------:|----------------------:|
| B, robust to conditions only | 34.3 | 20.3 | 9.3 |
| B, robust to conditions and build error | 33.7 | 26.5 | **19.5** |

On paper these two look almost identical. Add realistic build error, tested
on shapes the optimizer never saw, and the design that expected the error
keeps more than double the reliable worst-case performance. The cost on
paper is almost nothing.

### 3.8 Putting the measured confidence inside the optimizer

Section 3.1 showed the confidence score is a reliable warning about drag
error. Section 3.3 showed the warning is mostly about XFoil's physics. And
drag is what carries the L/D error. So put the warning inside the optimizer.

Alongside the worst-case L/D, the optimizer gets a second reward for landing
where NeuralFoil is confident, controlled by one dial called `w_conf`. The
grid, 5 by 5, and the three starting shapes match the robust airfoil B from
section 3.5. The thickness range is a little wider, 8 to 16 percent instead
of 9 to 13, and each dial setting starts from the previous one.

| `w_conf` | Worst-case L/D | Mean confidence | Lowest confidence | Drag error expected there (section 3.1) |
|---:|---:|---:|---:|---:|
| 0 (trust the model blindly) | 38.5 | 0.16 | 0.01 | about 32% |
| 0.5 | 37.7 | 0.96 | 0.89 | about 9% |
| 1 | 37.7 | 0.96 | 0.90 | about 9% |
| 2 | 37.5 | 0.96 | 0.92 | about 9% |
| 4 | 36.7 | 0.97 | 0.93 | about 9% |
| 8 | 35.2 | 0.98 | 0.94 | about 9% |

With the dial off, the optimizer finds airfoil B again: a predicted
worst-case L/D of 38.5 against B's 38.2, at a mean confidence of 0.16
against B's 0.16. At that confidence the benchmark says NeuralFoil's drag is
wrong by about a third on average. So 38.5 is a number the model itself
won't stand behind. The optimizer went there because, as far as it can tell,
a whole family of shapes gives nearly the same worst-case L/D. Some sit
inside the region where the model was checked and some sit well outside it.
With nothing to break the tie, it picked one outside.

A small nudge on the dial, `w_conf` = 0.5, breaks the tie. The design moves
to a mean confidence of 0.96, where the measured drag error is about 9
percent, and the predicted worst-case L/D drops 2 percent, from 38.5 to
37.7. Push harder and you pay steadily more for a little more confidence: at
`w_conf` = 8 the predicted L/D is 35.2. The shapes themselves barely change,
and the two extremes are shown in the figure.

Two cautions. That "expected drag error" column is a population statistic
from 94 other airfoils, not a measurement of these designs, and none of them
have been tested. And as section 1.4 says plainly, penalizing a surrogate's
uncertainty inside an optimizer isn't new; trust-region methods and
Kriging-based search rules have done it for decades. What's specific here is
that the penalty has a measured scale. The confidence score has been
calibrated against 9,100 wind-tunnel points, including the split that shows
it flags XFoil's physics error and not just the network's. And the
experiment shows something worth knowing on its own: trusting the surrogate
blindly lands you, by default, in the region where it's known to be least
reliable, and leaving that region is nearly free.

---

## 4. Discussion

Know when to trust the stand-in, and know what to trust it about. From
Re = 100,000 up, NeuralFoil's drag lands within 7 to 10 percent of
measurement and its lift within a few hundredths for ordinary shapes. That
holds in two separate tunnels. From Re = 200,000 up its drag error is no
bigger than the disagreement between those tunnels, so at those speeds it
has reached the limit of what the experiments can check. Below Re = 100,000,
for very cambered high-lift shapes, and for L/D in general, it turns
optimistic.

The split changes what all that means. NeuralFoil is a faithful copy of
XFoil, within 3 percent on drag and 0.01 on lift, and it's marginally closer
to reality than XFoil itself. The 11 to 12 percent drag error, the Re =
60,000 over-prediction and the high-camber lift error are XFoil's physics,
inherited whole. So anyone who wants better slow-speed drag than this won't
get it from a bigger network trained on more XFoil. They'll need better
physics to train on, or a correction from experiment.

The confidence score, meanwhile, is more useful than it was designed to be.
It flags the conditions where XFoil's answer is shaky, and those are the
same conditions where XFoil's physics is wrong. A designer can watch it as a
live warning light for drag and L/D. Better still, run the fitted model from
section 3.4, then send only the doubtful, high-value designs off for slower
analysis or a real tunnel. If the blade will fly rough or dirty, use the
transition inputs, because they work. And when a design finally gets built,
measure the built shape and run it again. A few tenths of a percent of chord
in build error moves the drag prediction by about 5 percent.

Two lessons here generalize. Points inside one polar are not independent,
and treating them as if they were understates the uncertainty by a factor of
two to five. And a correlation measured at the point level can come from
completely different structures underneath. The UIUC and Princeton sets
reach the same conclusion about the confidence score for different reasons,
and only the breakdown in section 3.4 shows that the airfoil-level version
belonged to one sample.

The design results point the same way from another angle. Designs picked for
robustness, whether against changing conditions or against build error, tend
to keep the flow attached instead of separated. And the confidence-aware
optimizer shows that staying inside NeuralFoil's checked region costs almost
nothing in predicted performance.

---

## 5. Limitations

- **Most of this is a validation and framework study, not a discovery.** The
  shape description, the surrogate and the base optimization methods were
  all built by other people, and as section 1.4 sets out, robust design and
  uncertainty-penalized optimization are established fields. What's new is
  the two-tunnel benchmark, the XFoil split, the experimental noise floor,
  the build-error separation, the honest statistics and error model, and the
  measured meaning of the confidence score. Any specific optimized shape
  still needs higher-fidelity or physical confirmation.
- **Every L/D number here is an estimate**, with a typical measured error of
  15 percent and an average over-prediction of 15 percent. Trust the
  relative conclusions, like robust beating peak, not the raw values.
- **The sample is 94 airfoils, not 9,100 independent points** (section 2.6),
  and it covers their plain form only. Flapped and gurney-flap versions and
  UIUC Volume 5 aren't in it. Two UIUC airfoils were dropped because the
  17-number description couldn't reproduce them. The Princeton data are
  older, that tunnel's turbulence level isn't recorded, and the two tunnels
  disagree with each other by 12 percent in drag, which bounds every number
  in this paper. Where only design coordinates exist, which is all of the
  UIUC set, part of every reported error belongs to whoever built the model.
  The Princeton comparison puts that part at roughly 2 percentage points of
  drag error. Part belongs to the shape description too. These are pipeline
  errors, not pure network errors.
- **The XFoil split covers converged points only**, 96.5 percent of them.
  XFoil's failures cluster where NeuralFoil's confidence is low, so the
  split is least complete exactly where the errors are biggest.
- **The fitted error model is descriptive.** It explains a modest share of
  the point-to-point variation, with a rank correlation of 0.39 out of
  sample. Its value is in the calibrated average and the percentile bands,
  not in nailing individual points.
- **The "expected drag error" attached to the optimized designs is a
  population statistic**, not a measurement of those designs.
- **This is two-dimensional, steady analysis only.** No three-dimensional
  wing, no spinning propeller or turbine, no unsteady gusts.
- **None of the optimized airfoils have been physically tested.**

---

## 6. Conclusion

This paper tests NeuralFoil against 9,100 wind-tunnel measurements on 94
airfoils in two separate tunnels below Re = 500,000, runs XFoil at every one
of those conditions, and puts an honest error bar on every number.

The stand-in is accurate on drag from Re = 100,000 up, optimistic below that
and for high-lift shapes, and the same in both labs. From Re = 200,000 up
its drag error is no bigger than the disagreement between those labs. Almost
all of its error is XFoil's, inherited faithfully; the network itself adds
about 3 percent on drag and 0.01 on lift, and lands slightly closer to
reality than the solver it copies. Its confidence score is a usable warning
about drag error, and about XFoil's physics error, but not about lift, and
the airfoil-level version of that result belongs to one sample. A fitted
model turns confidence, Reynolds number, angle and camber into an expected
drag error that carries over between tunnels. The transition inputs
reproduce trip-strip measurements. Running the model on a wing's measured
shape instead of its drawing removes about a sixth of the drag error. And
fed back into the optimizer with its measured meaning, the confidence score
moves designs out of the least reliable region for about 2 percent of
predicted performance.

Three things would push this further. A third tunnel, with documented
turbulence and flapped configurations. A training set for the surrogate that
includes experimental or higher-fidelity slow-speed data, since the split
here shows more XFoil won't help. And a real wind-tunnel test of 3D-printed
robust and peak-tuned airfoils, with their built shapes measured, which
would turn this validation study into an original experimental result.

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

*Reproducibility: all code, configuration files, extracted data as CSV, the
fitted error model as JSON, and every figure are in the project repository.
Software versions are pinned and random seeds are fixed. Every study writes
its data to disk before making any plots, and a test suite pins every number
quoted in this paper to those data files.*
