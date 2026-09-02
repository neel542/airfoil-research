# Robust, multi-objective airfoil design at low Reynolds number, with a 55-airfoil wind-tunnel benchmark of the NeuralFoil surrogate below Re = 500,000

**Author:** Neel Madhav
**Date:** 2026

---

## Abstract

Small drones and little wind turbines fly slowly. Flying slowly puts them in
what engineers call a **low Reynolds number** range (Re is roughly 50,000 to
500,000). The air behaves oddly there: a thin layer near the wing can peel off
and curl into a bubble, and that bubble is where most of the drag comes from. A
wing tuned for one exact speed can also fail badly the moment conditions change
even slightly.

To design wings fast, people now use quick neural stand-ins instead of slow flow
solvers. The best-known one is called **NeuralFoil**. It is fast and it plugs
directly into an optimizer. But there was a gap: NeuralFoil had only ever been
checked against real wind-tunnel measurements at one high speed
(Re = 1,800,000), far above where small drones and turbines actually fly.

This project does three things. First, it builds a pipeline that designs wings
for good *worst-case* performance across a whole range of conditions, balancing
four goals at once: efficiency, safety, structural strength, and quietness.
Second, it makes the wing hold up even when it is built slightly wrong, the way
a real 3D-printed part would be. Third, and most important, it benchmarks
NeuralFoil against **4,400 measured wind-tunnel points on 55 airfoils** from
the public UIUC low-speed airfoil archive, at Re = 60,000 to 500,000, plus 900
points with a boundary-layer trip and 300 lift sweeps through stall.

The result: NeuralFoil predicts lift to within about 0.07 in lift coefficient
and drag to within about 12 percent on average (8 percent in the typical case),
with drag error rising to about 22 percent at Re = 60,000. Maximum lift is
over-predicted by about 6 percent and the stall angle is placed within about
1.5 degrees. Modelling a trip strip by forcing transition at its location
removes most of an otherwise large drag bias. The most useful finding
concerns NeuralFoil's own confidence score: it tracks **drag** error closely
(from 9 percent error when confidence is above 0.95 to 32 percent when it is
below 0.5) but says nothing about lift error. This project uses that score
inside the optimizer, and shows that a small confidence reward moves the
design out of the region where the surrogate's drag is least reliable for a
cost of about 2 percent in predicted worst-case L/D.

---

## 1. Introduction

### 1.1 Why this matters

A drone that delivers medicine, a small hobby plane, and a rooftop wind turbine
all share one quiet problem. They move slowly, and at slow speed air behaves
very differently than it does around a full-size airplane. This is the **low
Reynolds number** regime. Here, a thin layer of slow air near the wing's surface
can lift off the surface and roll up into a **laminar separation bubble**. That
bubble adds drag, and it can make the wing stall without warning. Designing good
wings in this regime is genuinely hard, and it is worth doing well, because
these are real machines that do useful jobs for people.

### 1.2 One setting is not enough

A wing tuned for one exact speed and angle can perform poorly the moment
conditions shift even a little. Real drones and turbines are always changing
speed and angle as they fly, so a wing that is only good at one setting is close
to useless in practice. What you actually want is **robustness**: a wing that
performs reasonably well across the whole range of conditions it will really
face, not a wing that is perfect at one point and bad everywhere else.

### 1.3 A fast stand-in nobody had fully checked

To find a good wing shape, you need to test thousands of candidate shapes, and
running a full physics solver on each one takes too long. So the field turned to
**NeuralFoil** (Sharpe & Hansman, 2025), a neural network trained to copy an
older solver called XFoil almost instantly. NeuralFoil is also
**differentiable**, meaning the optimizer can tell exactly how a small change in
shape will change performance, which is what makes fast optimization possible at
all.

Here is the catch. NeuralFoil's only published comparison to real wind-tunnel
data was at Re = 1,800,000, roughly ten times higher than the speed small drones
actually fly at. Nobody had checked whether it is trustworthy in the slow,
messy, low-Re range where it is used the most. That is the gap this project
closes.

### 1.4 What this project contributes

1. A repeatable, easy-to-reconfigure pipeline that designs wings for good
   **worst-case** performance across a whole range of speeds and angles.
2. A **multi-objective** version that balances efficiency against safety
   (stall margin), structural strength (spar depth), and quietness.
3. **Manufacturing-tolerance robustness**: wings that keep performing even when
   built imperfectly, tested on error patterns the optimizer never saw.
4. **A 55-airfoil wind-tunnel benchmark of NeuralFoil below Re = 500,000**,
   built from the official UIUC low-speed airfoil archive: 4,400 clean
   measurements, 900 with a boundary-layer trip, and 300 lift sweeps through
   stall. It gives a measured error bar for lift, drag, L/D and maximum lift
   by Reynolds number, and shows what NeuralFoil's confidence score does and
   does not predict.
5. **An uncertainty-aware optimizer** that uses that measured reliability
   inside the design process itself, steering the design out of the region
   where the surrogate's drag prediction is least trustworthy.

---

## 2. Background and methods

### 2.1 How the wing shape is described

Every wing shape here is written as just 17 numbers: 8 that shape the top
surface, 8 that shape the bottom surface, and 1 that adjusts the leading edge.
This way of describing a shape is called the Kulfan, or CST, basis. It is
smooth, uses very few numbers, always produces a shape you could actually build,
and, importantly, it plays nicely with calculus: small changes in the numbers
produce smooth, predictable changes in the shape. That last property is exactly
what a gradient-based optimizer needs to work well.

### 2.2 How the airflow is estimated

Lift, drag, and other airflow numbers all come from NeuralFoil (specifically its
`large` model). Because NeuralFoil is differentiable, the optimizer can work out
exactly how a tiny change in shape changes lift and drag, without having to try
many nearby shapes one by one. That turns what would be an hours-long search
over 25 different flying conditions into something that finishes in seconds.

### 2.3 Chasing the worst case, not the average

The goal here is to maximize the *worst-case* lift-to-drag ratio (L/D) across
the whole range of conditions, not just the average. Here is the trick in plain
terms. Imagine you want to raise your weakest subject's grade in school. You
would not average all your grades together; you would push up the lowest one
specifically. The math here does the same thing: it adds a helper number, call
it *g*, that has to stay at or below the L/D at every single flying condition.
Then it pushes *g* as high as it possibly can. Since *g* can never be higher
than the worst point, pushing *g* up pushes the worst case up too, and it does
this with equations smooth enough for the optimizer to solve quickly.

This search space also has more than one "hilltop," meaning the optimizer can
get stuck on a small hill instead of finding the tallest one. To fight that,
each design is solved starting from several different seed shapes, and only the
best result is kept. Whole families of related designs are built by starting
each new solve from the previous one, a trick called continuation.

### 2.4 Balancing four goals at once

Four real, physical goals are combined into one weighted score, with each goal
scaled so the weights are easy to compare fairly:

| Goal | What is measured | Direction |
|-----------|--------|-----------|
| Efficiency | worst-case L/D across the whole range | higher is better |
| Safety | lift kept at a steep 12-degree angle, at low speed | higher is better |
| Structure | maximum wing thickness (room for a strong spar) | higher is better |
| Quietness | a measure of trailing-edge turbulence thickness | lower is better |

### 2.5 Designing for a wing that is built imperfectly

A real wing, especially a 3D-printed or hand-built one, never matches the
design exactly. To account for that, small random bumps are added to the shape
numbers, matching about a half a percent of the wing's chord length in surface
error, which is realistic for a field-built part. A design is optimized to hold
its worst-case L/D across both the range of flying conditions *and* a set of
these random build errors. Then it is tested against a completely fresh, larger
set of build errors the optimizer never saw, to make sure the result is
trustworthy and not just memorized.

### 2.6 Checking NeuralFoil against real measurements

Measured lift and drag polars come from the UIUC Low-Speed Airfoil Tests
archive (Selig et al., 1995, 1996; Lyon et al., 1998; Selig & McGranahan,
2004), which publishes its wind-tunnel tables as plain text files. Every
airfoil in those four volumes whose design coordinates exist in the
AeroSandbox airfoil database was included, in its plain form (no flaps or
gurney flaps, which the surrogate cannot represent): 57 airfoils, 91 data
files, 5,441 drag-run points at Re = 40,000 to 500,000, plus the companion
lift runs (21,000 points) that sweep through stall and back. Clean runs and
runs with a boundary-layer trip are separate files in the archive and are kept
separate here; a few files that mix several trip heights are left out. A
parser script turns the files into one table, so every number is traceable
to its source file and run.

Each airfoil is described with the same 17-number Kulfan basis used for the
designs. For 55 of the 57 airfoils that description reproduces the true shape
to better than 0.5 percent of the chord, the manufacturing-error scale used
elsewhere in this project; the two that do not (A18 and BE50, at 0.9 to 1.3
percent) are reported but left out of the benchmark statistics, because their
disagreement with experiment cannot be separated from the shape error. The
E387, the most-tested low-Re reference airfoil, is reproduced to 0.15 percent.
Even that small a change matters aerodynamically: XFoil run on the true E387
coordinates and on the 17-number version gives lift coefficients about 3
percent apart and drag coefficients 4 to 7 percent apart, which is roughly
half of the E387 lift error and a third of the drag error reported in section
3.4. The errors reported there are therefore what a designer using this
pipeline actually experiences, and an upper bound on the neural network's own
error.

NeuralFoil (the `large` model, with the bigger `xxlarge` as a check) was run
at every measured condition. For clean runs it was run with free transition.
For tripped runs it was run twice: with free transition, and with transition
forced at the trip locations given in the file header (2 percent of chord on
the upper surface and 5 percent on the lower for most tripped runs), to test
whether that input does what it should. For the E387, a
separately built copy of the XFoil solver was also run at the same conditions,
so the surrogate can be compared against the solver it was trained to copy.

---

## 3. Results

### 3.1 Peak performance versus a wing that holds up everywhere

| Wing | Goal | Peak L/D | Worst-case L/D | Max thickness |
|---------|-----------|---------:|---------------:|--------------:|
| A (tuned for one setting) | best L/D at one condition | 232.9 † | 7.1 | 13% |
| B (robust) | best worst-case L/D | 105.1 | **38.2** | 9% |

Wing B gives up about 55 percent of A's headline peak L/D, but in return it gets
a **5.4 times higher worst-case** L/D. Wing A is a razor-thin peak that falls
apart the moment conditions shift away from its one perfect setting. Wing B is a
broad, flat plateau that performs decently almost everywhere.

† See section 3.4. Wing A's peak number is produced at a condition where
NeuralFoil itself reports almost no confidence, so 233 should be read as an
optimistic ceiling, not a real, trustworthy value.

### 3.2 Balancing four goals at once

| Setting | Worst-case L/D | Stall angle | Max thickness | Quietness score |
|---------|---------------:|------------:|--------------:|------------:|
| Efficiency-first | **38.5** | 9.0 degrees | 8.0% | 15.1 × 10⁻³ |
| Balanced | 33.9 | 10.0 degrees | 10.4% | 11.9 × 10⁻³ |
| Community-first | 17.8 | **11.5 degrees** | **13.7%** | **11.1 × 10⁻³** |

Structural strength and low-speed efficiency pull hard against each other.
Forcing the wing thick enough for a strong spar (13.7 percent) costs about 54
percent of its worst-case L/D, but it buys a 2.5-degree bigger safety margin
before stall, and a quieter trailing edge.

### 3.3 Does the robust wing survive being built badly?

| Design | On paper: worst-case L/D | Actually built, average | Actually built, worst 5% of cases |
|--------|---------------------------:|------------:|----------------------:|
| B, robust to conditions only | 34.3 | 20.3 | 9.3 |
| B, robust to conditions and build error | 33.7 | 26.5 | **19.5** |

On paper the two designs look almost identical. But once real build error is
added, the wing that was designed to expect that error keeps **more than
double** the reliable worst-case performance, for almost no cost in the
on-paper numbers.

### 3.4 The main result: checking NeuralFoil against real measurements

**The reference airfoil first.** The Eppler E387 has been measured in more
wind tunnels than any other low-Re section, so it is the natural place to
start. Against its 112 clean measurements at six Reynolds numbers (errors are
absolute values, averaged; bias is the signed average):

| Re | Lift error (ΔCL) | Lift error, relative | Drag error (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|
| 100,000 | 0.057 | 11% | 22 | −5 | 0.88 |
| 200,000 | 0.030 | 5.4% | 12 | 0 | 0.82 |
| 300,000 | 0.025 | 4.9% | 12 | −1 | 0.81 |
| 350,000 | 0.024 | 4.7% | 12 | −3 | 0.81 |
| 460,000 | 0.023 | 5.1% | 12 | −4 | 0.81 |
| 500,000 | 0.023 | 5.1% | 12 | −5 | 0.81 |

Lift is right to about 5 percent from Re = 200,000 up, and to 11 percent at
Re = 100,000. Drag is the harder number: about 12 percent at Re = 200,000 and
above, and 22 percent at Re = 100,000, where the laminar separation bubble
sits in the middle of the low-drag range. The drag charts show both NeuralFoil
and XFoil missing that bubble drag. There is no large systematic drag bias
for the E387: 3 percent low overall.

**Fifty-five airfoils.** The E387 is a well-behaved shape. Across the
55-airfoil benchmark set (4,428 clean measurements), the picture is broader:

| Re | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) | Drag bias (%) | Confidence |
|---:|---:|---:|---:|---:|---:|---:|
| 60,000 | 565 | 0.084 | 22 | 15 | +13 | 0.92 |
| 100,000 | 1,105 | 0.085 | 14 | 10 | +2 | 0.91 |
| 200,000 | 1,183 | 0.071 | 10 | 7 | 0 | 0.91 |
| 300,000 to 400,000 | 1,187 | 0.061 | 10 | 7 | −4 | 0.91 |
| 400,000 to 500,000 | 388 | 0.049 | 9 | 7 | −7 | 0.95 |
| **All** | **4,428** | **0.072** | **12** | **8** | **0** | **0.91** |

Three things stand out.

1. **Drag is good from Re = 100,000 up and degrades below it.** The typical
   (median) drag error is 7 percent from Re = 200,000 up, 10 percent at
   Re = 100,000, and 15 percent at Re = 60,000, where NeuralFoil over-predicts
   drag by about 13 percent on average. The bias changes sign with Reynolds
   number: high at the low end, about 7 percent low at the high end, and
   close to zero overall. The bigger `xxlarge` model is no more accurate
   than `large`.
2. **Lift error is concentrated in highly cambered shapes.** The mean lift
   error of 0.072 hides a split: 0.066 for the 45 airfoils with less than 5
   percent camber, and 0.113 for the ten with more (among them the S1223,
   S1210, FX 63-137 and NACA 6409). Those are the high-lift shapes near the
   edge of what XFoil, and so NeuralFoil, handles well.
3. **Because L/D divides lift by drag, its error is larger than either.**
   Over the 3,151 clean points with a lift coefficient above 0.2, NeuralFoil's
   L/D is off by 15 percent in the typical case (21 percent on average), and
   the error leans one way: NeuralFoil over-predicts L/D by 15 percent on
   average, most strongly at Re = 100,000 (20 percent). Every L/D chart in
   this project is drawn with that band, and quoted L/D values should
   be read as optimistic.

**Trip strips.** Fifteen of the benchmark airfoils were also measured with a
zigzag trip near the leading edge on both surfaces, which forces the boundary
layer turbulent the way roughness or insects would on a real blade. Against
those 885 tripped measurements, NeuralFoil run normally under-predicts drag by
15 percent on average, as it should, since nothing told it about the trip.
Run with transition forced at the trip locations, the bias falls to 6 percent
and the lift error drops by a quarter; on the three airfoils of the 2004
report, where the trip is documented most precisely, it falls from 19 percent
to 2 percent and the lift error halves. NeuralFoil's transition inputs do what
they claim, which had not been checked against experiment either.

**Through stall.** The archive's lift runs sweep to 15 to 24 degrees, past
maximum lift, stepping up and then back down. Against the 298 clean runs on
the 55 benchmark airfoils where the sweep passed the stall (Re = 30,000 to
500,000), NeuralFoil over-predicts maximum lift by 0.06 on average (about 6
percent) and puts the stall about 1 degree early, within 1.5 degrees in the
typical case. Past the stall its lift error nearly doubles (0.08 to 0.15) and
its confidence drops from 0.89 to 0.52, so the warning light works there
too. Its pitching-moment coefficient is rougher: off by 0.018 on average
against typical values near 0.08. The high-lift and thick shapes are again
the outliers, by 0.15 to 0.17 in maximum lift. One thing no steady model can
reproduce is stall hysteresis, the gap between the lift on the way up and on
the way down; it is small in the typical run (0.05) but large at the lowest
Reynolds numbers and on the high-lift shapes, where the two curves can
differ by 0.5 or more.

**What the confidence score actually tells you.** NeuralFoil reports a
confidence number with every prediction. Binning the 3,995 attached-flow
benchmark points by that number:

| Confidence | Points | Lift error (ΔCL) | Drag error, mean (%) | Drag error, median (%) |
|---|---:|---:|---:|---:|
| below 0.5 | 222 | 0.069 | 32 | 27 |
| 0.5 to 0.7 | 102 | 0.085 | 31 | 27 |
| 0.7 to 0.8 | 62 | 0.096 | 26 | 20 |
| 0.8 to 0.9 | 157 | 0.074 | 21 | 17 |
| 0.9 to 0.95 | 387 | 0.082 | 15 | 11 |
| above 0.95 | 3,065 | 0.071 | 9 | 7 |

The score is a drag-error indicator and nothing else. Drag error falls
steadily from 32 percent to 9 percent as confidence rises (correlation
r = −0.43 across all points, with the same sign in every Reynolds-number band
and at both positive and negative angles), and at the airfoil level the
shapes NeuralFoil is least sure about are the ones it gets most wrong (rank
correlation −0.65). Lift error is flat across the bins, and its correlation
with confidence is exactly zero (r = 0.00, with a 95 percent interval of
−0.03 to +0.03). So when NeuralFoil says it is unsure, believe it about drag,
and therefore about L/D, but not about lift. Since it is the drag that
carries the L/D error, this is the useful direction. It is the basis for the
next section.

### 3.5 Using the finding to design better wings

Section 3.4 showed that NeuralFoil's confidence score is a reliable warning
about its drag error, and drag is what carries the L/D error. This section
puts that warning inside the optimizer. Alongside the worst-case L/D, the
optimizer is given a second reward for landing where NeuralFoil is confident,
controlled by one dial, `w_conf`. Everything else (the 5-by-5 optimization
grid, the three starting shapes, the thickness limits) is the same as for the
robust airfoil B in section 3.1, and each dial setting is also started from
the previous one.

| `w_conf` | Worst-case L/D | Mean confidence | Lowest confidence | Drag error expected at that confidence (section 3.4) |
|---:|---:|---:|---:|---:|
| 0 (trust the model blindly) | 38.5 | 0.16 | 0.01 | about 32% |
| 0.5 | 37.7 | 0.96 | 0.89 | about 9% |
| 1 | 37.7 | 0.96 | 0.90 | about 9% |
| 2 | 37.5 | 0.96 | 0.92 | about 9% |
| 4 | 36.7 | 0.97 | 0.93 | about 9% |
| 8 | 35.2 | 0.98 | 0.94 | about 9% |

With the dial off, the optimizer finds the same design as airfoil B: a
predicted worst-case L/D of 38.5, at a mean confidence of 0.16. Section 3.4
says that at that confidence NeuralFoil's drag is wrong by about a third
on average, so the 38.5 is a number the model itself does not stand behind.
The optimizer went there because, as far as the surrogate can tell, there is
a whole family of shapes with almost the same worst-case L/D, some inside the
region it was validated in and some well outside it, and with nothing to
break the tie it picked one outside.

A small nudge on the dial (`w_conf` = 0.5) breaks the tie. The design moves
to a mean confidence of 0.96, where the measured drag error is about 9
percent, and the predicted worst-case L/D drops by 2 percent, from 38.5 to
37.7. Pushing the dial harder buys a little more confidence for a steadily
larger price: at `w_conf` = 8 the predicted L/D is 35.2. The shape changes
are modest; the two extremes are shown in the figure.

One caution: the "expected drag error" column is a population statistic from
55 other airfoils, not a measurement of these designs, which have not been
tested. What the experiment does establish is that the standard approach,
trusting the surrogate blindly, lands by default in the region where it is
known to be least reliable, and that leaving that region is nearly free. As
far as this project's search of prior work could tell, no earlier
airfoil-design study has fed a surrogate's own measured reliability back into
the design loop this way.

---

## 4. Discussion

The practical rule that comes out of all this is simple: know when to trust
the fast stand-in, and know what to trust it about. From Re = 100,000 up,
NeuralFoil's drag is typically within 7 to 10 percent of measurement and its
lift within a few hundredths of the lift coefficient for ordinary shapes,
which makes it a genuinely useful design tool. Below Re = 100,000, for highly
cambered high-lift shapes, and for L/D in general, it turns optimistic. Its
own confidence score flags the drag problem, though not the lift one. So a
designer can watch that score as a live warning light for drag and L/D, and
send only the most doubtful, highest-value final designs off for slower,
higher-fidelity testing or an actual wind tunnel. If the blade will fly rough
or dirty, the transition inputs should be used; they work.

The robustness results point the same direction from a different angle.
Designs chosen to be robust, whether to changing flying conditions or to
build error, tend to keep the airflow attached rather than separated, and the
uncertainty-aware optimizer shows that staying inside NeuralFoil's validated
region costs almost nothing in predicted performance.

---

## 5. Limitations, stated honestly

- **Most of this project is a validation and framework study**, not a brand-new
  discovery. The shape description, the neural stand-in, and the base
  optimization method were all built by other people. Much of this project's
  value is in combining them carefully and making them robust. The two
  genuinely new pieces are (a) the 55-airfoil experimental benchmark and what
  it shows about the confidence score and the transition inputs, and (b) the
  uncertainty-aware optimizer that uses that measured reliability inside the
  design loop. Both still rely on the same underlying model, so any specific
  optimized shape needs higher-fidelity or physical confirmation before it is
  fully trusted.
- **Every L/D number here is an estimate**, with a typical measured error of
  15 percent and an average over-prediction of 15 percent. The conclusions
  worth trusting most are the *relative* ones, such as robust beating peak, or
  manufacturing-aware beating nominal, not the raw numbers themselves.
- **The benchmark covers 55 airfoils from four of the five UIUC volumes**, in
  their plain configuration only; flapped and gurney-flap variants, Volume 5
  and the SoarTech collection are not included. Two
  airfoils had to be left out because the 17-number shape description could
  not reproduce them. The wind-tunnel models themselves differ from their
  design shapes by a few hundredths to a few tenths of a percent of chord,
  which sets a floor on how well any prediction can agree. Part of every
  reported error is the shape description itself (section 2.6), so the
  numbers are pipeline errors, not pure network errors.
- **The "expected drag error" attached to the optimized designs is a
  population statistic**, not a measurement of those designs.
- **This is two-dimensional, steady analysis only.** It does not model a full
  three-dimensional wing, spinning propellers or turbines, or unsteady gusts.
- **None of the optimized wings from this project have been physically
  tested.** The next genuinely new step would be to 3D-print the robust and the
  peak-tuned wings and measure them in a real wind tunnel.

---

## 6. Conclusion and what comes next

This project delivers a repeatable, honest, uncertainty-aware pipeline for
designing wings at low speed, and it closes a real gap by benchmarking
NeuralFoil against 4,400 wind-tunnel measurements on 55 airfoils below
Re = 500,000. The stand-in turns out to be accurate on drag from Re = 100,000
up, optimistic below that and for high-lift shapes, and its confidence score
is a trustworthy warning about drag error but not about lift. Its transition
inputs reproduce trip-strip measurements. Feeding that warning back into the
optimizer moves designs out of the surrogate's least reliable region for
about 2 percent of predicted performance. The most valuable next steps are
(1) extending the benchmark to Volume 5 and to flapped configurations, and
(2) an actual physical wind-tunnel test of 3D-printed
robust-versus-peak-tuned wings, which would turn this validation study into a
fully original experimental result.

---

## Author Contributions

The author designed and directed this study: choosing the question, setting its
goals and scope, and reviewing and interpreting every result. AI tools were used
for parts of the code implementation and figure generation, under the author's
direction.

---

## Acknowledgments

All experimental data used in this work are from public sources and are cited
below.

---

## References

1. Sharpe, P. D., & Hansman, R. J. (2025). *NeuralFoil: An airfoil aerodynamics
   analysis tool using physics-informed machine learning.* arXiv:2503.16323.
2. Selig, M. S., & McGranahan, B. D. (2004). *Wind Tunnel Aerodynamic Tests of Six
   Airfoils for Use on Small Wind Turbines.* NREL/SR-500-34515.
3. McGhee, R. J., Walker, B. S., & Millard, B. F. (1988). *Experimental Results for
   the Eppler 387 Airfoil at Low Reynolds Numbers in the Langley Low-Turbulence
   Pressure Tunnel.* NASA TM-4062.
4. Kulfan, B. M. (2008). *Universal parametric geometry representation method.*
   Journal of Aircraft, 45(1), 142-158.
5. Drela, M. (1989). *XFOIL: An analysis and design system for low Reynolds number
   airfoils.* In Low Reynolds Number Aerodynamics (pp. 1-12). Springer.
6. Sharpe, P. D. (2021). *AeroSandbox: A differentiable framework for aircraft
   design optimization.* (MIT master's thesis / software.)
7. Selig, M. S., Guglielmo, J. J., Broeren, A. P., & Giguère, P. (1995).
   *Summary of Low-Speed Airfoil Data, Volume 1.* SoarTech Publications.
8. Selig, M. S., Lyon, C. A., Giguère, P., Ninham, C. P., & Guglielmo, J. J.
   (1996). *Summary of Low-Speed Airfoil Data, Volume 2.* SoarTech Publications.
9. Lyon, C. A., Broeren, A. P., Giguère, P., Gopalarathnam, A., & Selig, M. S.
   (1998). *Summary of Low-Speed Airfoil Data, Volume 3.* SoarTech Publications.
   Tabulated data for Volumes 1 to 4 from the UIUC Low-Speed Airfoil Tests
   archive, https://m-selig.ae.illinois.edu/pd.html

---

*Reproducibility: all code, configuration files, extracted data (as CSV files),
and figures are available in the project repository. Software versions are
pinned, and random seeds are fixed, so every result here can be reproduced
exactly. Every study saves its data to disk before making any plots.*
