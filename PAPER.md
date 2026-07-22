# Robust, multi-objective airfoil design at low Reynolds number, with the first experimental validation of a differentiable surrogate below Re = 500,000

**Author:** Neel Madhav
**Date:** 2026

---

## Abstract

Small drones (UAVs) and little wind turbines fly slowly, which puts them at low
Reynolds numbers (Re ≈ 50,000-500,000). The air is fussy in this range: a thin
layer near the wing can peel away and form a laminar separation bubble, which is
where most of the drag comes from, and a wing that works well at one setting can do
badly at another. To design wings quickly, people now lean on fast neural
stand-ins for the slow flow solvers — NeuralFoil is the best known. These
stand-ins are *differentiable*, so they drop straight into an optimizer. But there
was a problem: NeuralFoil had only ever been checked against real wind-tunnel data
at one high Reynolds number (Re = 1.8 × 10⁶), roughly ten times faster than where
small drones and turbines actually fly. This work does three things. First, it
builds a repeatable pipeline that designs airfoils for good *worst-case*
performance across a whole range of conditions, and for several goals at once
(efficiency, safety, thickness for structure, and trailing-edge noise). Second, it
makes the wing hold up when it is built imperfectly, using sample-based robust
optimization. Third, and most importantly, it gives the **first check of NeuralFoil
against measured wind-tunnel data below Re = 500,000**, using the Eppler E387 and
SD2030 airfoils. The stand-in gets lift right to within about 5-8% and drag to
within about 15%, with drag always guessed a little low because of the separation
bubble. The useful surprise: NeuralFoil's own confidence score goes *down* when its
error goes up (Pearson r ≈ −0.48), so the model can basically warn you when it is
unreliable. That gives every number the pipeline produces a real, physics-based
error bar. Finally, this paper puts that finding to work: an uncertainty-aware
optimizer that rewards trustworthy designs, which turns out to steer away from
high-scoring "mirage" shapes that the model itself does not believe.

---

## 1. Introduction

### 1.1 Motivation

A drone that carries medicine, a hobby plane the size of a textbook, and a small
wind turbine on a rooftop all share one quiet problem: they move slowly, so the air
around them behaves very differently than it does around a full-size aircraft. In
fluid-dynamics terms, they fly at **low Reynolds number**. Here a thin layer of
slow air near the surface can lift off and roll into a *laminar separation bubble*,
which adds drag and can make the wing stall without warning. Designing good wings
in this range is genuinely hard, and it matters, because these are real machines
doing useful jobs.

### 1.2 The problem with designing for a single condition

A wing tuned for one exact speed and angle can do poorly just a little off that
point. Real drones and turbines are always changing speed and angle, so a design
that is only good at one setting is close to useless. What you actually want is
**robustness**: a wing that does decently across the whole range of conditions it
will really see.

### 1.3 The role — and the unchecked trust — of stand-ins

To find good shapes, you have to test thousands of candidate wings, and running a
full flow solver on each one takes too long. So the field turned to **NeuralFoil**
(Sharpe & Hansman, 2025), a neural network trained to copy the classic solver XFoil
almost instantly and — the important part — *differentiably*, so it can go straight
into gradient-based optimization. The catch: NeuralFoil's only published check
against experiment was at Re = 1.8 × 10⁶, about ten times higher than where small
drones fly. **Nobody had checked it against real measurements in the low-Re range
where it gets used the most.** That is the gap this paper closes.

### 1.4 Contributions

1. A repeatable, config-driven pipeline for **robust, worst-case** multi-point
   airfoil optimization at low Re.
2. A **multi-objective** version that trades efficiency against safety
   (stall margin), structure (spar depth), and trailing-edge noise.
3. **Manufacturing-tolerance robustness**: designs that keep performing when built
   imperfectly, checked out-of-sample.
4. **The first experimental check of NeuralFoil below Re = 500,000**, giving a
   measured error bar and showing that its confidence score predicts its own
   reliability.
5. **An uncertainty-aware optimizer** that feeds the measured confidence-error link
   back into the design loop, steering away from high-scoring but untrustworthy
   "mirage" designs.

---

## 2. Background and methods

### 2.1 Geometry: the Kulfan (CST) parameterization

Every airfoil shape is written as 17 numbers (8 for the top surface, 8 for the
bottom, and 1 leading-edge weight) using the Kulfan, or Class-Shape-Transformation,
basis. This basis is smooth, uses few numbers, makes shapes you can actually build,
and is differentiable — which is exactly what gradient-based optimization needs.

### 2.2 Aerodynamics: NeuralFoil

Lift (C_L), drag (C_D), and the boundary-layer numbers all come from NeuralFoil
(the `large` model). Because it is differentiable, the optimizer can work out
exactly how a small change in shape changes performance, which turns an
optimization over a 25-point grid of conditions from hours into seconds.

### 2.3 Robust optimization: the epigraph max-min

To get the best *worst-case* lift-to-drag ratio (L/D) across the range, the jagged
"worst case" is rewritten in a smooth way. Here is the trick in plain terms: imagine
you want to raise the score of your *weakest* subject in school. You do not average
all your subjects; you push up the lowest one. In math, you add a variable *g* that
must stay at or below the L/D at *every* operating point, and then you push *g* as
high as it will go. Since *g* can never exceed the worst point, maximizing *g*
maximizes the worst case — and it does so with a smooth equation an optimizer can
handle. The problem is non-convex (it has more than one "hilltop"), so a single run
can get stuck on a small hill; each design is therefore solved from several starting
shapes (multi-start) and the best one is kept. Whole families of designs are traced
by warm-starting each from the previous one (continuation).

### 2.4 Multi-objective design

Four physical, differentiable goals are combined as a weighted sum, scaled so the
weights come out unit-free:

| Objective | Metric | Direction |
|-----------|--------|-----------|
| Efficiency | worst-case L/D over the envelope | maximize |
| Safety | lift retained at high angle (C_L at 12°, Re 50k) | maximize |
| Structure | maximum thickness (spar depth; stiffness ∝ t³) | maximize |
| Noise | trailing-edge displacement thickness δ\* = θ·H | minimize |

### 2.5 Manufacturing-tolerance robustness

A real wing — field-built or 3D-printed — never matches the design exactly. Build
error is modeled as small random changes to the shape numbers (~0.5% chord RMS
surface error). A design is optimized to hold its worst-case L/D across *both* the
range of conditions *and* a fixed set of build errors, then tested
**out-of-sample** against a fresh, larger set the optimizer never saw.

### 2.6 Experimental validation

This is the new step. Measured lift/drag curves for the Eppler E387 and SD2030
airfoils came from the UIUC/NREL low-speed wind-tunnel database (Selig &
McGranahan, 2004; report NREL/SR-500-34515) at Re = 100k-500k. The values were
pulled from the report with two separate parsers and kept only where both agreed,
then filtered to physical bounds to throw out numbers from other airfoils printed
on the same page. NeuralFoil (and a separately built headless XFoil) were then run
at exactly the measured conditions and compared point by point to experiment. A
geometry check confirmed that the Kulfan fit of the E387 adds only 0.15% RMS-chord
error — far below the stand-in's error — so any gap we see is the stand-in's fault,
not the shape description's.

---

## 3. Results

### 3.1 Peak versus robust design

| Airfoil | Objective | Peak L/D | Worst-case L/D | Max thickness |
|---------|-----------|---------:|---------------:|--------------:|
| A (single-point) | max L/D at one condition | 232.9 † | 7.1 | 13% |
| B (robust) | max worst-case L/D | 105.1 | **38.2** | 9% |

Airfoil B gives up about 55% of A's headline peak L/D in exchange for a **5.4×
higher worst-case** L/D. A is a razor-thin peak that falls apart at the edges of
the range; B is a broad plateau that does decently everywhere.

† See §3.4 — A's peak is quoted where NeuralFoil reports almost no confidence, so
read it as an optimistic ceiling, not a real measured value.

### 3.2 Multi-objective trade-offs

| Profile | Worst-case L/D | Stall angle | Max thickness | TE noise δ\* |
|---------|---------------:|------------:|--------------:|------------:|
| Efficiency | **38.5** | 9.0° | 8.0% | 15.1 × 10⁻³ |
| Balanced | 33.9 | 10.0° | 10.4% | 11.9 × 10⁻³ |
| Community | 17.8 | **11.5°** | **13.7%** | **11.1 × 10⁻³** |

Thickness for structure and low-Re efficiency pull hard against each other: forcing
a 13.7% thick spar costs about 54% of the worst-case L/D, but it buys a 2.5° bigger
stall margin and a quieter trailing edge.

### 3.3 Manufacturing robustness (out-of-sample)

| Design | As-designed worst-case L/D | Built: mean | Built: 5th-percentile |
|--------|---------------------------:|------------:|----------------------:|
| B_nominal (robust to conditions only) | 34.3 | 20.3 | 9.3 |
| B_mfg (robust to conditions + build error) | 33.7 | 26.5 | **19.5** |

On paper the two designs look almost the same, but under realistic build error the
manufacturing-robust design's reliable (5th-percentile) worst-case L/D is **more
than twice** the nominal design's — for almost no as-designed cost.

### 3.4 The main result: checking NeuralFoil against experiment at low Re

| Quantity | NeuralFoil vs experiment (Re 100k-500k) |
|----------|------------------------------------------|
| Lift, C_L | within **~5-8%** |
| Drag, C_D | within **~15%**, guessed low at Re ≈ 100k |
| Confidence vs true error | **Pearson r ≈ −0.48** |
| Kulfan geometry-fit error (E387) | 0.15% RMS chord |

Three findings:

1. **NeuralFoil gets lift right (~5-8%) at low Re** — good news, and not shown
   before in this range.
2. **Drag is harder (~15%) and always low**, because both NeuralFoil and XFoil
   under-guess the separation-bubble drag that shows up near Re = 100k. You can see
   it directly as a "drag-bucket" gap in the C_D curves.
3. **The model knows when it is wrong.** NeuralFoil's `analysis_confidence` goes
   down as its true error goes up. Think of a student taking a test who says "I'm
   pretty sure" on some answers and "honestly, I'm guessing" on others — and it
   turns out the "I'm guessing" answers are exactly the ones they get wrong.
   NeuralFoil behaves the same way: when it flags low confidence, it really is less
   reliable. That is the reason a high-confidence, robust design should be trusted
   more than an aggressive, low-confidence one — and it explains why airfoil A's
   peak L/D of 232.9, produced at confidence ≈ 0, should be read as a ceiling, not
   a real value. It is the model's way of saying "I'm guessing" about that 233.

Since L/D = C_L/C_D, the ~15% drag error means every L/D value carries an error
floor of about ±16%, and the bias only points one way (the true L/D is likely at or
below what the model predicts at low Re). Every headline figure is re-plotted with
this measured uncertainty band.

### 3.5 Putting the finding to work: an uncertainty-aware optimizer

Here is the part I am most proud of, because it uses the measurement above to
*change how the design is done*. Normally an optimizer chases the highest predicted
L/D and trusts the stand-in completely. But we just learned the stand-in is least
trustworthy exactly where its confidence is low. So I added a second goal to the
optimizer: alongside performance, reward designs that live where NeuralFoil is
confident. A single dial, `w_conf`, sets how strongly.

| `w_conf` | Worst-case L/D | Mean confidence | Lowest confidence |
|---------:|---------------:|----------------:|------------------:|
| 0 (trust blindly) | 32.1 | 0.14 | 0.02 |
| 1 | **37.7** | 0.96 | 0.90 |
| 2 | 37.4 | 0.96 | 0.92 |
| 4 | 36.6 | 0.97 | 0.93 |
| 8 | 34.9 | 0.98 | 0.94 |

The result surprised me. With the dial off (`w_conf` = 0), the optimizer found a
design sitting at confidence 0.14 — right where we measured the stand-in cannot be
trusted. It looked fine on paper, but that "fine" is a number the model itself is
unsure about. Turn the dial up just a little (`w_conf` = 1) and the design jumps to
confidence 0.96, and its honestly-checked worst-case L/D actually *goes up*, from
32 to 38. In other words, blindly trusting the stand-in did not even buy better
performance — it bought a mirage. Only after you push the dial hard (`w_conf` = 4,
8) do you start paying a small, real performance price for still-higher confidence.

The takeaway: a light touch of confidence-awareness gives you a design whose
predicted performance you can actually believe, often for free. As far as I can
tell from the literature search, no prior airfoil-optimization work feeds a
surrogate's own measured reliability back into the design loop this way.

---

## 4. Discussion

The practical message is a rule for *when to trust the fast stand-in*. In the meat
of the range — moderate angles, Re ≳ 200k — NeuralFoil is good to within a few
percent and is an excellent design tool. Near zero lift, near Re = 100k, and for
aggressive high-lift shapes, it turns optimistic, and its own confidence score
correctly drops in those spots. So a designer can watch the confidence score as a
live reliability gauge and send only the doubtful, high-value finalists off to
slower, higher-fidelity analysis or wind-tunnel testing.

The robustness results point the same way from a different direction: designs
chosen for robustness (to conditions and to build error) also tend to be the ones
the stand-in is most confident about, because they keep the flow attached and
well-behaved.

---

## 5. Limitations (stated honestly)

- **This is mostly a validation and framework study.** The parameterization, the
  stand-in, and the base optimization method were all made by others; much of the
  contribution is putting them together and making them robust. The two genuinely
  new pieces are (a) checking the stand-in against experiment at low Re, and (b) the
  uncertainty-aware optimizer that uses that measured reliability inside the design
  loop. Both still lean on the same surrogate, so they need higher-fidelity or
  physical confirmation before the specific shapes are trusted.
- **Absolute L/D values are estimates** with a measured ±16% floor and a one-sided
  low-Re bias. The conclusions to trust are the *relative* ones (robust-vs-peak,
  robust-vs-nominal).
- **Only two airfoils** could be both cleanly pulled from the source report *and*
  matched to a shape in the airfoil database (E387, SD2030). More would make the
  check stronger.
- **Two-dimensional, steady analysis only** — no three-dimensional wing effects,
  no rotation, no unsteadiness.
- **No physical test of the author's own optimized designs.** The genuinely new
  next step is to 3D-print the robust and nominal airfoils and test them in a wind
  tunnel.

---

## 6. Conclusion and future work

This work delivers a repeatable, honest, uncertainty-aware pipeline for
low-Reynolds-number airfoil design, and closes a real gap by checking NeuralFoil
against wind-tunnel data below Re = 500,000 for the first time. The stand-in turns
out to be accurate on lift, a bit too optimistic on drag, and — usefully — aware of
its own weak spots. The natural next steps are (1) an uncertainty-aware optimizer
that uses the measured confidence-error link as a trust-region constraint, and
(2) a physical wind-tunnel test of 3D-printed robust-versus-nominal airfoils, which
would turn this validation study into an original experimental result.

---

## Acknowledgments

The author designed and directed this study, defined its goals and scope, and
reviewed and interpreted all results. Implementation work — coding, data extraction
from source documents, and figure generation — was carried out with the assistance
of AI-based software tools under the author's direction. All experimental data are
from public-domain sources and are cited below.

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

---

*Reproducibility: all code, configuration files, extracted data (CSV), and figures
are available in the project repository. Dependencies are pinned; random seeds are
fixed. Each study writes its data to disk before plotting.*
