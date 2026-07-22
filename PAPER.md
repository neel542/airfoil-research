# Robust, multi-objective airfoil design at low Reynolds number, with the first check of a neural surrogate against real wind-tunnel data below Re = 500,000

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
a real 3D-printed part would be. Third, and most important, it gives the
**first check of NeuralFoil against real wind-tunnel data below Re = 500,000**,
using two classic reference wings, the Eppler E387 and the SD2030.

The result: NeuralFoil gets lift right to within about 5 to 8 percent, and drag
to within about 15 percent, though it always guesses drag a little low. The
useful surprise is that NeuralFoil's own confidence score drops exactly when its
error goes up, so the model can warn you when it is unreliable. This project
uses that warning inside the optimizer itself, and it changes the design that
comes out.

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
4. **The first check of NeuralFoil against real experiment below
   Re = 500,000**, giving an honest, measured error bar, and showing that its
   confidence score genuinely predicts its own reliability.
5. **An uncertainty-aware optimizer** that uses that measured reliability
   inside the design process itself, steering away from designs that look
   great on paper but that the model itself does not actually trust.

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

This is the new step nobody else had done. Real, measured lift and drag numbers
for the Eppler E387 and SD2030 wings came from a public wind-tunnel report
(Selig & McGranahan, 2004, NREL/SR-500-34515), covering speeds from Re = 100,000
to Re = 500,000. These numbers were pulled out of the report using two separate
computer programs, and only numbers both programs agreed on were kept. Numbers
that clearly belonged to a different wing printed on the same page were thrown
out. NeuralFoil, and a separately built copy of the older XFoil solver, were
then run at exactly the same conditions as the real measurements and compared
point by point.

One more check: does describing the E387 with only 17 numbers lose any real
shape detail? The answer is no. The reconstructed shape is off from the true
shape by only 0.15 percent of the chord length on average, which is far smaller
than NeuralFoil's own error. So any mismatch found below is NeuralFoil's fault,
not a problem with how the shape was described.

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

| What was measured | NeuralFoil vs real experiment (Re 100k to 500k) |
|----------|------------------------------------------|
| Lift | within about **5 to 8 percent** |
| Drag | within about **15 percent**, always guessed a little low near Re = 100,000 |
| Confidence score vs true error | strongly linked (Pearson r about −0.48) |
| Shape-description error (E387) | 0.15% of chord |

Three things stand out:

1. **NeuralFoil gets lift right, to within 5 to 8 percent, even at low speed.**
   That is good news, and nobody had shown it before in this speed range.
2. **Drag is the harder number, off by about 15 percent, and always guessed a
   little low** near Re = 100,000. This happens because both NeuralFoil and the
   older XFoil solver miss the extra drag from that separation bubble mentioned
   earlier. You can see the gap directly in one of the drag charts.
3. **The model knows when it is guessing.** NeuralFoil's own confidence score
   drops exactly when its real error goes up. Think of a student taking a test
   who says "I'm sure" on some answers and "honestly, I'm just guessing" on
   others, and it turns out the guessed answers really are the wrong ones.
   NeuralFoil behaves the same way. This is exactly why wing A's headline peak
   L/D of 232.9, produced where NeuralFoil's confidence is close to zero, should
   be read as an upper limit and not a trustworthy number. The model itself is
   quietly saying "I'm guessing" about that 233.

Since L/D is just lift divided by drag, a 15 percent drag error means every L/D
number in this project carries an error of at least about 16 percent, and that
error leans one direction: the true L/D is probably at or below what the model
predicts, not above it. Every main chart in this project is redrawn with that
honest error band included.

### 3.5 Using the finding to design better wings

This is the part of the project that goes beyond just checking a tool: it uses
what was learned to actually change how the design is done. Normally an
optimizer chases the single highest predicted L/D and simply trusts NeuralFoil's
number completely. But the previous section just showed that NeuralFoil is
least trustworthy exactly where its confidence is low. So a second goal was
added to the optimizer: alongside chasing performance, also reward designs that
land in the part of the map where NeuralFoil is confident. One single dial,
called `w_conf`, controls how strongly this new goal is weighted.

| `w_conf` dial setting | Worst-case L/D | Average confidence | Lowest confidence seen |
|---------:|---------------:|----------------:|------------------:|
| 0 (trust the model blindly) | 32.1 | 0.14 | 0.02 |
| 1 | **37.7** | 0.96 | 0.90 |
| 2 | 37.4 | 0.96 | 0.92 |
| 4 | 36.6 | 0.97 | 0.93 |
| 8 | 34.9 | 0.98 | 0.94 |

The result was a genuine surprise. With the dial turned off, the optimizer found
a design sitting at a confidence of just 0.14, right in the zone where
NeuralFoil is known to be unreliable. It looked fine on paper, but that "fine"
number is one the model itself does not really believe. The moment the dial is
turned up even a little, the design jumps to a confidence of 0.96, and its
honestly re-checked worst-case L/D actually goes up too, from 32 to 38. In other
words, blindly trusting the surrogate did not even buy better performance. It
bought a mirage. Only once the dial is pushed much harder does a small, real
performance cost start to appear in exchange for even higher confidence.

The takeaway is simple: a small amount of confidence-awareness gives a design
whose predicted performance can actually be believed, and it often costs
nothing to get it. As far as this project's search of prior work could tell, no
earlier airfoil-design study has fed a surrogate model's own measured
reliability back into the design process this way.

---

## 4. Discussion

The practical rule that comes out of all this is simple: know when to trust the
fast stand-in. In the middle of the range, at moderate angles and speeds above
roughly Re = 200,000, NeuralFoil is accurate to within a few percent and is a
genuinely excellent design tool. Near zero lift, near Re = 100,000, and for very
aggressive high-lift shapes, it turns optimistic, and its own confidence score
correctly drops exactly in those spots. So a designer can watch that confidence
score as a live warning light, and send only the most doubtful, highest-value
final designs off for slower, higher-fidelity testing or an actual wind tunnel.

The robustness results point the same direction from a different angle. Designs
chosen to be robust, whether to changing flying conditions or to build error,
also tend to be the designs NeuralFoil is most confident about, because they
keep the airflow smooth and attached to the surface rather than separating and
becoming chaotic.

---

## 5. Limitations, stated honestly

- **Most of this project is a validation and framework study**, not a brand-new
  discovery. The shape description, the neural stand-in, and the base
  optimization method were all built by other people. Much of this project's
  value is in combining them carefully and making them robust. The two
  genuinely new pieces are (a) checking the stand-in against real measurements
  at low speed, and (b) the uncertainty-aware optimizer that uses that
  measured reliability inside the design loop. Both still rely on the same
  underlying model, so they need further, higher-fidelity, or physical
  confirmation before any specific shape should be fully trusted.
- **Every L/D number here is an estimate**, with a measured error floor of
  about 16 percent that leans toward overstating performance at low speed. The
  conclusions worth trusting most are the *relative* ones, such as robust
  beating peak, or manufacturing-aware beating nominal, not the raw numbers
  themselves.
- **Only two wings** could be both cleanly pulled from the source report and
  matched to a shape already available in the airfoil database: the E387 and
  the SD2030. Checking more wings would make this result stronger.
- **This is two-dimensional, steady analysis only.** It does not model a full
  three-dimensional wing, spinning propellers or turbines, or unsteady gusts.
- **None of the optimized wings from this project have been physically
  tested.** The next genuinely new step would be to 3D-print the robust and the
  peak-tuned wings and measure them in a real wind tunnel.

---

## 6. Conclusion and what comes next

This project delivers a repeatable, honest, uncertainty-aware pipeline for
designing wings at low speed, and it closes a real gap by checking NeuralFoil
against real wind-tunnel data below Re = 500,000 for the first time. The
stand-in turns out to be accurate on lift, a little too optimistic on drag, and
usefully self-aware of its own weak spots. The most valuable next steps are (1)
pushing the uncertainty-aware optimizer further, and (2) an actual physical
wind-tunnel test of 3D-printed robust-versus-peak-tuned wings, which would turn
this validation study into a fully original experimental result.

---

## Acknowledgments

The author designed and directed this study: choosing the question, setting its
goals and scope, and reviewing and interpreting every result. AI tools
partially helped with coding and with figure generation, under the author's
direction. All experimental data used are from public sources and are cited
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

---

*Reproducibility: all code, configuration files, extracted data (as CSV files),
and figures are available in the project repository. Software versions are
pinned, and random seeds are fixed, so every result here can be reproduced
exactly. Every study saves its data to disk before making any plots.*
