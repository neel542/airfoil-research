# Plain-language summary

*This is the short, non-technical version of `PAPER.md`. Every number here
is pinned to the data files by `tests/test_claims.py`.*

## The question

Small drones and small wind turbines fly slowly, and slow air behaves oddly
around a wing: a thin layer near the surface peels off and rolls into a
bubble, and that bubble is where most of the drag comes from. Designing
wings for this regime means testing thousands of candidate shapes, and a
full physics solver is too slow for that. So engineers use **NeuralFoil**, a
neural network trained to imitate an older solver called XFoil in a fraction
of a second. NeuralFoil also reports a confidence score with every answer.

The catch: NeuralFoil had only ever been compared with a real wind tunnel at
one high speed, far above where small drones fly. Nobody knew how wrong it
was in the slow regime where it is used most, or what its confidence score
actually meant.

## What was done

1. **Two wind tunnels, 94 airfoils, 9,100 measurements.** Every published
   low-speed wind-tunnel measurement from the University of Illinois archive
   and from an older Princeton data set was compared with NeuralFoil's
   prediction at the same condition.
2. **XFoil at every one of those points too.** NeuralFoil is a copy of
   XFoil, so its error is XFoil's error plus the copying error. Running the
   original at every point separates the two.
3. **The same airfoils in both tunnels.** Fifteen airfoils were measured in
   both laboratories, which shows how much two good experiments disagree with
   each other.
4. **Honest statistics.** Points along one measurement sweep are not
   independent, so the uncertainty was computed by resampling whole airfoils,
   and a small formula was fitted and cross-checked that predicts how wrong a
   NeuralFoil drag number is likely to be.
5. **A design pipeline** that uses all of this to design wings that perform
   well everywhere, not only at one speed, survive being built imperfectly,
   and stay inside the region where the model is known to be reliable.

## What was found

- **NeuralFoil's drag is off by about 11 to 12 percent on average**, 8 percent
  in the typical case, and its lift by a few hundredths of the lift
  coefficient. Both tunnels give the same answer. At the slowest speeds the
  drag error grows to about 20 percent.
- **Almost all of that error is XFoil's, not the network's.** XFoil itself
  is off by 12 percent at the same points. The network differs from XFoil by
  under 3 percent, and it is actually slightly closer to the measurements
  than XFoil is, because it smooths out XFoil's scatter. A bigger network
  trained on more XFoil would not help; a better physics model would.
- **Two wind tunnels disagree with each other by 12 percent** on the same
  airfoils. From medium speeds up, NeuralFoil is as close to each tunnel as
  the tunnels are to each other. At those speeds, no model can be shown to
  do better without better experiments.
- **The confidence score is a warning about drag, not about lift.** When it
  is high, drag error is about 9 percent; when it is low, about 32 percent.
  Lift error does not change with the score at all. The score turns out to
  flag the conditions where XFoil's own physics goes wrong, which is more
  than it was designed to do.
- **Honest uncertainty bands are two to five times wider** than the naive
  ones, but every headline result survives. One result that looked solid in
  the first tunnel, a relationship between an airfoil's average confidence
  and its average error, did not hold in the second tunnel, and the
  analysis shows why: the second set had almost no spread in confidence
  between airfoils.
- **A small fitted formula** turns a NeuralFoil prediction's confidence,
  speed, angle and camber into an expected drag error. Tested on airfoils it
  never saw, and trained on one tunnel then tested on the other, it beats a
  fixed guess by about 14 percent and its 80 percent band covers 80 percent
  of cases.
- **Part of every "model error" is build error.** The Princeton models were
  measured after they were built. Running NeuralFoil on the real shape
  rather than the drawing removes about a sixth of the drag error.
- **Designing for the worst case works.** A wing tuned for one condition has
  a spectacular peak and collapses everywhere else; the robust wing gives up
  half the peak and gains five times the worst-case performance. Adding a
  small reward for staying where NeuralFoil is confident moves the design
  out of the model's least reliable region for about 2 percent of predicted
  performance.

## What is new and what is not

New: the two-tunnel benchmark itself, the split of the error into XFoil's
and the network's, the experimental noise floor, the build-error
separation, the honest statistics and error model, and the experimentally
measured meaning of the confidence score. Not new: the wing-shape
description, NeuralFoil, XFoil, and the robust and multi-objective
optimization methods, which are standard tools applied carefully.

## What is next

A third wind tunnel with documented turbulence, a surrogate trained on
better-than-XFoil low-speed data, and a real wind-tunnel test of 3D-printed
wings from this pipeline with their as-built shapes measured.
