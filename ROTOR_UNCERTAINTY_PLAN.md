# Implementation plan: propagating the surrogate error into a rotor design

Written 2026-09-10. Everything here comes from what two reviewers actually asked for.
Nothing is invented.

---

## Where the feedback came from

**Bharath Govindarajan**, Associate Professor, Aerospace Engineering, IIT Madras.
Three replies, 2026-09-09 and 2026-09-10.

| # | What he said | Status |
|---|---|---|
| G1 | The 11 percent drag error carries into rotor torque and rotor power. "There is the very real possibility that this error / uncertainty would build up over the course of the calculation." | **Not done.** Phase 1 |
| G2 | In a design loop, power uncertainty becomes battery or fuel weight uncertainty, which becomes payload and total takeoff weight uncertainty. | **Not done.** Phase 1 |
| G3 | Hover: induced power is about 60 percent, profile takes the rest. The profile fraction rises with airspeed. Confirms the error is diluted, "But a 11% error at any part of the toolchain will have to be investigated." | **Not done.** Phase 1 |
| G4 | "Asking how well we predict the mean and the standard deviation is a worthwhile pursuit." Low Re lift and drag are non-linear and unsteady, so a single number for a time-varying value may not be the right target. | **Partly done.** Phase 2 |
| G5 | XFoil and NeuralFoil carry modelling assumptions that limit them against tunnel data, and both do better at higher Reynolds number. | **Data exists, not stated as a finding.** Phase 2 |
| G6 | "Uncertainty quantification is an important aspect of design, and it tells you how much of the unknowns early-on will propagate till the end." Twice called this a worthwhile effort in itself. | Framing for Discussion and Conclusion. Phase 3 |

**Nikhil Khobragade**, IIT Madras. Replies 2026-09-06 to 2026-09-07.

| # | What he said | Status |
|---|---|---|
| K1 | Boundary layer trips have to be handled separately from clean runs. | **Done.** 2,629 tripped points held out, free transition 17.0 percent, forced at trip 13.8 percent |
| K2 | Do the model errors sit inside the measurement error bars? | **Done.** 3.7 percent repeatability from repeat runs of the same model, vs 11.7 percent model error |

---

## Phase 0: fix the numbers before anything else

Two places state a sample size that does not match the figures attached to it. Both must be
correct before this work is shown to anyone, and certainly before an endorsement is requested.

**0.1** `PAPER.md:534`. The table row reads `| **Both** | **all** | **9,130** | **11.2** |
**12.1** | **2.8** |`. Those three percentages are computed on the 8,814 points where XFoil
converged, not on all 9,130. Line 519 already says so correctly. Change the row's n to 8,814
and add a footnote naming the 96.5 percent convergence rate.

**0.2** The site. `scrollcraft/builds/airfoil-console/data.js` already carries `"conv":8814`
and `page.js` never renders it, so the site pairs 9,130 with 11.2 percent and never shows the
convergence figure at all. Surface it wherever the decomposition numbers appear.

**0.3** Grep the whole repo for `9,130` and `9130` and check every remaining instance is
attached to a figure that really was computed on all 9,130. README, METHODS and SUMMARY
included.

Verification: after the edit, every reported statistic can be traced to a CSV whose row count
matches the n printed next to it.

Effort: 1 to 2 hours.

---

## Phase 1: the rotor propagation study

This is the new work and it is the centrepiece. New file, `rotor_uncertainty.py`.

### 1.1 Pick a rotor that is actually inside the validated envelope

Hard constraint, and it is what makes the study honest: **the blade sections must operate in
the Reynolds range the benchmark actually covers**, roughly 60,000 to 500,000. A rotor whose
sections sit at Re = 2,000,000 tells us nothing about an error measured at Re = 100,000.

Choose a small multirotor or eVTOL lift rotor. Two blades, radius in the 0.15 to 0.3 m range,
RPM set so that chord Reynolds number at 75 percent span lands mid-envelope. Use E387 or
SD7003 as the section, because both are in the benchmark set with measured data in both
archives, so the section error is known rather than assumed.

Record the resulting Re distribution along the blade. If any station falls outside the
validated range, say so and exclude it or flag it.

### 1.2 Blade element momentum theory, hover

Standard BEMT. Discretise the blade into elements, solve for inflow at each, get local angle
of attack and Reynolds number, look up Cl and Cd from NeuralFoil, integrate to thrust and
torque, iterate until inflow converges. Trim to a fixed thrust.

Outputs: rotor power, and the split between induced power and profile power.

**Verification gate.** Govindarajan gave a number: about 60 percent induced in hover. If the
BEMT produces a wildly different split at the design point, the rotor is unrepresentative and
should be retuned before going further. This is a real check, not a formality, and it is the
reason to ask him in the first place.

### 1.3 Propagate the measured error, both correlation bounds

This is the intellectually interesting part and it decides the answer.

`error_model.json` already holds a fitted Gamma model of the drag residuals. Draw from it and
perturb the section Cd, then re-run the BEMT. Monte Carlo, on the order of 1,000 draws.

The question Govindarajan raised is whether the error builds up or washes out, and that
depends entirely on an assumption nobody has stated:

- **Fully correlated.** The error is a property of the airfoil at a given Re and alpha, and
  every blade element shares the same airfoil. If one element's Cd is 11 percent high, they
  all are. The error carries straight through at full strength into profile power.
- **Independent per element.** The errors partially cancel across the blade and the effect on
  power shrinks by roughly the square root of the element count.

The truth is much closer to correlated, because the error source is a shared modelling
assumption rather than random noise. **Run both and report both as bounds.** The correlated
case is the honest headline. The gap between them is itself a result worth showing.

**Analytic cross-check:** in the fully correlated case, the fractional power error should come
out close to (profile fraction) times (Cd error), so roughly 0.40 x 11 percent, about 4.4
percent in hover. If the Monte Carlo does not land near that, one of the two is wrong. Find
out which before writing anything down.

### 1.4 The weight closure loop, where it compounds

This is G2 and it is where Govindarajan's "build up over the course of the calculation" should
actually show itself.

Given hover power P and a required endurance t, energy is P x t. Battery mass follows from
specific energy, usable fraction and drivetrain efficiency. Total mass is payload plus
structure plus battery. But thrust has to equal weight, so a heavier battery needs more
thrust, which needs more power, which needs more battery. Iterate to a fixed point.

Report the **amplification factor**: the ratio of percent error in takeoff weight to percent
error in section Cd. If that number is above one, the error genuinely compounds and
Govindarajan's warning is quantified. If it is below one, say that plainly instead.

Run the loop at the 5th, 50th and 95th percentile of the drag error and report the spread in
takeoff weight in grams, not only in percent. A design number in grams is what makes this
land.

### 1.5 Honest limitations, written at the same time as the code

- BEMT is itself a model with its own error, which is not quantified here. This propagates
  aerodynamic uncertainty through a simplified rotor model. It does not predict real rotor
  power.
- Hover only. Govindarajan said the profile fraction rises with airspeed, so hover is the
  mild case and forward flight would be worse. Say so, and do not claim forward flight numbers
  without running them.
- The Gamma error model was fitted to tunnel-versus-model residuals in a wind tunnel, not on a
  rotating blade. Rotational effects on the boundary layer are not in it.

Deliverables: `rotor_uncertainty.py`, `data/rotor_propagation.csv`,
`data/rotor_weight_closure.csv`, two figures (power distribution, weight amplification).

Effort: this is the large one. 15 to 25 hours honestly, most of it in getting BEMT to converge
and verifying it rather than in the Monte Carlo.

---

## Phase 2: mean and spread, and the Reynolds trend

### 2.1 Report the distribution, not one number (G4)

The paper currently leads with a single pooled percentage. Govindarajan asked how well the
mean **and the standard deviation** are predicted. Restate the headline results as a
distribution: median, interquartile range, 90th percentile, plus the fitted Gamma parameters
already in `error_model.json`. Section 3.4 has some of this; it needs to be promoted, not
buried.

### 2.2 State the Reynolds trend as a finding (G5)

He said both tools do better at higher Reynolds number. The data already proves it and it is
scattered across tables. Pull `uiuc_validation_by_Re.csv`,
`soartech8_validation_by_Re.csv` and `xfoil_decomposition_by_Re.csv` into one figure: error
against Reynolds number, both archives, with the network and XFoil separated. At n_crit 9 the
SoarTech8 numbers run 16.8 percent at Re = 60,000 down to 9.0 percent at Re = 300,000. That is
a clean monotonic trend and it deserves a figure rather than three table rows.

Then connect it to the rotor: name which part of the blade sits at which Reynolds number, so
the reader sees that the inboard sections are the least trustworthy.

### 2.3 What we will NOT do, and why

Govindarajan raised genuine flow unsteadiness as distinct from measurement scatter. **We
cannot separate those two with this data.** Repeat runs of the same model in the same tunnel
give 3.7 percent, and that figure contains both effects mixed together. Separating them needs
time-resolved measurements the archives do not contain.

Write this as a stated limitation. Do not construct an analysis that appears to separate them.

Effort: 3 to 5 hours, mostly reorganising results that already exist.

---

## Phase 3: paper integration

- New results subsection for the rotor propagation, after 3.4.
- Discussion picks up G6: the point of a benchmark is what the number does downstream, and the
  chain now runs drag error, rotor power, battery weight, takeoff weight, with a figure at
  each step.
- Limitations gains 1.5 and 2.3.
- Acknowledgments: both reviewers, by name, for correspondence. Ask permission first.
- Conclusion states the amplification factor. That is the sentence the paper has been missing.

Effort: 5 to 8 hours.

---

## Phase 4: site

`scrollcraft/builds/airfoil-console/STORY.md` is written and still unapproved. It already
identified that the site had no statement of consequence at the end. Phase 1 supplies exactly
that, with numbers. Fold the rotor chain into the final section rather than treating the story
rewrite and this work as two separate jobs.

Blocked on approval of STORY.md.

---

## Order and gating

1. Phase 0 first, always. The numbers must be right.
2. Phase 2.2 next, because it is cheap and the figure is needed for Phase 1's framing anyway.
3. Phase 1, the long build.
4. Phases 2.1, 3, 4.

**Do not request the arXiv endorsement until Phase 0 and Phase 1 are done.** The reason to ask
Govindarajan is that his input changed the work. That is only true once the work has changed.

---

## The one thing that could invalidate this

If the BEMT cannot be made to reproduce roughly the 60 percent induced split Govindarajan
quoted, the rotor model is wrong and every number downstream is decoration. That check comes
before the Monte Carlo, not after.
