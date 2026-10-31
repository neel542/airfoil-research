# Improving NeuralFoil's drag error: research plan

Neel is not entering ISEF (decided 29 Oct 2026), so Claude drafted steps 1 and 3 to 5 at his
request. Anything that uses this work must say so.
Fill the steps in order. Don't start step 5 before steps 1 to 4 are written and dated.

---

## Step 1. The question (one sentence)
What exactly will be improved, measured how, on what data?

> Can a small correction, learned from one wind tunnel's measurements, cut NeuralFoil's mean drag
> error on airfoils measured in the other tunnel, scored by benchmark/testbench.py?

## Step 2. The baseline to beat (already measured, do not change)
These are the numbers from the current paper. Any "improvement" is measured against them.

Measured by `benchmark/testbench.py` (file `data/testbench_results.csv`), which also checks
them against `data/clustered_statistics.csv`. Mean drag error, with 95% interval from resampling airfoils:

| Measure | Princeton (learn on UIUC) | UIUC (learn on Princeton) | Both, 10-fold by airfoil |
|---|---|---|---|
| NeuralFoil mean drag error | 11.10% (10.28 to 12.01) | 12.32% (11.35 to 13.37) | 11.69% (11.09 to 12.30) |
| NeuralFoil median drag error | 7.49% | 8.15% | 7.82% |
| XFoil mean drag error | 11.37% | 12.90% | 12.10% |
| Points / airfoils | 4,702 / 54 | 4,428 / 55 | 9,130 / 94 |
| Repeat-run noise floor (same tunnel) | 3.7% | | |

An idea counts as an improvement only if its "vs_neuralfoil" interval is entirely below zero.

## Step 3. Why does the error happen? (your hypothesis)
What do you think causes most of the error, and what evidence from your own results points to it?

> Part of the error is a steady bias, not random scatter, and it depends on speed and lift the
> same way in both tunnels (measured 29 Oct with data/xfoil_decomposition.csv):
> - At Re 60,000 NeuralFoil's drag is too high by 12 to 13 percent on average in both tunnels.
> - At Re 300,000 to 500,000 it is too low by 4 to 7 percent.
> - At zero or negative lift it is too high by 10 to 14 percent; at mid lift (CL 0.3 to 0.9) slightly low.
> Because NeuralFoil copies XFoil (errors correlate at r = 0.95), this is most likely XFoil's
> transition and separation-bubble model being wrong in a consistent direction. A bias that repeats
> across two independent tunnels should be learnable in one and removable in the other.
> What argues against it: most of the error is scatter (the bias is small outside low Re and low lift),
> and the two archives differ from each other by 10.9 percent, so some bias may be tunnel-specific.

## Step 4. What will you change, and what do you predict?
- The change:
- Your prediction, as a number (for example "mean drag error drops from X% to Y%"):
- What result would mean the idea failed:
- Date written:

> - The change: multiply NeuralFoil's drag by exp(f), where f is a small linear model of the
>   log ratio log(CD_tunnel / CD_NeuralFoil), fitted by ridge least squares on the training tunnel only.
>   Inputs use only what is known at prediction time, never the measured lift:
>   A: log Re, NeuralFoil CL, CL squared, log Re times CL
>   B: A plus camber and thickness
>   C: B plus NeuralFoil's confidence (cannot be used on the locked LTPT test, which has no confidence)
> - Prediction: the best of A to C lowers mean drag error by at least 1 percentage point in BOTH
>   cross-tunnel tests (from 11.10 and 12.32 percent), with the vs_neuralfoil interval below zero;
>   most of the gain at Re 60,000.
> - Failure: the interval includes zero in either cross-tunnel test, or the error rises in one of them.
> - Written 29 October 2026, before any correction was fitted.

## Step 5. The rules of the test (write before running anything)
- Which data is used to build or tune the change?
- Which data is kept untouched for the final test? (another tunnel? airfoils never seen?)
- How are error bars made? (whole airfoils resampled, as in the paper?)
- How many tries are allowed before you report?

> - Build and tune: only the UIUC->Princeton and Princeton->UIUC splits and the 10-fold airfoil CV.
> - Locked: NASA Langley LTPT E387 (data/ltpt_neuralfoil_validation.csv, 234 points, a third tunnel
>   never used for fitting). It is scored once, at the end, with the chosen model fitted on both tunnels.
> - Error bars: resample whole airfoils (the bench). LTPT has one airfoil, so it gets a plain mean only.
> - Tries: these three models, ridge strength fixed in advance at 1.0 on standardized inputs.
>   No other models get added after seeing results; anything new is a new, dated plan entry.

## Step 6. Run it
Log every run here with the date, what changed, and the result. Failed runs stay in the log.

| Date | What I ran | Result | Note |
|---|---|---|---|
| 29 Oct | Corrections A, B, C on the bench (`benchmark/drag_correction.py`) | A: Princeton 11.10 -> 9.92% (change -1.19, interval -1.59 to -0.80); UIUC 12.32 -> 11.46% (-0.86, interval -1.56 to -0.11); 10-fold 11.69 -> 10.48% | B and C fail Princeton->UIUC: their intervals include zero |
| 29 Oct | A by Re band, cross-tunnel | Re 60k: Princeton 16.8 -> 13.2%, UIUC 21.7 -> 17.9%. Re 300k to 500k (UIUC): 9.8 -> 10.2% and 9.2 -> 10.3%, slightly worse | |
| 29 Oct | Locked test: A fitted on both tunnels, scored once on NASA LTPT E387 | 234 points: mean 13.0 -> 12.6%, median 11.2 -> 10.2%, better on 53% of points. By pressure: 15 psi 13.4 -> 12.5%, 10 psi 12.9 -> 12.0%, 5 psi 11.1 -> 11.5%, 3 psi 19.1 -> 19.5% | One airfoil, so no interval |

## Step 7. What the result means
Did it beat the baseline by more than the noise floor? On both tunnels? Where does it still fail?

> Partly confirmed. The simplest correction (A: speed and lift only) beats NeuralFoil in both
> cross-tunnel tests, with the whole interval below zero both times, so the shared bias is real
> and does carry from one tunnel to the other. But it misses the 1-point target on the UIUC side
> (0.86 points), so the prediction as written failed there. Overall the gain is about 1.2 points,
> 11.7 to 10.5 percent, roughly a tenth of the error. Most of the error is scatter that a smooth
> correction cannot touch.
> Where it works: slow speeds. At Re 60,000 the error drops by 3.6 to 3.8 points in both tunnels.
> Where it hurts: the fastest UIUC band (Re 300,000 to 500,000) gets 0.4 to 1.1 points worse.
> Adding camber, thickness or confidence (B, C) made the cross-tunnel result worse: those let the
> model learn things specific to one tunnel. Simpler transferred better.
> The locked third tunnel agrees only weakly: 13.0 to 12.6 percent on one airfoil, better at the
> two higher pressures and slightly worse at the two lower ones. Not strong evidence either way.
> Honest summary: a real but small improvement, concentrated at low Reynolds number. Next idea
> should target the low-Re band directly, or stop correcting everywhere above Re 300,000.

## Step 8. Check before telling anyone
- [ ] Numbers re-run from scratch match
- [ ] Tested on data that was never used for tuning
- [ ] A professor or mentor has looked at it
- [ ] Only then: write up and think about publishing

## Time budget (studies first)
Hours per week I will give this:
Day(s):

---

## Entry 2 (written 31 October 2026, before running)

Why: entry 1 showed correction A helps below Re 150,000 (Re 60k: -3.6 to -3.8 points) and hurts
above Re 300,000 (+0.4 to +1.1 points). I saw those band results before writing this, so the 150,000
cutoff is informed by data from the same tunnels. It is not blind. The only blind check left is
that it must hold in both cross-tunnel directions.

- The change: correction A (log Re, NeuralFoil CL, CL squared, log Re times CL), fitted ONLY on
  training points with Re <= 150,000, applied ONLY to test points with Re <= 150,000. Everything above
  150,000 keeps NeuralFoil's drag unchanged. Ridge 1.0. Nothing else varied.
- Prediction: the vs_neuralfoil interval is below zero in both cross-tunnel tests; the mean improves
  by at least 1.2 points in both (entry 1 gave 1.19 and 0.86); no Re band above 150,000 changes.
- Failure: either cross-tunnel interval includes zero, or the gain over entry 1 is under 0.2 points
  (then the gate adds nothing).
- Locked LTPT E387: it was already scored once under entry 1, so this is a second look, not a clean
  test. It is reported as such.

### Entry 2 result (31 October 2026)

| | Princeton (learn on UIUC) | UIUC (learn on Princeton) | Both, 10-fold |
|---|---|---|---|
| Entry 1, correction A everywhere | -1.19 (-1.59 to -0.80) | -0.86 (-1.56 to -0.11) | -1.21 |
| Entry 2, A below Re 150,000 only | -0.82 (-1.14 to -0.50) | -0.92 (-1.30 to -0.50) | -0.96 |

Against what was written down:
- Both intervals below zero: yes.
- At least 1.2 points better in both: no (0.82 and 0.92).
- Gain over entry 1 at least 0.2 points: no. Princeton got 0.37 points worse, UIUC 0.06 better.
  That meets the failure condition, so entry 2 failed.
- Nothing above Re 150,000 changes: true, except that the Princeton "150k" band, which straddles the
  cutoff, got worse (10.2 to 10.8 percent).
- Re 60,000 still improves a lot (Princeton 16.8 to 12.5, UIUC 21.7 to 17.4 at 40k to 60k).
- LTPT second look (not clean): mean 13.0 to 12.7 percent, better on only 41 percent of points.

What it means: the gate removed the harm above Re 300,000 but also removed gains in the middle bands
(Princeton 300k, 200k to 400k in UIUC), and the cutoff sits on a band where the correction misfires.
Entry 1 (correct everywhere) is the better model. Keep it as the result: about 1.2 points, 11.7 to 10.5
percent. A smooth correction tops out near 10 percent error; the rest is scatter that needs a different
kind of idea, not another cutoff.

---

## Entry 3 (written 31 October 2026, before running)

Why: entries 1 and 2 say a smooth correction on speed and lift is worth about 1.2 points and no more.
After it, the mean drag error is about 10.5 percent. Before trying another correction, find out what
the remaining error is made of. The decomposition below is run on the out-of-fold residuals of
correction A (airfoil-CV, so no residual comes from a model that saw its own point).

Residual r = log(CD_tunnel / CD_corrected), corrected by A, 10-fold by airfoil.

Question: how much of r is a constant offset per airfoil, how much per polar (one airfoil at one Re),
and how much scatter inside a polar? Does the offset follow airfoil shape? Where in lift and Re does the
scatter sit? Would a more flexible smooth correction (E) get past A?

- Analysis 1: sums of squares of r split into between-airfoil, between-polar-within-airfoil, within-polar.
- Analysis 2: the Spearman correlation of each airfoil's mean r with its camber and with its thickness.
- Analysis 3: share of the total squared residual coming from points with CL <= 0 or CL >= 1.2.
- Candidate E: A plus CL^3, logRe^2, logRe^2 * CL, logRe * CL^2. Same ridge (1.0), same splits, bench scoring.
  LTPT is not touched in this entry (it has already been scored twice).

Predictions:
- P1: within-polar scatter is more than 50 percent of the residual sum of squares, and between-airfoil
  is under 25 percent.
- P2: both Spearman correlations in analysis 2 are below 0.3 in size.
- P3: the points with CL <= 0 or CL >= 1.2 (about 26 percent of all points) hold more than 35 percent
  of the squared residual.
- P4: E gains less than 0.4 points beyond A in the 10-fold test (E's change from NeuralFoil is above
  -1.6, where A's is -1.21), and E is worse than A in at least one of the two cross-tunnel directions.

What would mean the predictions were wrong, and what to do:
- P1 false (airfoil or polar offsets are large): the error is mostly predictable per airfoil, so
  chase an airfoil-level variable (shape, edge radius, trailing-edge thickness) next.
- P3 false: the scatter is not tied to lift extremes; look at angle of attack relative to stall.
- P4 false (E gains 0.4 or more in both directions): a more flexible smooth model is worth pursuing.

### Entry 3 result (31 October 2026)

What the leftover error is made of (correction A, out-of-fold, 9,130 points, 94 airfoils, 667 polars at about 14 points each):

| Part of the residual | Share of the sum of squares |
|---|---|
| An offset per airfoil | 8% |
| An offset per polar (one airfoil at one Re) | 14% |
| Scatter inside a polar | 78% |

Against what was written down:
- P1 (within-polar over 50 percent, between-airfoil under 25 percent): true, 78 and 8.
  Small caveat: with about 14 points per polar, a polar's own mean soaks up some scatter, so the 14
  percent between-polar share is partly noise (roughly 6 points of it).
- P2 (both correlations under 0.3): half wrong. Camber 0.23, thickness 0.41. A thicker airfoil tends to
  have its drag under-predicted by NeuralFoil's corrected value (about 0.9 percent per extra 1 percent
  of chord thick). Post-hoc, per tunnel (not in the plan): thickness 0.28 in Princeton, 0.48 in UIUC,
  same sign in both. But the airfoil offset is only 8 percent of the residual: removing every airfoil's
  offset entirely would lower the mean error from 10.56 to 10.29 percent, at most 0.27 points.
- P3 (CL extremes hold over 35 percent): false, but close. The 26 percent of points with CL at most 0 or
  at least 1.2 hold 32 percent of the squared residual. Mean error by lift: 12.0 percent at CL below 0,
  8.7 percent at CL 0.3 to 0.6, 13.2 percent above 1.2. A mild U shape, not a concentration.
- P4 (E gains under 0.4 points beyond A; E worse than A in one direction): true. E gains 0.10 points
  beyond A in the 10-fold test (-1.31 versus -1.21). E is slightly worse than A for Princeton (-1.14
  versus -1.19) and better for UIUC (-1.18 versus -0.86).

Erratum (31 October 2026, found while writing the paper): the "10.56 to 10.29 percent" above is the mean
absolute LOG residual, not the paper's mean relative drag error. In the paper's own measure, removing every
airfoil's offset takes 10.48 to 10.22 percent, 0.26 points. The conclusion (at most about a quarter of a
point) stands.

What it means:
- A smooth correction on speed and lift has hit its limit at about 1.2 points. More flexible terms add
  0.1.
- About four fifths of the error left is scatter inside a single polar: the same airfoil at the same
  speed, angle to angle. Repeat runs in one tunnel disagree by 3.7 percent, so noise accounts for some
  of it, but not for a 10 percent error.
- Airfoil shape is not the lever either: even a perfect shape-based correction is worth 0.3 points.
- Where the scatter comes from is the open question. The laminar-run test (benchmark/laminar_run.py)
  already ruled out one cause. The next candidates are the angle-to-angle jumps where transition or the
  separation bubble moves, which would need NeuralFoil's or XFoil's per-point boundary-layer state as
  input, not a smooth function of Re and CL.

---

## Entry 4 (written 31 October 2026, before running)

Question: does NeuralFoil's own picture of the boundary layer explain the scatter that is left after
correction A? Entry 3 found 78 percent of the leftover squared error is scatter inside one polar. The
paper (section 3.2, benchmark/laminar_run.py) found that drag error is larger where XFoil's laminar run on the
suction side is short (partial rank correlation -0.17, strongest at Re below 130,000, about -0.28). Entry 4
puts that idea inside the correction instead of correlating it, and adds a measure of how fast transition
moves from one angle to the next.

Inputs, all computed by NeuralFoil only (fast, and known when you make a prediction; XFoil is not used):
- xs, xp: transition location on the suction and pressure side. The surface is chosen by the sign of
  NeuralFoil's own CL (suction = upper if CL >= 0, lower if CL < 0), never by "upper", because the paper
  showed the other choice reverses the sign of the effect below zero lift.
- Hs, logtheta_s: shape factor and log momentum thickness at the trailing-edge station (the last of
  NeuralFoil's 32 stations) on the suction side.
- jump_s, jump_p: |d xtr / d alpha| on the suction and pressure side, from NeuralFoil at alpha plus and
  minus 0.5 degrees (a fixed stencil, so it does not depend on how the tunnel spaced its angles).

Candidates (ridge 1.0 on standardized inputs, same target log(CD_tunnel / CD_NeuralFoil), bench splits):
- F = A + xs, xs^2, xp, Hs, logtheta_s
- G = F + jump_s, jump_p, xs * logRe

Not used: LTPT (already scored twice), NeuralFoil's confidence (tried in C, failed to transfer).

Checks before anything is scored: the features come from re-running NeuralFoil with the validation scripts'
settings (large model, n_crit 9, free transition); every re-computed CD must match the stored NF_CD to
within one part in a million, or the run stops.

Predictions:
- P1: F and G both fail the success rule: neither is 0.5 points or more better than A in the 10-fold
  test (A is -1.21, so neither reaches -1.71). Expected gain beyond A is 0.1 to 0.4 points, because
  a rank correlation of -0.17 explains about 3 percent of variance.
- P2: G is within 0.15 points of F in the 10-fold test (the jump terms add nothing).
- P3: after G the within-polar share of the leftover squared error is still above 70 percent.
- P4: inside a polar (polar means removed), the Spearman correlation between |residual| and xs, and
  between |residual| and jump_s, are each below 0.15 in size.
- P5: measurement noise explains under 15 percent of the within-polar variance. Estimate: the two paired
  repeat models (data/repeatability_same_model.csv) give a single-measurement noise of about 0.033 in
  log units, against a within-polar residual sd of about 0.14.

Success rule: F or G is at least 0.5 points better than A in the 10-fold test, lower than A in both
cross-tunnel directions, and has both cross-tunnel intervals below zero.

If it succeeds: the boundary-layer state is worth adding to the correction; fit on both tunnels and score
once on LTPT as a third look (labelled as such).
If it fails, as predicted: NeuralFoil's own boundary-layer state does not carry the scatter. Then what is
left is measurement scatter and physics neither NeuralFoil nor XFoil represent, and the correction work
stops at A: a 1.2-point gain, 11.7 to 10.5 percent.

### Entry 4 result (31 October 2026)

Features: NeuralFoil re-run at all 9,130 points. The re-computed drag matched the stored drag to 4e-14, so
the geometry and settings are the same as in the validations. One caveat, not tested: the trailing-edge
shape factor Hs has a long tail (a few points reach 38, where NeuralFoil has the layer separated), and it
went in unchanged.

Mean drag error, and the change from NeuralFoil in points (95 percent interval from resampling airfoils):

| | Princeton (learn on UIUC) | UIUC (learn on Princeton) | Both, 10-fold |
|---|---|---|---|
| NeuralFoil | 11.10% | 12.32% | 11.69% |
| A (speed and lift) | 9.92% (-1.19) | 11.46% (-0.86) | 10.48% (-1.21) |
| F (A + boundary-layer state) | 10.23% (-0.87) | 11.28% (-1.04) | 10.20% (-1.49) |
| G (F + how fast transition moves) | 10.31% (-0.79) | 11.22% (-1.09) | 10.03% (-1.67) |

Against what was written down:
- P1 (neither reaches 0.5 points beyond A): true. F gains 0.28 and G 0.46 in the 10-fold test. The expected
  range was 0.1 to 0.4, so G is above it. Both fail the success rule, which needs both cross-tunnel
  directions: they are better than A learning on Princeton and testing on UIUC, and WORSE than A learning
  on UIUC and testing on Princeton.
- P2 (G within 0.15 of F): false. G is 0.18 better, just outside.
- P3 (within-polar share above 70 percent after G): true, 78.6 percent (A was 77.9). The scatter inside a
  polar did not shrink.
- P4 (both within-polar correlations under 0.15): false. Inside a polar, the size of the leftover error
  correlates -0.22 with the suction-side transition location and -0.17 with how fast transition moves.
  Where transition sits well forward, or is moving fast, the error is larger.
- P5 (noise under 15 percent of within-polar variance): true, about 5 percent. Two repeat runs of one model
  differ by 3.7 percent, which is a single-measurement noise of about 0.033 in log units, against a
  leftover within-polar sd of about 0.14. Measurement noise is not what is left over. (Only two model pairs
  back this, so it is a rough figure.)

What it means:
- By the success rule written in advance, this fails, and the headline result stays correction A:
  11.7 to 10.5 percent.
- My written interpretation of a failure was too strong. I said the boundary-layer state "does not carry the
  scatter". The data say it carries a little of it inside a mix of the two tunnels (0.3 to 0.5 points) and that
  gain does not transfer reliably from one tunnel to the other.
- The within-polar correlations are with the SIZE of the error, not its direction. A correction can only shift
  the average error, and the average does not move with transition location. The size does. That points
  away from a correction and toward an error bar: the boundary-layer state may tell a designer where
  NeuralFoil is least reliable, which is a different and probably better use of it.
- It is not measurement noise (5 percent), and it is not airfoil shape (entry 3). What remains is how the
  flow itself behaves from one angle to the next in ways neither NeuralFoil nor XFoil carries.

Next (not run; would need a new dated entry): add the boundary-layer state to the fitted error model
(benchmark/fit_error_model.py), which predicts the size of the drag error, and test whether it predicts
held-out error better across tunnels than the confidence score alone.

---

## Entry 5 (written 31 October 2026, before running)

Question: does NeuralFoil's boundary-layer state help predict how big its drag error will be at a given
point, on data from a tunnel the error model never saw?

Why: entry 4 found that inside one polar, the size of the leftover error correlates -0.22 with the
suction-side transition location and -0.17 with how fast transition moves, while a mean correction gained
almost nothing from the same inputs. A correction moves the average; these inputs seem to move the spread.
The repo already has an error model (benchmark/fit_error_model.py): a Gamma regression that predicts the
expected size of |CD_NF - CD_tunnel| / CD_tunnel from NeuralFoil's confidence, Reynolds number, angle of
attack, camber and thickness. Entry 5 adds the boundary-layer state to it and asks if it predicts better.

What is scored: the size of NeuralFoil's own relative drag error, y = max(|CD_NF - CD_tunnel| / CD_tunnel,
1e-4), at the bench points with |CL_tunnel| > 0.1 (attached flow, the same domain the existing error model
is defined on). Not the error after correction A. The fitted models are Gamma regressions with a log link,
fitted exactly as fit_error_model.py fits them (same iteratively reweighted least squares, no penalty).
Inputs are the ones entry 4 computed (data/nf_bl_state.csv): xs, xs^2, xp, log10(Hs), logtheta_s, jump_s,
jump_p. Hs enters as log10(Hs) because it has a long tail (up to 38); that choice is made now, before any
score.

Models, fitted on the training part only:
- baselines, as in fit_error_model.py: constant; confidence_only; confidence_Re (confidence and log Re);
  full (confidence, log Re, alpha, alpha^2, camber, thickness)
- candidates: confidence_Re_BL (confidence_Re plus the seven boundary-layer inputs); full_BL (full plus the
  seven); BL_Re (log Re plus the seven, no confidence)

Splits, the same as the bench: learn on UIUC and score on Princeton, learn on Princeton and score on UIUC,
and 10 folds over airfoils from both tunnels (same fold assignment as testbench.py). Primary score: mean
Gamma deviance on the held-out points (lower is better; it is the score the model is fitted to). Intervals
come from resampling whole airfoils (2,000 resamples) on the paired difference from the baseline.
Also reported: mean absolute error of the predicted size, Spearman correlation between predicted and actual
size, and how often the actual error falls under the model's 80 percent band (should be near 0.80).

The bar to beat is confidence_Re, not confidence_only: Reynolds number is the obvious confound, because
transition moves forward as Re falls, so a gain over confidence_only could just be Re leaking in.

Not used: LTPT (one airfoil, already scored twice), the correction models, XFoil.

Limit, stated now: the idea came from looking at these same points (the correlations in entry 4). The
folds and the cross-tunnel splits keep the fitting honest, but they do not make this a clean confirmation of
a hypothesis formed in advance. A further tunnel would be the clean test.

Predictions:
- P1: confidence_Re_BL has lower mean deviance than confidence_Re in all three splits (the sign is right
  everywhere), because the within-polar correlations are real.
- P2: the gain is small: the 10-fold reduction in mean deviance is between 1 and 4 percent of
  confidence_Re's.
- P3: in at least one of the two cross-tunnel splits the interval on the deviance difference includes zero,
  so the success rule below fails.
- P4: BL_Re scores worse than confidence_Re in the 10-fold test (confidence still carries something the
  seven boundary-layer numbers do not).
- P5: full_BL gains less over full than confidence_Re_BL gains over confidence_Re, as a share of deviance
  in the 10-fold test, because alpha, camber and thickness already stand in for part of the boundary-layer
  state.

Success rule (all three): confidence_Re_BL has lower mean deviance than confidence_Re with the whole
interval below zero in both cross-tunnel splits; its 10-fold mean deviance is at least 2 percent lower; and
its Spearman correlation is higher in both cross-tunnel splits.

If it succeeds: the boundary-layer state goes into the published error model (fit on both tunnels, the
coverage of its bands checked, the JSON and the site's error bar updated), and the paper says the error
bar, not the correction, is where NeuralFoil's own boundary-layer state is useful.
If it fails, as predicted: the existing error model stays as it is, and the work that can be done from
NeuralFoil's own outputs is finished. The next step would need new measurements (a third tunnel) or a
physical model of the bubble, neither of which is a script.

### Entry 5 result (31 October 2026)

Points: 8,211 attached-flow points, 94 airfoils (3,995 UIUC, 4,216 Princeton). Sanity check: the four
baselines on this point set score a 10-fold mean absolute error of the predicted size within 0.0006 of
fit_error_model.py's published run (constant 0.0862 against 0.0867, confidence_only 0.0789 against 0.0794,
confidence_Re 0.0766 against 0.0771, full 0.0739 against 0.0742). The point set and folds differ slightly,
so they are not identical, but the setup is the same.

Mean Gamma deviance on held-out points (lower is better):

| Model | Learn UIUC, score Princeton | Learn Princeton, score UIUC | 10-fold |
|---|---|---|---|
| constant | 1.137 | 1.184 | 1.150 |
| confidence_only | 1.055 | 0.982 | 1.016 |
| confidence_Re (the bar) | 1.023 | 0.929 | 0.977 |
| confidence_Re_BL | 0.942 | 0.917 | 0.905 |
| BL_Re | 0.974 | 0.977 | 0.943 |
| full (the published model) | 0.923 | 0.902 | 0.902 |
| full_BL | 0.940 | 0.923 | 0.896 |

confidence_Re_BL minus confidence_Re, with the 95 percent interval from resampling airfoils:
- Learn UIUC, score Princeton: -0.081 (-0.116 to -0.048), 7.9 percent lower
- Learn Princeton, score UIUC: -0.012 (-0.050 to +0.025), 1.3 percent lower
- 10-fold: -0.072 (-0.099 to -0.046), 7.4 percent lower

Spearman correlation of predicted with actual error size, confidence_Re_BL against confidence_Re: 0.350
against 0.229, 0.368 against 0.360, 0.388 against 0.293. The 80 percent band covers 79.8, 78.3 and 79.1
percent of held-out points for confidence_Re_BL (confidence_Re: 81.3, 81.6, 81.6).

Against what was written down:
- P1 (lower deviance in all three splits): true.
- P2 (10-fold reduction between 1 and 4 percent): false. It is 7.4 percent.
- P3 (an interval includes zero in a cross-tunnel split): true, learn Princeton and score UIUC.
- P4 (BL_Re worse than confidence_Re in the 10-fold test): false. BL_Re is 3.5 percent better in the 10-fold
  test (interval -0.069 to -0.0003) and better learning on UIUC, but worse learning on Princeton (+0.048,
  interval +0.001 to +0.095). Confidence adds to it (confidence_Re_BL is better still), so the two carry
  partly different information.
- P5 (full_BL gains less over full than confidence_Re_BL over confidence_Re): true. 0.7 percent against
  7.4 percent in the 10-fold test.

Success rule: FAILS. The 10-fold test is 7.4 percent lower (needs 2) and the Spearman correlation is higher
in both cross-tunnel splits, but the learn-Princeton, score-UIUC interval reaches +0.025, so not both
cross-tunnel intervals are below zero. By the rule written in advance, the error model is not updated.

Post hoc, not in the plan, run after seeing the table because the published model is "full", not
confidence_Re (deviance change, interval):
- full_BL minus full: +0.017 (+0.003 to +0.031) learning on UIUC, +0.021 (+0.006 to +0.037) learning on
  Princeton, -0.006 (-0.013 to +0.002) in the 10-fold test.
- confidence_Re_BL minus full: +0.019 (+0.002 to +0.038), +0.014 (-0.014 to +0.041), +0.003 (-0.010 to
  +0.017).

What it means:
- The boundary-layer state does say something about how big the error is. The gain over the Re-and-confidence
  model is real in the pooled test and in one tunnel direction.
- But the published model already gets that gain from angle of attack, camber and thickness: it scores as well
  as confidence_Re_BL everywhere (post hoc: confidence_Re_BL is never better than it, and is worse by
  0.019 learning on UIUC). Adding the state on top of the published model makes tunnel-to-tunnel prediction
  worse in both directions, with the whole interval above zero. That is consistent with the state mostly
  repeating what alpha and shape already tell the model, and overfitting on top of them.
- The gain is lopsided: 7.9 percent learning on UIUC, 1.3 percent learning on Princeton. Entry 4's
  correction was lopsided the other way round. I have no explanation for either, so this should not be read
  as a finding about either tunnel.
- The 80 percent band under-covers slightly with the state in (78 to 80 percent against 81 to 82), so it
  would also need re-calibrating if it were used.
- Limit, as stated before running: the idea came from these points.

Where this leaves the work. From NeuralFoil's own outputs, the correction work ends at A: 11.7 to 10.5
percent mean drag error, and the error model stays as published. Entries 2 to 5 each tried to go further and
none passed its written rule. The honest framing for the paper is one positive result (A), plus four
recorded attempts that did not add to it. Going further needs something new: a third tunnel's data, or a
physical model of the laminar separation bubble. Neither is a regression on this data.
