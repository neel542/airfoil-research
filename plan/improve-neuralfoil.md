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
