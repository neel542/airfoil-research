# The short version

*Plain language. Same numbers as `PAPER.md`, all of them pinned to the data
files by `tests/test_claims.py`.*

## The problem

Small drones and rooftop wind turbines fly slowly, and slow air misbehaves.
A thin layer of air near the wing peels away from the surface and rolls into
a bubble. That bubble is where most of the drag comes from.

Finding a good wing shape means testing thousands of candidates, and a real
physics solver is far too slow for that. So the field uses **NeuralFoil**: a
neural network that imitates an older solver, XFoil, in a fraction of a
second. It reports a confidence score alongside every answer.

Nobody had checked it where it matters. Its one published wind-tunnel
comparison sits at a speed roughly ten times faster than a drone flies.
Whether it can be trusted in the slow regime, and whether that confidence
score means anything at all, were open questions.

## The test

Two wind tunnels. 94 airfoils. 9,100 measurements.

Every published low-speed measurement from the University of Illinois
archive and from an older Princeton data set went up against NeuralFoil's
prediction at the identical condition. Then XFoil itself was run at all
9,100 of those points, because NeuralFoil is a copy of XFoil and its error
is really two errors stacked: XFoil's, plus the copying. Running the
original pulls them apart.

Fifteen of the airfoils were tested in both tunnels. That is rarer than it
sounds, and it answers a question no model can: how much do two careful
experiments disagree with each other?

## What came back

**The drag error is 11 to 12 percent.** Eight percent in the typical case,
and a few hundredths of a lift coefficient on lift. Both tunnels agree. At
the slowest speeds the drag error climbs to roughly 20 percent.

**Almost none of that is the network's fault.** XFoil misses the tunnels by
12 percent at the very same points. NeuralFoil differs from XFoil by under 3
percent, and it lands slightly *closer* to the measurements than XFoil does,
because fitting a network to hundreds of thousands of XFoil runs smooths
away some of XFoil's own scatter. So a bigger network trained on more XFoil
will not fix this. Better physics would.

**The two tunnels disagree with each other by 12 percent.** Above medium
speeds, NeuralFoil sits as close to each tunnel as the tunnels sit to each
other. No model can be shown to beat that, because the experiments cannot
resolve the difference.

**The confidence score warns about drag. Not lift.** High confidence, and
the drag error runs about 9 percent. Low confidence, about 32 percent. Lift
error ignores the score completely. What the score really detects is where
XFoil's own physics breaks down, which is both more than it was built to do
and more useful.

**Honest error bars are two to five times wider** than the naive ones, and
everything survives them. One result did not. In the first archive, the
airfoils the model felt least sure about were also the ones it got most
wrong; in the second archive that vanished. The reason is visible in the
data. Nearly every Princeton airfoil sits at high confidence, so there was
no spread left to correlate.

**A small formula now does the work of the table.** Feed it a prediction's
confidence, speed, angle and camber, and it returns the drag error to
expect. It was tested only on airfoils it had never seen, then trained on
one tunnel and tested on the other. It beats a fixed guess by 14 percent,
and its 80 percent band really does cover 80 percent of cases.

**Some of the "model error" is build error.** Princeton measured its models
after building them, and running NeuralFoil on the real shape instead of the
drawing removes about a sixth of the drag error. The wing in the tunnel was
never quite the wing on paper.

**Robust beats peak.** A wing tuned for one condition posts a spectacular
number and falls apart everywhere else. The robust wing surrenders half that
peak and holds five times the worst-case performance. Adding a small reward
for staying where NeuralFoil is confident costs about 2 percent of predicted
performance.

## Credit where it is due

New here: the two-tunnel benchmark, the split of the error into XFoil's part
and the network's, the experimental noise floor, the build-error separation,
the honest error bars and fitted model, and a measured meaning for the
confidence score.

Not new: the wing-shape description, NeuralFoil, XFoil, and the robust and
multi-objective optimization methods. Those are standard tools, applied
carefully.

## Next

A third wind tunnel, with its turbulence documented. A surrogate trained on
something better than XFoil at low speed. And a real test: 3D-print the
wings this pipeline designs, measure the shapes that actually come off the
printer, and put them in a tunnel.
