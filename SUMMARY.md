# The short version

Plain language. Same numbers as the paper, all of them pinned to the data
files by the test suite.

## The problem

Small drones and rooftop wind turbines fly slowly, and slow air misbehaves.
A thin sheet of air near the wing lifts off the surface and curls into a
bubble. That bubble is where most of the drag comes from.

To find a good wing shape you have to try thousands of them. A real physics
solver is way too slow for that, so the field reaches for NeuralFoil
instead. It's a neural network that copies an older solver called XFoil, and
it answers in a fraction of a second. It also hands you a confidence score
with every answer.

Here's the thing. Nobody had checked it where it counts. Its one published
wind-tunnel comparison sits at a speed about ten times faster than a drone
flies. So could you trust it down in the slow regime where people actually
use it? And did that confidence score mean anything? Nobody knew.

## The test

Two wind tunnels. 94 airfoils. 9,100 measurements.

Every published low-speed measurement from the Illinois archive and from an
older Princeton data set went head to head with NeuralFoil's prediction at
the exact same condition. Then XFoil itself got run at all 9,100 of those
points too. That part matters. NeuralFoil is a copy of XFoil, so its error
is really two errors stacked on top of each other: whatever XFoil gets
wrong, plus whatever goes missing in the copying. Running the original pulls
them apart.

Fifteen airfoils showed up in both tunnels. That's rarer than it sounds, and
it answers something no model can. How far apart do two careful experiments
land?

## What came back

NeuralFoil's drag is off by 11 to 12 percent on average, and about 8 percent
in a typical case. Lift is off by a few hundredths. Both tunnels say the
same thing. Down at the slowest speeds the drag error climbs to roughly 20
percent.

But almost none of that is the network's fault. XFoil misses the tunnels by
12 percent at those same points. NeuralFoil differs from XFoil by under 3
percent, and it actually lands a little closer to the measurements than
XFoil does. Odd, but it makes sense. Fit a network to hundreds of thousands
of XFoil runs and you smooth away some of XFoil's own scatter. So a bigger
network trained on more XFoil won't fix this. Better physics would.

Then there are the tunnels themselves. They disagree with each other by 12
percent. Above medium speeds NeuralFoil sits as close to each tunnel as the
tunnels sit to each other, which means no model can be shown to beat it
there. The experiments just can't tell the difference.

The confidence score turned out to be a drag warning and nothing else. High
confidence, drag error around 9 percent. Low confidence, around 32 percent.
Lift error? Ignores the score completely. And what the score is really
picking up on is where XFoil's physics falls apart, which is more than it
was built to do.

The error bars got two to five times wider once I counted airfoils instead
of data points, because a hundred readings off one wing aren't a hundred
independent facts. Everything survived that except one result. In the first
archive, the airfoils the model felt shakiest about were the ones it got
most wrong. In the second archive that just vanished. The data shows why.
Nearly every Princeton airfoil sits at high confidence, so there was no
spread left to measure.

There's a small formula now that does the job of the old lookup table. Give
it a prediction's confidence, speed, angle and camber and it tells you the
drag error to expect. It only ever got tested on airfoils it hadn't seen,
and it was trained on one tunnel then tested on the other. It beats a flat
guess by 14 percent, and its 80 percent band really does cover 80 percent of
cases.

One more finding, and it's an easy one to miss. Some of what looks like
model error is build error. Princeton measured its models after building
them, and running NeuralFoil on the real shape instead of the drawing wipes
out about a sixth of the drag error. The wing in the tunnel was never quite
the wing on paper.

On the design side, a wing tuned for a single condition posts a spectacular
number and falls apart everywhere else. The robust wing gives up half that
peak and holds five times the worst-case performance. And nudging the
optimizer to stay where NeuralFoil is confident costs about 2 percent.

## Credit where it's due

What's new here: the two-tunnel benchmark, and splitting the error into
XFoil's share and the network's. The experimental noise floor. Pulling build
error out of model error. Honest error bars with a fitted model behind them.
And an actual measured meaning for that confidence score.

What isn't: the wing-shape math, NeuralFoil, XFoil, and the robust and
multi-objective optimization methods. Standard tools. Applied carefully, but
standard.

## Next

A third wind tunnel, one that documents its turbulence. A surrogate trained
on something better than XFoil at low speed. And the real test: print the
wings this pipeline designs, measure what actually comes off the printer,
and put them in a tunnel.
