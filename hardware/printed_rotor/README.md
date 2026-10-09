# Printed E387 test rotors

Five parts for the thrust stand. Four are the same blade at different pitches: the E387 (in the
benchmark, measured in both tunnels), 40 mm chord, 240 mm across, two blades, no twist and no
taper. The fifth is the bare hub.

`printed_rotor.py` made these files and predicted what the stand should read **before anything was
printed**. The predictions are in `data/printed_rotor_prediction.csv` (each rotor at its own pitch)
and `data/printed_rotor_zero_thrust.csv` (the torque at zero thrust), drawn in
`figures/37_printed_rotor_prediction.png`. `data/printed_rotor_design.json` records the SHA-256 of
every STL, so each prediction is tied to the exact file it was made for. If a file changes, rerun
the script and commit before testing.

| File | Pitch | What it is for | Max rpm | Solid mass |
|---|---|---|---|---|
| `rotor_Z-4.6.stl` | -4.6° | zero-thrust set, just below zero lift | 7,750 | 37.3 g |
| `rotor_Z-3.6.stl` | -3.6° | zero-thrust set, at NeuralFoil's zero lift | 7,750 | 37.3 g |
| `rotor_Z-2.6.stl` | -2.6° | zero-thrust set, just above zero lift | 7,750 | 37.3 g |
| `rotor_L+4.0.stl` | +4.0° | lifting rotor, about 60 percent of its power induced | 6,250 | 37.3 g |
| `hub_tare.stl` | none | the hub alone; its torque is subtracted from every rotor | 7,750 | 8.9 g |

Print **two** of `rotor_Z-3.6.stl`: one to spin, one to cut open and measure. One of each of the others.

## Before you print

Measure the motor's prop adapter with the caliper, the threaded part the propeller slides onto.
The files have a 6.0 mm hole. If the adapter is different, regenerate them:

```bash
.venv/bin/python printed_rotor.py --bore 5.0 --stl-only
```

The hole does not touch the blades, so the predictions still stand. Commit the new files anyway,
so their hashes are on record before testing.

## Print settings (Bambu Studio, PETG)

- Leave each part as it loads: hub face down. Do not rotate or scale it.
- 0.08 mm layers.
- Sparse infill 100 percent. The speed limits assume a solid part.
- Supports: tree, build plate only. The blades sit a few millimetres above the bed.
- Same spool and the same settings for every part, so the rotors can be compared with each other.

## After printing

1. Take the supports off. Wet-sand both faces of each blade, 400 grit then 800, with light strokes
   along the span, until no layer step can be felt with a fingernail. Do not round the leading edge
   or thin the trailing edge.
2. Weigh it. A solid rotor is about 37 g. If it is clearly lighter, there are voids inside: do not
   spin it.
3. Balance it. Put the hub on a smooth rod resting across two level edges. If one blade sinks,
   sand a little off that blade's tip until the rotor stays wherever it is left.
4. With the caliper, measure the blade thickness 90 mm from the centre. The design is 3.66 mm.
   Write it down for every rotor.
5. Cut the spare Z-3.6 straight across one blade at 90 mm from the centre. Put the cut face on a
   scanner, or photograph it straight down beside a ruler. That picture is the printed shape at
   75 percent span, and the paper already shows the built shape matters.

## Mounting and direction

- The small dimple on the hub marks the top. The curved side of each blade faces the same way.
- Seen from the dimple side, the rotor turns **counter-clockwise**. The rounded edge goes first.
  If the motor spins the other way, swap any two of its three wires.
- L+4.0 pulls toward the dimple side. The Z rotors make almost no thrust and it can be either sign,
  so the thrust load cell has to read in both directions.

## Safety

- The max rpm in the table is a hard limit. Each one keeps the blade at no more than a fifth of a
  conservative strength for printed PETG, counting centrifugal pull and bending together.
- First spin of every new part: inside a closed box (plywood or polycarbonate), nobody standing in
  line with the blades, an adult in the room, safety glasses on. Go up 1,000 rpm at a time to the
  part's limit and hold it there for a minute. Measure only after that.
- Retire any part that has been dropped, has a crack, or shows white stress marks near the hub.

## What the stand has to resolve

From the NeuralFoil prediction at 28 C and 99.2 kPa. Force is what a load cell would see on a
torque arm 25 mm from the shaft, so it changes with the arm you build.

| Rotor | rpm | Re at 75% span | Thrust | Torque | Force on a 25 mm arm |
|---|---|---|---|---|---|
| Z-3.6 | 3,000 | 70,000 | about 0 | 6.4 N mm | 26 g |
| Z-3.6 | 7,750 | 181,000 | about 0 | 30.3 N mm | 124 g |
| L+4.0 | 3,000 | 70,000 | 0.51 N (51 g) | 9.4 N mm | 38 g |
| L+4.0 | 6,250 | 147,000 | 2.38 N (243 g) | 35.0 N mm | 143 g |

The Z rotors' thrust stays within about 0.13 N (13 g) either way, so the thrust cell has to
resolve a few grams. At 6,000 rpm the zero-thrust torque predicted from NeuralFoil, the UIUC
polars and the Princeton polars is 20.6, 17.4 and 15.4 N mm. Telling those apart means
resolving about 1 N mm (4 g on a 25 mm arm) after the hub is subtracted.

Every run, record: rpm, thrust, torque, air temperature and pressure (the BMP280 from the CanSat
parts will do), and the bare hub's torque at the same rpm, which is subtracted from each rotor.
Compare in coefficient form against Reynolds number at 75 percent span, which takes the weather
out; both columns are in the prediction files.
