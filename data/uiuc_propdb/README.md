# UIUC Propeller Database, the subset used here

Static (zero airspeed) thrust and power measurements of 3D-printed propellers whose
airfoil section is known exactly, from Volume 2 of the UIUC Propeller Database.

Source: https://m-selig.ae.illinois.edu/props/propDB.html (UIUC-propDB.zip, Volume 2,
"UIUC PhD dissertation by Robert Deters and following tests, 2009-2015").

Cite:

- Deters, R. W., Ananda, G. K., and Selig, M. S., "Reynolds Number Effects on the
  Performance of Small-Scale Propellers," AIAA Paper 2014-2151, 32nd AIAA Applied
  Aerodynamics Conference, Atlanta, GA, June 2014.
- Deters, R. W., "Performance and Slipstream Characteristics of Small-Scale Propellers
  at Low Reynolds Numbers," Ph.D. Dissertation, University of Illinois at
  Urbana-Champaign, 2014.
- Brandt, J. B. and Selig, M. S., "Propeller Performance Data at Low Reynolds Numbers,"
  AIAA Paper 2011-1255, 2011 (Volume 1 of the database and the test rig).

## Propellers

Both use the SDA1075 airfoil over the whole blade, from the 30 percent station outward;
inboard of that is the printed hub. Blades are rectangular with a constant geometric
pitch, so the twist is helical.

| Prefix | c/R | Diameters | Variants |
|---|---|---|---|
| `da4002` | 0.18 | 5 in, 9 in | pitch 2.85, 4.76, 6.75, 8.95 in (9 in); 1.58, 2.65, 3.75, 4.92 in (5 in); 2 blades |
| `da4022` | 0.23 | 5 in, 9 in | pitch 6.75 in (9 in), 3.75 in (5 in); 2, 3 and 4 blades |

## Files

- `*_geom.txt`: `r/R  c/R  beta`, the chord and pitch angle (degrees, from the plane of
  rotation) at 18 stations, measured on the built propeller with PropellerScanner. The
  3- and 4-blade variants use the same blades as the 2-blade case and so share its file.
- `da4002_geom.txt`: the designed geometry of the DA4002, 15 stations from r/R = 0.30,
  for the sensitivity check between drawing and built article.
- `*_static_*.txt`: `RPM  CT  CP` at zero airspeed, one row per RPM setting. Propeller
  convention: CT = T / (rho n^2 D^4), CP = P / (rho n^3 D^5), n in revolutions per second.
- `sda1075.dat`: the SDA1075 coordinates, 61 points, transcribed with pdftotext from
  Table 4 of Deters, Ananda and Selig (2014). Selig format, upper surface from the
  trailing edge to the leading edge then the lower surface back. The trailing edge is
  deliberately blunt (0.011c) so the 5-inch blades could be printed.

Files are unmodified copies. The four-character code before `_static` or the RPM in a
file name is the UIUC run identifier.
