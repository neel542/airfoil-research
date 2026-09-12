"""
build_paper_html.py
=============================================================================
Turn PAPER.md into a single self-contained HTML file with the figures embedded
inline (as base64), styled like an academic paper. Open the result in any
browser and use "Print -> Save as PDF" to get a shareable PDF. No external
tools or internet needed.
"""

import os, re, base64, markdown

OUT = os.path.dirname(os.path.abspath(__file__))
MD  = os.path.join(OUT, "PAPER.md")
FIG = os.path.join(OUT, "figures")
HTML_OUT = os.path.join(OUT, "PAPER.html")

# Which figures to embed, where (after the line containing the key phrase), and
# their captions.
FIGURE_INSERTS = [
    ("missing that bubble drag.", [
        ("12_e387_CL_polars.png",
         "Figure 1. Lift coefficient versus angle of attack for the clean Eppler "
         "E387 at Re of about 100k, 200k and 500k: NeuralFoil, headless XFoil, and "
         "wind-tunnel measurement (UIUC; Selig & McGranahan 2004)."),
        ("13_e387_CD_polars.png",
         "Figure 2. Drag coefficient (log scale) for the same runs. At Re of about "
         "100k the measured drag sits above both predictions across the low-drag "
         "range: the laminar separation bubble that both NeuralFoil and XFoil "
         "under-predict."),
    ]),
    ("should be read as optimistic.", [
        ("17_multifoil_parity.png",
         "Figure 3. NeuralFoil versus wind tunnel for all 4,428 clean benchmark "
         "points on 55 airfoils, coloured by Reynolds number. Lift scatters within "
         "about 0.05 for most points; drag within about 20%."),
        ("21_uiuc_error_by_airfoil.png",
         "Figure 4. Mean drag and lift error by airfoil, clean runs. Grey bars are "
         "the two airfoils excluded from the benchmark because the 17-parameter "
         "shape description cannot reproduce them."),
    ]),
    ("had not been checked against experiment either.", [
        ("23_uiuc_tripped_vs_forced_transition.png",
         "Figure 5. Boundary-layer trip runs near Re = 200k. Diamonds: clean "
         "measurement; squares: tripped measurement. Running NeuralFoil with "
         "transition forced at the trip location (dashed) reproduces the tripped "
         "drag that a free-transition run (solid) misses."),
    ]),
    ("the two branches can differ.", [
        ("24_uiuc_stall_parity.png",
         "Figure 6. Maximum lift and the angle it occurs at, NeuralFoil versus wind "
         "tunnel, for the 298 clean lift sweeps that passed through stall."),
        ("25_uiuc_lift_curves.png",
         "Figure 7. Three lift curves through stall. Triangles: increasing angle; "
         "open circles: decreasing angle (the gap is stall hysteresis); line: "
         "NeuralFoil; grey band: NeuralFoil's confidence, which collapses past "
         "the stall."),
    ]),
    ("Section 3.9 uses the score inside the optimizer.", [
        ("22_uiuc_confidence_calibration.png",
         "Figure 8. What NeuralFoil's confidence score tracks in the UIUC data. Left: "
         "lift error is flat across confidence bins. Middle: drag error falls "
         "from 32% to 9% as confidence rises. Right: the same relationship at the "
         "airfoil level (section 3.4 shows this panel does not replicate in the "
         "Princeton data, and why)."),
    ]),
    ("doubles the error.", [
        ("26_two_tunnels_and_ncrit.png",
         "Figure 9. Left: NeuralFoil's drag error by Reynolds number against the "
         "UIUC tunnel (red) and the Princeton tunnel (purple), mean and median. "
         "Middle and right: sweeping the transition parameter n_crit on the "
         "Princeton data; 9, the value used throughout, gives the lowest error "
         "and the smallest bias."),
    ]),
    ("error is the model's.", [
        ("27_tunnel_vs_tunnel.png",
         "Figure 10. The same airfoil in two wind tunnels. Diamonds: UIUC; squares: "
         "Princeton; dashed: NeuralFoil on the design coordinates; solid: NeuralFoil "
         "on the measured shape of the Princeton model. For the E387 at Re = 100k "
         "the two experiments differ by more than either differs from the model."),
    ]),
    ("the airfoil it was meant to be.", [
        ("28_measured_vs_design_geometry.png",
         "Figure 11. Left: drag error per Princeton model with design versus "
         "measured coordinates; points below the line improved. Right: the "
         "as-built deviation of each model against how much that deviation alone "
         "moves NeuralFoil's drag; the dotted line is the 0.5% chord build-error "
         "scale used in section 2.5."),
    ]),
    ("next to NeuralFoil's 0.079.", [
        ("32_error_vs_reynolds.png",
         "Figure 12. Mean drag error against Reynolds number in both archives, with "
         "XFoil's error and the network's own emulation error separated. Both tools "
         "improve steadily as the air speeds up, by a factor of 2.4 in the UIUC set "
         "and 1.9 in the Princeton set, while the part contributed by the network "
         "stays flat and small at every speed. The shaded band is the Reynolds range "
         "the rotor blade of section 3.5 spans, root at its left edge and tip at its "
         "right."),
    ]),
    ("relies on that.", [
        ("29_xfoil_decomposition.png",
         "Figure 13. Where the drag error comes from. Left and middle: mean drag "
         "error by Reynolds number in each tunnel for NeuralFoil versus tunnel "
         "(blue), XFoil versus tunnel (green) and NeuralFoil versus XFoil (grey, "
         "the network's own contribution), on the points where XFoil converged. "
         "Right: the same three errors by confidence bin, both tunnels pooled; "
         "the score tracks XFoil's error against experiment at least as closely "
         "as the network's error against XFoil."),
    ]),
    ("score itself.", [
        ("30_clustered_statistics.png",
         "Figure 14. Honest uncertainty. Left and middle: point estimates with "
         "airfoil-cluster bootstrap intervals (thin) and naive point-bootstrap "
         "intervals (thick), with the width ratio labelled; clustered intervals "
         "are two to five times wider, and every headline result survives. Right: "
         "the confidence-versus-drag-error correlation split by level; in the "
         "Princeton data it lives entirely within polars, which is why the "
         "airfoil-level result does not replicate there."),
    ]),
    ("model without this paper.", [
        ("31_error_model.png",
         "Figure 15. The fitted error model. Left: held-out calibration by decile "
         "of predicted drag error, whole airfoils held out. Middle: multiplicative "
         "effect of each feature. Right: expected drag error (solid) and its 80th "
         "percentile (dotted) against confidence at four Reynolds numbers for a "
         "representative section."),
    ]),
    ("sits on a razor-thin peak.", [
        ("18_LD_vs_AoA_uncertainty.png",
         "Figure 16. Lift-to-drag ratio versus angle of attack for airfoil A "
         "(single-point) and B (robust) at three Reynolds numbers. Hatched bands "
         "show the measured 15% surrogate uncertainty; both airfoils sit below "
         "NeuralFoil's validated confidence range, so the band is a lower bound."),
        ("19_tradeoff_uncertainty.png",
         "Figure 17. Peak versus worst-case L/D across the design family. Error "
         "bars are one-sided (true values trend downward at low Re). Airfoil A's "
         "peak near 233 is produced where NeuralFoil reports near-zero confidence."),
    ]),
    ("and no drag error can touch it.", [
        ("33_rotor_power_uncertainty.png",
         "Figure 18. Left: every blade station of the design rotor sits inside the "
         "benchmark's Reynolds range, and the fitted error model's expectation "
         "(dotted) is worst on the inboard blade. Centre: only the 37 percent of "
         "hover power that is profile power can respond to a drag error. Right: "
         "1,000 Monte Carlo draws of the measured error through the rotor, with the "
         "error correlated along the blade and independent along it."),
    ]),
    ("cares about drag.", [
        ("34_weight_amplification.png",
         "Figure 19. Left: the take-off mass that closes the design loop, against the "
         "section drag error carried into it. Right: the same three cases as a chain. "
         "The closure loop roughly doubles the hover-power error, and take-off weight "
         "still moves about two thirds of a percent per percent of section drag."),
    ]),
    ("propagate into the 1.7.", [
        ("35_forward_flight.png",
         "Figure 20. The same rotor in level forward flight. Left: shaft power and "
         "its three parts against speed, showing the power bucket with its minimum "
         "at 12 m/s. Right: the profile share of shaft power climbs from 41 percent "
         "in hover to 68 percent at 14 m/s, and the power error caused by a fixed "
         "11.7 percent section drag error climbs with it, to 1.7 times the hover "
         "value."),
    ]),
    ("the two extremes are shown in the figure.", [
        ("20_trust_vs_performance.png",
         "Figure 21. The confidence-aware optimizer. Left: worst-case L/D against "
         "mean confidence as the weight w_conf is turned up; the first step, from "
         "confidence 0.16 to 0.96, costs 2% of predicted L/D. Right: the two "
         "extreme shapes."),
    ]),
]


def check_markers(text):
    """A renamed section silently drops a figure, so refuse to build instead."""
    missing = [m for m, _ in FIGURE_INSERTS if m not in text]
    if missing:
        raise SystemExit("unknown figure marker, the paper no longer contains:\n  "
                         + "\n  ".join(missing))


def img_tag(fname, caption):
    path = os.path.join(FIG, fname)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{fname}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


# Read markdown, insert figure blocks after the marker lines.
_raw = open(MD, encoding="utf-8").read()
check_markers(_raw)
lines = _raw.split("\n")
out_lines = []
for line in lines:
    out_lines.append(line)
    for marker, figs in FIGURE_INSERTS:
        if marker in line:
            for fname, cap in figs:
                out_lines.append("")
                out_lines.append(f"<!--FIGURE:{fname}|{cap}-->")
                out_lines.append("")

md_text = "\n".join(out_lines)
body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# Replace figure placeholders with embedded images.
def _sub(m):
    fname, cap = m.group(1).split("|", 1)
    return img_tag(fname, cap)
body = re.sub(r"<!--FIGURE:(.*?)-->", _sub, body)

CSS = """
:root { color-scheme: light; }
body { font-family: Georgia, 'Times New Roman', serif; max-width: 820px;
       margin: 0 auto; padding: 48px 40px; line-height: 1.55; color: #1a1a1a;
       background: #fff; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 4px; }
h2 { font-size: 1.25rem; border-bottom: 1px solid #ddd; padding-bottom: 4px;
     margin-top: 34px; }
h3 { font-size: 1.05rem; margin-top: 22px; color: #333; }
p, li { font-size: 0.98rem; }
strong { color: #000; }
hr { border: none; border-top: 1px solid #e0e0e0; margin: 26px 0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.9rem; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
th { background: #f3f3f3; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
       font-family: 'SF Mono', Menlo, monospace; font-size: 0.85em; }
figure { margin: 22px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; border: 1px solid #e5e5e5; }
figcaption { font-size: 0.82rem; color: #555; margin-top: 8px; text-align: left;
             font-style: italic; }
@media print {
  body { padding: 0; max-width: none; }
  h2 { page-break-after: avoid; }
}
"""

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Airfoil paper - Neel Madhav</title>
<style>{CSS}</style></head>
<body>{body}</body></html>"""

open(HTML_OUT, "w", encoding="utf-8").write(html)
kb = os.path.getsize(HTML_OUT) / 1024
print(f"wrote PAPER.html ({kb:.0f} KB, figures embedded)")
print("Open it in a browser, then Print -> Save as PDF.")
