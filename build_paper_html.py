"""
build_paper_html.py
=============================================================================
Turn PAPER.md into a single self-contained HTML file with the figures embedded
inline (as base64), styled like an academic paper. Open the result in any
browser and use "Print -> Save as PDF" to get a shareable PDF. No external
tools or internet needed.
"""

import os, re, base64, markdown

OUT = "/Users/neelmadhav/Airfoil research"
MD  = os.path.join(OUT, "PAPER.md")
FIG = os.path.join(OUT, "figures")
HTML_OUT = os.path.join(OUT, "PAPER.html")

# Which figures to embed, where (after the line containing the key phrase), and
# their captions.
FIGURE_INSERTS = [
    ("razor-thin peak", [
        ("18_LD_vs_AoA_uncertainty.png",
         "Figure 1. Lift-to-drag ratio versus angle of attack for airfoil A "
         "(single-point) and B (robust) at three Reynolds numbers. Hatched bands "
         "show the measured +/-16% surrogate uncertainty; both airfoils sit below "
         "NeuralFoil's validated confidence range, so the band is a lower bound."),
        ("19_tradeoff_uncertainty.png",
         "Figure 2. Peak versus worst-case L/D across the design family. Error "
         "bars are one-sided (true values trend downward at low Re). Airfoil A's "
         "peak near 233 is produced where NeuralFoil reports near-zero confidence."),
    ]),
    ("not the shape description's", [
        ("12_e387_CL_polars.png",
         "Figure 3. Lift coefficient versus angle of attack for the Eppler E387: "
         "NeuralFoil, headless XFoil, and wind-tunnel measurement. Lift agrees to "
         "within about 5-8%."),
        ("13_e387_CD_polars.png",
         "Figure 4. Drag coefficient (log scale). The wind-tunnel points sit above "
         "the predictions near Re = 100k, angle 1-5 degrees: this is the laminar "
         "separation bubble that both NeuralFoil and XFoil under-predict."),
        ("17_multifoil_parity.png",
         "Figure 5. NeuralFoil versus experiment for both airfoils (E387 and "
         "SD2030). Lift falls within +/-0.05; drag scatters within about +/-20%."),
        ("16_e387_confidence_error.png",
         "Figure 6. NeuralFoil's self-reported confidence versus its true error. "
         "The negative trend means the model can flag when it is unreliable."),
    ]),
    ("no prior airfoil-optimization work feeds", [
        ("20_trust_vs_performance.png",
         "Figure 7. The uncertainty-aware optimizer. Left: turning up the "
         "confidence dial moves the design from an untrustworthy corner "
         "(confidence 0.14) to a trustworthy one (0.96); worst-case L/D actually "
         "improves at first, then trades off slowly. Right: the blind-optimum and "
         "trust-aware shapes differ."),
    ]),
]


def img_tag(fname, caption):
    path = os.path.join(FIG, fname)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{fname}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


# Read markdown, insert figure blocks after the marker lines.
lines = open(MD, encoding="utf-8").read().split("\n")
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
