"""
langley_ltpt_parse.py
=============================================================================
Parse Table B1 of McGhee, Walker and Millard (1988), "Experimental Results
for the Eppler 387 Airfoil at Low Reynolds Numbers in the Langley
Low-Turbulence Pressure Tunnel", NASA TM-4062 (NTRS 19890001471), into
data/langley_ltpt_e387/e387_ltpt.csv: one row per tabulated point, with
Re, Mach, alpha_deg, cl, cd, cm and the PDF page it came from.

The PDF is a 1-bit, 300 dpi scan of a line-printer listing. Its embedded
text layer is OCR from 2004 and too noisy to parse on its own: decimal
points come out as bullets, commas or the letter o, 0 as O or Q, 1 as I, i
or l, 9 as q, and on four of the nine pages the columns are split onto
separate lines. Nothing here is typed by hand. Every digit is read twice,
and a third time when the first two disagree:

  reader 1  the scan itself. Each page image is taken out of the PDF with
            pdfimages (no resampling), deskewed, and cut into lines, tokens
            (gaps wider than 40 px) and glyphs (connected components, with
            the pieces of a broken glyph merged). Digits are matched against
            templates learned from the scan: the first set is the mean of
            every glyph the text layer labels cleanly, then refined twice on
            the glyphs the templates themselves read confidently.
  reader 2  the 2004 text layer. pdftotext -bbox gives each OCR word with its
            box; the words that fall on an image token are that token's text.
            The OCR was run on differently rotated copies of the pages, so
            each page's word frame is registered to the deskewed image by a
            search over rotation and vertical shift (see register()). Its
            digits are used only when, after the usual OCR substitutions (O
            to 0, I to 1, q to 9 and so on), there are exactly as many as the
            image token has.
  reader 3  nearest neighbours (k = 9) among glyphs on which readers 1 and 2
            agree, on a more heavily smoothed bitmap.

A digit is accepted when readers 1 and 2 agree; otherwise when reader 3
sides with one of them by at least 7 votes of 9 (and, if it sides with
reader 1, reader 1 was not itself ambiguous). Anything else leaves the
value empty, the column name goes in the row's "unread" field, and the case
is printed. Every disagreement that was settled is printed too. An unread
cl, cd or cm is then settled from a figure legend when one of its candidate
readings is exactly the value a legend prints for a curve whose other two
coefficients match the row (see legend_fill()); such rows are marked in
"legend_filled" and left out of the legend cross-check.

Block conditions (PTINF, M, R) are read the same way from each block's
header line. HYSTERESIS and TURBULATOR come from text-layer words, assigned
to the block whose header is nearest above them.

Checks that stop the script (see parse() and check()):
  - the PDF's SHA-256 matches data/langley_ltpt_e387/source.json
  - on every page at least 90 percent of the registered text-layer digits
    agree with templates bootstrapped from the least skewed page
  - every block has a readable header; no line between a header and its
    first data row looks like a data row that failed to parse; the image has
    at least as many data rows on each page as the text layer has
    four-field rows (the text layer splits rows, it never merges them)
  - image and text layer agree on PTINF, M and R wherever both read them;
    every R is one of the abstract's 60,000 / 100,000 / 200,000 / 300,000 /
    460,000 and every M is in the abstract's 0.03 to 0.13
  - within each block alpha is monotone in the sweep direction (increasing,
    or decreasing for hysteresis runs), allowing repeated points
  - values in physical ranges: -4 < alpha < 17, -0.2 < cl < 1.4,
    0.004 < cd < 0.1, -0.2 < cm < 0
  - repeated alphas inside a block give cl within 0.01, except at R = 60,000
    where the report says two flow regimes occur at the same alpha
  - at least LEGEND_MIN_MATCH of the (cl, cd, cm) triples printed in the
    legends of the report's pressure-distribution figures (read from the text
    layer of PDF pages 5 to 94) match a parsed row on cl and cm

Usage:  .venv/bin/python -I langley_ltpt_parse.py
"""

import hashlib
import html
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "langley_ltpt_e387")
PDF = os.path.join(SRC, "19890001471.pdf")
TEXT_LAYER = os.path.join(SRC, "table_b1_textlayer.txt")
OUT_CSV = os.path.join(SRC, "e387_ltpt.csv")

PAGES = list(range(95, 104))          # PDF pages of Table B1 (report pages 91 to 99)
PAGE_OFFSET = 4                       # report page = PDF page - 4
FIG_PAGES = (5, 94)                   # PDF pages searched for figure legend triples
R_SET = {60000, 100000, 200000, 300000, 460000}
MARGIN = 0.04                         # template score margin below which reader 1 is ambiguous
KNN, KNN_VOTES = 9, 7
LEGEND_MIN_MATCH = 0.8

DATA_SHAPES = [re.compile(r"^-?D{0,2}\.DD$"), re.compile(r"^-?D?\.DDD$"),
               re.compile(r"^(\.DDDD|\*{4})$"), re.compile(r"^-?\.DDDD$")]
COLS = ("alpha", "cl", "cd", "cm")


# ─────────────────────────────────────────────────────────────────────────────
# Source and text layer
# ─────────────────────────────────────────────────────────────────────────────
def verify_pdf():
    with open(os.path.join(SRC, "source.json"), encoding="utf-8") as f:
        want = json.load(f)["pdf_sha256"]
    with open(PDF, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != want:
        raise SystemExit(f"PDF hash {got} does not match source.json {want}")


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def text_layer():
    """pdftotext -layout of each Table B1 page, saved for the record."""
    pages = {p: run(["pdftotext", "-layout", "-f", str(p), "-l", str(p), PDF, "-"]) for p in PAGES}
    with open(TEXT_LAYER, "w", encoding="utf-8") as f:
        for p in PAGES:
            f.write(f"===== PDF page {p}\n{pages[p]}")
    return pages


WORD = re.compile(r'<page width="([\d.]+)" height="([\d.]+)"|'
                  r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>')


def text_words(p, shape):
    """OCR words of PDF page p with their centres in image pixels."""
    out, sx, sy = [], None, None
    for m in WORD.finditer(run(["pdftotext", "-bbox", "-f", str(p), "-l", str(p), PDF, "-"])):
        if m.group(1):
            sx, sy = shape[1] / float(m.group(1)), shape[0] / float(m.group(2))
            continue
        x0, y0, x1, y1 = (float(m.group(k)) for k in range(3, 7))
        out.append(dict(x=(x0 + x1) / 2 * sx, y=(y0 + y1) / 2 * sy, text=html.unescape(m.group(7))))
    return out


def text_data_rows(txt):
    """Number of text-layer lines that split into four numeric-looking fields."""
    n = 0
    for ln in txt.splitlines():
        f = [s.replace(" ", "") for s in re.split(r"\s{3,}", ln.strip()) if s.strip()]
        if (len(f) == 4 and sum(bool(re.search(r"\d", x)) for x in f) >= 3
                and not re.search(r"[A-HJ-NP-Z]{2}", "".join(f))):
            n += 1
    return n


SUBST = {"O": "0", "o": "0", "D": "0", "I": "1", "i": "1", "l": "1", "t": "1",
         "q": "9", "g": "9", "B": "8", "S": "5", "s": "5", "Z": "2", "b": "6"}


def text_digits(field, n):
    """Digits of an OCR field after the usual substitutions, if there are exactly n."""
    if field is None:
        return None
    d = [c for c in (SUBST.get(c, c) for c in field) if c.isdigit()]
    return [int(c) for c in d] if len(d) == n else None


# ─────────────────────────────────────────────────────────────────────────────
# Image: deskew, lines, tokens, glyphs
# ─────────────────────────────────────────────────────────────────────────────
def page_images(tmp):
    run(["pdfimages", "-png", "-f", str(PAGES[0]), "-l", str(PAGES[-1]), PDF, os.path.join(tmp, "p")])
    files = sorted(f for f in os.listdir(tmp) if f.startswith("p-"))
    if len(files) != len(PAGES):
        raise SystemExit(f"expected {len(PAGES)} page images, got {len(files)}")
    out = {}
    for p, fn in zip(PAGES, files):
        im = np.array(Image.open(os.path.join(tmp, fn)).convert("L")) < 128
        if im.shape != (3300, 2544):
            raise SystemExit(f"page {p}: image {im.shape}, expected 3300 x 2544 at 300 dpi")
        out[p] = deskew(im)
    return out


def deskew(im):
    """Rotate so text lines are horizontal; returns (image, angle in degrees)."""
    small = im[::3, ::3].astype(np.uint8)
    a = max((ndi.rotate(small, a, reshape=False, order=0).sum(1).var(), a)
            for a in np.arange(-4, 4.001, 0.05))[1]
    a = max((ndi.rotate(small, b, reshape=False, order=0).sum(1).var(), b)
            for b in np.arange(a - 0.05, a + 0.051, 0.01))[1]
    return ndi.rotate(im.astype(np.uint8), a, reshape=False, order=0) > 0, a


def to_deskewed(y, x, angle, shape):
    """Original (y, x) to the frame of ndi.rotate(reshape=False) by angle."""
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    t = np.deg2rad(angle)
    dy, dx = y - cy, x - cx
    return cy + dy * np.cos(t) - dx * np.sin(t), cx + dy * np.sin(t) + dx * np.cos(t)


def lines_of(im):
    on = im.sum(1) > 2
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= 8:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def glyphs_of(strip, y0):
    """Connected components of a line strip, merged when they overlap in x."""
    lab, _ = ndi.label(strip, structure=np.ones((3, 3)))
    boxes = []
    for k, s in enumerate(ndi.find_objects(lab), start=1):
        if s is None or int((lab[s] == k).sum()) < 6:
            continue
        boxes.append([s[1].start, s[1].stop, s[0].start, s[0].stop])
    boxes.sort()
    merged = []
    for b in boxes:
        if merged:
            m = merged[-1]
            ov = min(m[1], b[1]) - max(m[0], b[0])
            if ov > 0.3 * min(m[1] - m[0], b[1] - b[0]):
                merged[-1] = [min(m[0], b[0]), max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3])]
                continue
        merged.append(b)
    return [dict(x0=b[0], x1=b[1], y0=b[2] + y0, y1=b[3] + y0) for b in merged]


def tokens_of(glyphs, gap=40):
    toks = []
    for g in glyphs:
        if toks and g["x0"] - toks[-1][-1]["x1"] <= gap:
            toks[-1].append(g)
        else:
            toks.append([g])
    return toks


def kind(g, line_top):
    """Shape class of a glyph: digit, dot, minus, star, wide, or other.

    Digits are 25 to 29 px tall and hang from the top of the line. A glyph
    that starts within 5 px of the top and is at least 15 px tall is a digit
    with a broken stroke. Decimal points are about 13 px square and sit low;
    minus signs are flat bars at mid height."""
    h, w = g["y1"] - g["y0"], g["x1"] - g["x0"]
    off = g["y0"] - line_top
    if h >= 15 and w > 26:
        return "wide"
    if h >= 20 or (h >= 15 and off <= 5 and w >= 12):
        return "digit"
    if h <= 11 and w >= 12 and w > 1.6 * h:
        return "minus"
    if h <= 18 and w <= 18:
        return "dot" if off >= 10 else "star"
    return "other"


def bitmap(im, g, line_top, sigma=0.8):
    """A fixed 24 x 32 px window on the glyph: centred in x, hung from the line top."""
    cx = (g["x0"] + g["x1"]) // 2
    v = ndi.gaussian_filter(im[max(line_top - 1, 0):line_top + 31, cx - 12:cx + 12].astype(np.float32), sigma)
    v -= v.mean()
    n = np.linalg.norm(v)
    return (v / n if n else v).ravel()


def read_page(im):
    """Lines of tokens of glyphs, with shape classes and bitmaps."""
    out = []
    for top, bot in lines_of(im):
        toks = []
        for t in tokens_of(glyphs_of(im[top:bot], top)):
            for g in t:
                g["kind"] = kind(g, top)
            # The listing prints **** where no drag was measured. Its asterisks
            # are 19 to 24 px tall, digits 25 to 29: a four-glyph token with no
            # decimal point whose glyphs are all that short is ****.
            if len(t) == 4 and all(g["kind"] != "dot" and 15 <= g["y1"] - g["y0"] <= 24 for g in t):
                for g in t:
                    g["kind"] = "star"
            for g in t:
                if g["kind"] in ("digit", "wide"):
                    g["bm"] = bitmap(im, g, top)
                    g["bm2"] = bitmap(im, g, top, sigma=1.6)
            toks.append(dict(glyphs=t, text=None, top=top, bot=bot,
                             x0=t[0]["x0"] - 8, x1=t[-1]["x1"] + 8))
        out.append(dict(top=top, bot=bot, tokens=toks))
    return out


def attach_text(lines, words, angle, dy, shape_):
    """Give each token the text-layer words whose centres fall on it, for a
    text-layer frame rotated by angle and shifted by dy px from the deskewed one."""
    ys, xs = to_deskewed(np.array([w["y"] for w in words]), np.array([w["x"] for w in words]), angle, shape_)
    ys = ys + dy
    order = np.argsort(xs)
    ys, xs, txt = ys[order], xs[order], [words[i]["text"] for i in order]
    for ln in lines:
        for tok in ln["tokens"]:
            hit = np.nonzero((ys >= tok["top"] - 6) & (ys <= tok["bot"] + 6)
                             & (xs >= tok["x0"]) & (xs <= tok["x1"]))[0]
            tok["text"] = "".join(txt[i] for i in hit) if len(hit) else None


def agreement(lines, reader):
    """(digits where reader and text layer agree, digits compared) over data tokens."""
    ag = n = 0
    for ln in lines:
        if not is_data_line(ln):
            continue
        for tok in ln["tokens"]:
            g = digits(tok)
            t = text_digits(tok["text"], len(g))
            if t:
                r = reader(tok)
                n += len(t)
                ag += sum(a == b for a, b in zip(r, t)) if r else 0
    return ag, n


def register(lines, words, shape_, reader):
    """Find where the 2004 OCR's word boxes sit on the deskewed image.

    The text layer was made on a differently rotated copy of the scan for some
    pages, so its frame is searched for: rotation -4 to 4 deg and vertical shift
    -24 to 24 px, keeping the one under which the most digits agree with the
    reader. Returns (angle, dy, agreeing digits, compared digits)."""
    best = None
    for a in np.arange(-4, 4.001, 0.25):
        for dy in range(-24, 25, 6):
            attach_text(lines, words, a, dy, shape_)
            ag, n = agreement(lines, reader)
            key = (ag, -abs(dy), -abs(a))
            if best is None or key > best[0]:
                best = (key, a, dy, ag, n)
    _, a, dy, ag, n = best
    attach_text(lines, words, a, dy, shape_)
    return a, dy, ag, n


def shape(tok):
    return "".join({"digit": "D", "dot": ".", "minus": "-", "star": "*"}.get(g["kind"], "?")
                   for g in tok["glyphs"])


def is_data_line(line):
    t = line["tokens"]
    return len(t) == 4 and all(p.match(shape(x)) for p, x in zip(DATA_SHAPES, t))


def digits(tok):
    return [g for g in tok["glyphs"] if g["kind"] == "digit"]


# ─────────────────────────────────────────────────────────────────────────────
# Readers
# ─────────────────────────────────────────────────────────────────────────────
def templates_from(pairs):
    T = np.zeros((10, len(pairs[0][0])), np.float32)
    for d in range(10):
        v = [b for b, c in pairs if c == d]
        if not v:
            raise SystemExit(f"no labelled glyphs for digit {d}")
        t = np.mean(v, axis=0)
        t -= t.mean()
        T[d] = t / np.linalg.norm(t)
    return T


def score(T, bm):
    s = T @ bm
    o = np.argsort(s)[::-1]
    return int(o[0]), int(o[1]), float(s[o[0]]), float(s[o[0]] - s[o[1]])


def data_tokens(pages):
    for p in PAGES:
        for ln in pages[p]:
            if is_data_line(ln):
                yield from ln["tokens"]


def learn_templates(pages):
    """Reader 1's templates: text-labelled glyphs first, then two self-refinements."""
    pairs = []
    for tok in data_tokens(pages):
        f = tok["text"]
        if f and re.fullmatch(r"[-\d.,•*°]+", f):
            g = digits(tok)
            t = text_digits(f, len(g))
            if t:
                pairs += [(x["bm"], c) for x, c in zip(g, t)]
    n0 = len(pairs)
    T = templates_from(pairs)
    for _ in range(2):
        conf = []
        for tok in data_tokens(pages):
            for g in digits(tok):
                d, _, s, m = score(T, g["bm"])
                if m >= 2 * MARGIN and s > 0.6:
                    conf.append((g["bm"], d))
        T = templates_from(conf)
    return T, n0, len(conf)


def knn_pool(pages, T):
    """Reader 3's reference set: glyphs where readers 1 and 2 agree."""
    X, y = [], []
    for tok in data_tokens(pages):
        g = digits(tok)
        t = text_digits(tok["text"], len(g))
        if t:
            for x, c in zip(g, t):
                if score(T, x["bm"])[0] == c:
                    X.append(x["bm2"])
                    y.append(c)
    return np.array(X, dtype=np.float64), np.array(y)


def knn(pool, bm2):
    X, y = pool
    with np.errstate(all="ignore"):      # numpy 2.0 on Accelerate warns spuriously in matmul
        s = X @ bm2.astype(np.float64)
    if not np.isfinite(s).all():
        raise SystemExit("non-finite neighbour score")
    o = np.argsort(s)[::-1]
    if s[o[0]] > 0.9999:          # the glyph itself is in the pool
        o = o[1:]
    votes = np.bincount(y[o[:KNN]], minlength=10)
    return int(votes.argmax()), int(votes.max())


def read_digit(T, pool, g, tdig, where, notes, stats):
    d1, d2, s, m = score(T, g["bm"])
    if tdig is not None and tdig == d1:
        stats["agree"] += 1
        return d1
    k, v = knn(pool, g["bm2"])
    if k == d1 and v >= KNN_VOTES and m >= MARGIN:
        stats["r1_r3"] += 1
        if tdig is not None:
            notes.append(f"{where}: image {d1}, text layer {tdig}, neighbours {k} ({v}/{KNN}); took {d1}")
        return d1
    if tdig is not None and k == tdig and v >= KNN_VOTES:
        stats["r2_r3"] += 1
        notes.append(f"{where}: image {d1} (runner-up {d2}, margin {m:.3f}), text layer {tdig}, "
                     f"neighbours {k} ({v}/{KNN}); took {tdig}")
        return tdig
    stats["unread"] += 1
    notes.append(f"{where}: image {d1} or {d2} (margin {m:.3f}), text layer "
                 f"{'illegible' if tdig is None else tdig}, neighbours {k} ({v}/{KNN}); unread")
    return sorted({d1, d2, k} | ({tdig} if tdig is not None else set()))


def read_token(T, pool, tok, where, notes, stats):
    """(value string or None, candidate strings when a digit is unread)."""
    if shape(tok) == "****":
        return "****", []
    g = digits(tok)
    t = text_digits(tok["text"], len(g))
    parts, k = [], 0
    for x in tok["glyphs"]:
        if x["kind"] == "digit":
            d = read_digit(T, pool, x, t[k] if t else None, f"{where} digit {k + 1}", notes, stats)
            parts.append([str(c) for c in d] if isinstance(d, list) else [str(d)])
            k += 1
        elif x["kind"] == "dot":
            parts.append(["."])
        elif x["kind"] == "minus":
            parts.append(["-"])
    if all(len(c) == 1 for c in parts):
        return "".join(c[0] for c in parts), []
    return None, ["".join(c) for c in itertools.product(*parts)]


def header_fields(T, pool, line):
    """PTINF, M and R from a block header line, as read by the image and by the text layer.

    The header prints as four tokens: RUNS n,n  |  PTINF = nn PSI  |  M = 0.nn
    |  R = nnn,nnn. In the last three the value is the first word after the
    equals sign; PSI comes after a word gap and is dropped."""
    t = line["tokens"]
    if len(t) != 4:
        return None
    img, txt = [], []
    for tok in t[1:]:
        gl = tok["glyphs"]
        # the field name's letters come out as digit-sized glyphs; the equals
        # sign is the first glyph that is not
        k = next((i for i, g in enumerate(gl) if g["kind"] != "digit"), None)
        if k is None:
            return None
        s, prev = "", None
        for g in gl[k + 1:]:
            if prev is not None and g["x0"] - prev["x1"] > 18:
                break
            if g["kind"] == "digit":
                d, d2, _, m = score(T, g["bm"])
                if m < MARGIN:                  # ambiguous: reader 3 may settle it
                    k3, v = knn(pool, g["bm2"])
                    d = d if k3 in (d, d2) and v >= KNN_VOTES and (d := k3) is not None else "?"
                s += str(d)
            elif g["kind"] == "dot":
                s += "."
            prev = g
        img.append(s)
        f = (tok["text"] or "").replace("PSI", "")
        f = re.sub(r"^[A-Za-z_]+", "", f)                       # the field name
        f = "".join(SUBST.get(c, c) for c in f)
        m = re.search(r"\d[\d.,•°]*", f[1:] if f[:1] in "=-•\"'.," else f)
        txt.append(m.group(0).replace("•", ".").replace(",", ".").replace("°", ".") if m else "")

    def conv(pt, M, R):
        rules = (("ptinf", pt, r"\d{1,2}", int),
                 ("M", M, r"0\.\d\d", float),
                 ("R", R, r"\d{2,3}\.\d{3}", lambda v: int(v.replace(".", ""))))
        return {k: (fn(v) if re.fullmatch(rx, v or "") else None) for k, v, rx, fn in rules}

    i, x = conv(*img), conv(*txt)
    if all(v is None for v in i.values()):
        return None
    return dict(image=i, text=x)


# ─────────────────────────────────────────────────────────────────────────────
# Parse
# ─────────────────────────────────────────────────────────────────────────────
def parse(pages, words, imgs, reg, n_text_rows, T, pool, notes, stats):
    rows, blocks = [], []
    for p in PAGES:
        lines = pages[p]
        idx = [i for i, ln in enumerate(lines) if is_data_line(ln)]
        groups = []
        for i in idx:
            if groups and i == groups[-1][-1] + 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        a, dy = reg[p]
        flags = [(to_deskewed(w["y"], w["x"], a, imgs[p][0].shape)[0] + dy, w["text"][:10]) for w in words[p]
                 if w["text"].startswith(("HYSTERESIS", "TURBULATOR"))]
        for b, g in enumerate(groups):
            stop = groups[b - 1][-1] if b else -1
            hdr = hline = None
            for i in range(g[0] - 1, stop, -1):
                hdr = header_fields(T, pool, lines[i])
                if hdr:
                    hline = i
                    break
            if hdr is None:
                raise SystemExit(f"page {p}: no header line above the block starting at line {g[0]}")
            for i in range(hline + 1, g[0]):
                t = lines[i]["tokens"]
                if len(t) == 4 and sum("." in shape(x) and len(digits(x)) >= 2 for x in t) >= 3:
                    raise SystemExit(f"page {p} line {i}: looks like a data row but did not parse "
                                     f"({[shape(x) for x in t]})")
            top = lines[hline]["top"]
            fl = {f for y, f in flags if top - 10 <= y < lines[g[0]]["top"]}
            blocks.append(dict(page=p, n_rows=len(g), header=hdr,
                               hysteresis="HYSTERESIS" in fl, turbulator="TURBULATOR" in fl))
            for i in g:
                rd = [read_token(T, pool, tok, f"page {p} line {i} {COLS[c]}", notes, stats)
                      for c, tok in enumerate(lines[i]["tokens"])]
                rows.append(dict(page=p, line=i, block=len(blocks) - 1, raw=[v for v, _ in rd],
                                 cands=[c for _, c in rd]))
        n_img = sum(len(g) for g in groups)
        if n_text_rows[p] > n_img:
            raise SystemExit(f"page {p}: the text layer has {n_text_rows[p]} four-field rows, "
                             f"the image only {n_img} data rows")
    return rows, blocks


def block_conditions(blocks, notes):
    """Settle PTINF, M and R per block: image and text layer must agree where both read."""
    for k, bl in enumerate(blocks):
        im, tx = bl["header"]["image"], bl["header"]["text"]
        for key in ("ptinf", "M", "R"):
            a, b = im[key], tx[key]
            if a is not None and b is not None and a != b:
                raise SystemExit(f"block {k} (page {bl['page']}): {key} reads {a} in the image, "
                                 f"{b} in the text layer")
            if a is None and b is not None:
                notes.append(f"block {k} (page {bl['page']}): {key} ambiguous in the image, text layer gives {b}")
            if a is None and b is None:
                notes.append(f"block {k} (page {bl['page']}): {key} unread by both")
            bl[key] = a if a is not None else b
            bl[key + "_both"] = a is not None and b is not None
    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────────────────────
def legend_triples():
    """(cl, cd, cm) triples printed in figure legends, from the text layer."""
    txt = run(["pdftotext", "-layout", "-f", str(FIG_PAGES[0]), "-l", str(FIG_PAGES[1]), PDF, "-"])
    pat = re.compile(r"(?<![\d.])(-?\d?\.\d{3})\s+(\.\d{4})\s+(-\.\d{4})(?!\d)")
    return [tuple(float(x) for x in m.groups()) for m in pat.finditer(txt)]


def check(d, blocks):
    bad = []
    for k, bl in enumerate(blocks):
        if bl["R"] not in R_SET:
            bad.append(f"block {k} (page {bl['page']}): R {bl['R']} not in {sorted(R_SET)}")
        if bl["M"] is None or not 0.03 <= bl["M"] <= 0.13:
            bad.append(f"block {k} (page {bl['page']}): M {bl['M']} outside 0.03 to 0.13")
    for k, g in d.groupby("block_id"):
        a = g.alpha_deg.dropna().to_numpy()
        hyst = bool(g.hysteresis.iloc[0])
        step = np.diff(a)
        wrong = step > 0.05 if hyst else step < -0.05
        if wrong.any():
            bad.append(f"block {k} (page {g.page.iloc[0]}): alpha not "
                       f"{'decreasing' if hyst else 'increasing'} at {a[1:][wrong].tolist()}")
    for col, lo, hi in (("alpha_deg", -4, 17), ("cl", -0.2, 1.4), ("cd", 0.004, 0.1), ("cm", -0.2, 0.0)):
        v = d[col].dropna()
        for i in v[(v <= lo) | (v >= hi)].index:
            bad.append(f"{col} = {d.loc[i, col]} outside ({lo}, {hi}): page {d.loc[i, 'page']}, "
                       f"alpha {d.loc[i, 'alpha_deg']}")
    for k, g in d.groupby("block_id"):
        if g.Re.iloc[0] == 60000:
            continue
        for a, gg in g.groupby("alpha_deg"):
            if len(gg) > 1 and gg.cl.max() - gg.cl.min() > 0.01:
                bad.append(f"block {k}: repeated alpha {a} gives cl {gg.cl.tolist()}")
    return bad


def legend_check(d):
    res = []
    for cl, cd, cm in legend_triples():
        m = d[np.isclose(d.cl, cl, atol=5e-4) & np.isclose(d.cm, cm, atol=5e-5)]
        res.append(dict(cl=cl, cd=cd, cm=cm, rows=len(m),
                        cd_ok=bool(np.isclose(m.cd, cd, atol=5e-5).any()),
                        where="; ".join(f"Re {r.Re} alpha {r.alpha_deg} cd {r.cd}" for r in m.itertuples())))
    return pd.DataFrame(res)


def legend_fill(rows, notes):
    """Settle an unread cl, cd or cm from a figure legend.

    The pressure-distribution figures print (cl, cd, cm) for each curve, and
    their text layer is a separate OCR of separately typeset numbers. A value
    is filled only when a legend triple matches the row exactly on the other
    two coefficients and its value for this one is exactly one of the
    candidate readings of the unread digit."""
    trip = legend_triples()
    n = 0
    for r in rows:
        for c in (1, 2, 3):
            if r["raw"][c] is not None or not r["cands"][c]:
                continue
            others = [j for j in (1, 2, 3) if j != c]
            if any(r["raw"][j] in (None, "****") for j in others):
                continue
            ov = [float(r["raw"][j]) for j in others]
            hits = {t[c - 1] for t in trip if all(abs(t[j - 1] - v) < 5e-5 for j, v in zip(others, ov))}
            cand = [x for x in r["cands"][c] if any(abs(float(x) - h) < 5e-5 for h in hits)]
            if len(cand) == 1:
                r["raw"][c] = cand[0]
                r["legend_filled"] = COLS[c]
                n += 1
                notes.append(f"page {r['page']} line {r['line']} {COLS[c]}: unread digit settled by a figure "
                             f"legend, {cand[0]} (candidates {', '.join(r['cands'][c])})")
    return n


def to_float(s):
    return np.nan if s is None or s == "****" else float(s)


def main():
    verify_pdf()
    texts = text_layer()
    n_text_rows = {p: text_data_rows(texts[p]) for p in PAGES}
    tmp = tempfile.mkdtemp()
    try:
        imgs = page_images(tmp)
    finally:
        shutil.rmtree(tmp)
    words = {p: text_words(p, imgs[p][0].shape) for p in PAGES}
    pages = {p: read_page(im) for p, (im, a) in imgs.items()}
    print("Deskew angles (deg): " + ", ".join(f"{p}: {imgs[p][1]:+.2f}" for p in PAGES))

    # Bootstrap on the least skewed page, where the text layer's frame and the
    # deskewed one differ by under 10 px anywhere on the table, so its words
    # land on the right tokens without registration.
    p0 = min(PAGES, key=lambda p: abs(imgs[p][1]))
    attach_text(pages[p0], words[p0], 0.0, 0, imgs[p0][0].shape)
    T0, _, _ = learn_templates({p: (pages[p] if p == p0 else []) for p in PAGES})
    reader = lambda tok: [score(T0, g["bm"])[0] for g in digits(tok)]
    reg = {}
    for p in PAGES:
        a, dy, ag, n = register(pages[p], words[p], imgs[p][0].shape, reader)
        reg[p] = (a, dy)
        print(f"  page {p}: text layer registered at {a:+.2f} deg, {dy:+d} px; "
              f"{ag} of {n} digits agree with the bootstrap templates ({ag / max(n, 1):.1%})")
        if n < 200 or ag / n < 0.9:
            raise SystemExit(f"page {p}: text layer registration failed")

    T, n_boot, n_conf = learn_templates(pages)
    pool = knn_pool(pages, T)
    print(f"Reader 1 templates: from {n_boot} text-labelled glyphs, refined on {n_conf}; "
          f"reader 3 pool: {len(pool[1])} glyphs where readers 1 and 2 agree")

    notes, stats = [], dict(agree=0, r1_r3=0, r2_r3=0, unread=0)
    rows, blocks = parse(pages, words, imgs, reg, n_text_rows, T, pool, notes, stats)
    n_fill = legend_fill(rows, notes)
    blocks = block_conditions(blocks, notes)

    recs = []
    for r in rows:
        bl = blocks[r["block"]]
        a, cl, cd, cm = (to_float(v) for v in r["raw"])
        recs.append(dict(Re=bl["R"], Mach=bl["M"], ptinf_psi=bl["ptinf"], alpha_deg=a, cl=cl, cd=cd, cm=cm,
                         hysteresis=bl["hysteresis"], turbulator=bl["turbulator"], block_id=r["block"],
                         page=r["page"], report_page=r["page"] - PAGE_OFFSET,
                         unread=";".join(c for c, v in zip(COLS, r["raw"]) if v is None),
                         legend_filled=r.get("legend_filled", "")))
    d = pd.DataFrame(recs)

    n_dig = sum(stats.values())
    print(f"\nDigits: {n_dig}. Readers 1 and 2 agree on {stats['agree']}; reader 3 settled "
          f"{stats['r1_r3']} for the image and {stats['r2_r3']} for the text layer; "
          f"{stats['unread']} unread, of which {n_fill} values were then settled by figure legends")
    print(f"\n{len(d)} rows in {len(blocks)} blocks")
    for k, bl in enumerate(blocks):
        g = d[d.block_id == k]
        both = "".join(c for c, key in (("P", "ptinf"), ("M", "M"), ("R", "R")) if bl[key + "_both"])
        print(f"  block {k:>2}  page {bl['page']}  R {bl['R']:>6}  M {bl['M']}  PTINF {bl['ptinf']:>2}  "
              f"rows {len(g):>2}  cd {g.cd.notna().sum():>2}  unread {int(g.unread.ne('').sum())}  "
              f"[both readers: {both or '-'}]  "
              f"{'hysteresis ' if bl['hysteresis'] else ''}{'turbulator' if bl['turbulator'] else ''}")
    if notes:
        print(f"\n{len(notes)} notes:")
        for n in notes:
            print("  " + n)

    bad = check(d, blocks)
    if bad:
        raise SystemExit("check failed:\n  " + "\n  ".join(bad))
    print("\nChecks passed: block structure, R and M, alpha monotone per block, ranges, repeated alphas")

    lg = legend_check(d[d.legend_filled.eq("")])     # filled rows would match by construction
    share = lg.rows.gt(0).mean() if len(lg) else 0.0
    print(f"Figure legends: {len(lg)} (cl, cd, cm) triples in the text layer of PDF pages "
          f"{FIG_PAGES[0]} to {FIG_PAGES[1]}; {int(lg.rows.gt(0).sum())} match a row on cl and cm "
          f"(rows filled from legends left out), "
          f"{int(lg.cd_ok.sum())} of those on cd as well")
    for r in lg[lg.rows.eq(0) | ~lg.cd_ok].itertuples():
        print(f"  legend cl {r.cl:.3f} cd {r.cd:.4f} cm {r.cm:.4f}: "
              f"{'no row' if r.rows == 0 else r.where + ' (cd differs)'}")
    if share < LEGEND_MIN_MATCH:
        raise SystemExit(f"only {share:.0%} of legend triples match a parsed row")

    d.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {os.path.relpath(OUT_CSV, HERE)}")
    return d, blocks


if __name__ == "__main__":
    main()
