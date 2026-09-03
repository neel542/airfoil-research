"""
soartech8_parse.py
=============================================================================
Parse the SoarTech 8 data set ("Airfoils at Low Speeds", Selig, Donovan and
Fraser, 1989; Princeton University 3 x 4 ft smoke tunnel, 1986-1989) into
tidy CSV files. The raw files are the official plain-text archive
Stec8.zip from https://m-selig.ae.illinois.edu/uiuc_lsat.html, unpacked in
data/soartech8/. Nothing is transcribed by hand.

Raw files
  ALL.PD        drag polars: for each airfoil/configuration block, the
                builder, then for each Reynolds number a list of
                (CL, CD, alpha) rows, the VAX file name and the figure
                number in the book. 127 blocks, 54 airfoils.
  *.NN[X]       lift curves (alpha, CL) at nominal Re = NN x 10k; header
                gives the airfoil label, builder and measured Re.
  ALL.DAP       measured ("profiler") coordinates of the actual wind-tunnel
                models, one block per model, with the model chord (inches).
  *.COR         design coordinates (Eppler order, TE -> upper -> LE ->
                lower -> TE).

Configuration labels are parsed from the airfoil name in ALL.PD:
  clean    no modifier, "REPEAT", or "F0" (flap at zero degrees)
  tripped  "UST h, w, N%" / "LST ..." (upper / lower surface trip strip at
           x/c = N%) and "BUMP SHOT TRIP N%" (upper surface)
  flap     NF3/NF6/PF3/PF6 (flap deflected), GFA/GFB/GFC (Gurney flaps)
  other    blocks that mix configurations across Re ("VARIOUS TRIPS",
           "TRIPS AT 300k", ...), blowing, clay leading edge, high
           turbulence, thickened trailing edges without measured geometry,
           and the "SANDED BALSA FINISH" model

Outputs
  data/soartech8_experimental.csv        drag polars, one row per point
  data/soartech8_experimental_lift.csv   lift curves, one row per point
  data/soartech8_measured_coords.csv     measured model coordinates (x/c, y/c)
  data/soartech8_design_coords.csv       design coordinates

The raw files are 1989 DOS text: CR/LF line ends, a Ctrl-Z end-of-file
marker in some files, and one stray control byte (0x18) inside a drag value
on line 551 of ALL.PD ("0.01\x180"). All control characters are stripped
before parsing, which reads that value as 0.0100 (its neighbours in the
polar are 0.0096 and 0.0109). No other byte was touched.
"""

import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "soartech8")
DATA = os.path.join(HERE, "data")

# Model key (ALL.PD name with modifiers stripped) -> block name in ALL.DAP.
# Only exact, unambiguous matches are used; anything else falls back to the
# design coordinates and is marked geom_source = "design".
DAP_NAME = {
    "E193 MODIFIED": "E193MOD", "NACA 0009": "NACA0009", "NACA 2.5411": "NACA2.5411",
    "NACA 6409": "NACA6409", "NACA 64A010": "NACA64a010", "WB135/35": "WB13535",
    "WB140/35/FB": "WB14035", "E214C F0": "E214C", "SD7032C F0": "SD7032C",
    "SD7003 REPEAT": "SD7003", "S2055 REPEAT": "S2055", "E387A REPEAT": "E387A",
    "SD6080 THICK T.E.": "SD6080TTE",
}
# Model key -> design coordinate file stem (*.COR)
COR_NAME = {
    "E193 MODIFIED": "E193MOD", "NACA 0009": "NACA0009", "NACA 2.5411": "NACA2411",
    "NACA 6409": "NACA6409", "NACA 64A010": "NACA64", "WB135/35": "WB13535",
    "WB140/35/FB": "WB14035", "HQ2/9A": "HQ2-9", "HQ2/9B": "HQ2-9",
    "E214C F0": "E214", "SD7032C F0": "SD7032", "FX63-137A": "FX63-137",
    "FX63-137B": "FX63-137", "MB253515": "MB253515",
}
# Model key -> AeroSandbox database name (for cross-referencing with the
# UIUC benchmark). None = not in the database.
ASB_NAME = {
    "AQUILA": None, "CLARK-Y": "clarky", "DAE51": "dae51", "DF101": "df101",
    "DF102": "df102", "DF103": None, "E193": "e193", "E193 MODIFIED": None,
    "E205": "e205", "E214": "e214", "E374": "e374", "E387": "e387", "FLAT PLATE": None,
    "FX60-100": "fx60100", "FX63-137": "fx63137", "HQ2/9": "hq209", "J5012": "j5012",
    "MB253515": None, "MILEY": "miley", "NACA 0009": "naca0009", "NACA 2.5411": "naca2411",
    "NACA 6409": "naca6409", "NACA 64A010": "naca64a010", "RG15": "rg15", "S2048": "s2048",
    "S2055": "s2055", "S2091": "s2091", "S3010": "s3010", "S3014": "s3014", "S3016": "s3016",
    "S3021": "s3021", "S4061": "s4061", "S4062": "s4062", "S4180": "s4180", "S4233": "s4233",
    "SD2030": "sd2030", "SD2083": "sd2083", "SD5060": "sd5060", "SD6060": "sd6060",
    "SD6080": "sd6080", "SD7003": "sd7003", "SD7032": "sd7032", "SD7037": "sd7037",
    "SD7043": "sd7043", "SD7062": "sd7062", "SD7080": "sd7080", "SD7084": "sd7084",
    "SD7090": "sd7090", "SD8000": "sd8000", "SD8020": "sd8020", "SD8040": "sd8040",
    "SPICA": None, "WB135/35": None, "WB140/35/FB": None,
}

CONTROL = re.compile(r"[\x00-\x1f\x7f]")   # CR, Ctrl-Z, and the one stray 0x18 byte

MODIFIER_RE = re.compile(
    r"\b(UST|LST|BUMP SHOT TRIP|REPEAT|HIGH TURBULENCE|THICK T\.E\.|CLAY L\.E\.|"
    r"BLOWING|TRIPS? AT|TWO UST|VARIOUS TRIPS|PLAIN & TWO TRIPS|LOOSE, TIGHT COVERING|"
    r"SANDED BALSA FINISH|NF\d|PF\d|GF[ABC]|F0)\b")


def model_key(label):
    """'E387A UST .020, .125, 20%' -> 'E387A'; keeps 'F0', 'REPEAT', 'MODIFIED',
    'THICK T.E.' because those name a distinct model/geometry file."""
    keep = {"E214C F0", "SD7032C F0", "SD7003 REPEAT", "S2055 REPEAT", "E387A REPEAT",
            "SD6080 THICK T.E.", "E193 MODIFIED"}
    for k in keep:
        if label.startswith(k):
            return k
    m = MODIFIER_RE.search(label)
    key = label[:m.start()] if m else label
    key = key.strip().rstrip(",")
    if key.startswith("NACA "):
        return key
    return key.split(" ")[0] if key not in ("FLAT PLATE",) else key


def family(key):
    """'E387A' -> 'E387', 'SD7032C F0' -> 'SD7032', 'HQ2/9A' -> 'HQ2/9'."""
    k = key.replace(" F0", "").replace(" REPEAT", "")
    if k in ("E193 MODIFIED", "SD6080 THICK T.E."):
        return k
    if k.startswith("NACA ") or k in ("FLAT PLATE", "MB253515", "WB135/35", "WB140/35/FB"):
        return k
    m = re.match(r"^([A-Z]+\d[\d/.-]*?)([A-D])?$", k)
    return m.group(1) if m else k


def classify(label):
    if re.search(r"\b(NF\d|PF\d|GF[ABC])\b", label):
        return "flap"
    if re.search(r"HIGH TURBULENCE|BLOWING|CLAY L\.E\.|VARIOUS TRIPS|PLAIN & TWO|TRIPS AT|"
                 r"UST AT|TWO UST|LOOSE, TIGHT|SANDED BALSA", label):
        return "other"
    if re.search(r"THICK T\.E\.", label) and label not in DAP_NAME:
        return "other"
    if re.search(r"\b(UST|LST|BUMP SHOT TRIP)\b", label):
        return "tripped"
    return "clean"


def trip_locations(label):
    xu = xl = np.nan
    m = re.search(r"UST[^%]*?(\d+)%", label)
    if m:
        xu = int(m.group(1)) / 100
    m = re.search(r"LST[^%]*?(\d+)%", label)
    if m:
        xl = int(m.group(1)) / 100
    m = re.search(r"BUMP SHOT TRIP (\d+)%", label)
    if m:
        xu = int(m.group(1)) / 100
    return xu, xl


def parse_all_pd(path):
    lines = [CONTROL.sub("", ln) for ln in open(path, errors="replace")]
    rows, i, nblocks = [], 0, 0
    while i < len(lines):
        if not lines[i].startswith("Airfoil"):
            i += 1
            continue
        label = lines[i][len("Airfoil"):].strip()
        builder = lines[i + 1][len("Builder"):].strip()
        n_re = int(lines[i + 2].split()[0])
        i += 3
        blocks = []
        for _ in range(n_re):
            Re = int(round(float(lines[i].split()[0])))
            n = int(lines[i + 1].split()[0])
            pts = [[float(v) for v in lines[i + 2 + k].split()[:3]] for k in range(n)]
            blocks.append((Re, pts))
            i += 2 + n
        m = re.match(r"Data file name\s+(\S*)", lines[i])
        datafile = m.group(1) if m else ""
        fig = lines[i + 1].replace("Fig.", "").strip()
        i += 2
        nblocks += 1
        key, cfg = model_key(label), classify(label)
        xu, xl = trip_locations(label) if cfg == "tripped" else (np.nan, np.nan)
        for Re, pts in blocks:
            for cl, cd, a in pts:
                rows.append(dict(airfoil_label=label, model=key, family=family(key),
                                 asb_name=ASB_NAME.get(family(key)) or "",
                                 builder=builder, config=cfg, xtr_upper=xu, xtr_lower=xl,
                                 Re=Re, alpha=a, CL=cl, CD=cd, datafile=datafile, fig=fig))
    return pd.DataFrame(rows), nblocks


def parse_coords_blocks(path, header_re):
    """Blocks of 'x y' lines separated by header lines matching header_re."""
    out, name, chord = {}, None, np.nan
    for ln in open(path, errors="replace"):
        ln = CONTROL.sub("", ln)
        m = header_re.match(ln)
        if m:
            name = m.group(1)
            chord = float(m.group(2)) if m.lastindex and m.lastindex >= 2 else np.nan
            out[name] = dict(chord=chord, pts=[])
            continue
        p = ln.split()
        if name and len(p) == 2:
            try:
                out[name]["pts"].append((float(p[0]), float(p[1])))
            except ValueError:
                pass
    return out


def parse_lift_files():
    rows = []
    for fn in sorted(os.listdir(RAW)):
        if not re.match(r"^[A-Z0-9.-]+\.\d{2}[A-Z]?$", fn):
            continue
        lines = [CONTROL.sub("", ln) for ln in open(os.path.join(RAW, fn), errors="replace")]
        label, builder = lines[1].strip(), lines[2].strip()
        label = re.sub(r"\s+", " ", label).replace("0 DEG FLAP", "F0").replace("FX 63-137", "FX63-137")
        m = re.match(r"\s*(\d+)K\s+([\d.]+)", lines[3])
        Re_nom, Re = int(m.group(1)) * 1000, int(round(float(m.group(2))))
        key, cfg = model_key(label), classify(label)
        for ln in lines[5:]:
            p = ln.split()
            if len(p) >= 2:
                try:
                    a, cl = float(p[0]), float(p[1])
                except ValueError:
                    continue
                rows.append(dict(file=fn, airfoil_label=label, model=key, family=family(key),
                                 asb_name=ASB_NAME.get(family(key)) or "", builder=builder,
                                 config=cfg, Re_nom=Re_nom, Re=Re, alpha=a, CL=cl))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd_df, nblocks = parse_all_pd(os.path.join(RAW, "ALL.PD"))
    print(f"ALL.PD: {nblocks} blocks, {len(pd_df)} drag-polar points, "
          f"{pd_df.model.nunique()} models, {pd_df.family.nunique()} airfoil families, "
          f"Re {pd_df.Re.min()}-{pd_df.Re.max()}")
    print(pd_df.groupby("config").agg(points=("CL", "size"), blocks=("airfoil_label", "nunique")))

    dap = parse_coords_blocks(os.path.join(RAW, "ALL.DAP"),
                              re.compile(r"^(\S+) profiler data\.\s+Chord = ([\d.]+)"))
    cor = {}
    for fn in sorted(os.listdir(RAW)):
        if fn.endswith(".COR"):
            pts = parse_coords_blocks(os.path.join(RAW, fn), re.compile(r"^([A-Za-z0-9./-]+)\s*$"))
            for k, v in pts.items():
                if len(v["pts"]) > 10:
                    cor[fn[:-4]] = v
    print(f"ALL.DAP: {len(dap)} measured models;  *.COR: {len(cor)} design coordinate sets")

    # geometry lookup for every model that has clean or tripped drag data
    geo_rows = []
    for key in sorted(pd_df.model.unique()):
        dn = DAP_NAME.get(key, key)
        if dn not in dap:                       # e.g. 'E387B' has a DAP block; 'SD7032A' does not
            dn = None
        fam = family(key)
        cn = COR_NAME.get(key) or COR_NAME.get(fam) or fam
        cn = cn if cn in cor else (key if key in cor else None)
        geo_rows.append(dict(model=key, family=fam, dap_name=dn or "", cor_name=cn or "",
                             asb_name=ASB_NAME.get(fam) or ""))
    geo = pd.DataFrame(geo_rows)
    pd_df = pd_df.merge(geo[["model", "dap_name", "cor_name"]], on="model", how="left")
    pd_df["geom_source"] = np.where(pd_df.dap_name != "", "measured",
                                    np.where(pd_df.cor_name != "", "design", "none"))
    pd_df.to_csv(os.path.join(DATA, "soartech8_experimental.csv"), index=False)
    print("\nmodels with clean/tripped data and their geometry source:")
    g = (pd_df[pd_df.config.isin(["clean", "tripped"])]
         .groupby(["model", "geom_source", "dap_name", "cor_name", "asb_name"], dropna=False)
         .agg(configs=("config", lambda s: "/".join(sorted(set(s)))), points=("CL", "size"))
         .reset_index())
    print(g.to_string(index=False))
    print("\nblocks classified 'other' (excluded):")
    print("  " + "\n  ".join(sorted(pd_df[pd_df.config == "other"].airfoil_label.unique())))

    mc = pd.DataFrame([dict(dap_name=k, chord_in=v["chord"], x=x, y=y)
                       for k, v in dap.items() for x, y in v["pts"]])
    mc.to_csv(os.path.join(DATA, "soartech8_measured_coords.csv"), index=False)
    dc = pd.DataFrame([dict(cor_name=k, x=x, y=y) for k, v in cor.items() for x, y in v["pts"]])
    dc.to_csv(os.path.join(DATA, "soartech8_design_coords.csv"), index=False)

    lift = parse_lift_files()
    lift = lift.merge(geo[["model", "dap_name", "cor_name"]], on="model", how="left")
    lift.to_csv(os.path.join(DATA, "soartech8_experimental_lift.csv"), index=False)
    print(f"\nlift files: {lift.file.nunique()} files, {len(lift)} points, "
          f"{lift.model.nunique()} models, Re {lift.Re.min()}-{lift.Re.max()}")
    print(lift.groupby("config").agg(points=("CL", "size"), files=("file", "nunique")))
    print("\nwrote data/soartech8_experimental.csv, soartech8_experimental_lift.csv, "
          "soartech8_measured_coords.csv, soartech8_design_coords.csv")
