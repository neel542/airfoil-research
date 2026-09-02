"""
uiuc_lsat_parse.py
=============================================================================
Parse the official UIUC Low-Speed Airfoil Tests (LSAT) ASCII polar files in
data/uiuc_lsat/ into two tidy tables: data/uiuc_experimental.csv (drag runs:
alpha, Cl, Cd) and data/uiuc_experimental_lift.csv (lift runs: alpha, Cl, Cm,
over a wider alpha range that usually passes through stall, with increasing
and decreasing sweeps), and derive the clean-only Eppler E387 drag file used
by e387_neuralfoil_validation.py.

Source: https://m-selig.ae.illinois.edu/pd.html  (Selig et al., "Summary of
Low-Speed Airfoil Data" Vols. 1-3, 1995-1998; Selig & McGranahan,
NREL/SR-500-34515, 2004, which the archive files as Vol. 4). Each *.DRG / *_drg.txt file is one
airfoil model in one configuration; it holds several Reynolds-number blocks of
(alpha, Cl, Cd, spanwise Cd's) from the drag-wake runs. Clean and tripped
configurations are separate files, and the trip description is in the header
"Comment:" line. The drag files are the runs where Cl and Cd were measured at
the same angle of attack; the lift files carry Cl and Cm only, but much wider
in alpha.
"""

import os
import re
import glob
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, "data", "uiuc_lsat")

# File stem -> AeroSandbox airfoil-database name (true design coordinates).
# Suffix letters (A, B, D, E ...) are UIUC model versions, not configurations.
ASB_NAME = {
    "e387": "e387", "sd2030": "sd2030", "fx63137": "fx63137",
    "A18": "a18", "BE50": "be50", "E374B": "e374", "E387A": "e387",
    "FX63137B": "fx63137", "K3311": "k3311", "MH45": "mh45", "N6409": "naca6409",
    "R140A": "r140", "RG15B": "rg15", "S1210": "s1210", "S1223": "s1223",
    "S6062": "s6062", "S7012B": "s7012", "S7055": "s7055", "SD6060": "sd6060",
    "SD7003": "sd7003", "SD7032D": "sd7032", "SD7032E": "sd7032",
    "SD7037A": "sd7037", "SD7037B": "sd7037", "SD8000": "sd8000",
    "SD8020": "sd8020", "SD8020T2": "sd8020",
    # Vol 2 (base models and trip variants only; F/GF/RTL suffixes are flaps
    # and gurney flaps, which the surrogate cannot represent)
    "7037B": "sd7037", "7037C": "sd7037", "7075AT1": "s7075", "7075AT2": "s7075",
    "7075AT3": "s7075", "7075BT2": "s7075", "E423": "e423", "MH32": "mh32",
    "NACA2414": "naca2414", "NACA2415": "naca2415", "RG15C": "rg15",
    "S4083A": "s4083", "S4083B": "s4083", "S5010": "s5010", "S8025": "s8025",
    "M665": "m665", "M685": "m685",
    # Vol 3
    "A18T": "a18", "AVISTAR": "avistar", "BW3": "bw3", "BW3T": "bw3",
    "CLARKYB": "clarky", "CLARKYBT": "clarky", "E231": "e231", "E387C": "e387",
    "E387D": "e387", "E472": "e472", "FALCON": "falcon", "GOE417A": "goe417a",
    "LRN1007B": "lrn1007", "PT40A": "pt40", "PT40B": "pt40", "RG14": "rg14",
    "S6063": "s6063", "S7012B0T": "s7012", "S7075A0": "s7075", "S7075A0T": "s7075",
    "S7075B0": "s7075", "S8036": "s8036", "S8037": "s8037", "S8052": "s8052",
    "SA7035": "sa7035", "SA7036A": "sa7036", "SA7036B": "sa7036", "SA7038": "sa7038",
    "SD7032T": "sd7032", "SD7037BT": "sd7037", "SD7037D": "sd7037", "SD7037D2": "sd7037",
    "SD7062B": "sd7062", "SD7062BT": "sd7062", "SD7080": "sd7080",
    "SG6040": "sg6040", "SG6040T": "sg6040", "SG6041": "sg6041", "SG6041T": "sg6041",
    "SG6042": "sg6042", "SG6042T": "sg6042", "SG6042T2": "sg6042", "SG6042T3": "sg6042",
    "SG6042T4": "sg6042", "SG6042T5": "sg6042", "SG6043": "sg6043", "SG6043T": "sg6043",
    "SG6043T2": "sg6043", "SG6043T3": "sg6043", "ULTIMATE": "ultimate", "USNPS4": "usnps4",
}


def parse_polar(path):
    """Return (header dict, list of blocks). Block = dict(Re, rows, run_file)."""
    lines = open(path, errors="replace").read().splitlines()
    hdr = {}
    for ln in lines[:4]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            hdr[k.strip().lower()] = v.strip()
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].strip().startswith("Average Reynolds"):
            Re = int(round(float(lines[i + 1].strip())))
            j, rows = i + 2, []
            while j < len(lines) and not lines[j].strip().startswith("Tabulated"):
                p = lines[j].split()
                if len(p) >= 3:
                    try:
                        rows.append([float(x) for x in p[:3]])
                    except ValueError:
                        pass
                j += 1
            m = re.search(r"data in file (\S+)", lines[j]) if j < len(lines) else None
            blocks.append(dict(Re=Re, rows=rows, run_file=m.group(1) if m else ""))
            i = j + 1
        else:
            i += 1
    return hdr, blocks


def classify(comment):
    """'clean', 'tripped' or 'other' from a UIUC header comment."""
    c = comment.strip().strip("'").lower()
    if "clean &" in c or "covering" in c or "open bay" in c:
        return "other"      # several configurations in one file, or a non-standard model
    if any(k in c for k in ("trip", "tape", "u.s.t", "l.s.t")):
        return "tripped"
    if "clean" in c or re.fullmatch(r"(obechi surface: )?flap = 0 degs", c):
        return "clean"
    return "other"


def trip_locations(comment):
    """(xtr_upper, xtr_lower) in x/c from a UIUC trip comment; NaN if unstated.
    Handles 'u.s.t. located at 2% chord', 'u.s. x/c = 2%', '2% u.s.', and a
    bare 'x/c = 68%' (surface not stated: taken as upper)."""
    c = comment.lower()
    def find(tag):
        m = (re.search(tag + r"(?:t\.)?[^&,;]*?(\d+(?:\.\d+)?)\s*%", c)
             or re.search(r"(\d+(?:\.\d+)?)\s*%\s*" + tag, c))
        return float(m.group(1)) / 100 if m else np.nan
    up, lo = find(r"u\.s\."), find(r"l\.s\.")
    if np.isfinite(up) or np.isfinite(lo):
        return up, lo
    m = re.search(r"x/c\s*=\s*(\d+(?:\.\d+)?)\s*%", c)
    return (float(m.group(1)) / 100, np.nan) if m else (np.nan, np.nan)


def file_kind(stem):
    u = stem.upper()
    if u.endswith(".DRG") or u.endswith("_DRG.TXT"):
        return "drag"
    if u.endswith(".LFT") or u.endswith("_LFT.TXT"):
        return "lift"
    return None


def file_key(stem):
    if stem.upper().endswith((".DRG", ".LFT")):
        return stem[:-4]
    return re.sub(r"_(c|tf)_(drg|lft)\.txt$", "", stem)


rows, lift_rows = [], []
for vol in ("vol1", "vol2", "vol3", "vol4"):
    for path in sorted(glob.glob(os.path.join(SRC, vol, "*"))):
        stem = os.path.basename(path)
        kind = file_kind(stem)
        if kind is None:
            continue
        key = file_key(stem)
        if key not in ASB_NAME:
            print(f"  skip {vol}/{stem} (no AeroSandbox geometry)")
            continue
        hdr, blocks = parse_polar(path)
        comment = hdr.get("comment", "")
        config = classify(comment)
        if config == "other":
            print(f"  skip {vol}/{stem} ({comment!r}: neither clean nor tripped)")
            continue
        xu, xl = trip_locations(comment) if config == "tripped" else (np.nan, np.nan)
        for b in blocks:
            base = dict(volume=vol, file=stem, airfoil_label=hdr.get("airfoil", ""),
                        asb_name=ASB_NAME[key], builder=hdr.get("builder", ""),
                        config=config, comment=comment, xtr_upper=xu, xtr_lower=xl,
                        Re=b["Re"], run_file=b["run_file"])
            if kind == "drag":
                for a, cl, cd in b["rows"]:
                    rows.append(dict(base, alpha=a, CL=cl, CD=cd))
            else:
                # Lift runs step alpha up and then back down (stall hysteresis);
                # label each point with its sweep direction.
                alphas = [r[0] for r in b["rows"]]
                for k, (a, cl, cm) in enumerate(b["rows"]):
                    if k == 0:
                        direction = "up"
                    else:
                        direction = "up" if a >= alphas[k - 1] else "down"
                    lift_rows.append(dict(base, alpha=a, CL=cl, CM=cm, sweep=direction))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "data", "uiuc_experimental.csv"), index=False)
print(f"wrote data/uiuc_experimental.csv: {len(df)} points, "
      f"{df.file.nunique()} files, {df.asb_name.nunique()} airfoils, "
      f"Re {df.Re.min()}-{df.Re.max()}")
lf = pd.DataFrame(lift_rows)
lf.to_csv(os.path.join(OUT, "data", "uiuc_experimental_lift.csv"), index=False)
print(f"wrote data/uiuc_experimental_lift.csv: {len(lf)} points, "
      f"{lf.file.nunique()} files, {lf.asb_name.nunique()} airfoils, "
      f"Re {lf.Re.min()}-{lf.Re.max()}, alpha {lf.alpha.min()}..{lf.alpha.max()}")

# Clean-only Eppler E387 (E), Vol. 4: the file e387_neuralfoil_validation.py reads.
e = df[(df.volume == "vol4") & (df.asb_name == "e387") & (df.config == "clean")].copy()
e387 = pd.DataFrame(dict(Re=e.Re, alpha=e.alpha, CL=e.CL, CD=e.CD,
                         Re_nom_k=(e.Re / 1e3).round().astype(int), airfoil="E387",
                         source="UIUC_LSAT_vol4_e387_c_drg.txt (NREL_SR-500-34515)"))
e387 = e387.sort_values(["Re", "alpha"]).reset_index(drop=True)
e387.to_csv(os.path.join(OUT, "data", "e387_experimental_NREL.csv"), index=False)
print(f"wrote data/e387_experimental_NREL.csv (clean E387 (E) only): {len(e387)} points, "
      f"Re_nom_k = {sorted(e387.Re_nom_k.unique())}")
