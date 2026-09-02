"""
uiuc_lsat_parse.py
=============================================================================
Parse the official UIUC Low-Speed Airfoil Tests (LSAT) ASCII polar files in
data/uiuc_lsat/ into one tidy table, data/uiuc_experimental.csv, and derive
the clean-only Eppler E387 file used by e387_neuralfoil_validation.py.

Source: https://m-selig.ae.illinois.edu/pd.html  (Selig et al., "Summary of
Low-Speed Airfoil Data" Vol. 1, 1995; Selig & McGranahan, NREL/SR-500-34515,
2004, which the archive files as Vol. 4). Each *.DRG / *_drg.txt file is one
airfoil model in one configuration; it holds several Reynolds-number blocks of
(alpha, Cl, Cd, spanwise Cd's) from the drag-wake runs. Clean and tripped
configurations are separate files, and the trip description is in the header
"Comment:" line. Only the drag files are used here, because they are the runs
where Cl and Cd were measured at the same angle of attack.
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


def trip_locations(comment):
    """(xtr_upper, xtr_lower) in x/c from a UIUC trip comment; NaN if unstated."""
    c = comment.lower()
    up = re.search(r"u\.s\.t\.[^,]*?(\d+(?:\.\d+)?)\s*%", c)
    lo = re.search(r"l\.s\.t\.[^,]*?(\d+(?:\.\d+)?)\s*%", c)
    if up or lo:
        return (float(up.group(1)) / 100 if up else np.nan,
                float(lo.group(1)) / 100 if lo else np.nan)
    m = re.search(r"x/c\s*=\s*(\d+(?:\.\d+)?)\s*%", c)   # surface not stated
    return (float(m.group(1)) / 100, np.nan) if m else (np.nan, np.nan)


rows = []
for vol in ("vol1", "vol4"):
    for path in sorted(glob.glob(os.path.join(SRC, vol, "*"))):
        stem = os.path.basename(path)
        if not (stem.upper().endswith(".DRG") or stem.endswith("_drg.txt")):
            continue
        key = stem[:-4] if stem.upper().endswith(".DRG") else stem.replace("_c_drg.txt", "").replace("_tf_drg.txt", "")
        if key not in ASB_NAME:
            print(f"  skip {vol}/{stem} (no AeroSandbox geometry)")
            continue
        hdr, blocks = parse_polar(path)
        comment = hdr.get("comment", "")
        config = "clean" if comment.strip().strip("'").lower() == "clean" else "tripped"
        xu, xl = trip_locations(comment) if config == "tripped" else (np.nan, np.nan)
        for b in blocks:
            for a, cl, cd in b["rows"]:
                rows.append(dict(volume=vol, file=stem, airfoil_label=hdr.get("airfoil", ""),
                                 asb_name=ASB_NAME[key], builder=hdr.get("builder", ""),
                                 config=config, comment=comment, xtr_upper=xu, xtr_lower=xl,
                                 Re=b["Re"], alpha=a, CL=cl, CD=cd, run_file=b["run_file"]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "data", "uiuc_experimental.csv"), index=False)
print(f"wrote data/uiuc_experimental.csv: {len(df)} points, "
      f"{df.file.nunique()} files, {df.asb_name.nunique()} airfoils, "
      f"Re {df.Re.min()}-{df.Re.max()}")

# Clean-only Eppler E387 (E), Vol. 4: the file e387_neuralfoil_validation.py reads.
e = df[(df.volume == "vol4") & (df.asb_name == "e387") & (df.config == "clean")].copy()
e387 = pd.DataFrame(dict(Re=e.Re, alpha=e.alpha, CL=e.CL, CD=e.CD,
                         Re_nom_k=(e.Re / 1e3).round().astype(int), airfoil="E387",
                         source="UIUC_LSAT_vol4_e387_c_drg.txt (NREL_SR-500-34515)"))
e387 = e387.sort_values(["Re", "alpha"]).reset_index(drop=True)
e387.to_csv(os.path.join(OUT, "data", "e387_experimental_NREL.csv"), index=False)
print(f"wrote data/e387_experimental_NREL.csv (clean E387 (E) only): {len(e387)} points, "
      f"Re_nom_k = {sorted(e387.Re_nom_k.unique())}")
