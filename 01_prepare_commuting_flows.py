"""
Tract-to-tract commuting flows for the counties of each study city.
Input: data/lodes_raw/LODES8/ (any subfolder layout): <st>_od_main_JT00_<year>.csv.gz and
<st>_od_aux_JT00_<year>.csv.gz for st = hi, fl, pa, mn, ga, ma and years 2013-2024
U.S. Census Bureau, LEHD Origin-Destination Employment Statistics, version 8,

Output: output/lodes_tracts/<City>.csv.gz, output/tract_centroids.csv
"""
import glob
import os

import pandas as pd

RAW = os.path.join("data", "lodes_raw", "LODES8")
OUT = os.path.join("output", "lodes_tracts")
YEARS = range(2013, 2025)
COLS = ["S000", "SE01", "SE02", "SE03", "SI01", "SI02", "SI03"]
CITIES = {
    "Honolulu": ("hi", ["15003"]),
    "Miami": ("fl", ["12086", "12011"]),
    "Pittsburgh": ("pa", ["42003"]),
    "Minneapolis": ("mn", ["27053", "27123"]),
    "Atlanta": ("ga", ["13121", "13089", "13067", "13063"]),
    "Boston": ("ma", ["25025", "25017", "25021"]),
}


def tract_flows(path, counties):
    parts = []
    for ch in pd.read_csv(path, dtype={"w_geocode": str, "h_geocode": str},
                          usecols=["w_geocode", "h_geocode"] + COLS, chunksize=2_000_000):
        ch["h_tract"], ch["w_tract"] = ch.h_geocode.str[:11], ch.w_geocode.str[:11]
        ch = ch[ch.h_tract.str[:5].isin(counties) & ch.w_tract.str[:5].isin(counties)]
        parts.append(ch.groupby(["h_tract", "w_tract"])[COLS].sum())
    return pd.concat(parts).groupby(level=[0, 1]).sum()


files = {os.path.basename(f): f for f in glob.glob(os.path.join(RAW, "**", "*.csv.gz"), recursive=True)}
missing = [f"{st}_od_main_JT00_{y}.csv.gz" for st, _ in CITIES.values() for y in YEARS
           if f"{st}_od_main_JT00_{y}.csv.gz" not in files]
missing += [f"{st}_xwalk.csv.gz" for st, _ in CITIES.values() if f"{st}_xwalk.csv.gz" not in files]
if missing:
    raise SystemExit(f"{len(missing)} LODES files not found under {RAW}, e.g. {missing[:3]}")

results = {}
for city, (st, counties) in CITIES.items():
    rows = []
    for y in YEARS:
        t = [tract_flows(files[n], counties) for n in
             (f"{st}_od_main_JT00_{y}.csv.gz", f"{st}_od_aux_JT00_{y}.csv.gz") if n in files]
        rows.append(pd.concat(t).groupby(level=[0, 1]).sum().reset_index().assign(year=y, jt="JT00"))
    results[city] = pd.concat(rows)[["year", "jt", "h_tract", "w_tract"] + COLS]

cent = []
for st, _ in CITIES.values():
    x = pd.read_csv(files[f"{st}_xwalk.csv.gz"], usecols=["trct", "blklatdd", "blklondd"], dtype={"trct": str})
    cent.append(x.groupby("trct")[["blklatdd", "blklondd"]].mean().reset_index())
cent = pd.concat(cent).drop_duplicates("trct").rename(columns={"trct": "tract", "blklatdd": "lat", "blklondd": "lon"})

os.makedirs(OUT, exist_ok=True)
for city, df in results.items():
    df.to_csv(os.path.join(OUT, f"{city}.csv.gz"), index=False)
cent.to_csv(os.path.join("output", "tract_centroids.csv"), index=False)
