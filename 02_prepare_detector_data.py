"""
Detector-day and hourly summaries of loop-detector flows.
Input: data/utd19/utd19_u.csv (UTD19, Loder et al., 2019, Scientific Reports 9, 16283;

Output: output/utd19/utd19_detector_day.csv.gz, output/utd19/utd19_hourly_profile.csv.gz
"""
import os

import pandas as pd

SRC = os.path.join("data", "utd19", "utd19_u.csv")
OUT = os.path.join("output", "utd19")
if not os.path.exists(SRC):
    raise SystemExit(f"Input file not found: {SRC}")
day_parts, hour_parts = [], []
for ch in pd.read_csv(SRC, chunksize=5_000_000, low_memory=False):
    ch.columns = [c.strip().lower() for c in ch.columns]
    ch = ch[ch["error"].fillna(0) == 0].dropna(subset=["flow"])
    h = (ch["interval"] // 3600).astype(int)
    ch = ch.assign(hour=h, am=ch.flow.where(h.between(7, 8)), pm=ch.flow.where(h.between(16, 17)))
    key = ["city", "detid", "day"]
    g = ch.groupby(key).agg({"flow": ["sum", "count"], "am": ["sum", "count"],
                             "pm": ["sum", "count"], "occ": ["sum", "count"]})
    g.columns = ["_".join(c) for c in g.columns]
    day_parts.append(g)
    hour_parts.append(ch.groupby(key + ["hour"]).flow.agg(["sum", "count"]))

d = pd.concat(day_parts).groupby(level=[0, 1, 2]).sum()
day = pd.DataFrame({
    "flow_day": d.flow_sum / d.flow_count,
    "flow_am": d.am_sum / d.am_count.replace(0, pd.NA),
    "flow_pm": d.pm_sum / d.pm_count.replace(0, pd.NA),
    "occ_day": d.occ_sum / d.occ_count.replace(0, pd.NA),
    "n_obs": d.flow_count}).reset_index()
hh = pd.concat(hour_parts).groupby(level=[0, 1, 2, 3]).sum().reset_index()
hh["flow"] = hh["sum"] / hh["count"]
hh["weekend"] = pd.to_datetime(hh.day, errors="coerce").dt.dayofweek >= 5
prof = hh.groupby(["city", "detid", "weekend", "hour"]).flow.agg(
    flow_mean="mean", flow_sd="std", n_days="count").reset_index()

os.makedirs(OUT, exist_ok=True)
day.round(3).to_csv(os.path.join(OUT, "utd19_detector_day.csv.gz"), index=False)
prof.round(3).to_csv(os.path.join(OUT, "utd19_hourly_profile.csv.gz"), index=False)
