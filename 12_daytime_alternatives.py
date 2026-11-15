"""
Daytime effect of each closure evaluated at all nine demand levels, hour by hour, compared with its
peak-hour effect, in cities with a beneficial closure at m = 1. Hours and matrices as the "directional"

Input: data/xu2024/ (Xu et al., 2024); output/work/<City>_meta.json and <City>_scan.csv
output/demand_variation.xlsx； source: UTD19, Loder et al., 2019)
Output: output/daytime_alternatives.xlsx (with --morning-peak: _am suffix, profile scaled so that
the morning peak hour carries reference demand); per-city checkpoints in output/work/
Usage: python 12_daytime_alternatives.py [City ...] [--morning-peak]   (re-running resumes from the checkpoints)
"""
import importlib.util
import json
import os
import sys

import pandas as pd

spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)
spec = importlib.util.spec_from_file_location("ca", "03_city_analysis.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

AM = "--morning-peak" in sys.argv  # morning peak hour = reference demand
CITIES = [a for a in sys.argv[1:] if not a.startswith("--")] or ["Honolulu", "Miami", "Minneapolis", "Atlanta", "Boston"]
SUF = "_am" if AM else ""
W = os.path.join("output", "work")
prof = pd.read_excel(os.path.join("output", "demand_variation.xlsx"), sheet_name="hourly_median")
hours = prof[prof.share_of_peak >= 0.6].set_index("hour").share_of_peak
if AM:
    hours = hours / hours.loc[7:10].max()


def part(h):
    return "morning" if h <= 10 else ("midday" if h <= 14 else "afternoon")


for city in CITIES:
    cf = os.path.join(W, f"{city}_daytime_alt{SUF}.csv")
    D = pd.read_csv(cf) if os.path.exists(cf) else pd.DataFrame(columns=["link_id", "hour", "matrix"])
    done = set(zip(D.link_id, D.hour, D.matrix))
    links = json.load(open(os.path.join(W, f"{city}_meta.json")))["sweep"]
    N = ns.Network(ns.load_links(city), ns.load_od(city))
    od = {"morning": N.base_od, "midday": (N.base_od + N.base_od.T) / 2, "afternoon": N.base_od.T.copy()}
    keys = sorted({(part(h), h) for h in hours.index})
    rows = []
    for mat, h in keys:
        if all((l, h, mat) in done for l in links):
            continue
        sh = hours[h]
        b = N.assign(ns.OD_MULTIPLIER * sh, od=od[mat], keep=True)
        for l in links:
            if (l, h, mat) in done:
                continue
            r = N.assign(ns.OD_MULTIPLIER * sh, od=od[mat], removed=[l], warm=b["state"])
            rows.append(dict(city=city, link_id=l, hour=h, share_of_peak=sh, matrix=mat, tstt_base=b["tstt"],
                             tstt_closed=r["tstt"], served=r["served"] - b["served"],
                             rgap=max(b["rgap"], r["rgap"]), veh_h=(b["tstt"] - r["tstt"]) / 60))
        D = pd.concat([D, pd.DataFrame(rows)]) if rows else D
        rows = []
        D.to_csv(cf, index=False)
    print(city, "done", flush=True)

out = []
for city in CITIES:
    cf = os.path.join(W, f"{city}_daytime_alt{SUF}.csv")
    if not os.path.exists(cf):
        continue
    D = pd.read_csv(cf)
    S = pd.read_csv(os.path.join(W, f"{city}_scan.csv"))
    pk = S[S.level == 1.0].set_index("link_id")
    D = D[[part(h) == m for h, m in zip(D.hour, D.matrix)]]
    total = D.drop_duplicates("hour").tstt_base.sum() / 60
    for l, g in D.groupby("link_id"):
        di = g.set_index("hour").veh_h
        out.append(dict(city=city, link_id=l, peak_bp=pk.loc[l, "delta"] * 1e4, peak_veh_h=pk.loc[l, "veh_h"],
                        daytime_veh_h=di.sum(), daytime_bp=di.sum() / total * 1e4,
                        hours_adding_time=int((di < 0).sum()), hours=len(di)))
if out:
    with pd.ExcelWriter(os.path.join("output", f"daytime_alternatives{SUF}.xlsx")) as xw:
        pd.DataFrame(out).to_excel(xw, sheet_name="summary", index=False)
