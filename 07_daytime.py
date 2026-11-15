"""
Daytime effect of the closure chosen at reference demand, hour by hour.
Each hour from 07:00 to 19:00 carries the reference OD scaled by its share of peak network volume.
Variant "morning": the morning OD in every hour.
Variant "directional": morning OD for 07-10, the mean of the OD and its transpose for 11-14,
and the transposed OD (return trips) for 15-19.

Input: data/xu2024/ (Xu et al., 2024); output/work/<City>_scan.csv (03_city_analysis.py);
output/demand_variation.xlsx (06_demand_variation.py; source: UTD19, Loder et al., 2019)
Output: output/daytime.xlsx (output/daytime_am.xlsx with --morning-peak)
Usage: python 07_daytime.py [City ...] [--morning-peak]
"""
import importlib.util
import os
import sys

import pandas as pd

spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)

DEFAULT = ["Honolulu", "Miami", "Pittsburgh", "Minneapolis", "Atlanta", "Boston"]  # cities of Tables S4-S6
AM = "--morning-peak" in sys.argv  # morning peak hour = reference demand
CITIES = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT
SUF = "_am" if AM else ""
W = os.path.join("output", "work")
prof = pd.read_excel(os.path.join("output", "demand_variation.xlsx"), sheet_name="hourly_median")
hours = prof[prof.share_of_peak >= 0.6].set_index("hour").share_of_peak
if AM:
    hours = hours / hours.loc[7:10].max()


def part(h):
    return "morning" if h <= 10 else ("midday" if h <= 14 else "afternoon")


for city in CITIES:
    cf = os.path.join(W, f"{city}_daytime{SUF}.csv")
    if os.path.exists(cf):
        continue
    rows = []
    S = pd.read_csv(os.path.join(W, f"{city}_scan.csv"))
    ok = S.groupby("link_id").served.min() > -1e-6
    s1 = S[S.link_id.isin(ok[ok].index) & (S.level == 1.0)]
    closure = int(s1.set_index("link_id").delta.idxmax())
    N = ns.Network(ns.load_links(city), ns.load_od(city))
    od = {"morning": N.base_od, "midday": (N.base_od + N.base_od.T) / 2, "afternoon": N.base_od.T.copy()}
    cache = {}
    for h, sh in hours.items():
        for variant in ["morning", "directional"]:
            mat = "morning" if variant == "morning" else part(h)
            key = (mat, h)
            if key not in cache:
                b = N.assign(ns.OD_MULTIPLIER * sh, od=od[mat], keep=True)
                r = N.assign(ns.OD_MULTIPLIER * sh, od=od[mat], removed=[closure], warm=b["state"])
                cache[key] = (b["tstt"], r["tstt"], r["served"] - b["served"], max(b["rgap"], r["rgap"]))
            tb, tr, sv, rg = cache[key]
            rows.append(dict(city=city, closure=closure, hour=h, share_of_peak=sh, variant=variant, matrix=mat,
                             tstt_base=tb, tstt_closed=tr, served=sv, rgap=rg, delta_bp=(tb - tr) / tb * 1e4,
                             veh_h=(tb - tr) / 60))
    pd.DataFrame(rows).to_csv(cf, index=False)
    print(city, "done", flush=True)

if not all(os.path.exists(os.path.join(W, f"{c}_daytime{SUF}.csv")) for c in DEFAULT):
    sys.exit()
D = pd.concat([pd.read_csv(os.path.join(W, f"{c}_daytime{SUF}.csv")) for c in DEFAULT])
summ = (D.groupby(["city", "closure", "variant"])
        .agg(daytime_veh_h=("veh_h", "sum"), hours_adding_time=("veh_h", lambda v: int((v < 0).sum())),
             hours=("veh_h", "size")).reset_index())
with pd.ExcelWriter(os.path.join("output", f"daytime{SUF}.xlsx")) as xw:
    summ.to_excel(xw, sheet_name="summary", index=False)
    D.to_excel(xw, sheet_name="by_hour", index=False)
