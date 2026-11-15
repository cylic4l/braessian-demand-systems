"""Two-peak check: for each closure that saves more than 0.002% at reference demand, the effect in the
08:00 and 17:00 hours combined, compared with the effect over 07:00-19:00 (return trips).

Input:  output/work/<City>_daytime_alt.csv and _alt_am.csv (12_daytime_alternatives.py);
        output/daytime_alternatives.xlsx and _am.xlsx (12_daytime_alternatives.py)
Output: output/two_peak_check.xlsx
"""
import os

import numpy as np
import pandas as pd

CITIES = ["Honolulu", "Miami", "Minneapolis", "Atlanta", "Boston"]
T = 0.002  # percent of total travel time
W = os.path.join("output", "work")


def part(h):
    return "morning" if h <= 10 else ("midday" if h <= 14 else "afternoon")


def cls(v):
    return np.where(v > T, "saves", np.where(v < -T, "adds", "within 0.002%"))


sheets = {}
for suf, scaling in [("", "busiest hour"), ("_am", "morning peak")]:
    A = pd.read_excel(os.path.join("output", f"daytime_alternatives{suf}.xlsx"))
    rows = []
    for city in CITIES:
        H = pd.read_csv(os.path.join(W, f"{city}_daytime_alt{suf}.csv"))
        H = H[H.matrix == H.hour.map(part)].drop_duplicates(["link_id", "hour"], keep="last")
        base = H.drop_duplicates("hour").set_index("hour").tstt_base
        for l, g in H.groupby("link_id"):
            a = A[(A.city == city) & (A.link_id == l)].iloc[0]
            if a.peak_bp / 100 <= T:
                continue
            v = g.set_index("hour").veh_h
            two = v[8] + v[17]
            rows.append(dict(city=city, link_id=int(l), scaling=scaling, peak_hour_pct=a.peak_bp / 100,
                             h08_veh_h=v[8], h17_veh_h=v[17],
                             two_peak_pct=two * 60 / (base[8] + base[17]) * 100,
                             day_pct=a.daytime_bp / 100))
    R = pd.DataFrame(rows)
    R["two_peak"] = cls(R.two_peak_pct)
    R["day"] = cls(R.day_pct)
    sheets[scaling] = R

with pd.ExcelWriter(os.path.join("output", "two_peak_check.xlsx")) as xw:
    for k, R in sheets.items():
        R.to_excel(xw, sheet_name=k.replace(" ", "_"), index=False)
        pd.crosstab(R.two_peak, R.day).to_excel(xw, sheet_name="counts_" + k.replace(" ", "_"))
