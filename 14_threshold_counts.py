"""City-level counts at different thresholds for a beneficial closure and for a tie.

Input:  output/work/<City>_scan.csv (03_city_analysis.py, full analysis);
        output/work20/<City>_scan.csv and <City>_top10.json (10_twenty_cities.py, common screen);
        output/twenty_cities.xlsx (10_twenty_cities.py)
Output: output/threshold_counts.xlsx (also the Mann-Whitney tests of Supplementary Note 1)

"""
import json
import os

import pandas as pd
from scipy.stats import mannwhitneyu

THRESHOLDS = [0.001, 0.002, 0.005, 0.01]  # percent of total travel time
FULL = ["Honolulu", "Pittsburgh", "Miami", "Minneapolis", "San_Francisco", "Boston", "New_Orleans", "Atlanta"]
W, W20 = os.path.join("output", "work"), os.path.join("output", "work20")


def savings(scan, links=None, levels=None):
    S = pd.read_csv(scan)
    ok = S.groupby("link_id").served.min() > -1e-6
    S = S[S.link_id.isin(ok[ok].index)]
    if links is not None:
        S = S[S.link_id.isin(links)]
    P = S.pivot_table(index="link_id", columns="level", values="delta") * 100  # percent
    return P[levels] if levels else P


def classify(P, t):
    mx = P.max()
    ben = mx > t
    if not ben.any():
        return "none"
    if not ben.all():
        return "changes"
    same = (P.ge(mx - t)).all(axis=1).any()
    return "same" if same else "changes"


rows = []
for city in FULL:  # full analysis, five screened levels
    P = savings(os.path.join(W, f"{city}_scan.csv"), levels=[0.7, 0.8, 1.0, 1.2, 1.3])
    for t in THRESHOLDS:
        rows.append(dict(analysis="full", city=city, threshold_pct=t, beneficial_at_m1=bool(P[1.0].max() > t),
                         group=classify(P, t)))

screen = pd.read_excel(os.path.join("output", "twenty_cities.xlsx"), sheet_name="summary").city
for city in screen:  # common screen, three levels
    if city in FULL:
        cand = json.load(open(os.path.join(W20, f"{city}_top10.json"))) if os.path.exists(
            os.path.join(W20, f"{city}_top10.json")) else None
        scan = os.path.join(W, f"{city}_scan.csv")
    else:
        cand, scan = None, os.path.join(W20, f"{city}_scan.csv")
    if not os.path.exists(scan) or (city in FULL and cand is None):
        for t in THRESHOLDS:
            rows.append(dict(analysis="screen", city=city, threshold_pct=t, group="missing"))
        continue
    P = savings(scan, links=cand)
    if not {0.7, 1.0, 1.3} <= set(P.columns) or P[[0.7, 1.0, 1.3]].isna().any().any():
        for t in THRESHOLDS:  # screen not finished for this city
            rows.append(dict(analysis="screen", city=city, threshold_pct=t, group="missing"))
        continue
    P = P[[0.7, 1.0, 1.3]]
    for t in THRESHOLDS:
        rows.append(dict(analysis="screen", city=city, threshold_pct=t, beneficial_at_m1=bool(P[1.0].max() > t),
                         group=classify(P, t)))

R = pd.DataFrame(rows)
summary = (R[R.group != "missing"].assign(beneficial=lambda d: d.group != "none",
                                           demand_dependent=lambda d: d.group == "changes")
           .groupby(["analysis", "threshold_pct"])[["beneficial", "demand_dependent", "beneficial_at_m1"]].sum()
           .astype(int).reset_index())
missing = R[R.group == "missing"].groupby(["analysis", "threshold_pct"]).city.count().rename("cities_missing")
summary = summary.merge(missing.reset_index(), how="left").fillna({"cities_missing": 0})
# cities with and without a beneficial closure in the common screen (threshold 0.002%)
# delay factor and travel-time correlation as reported by Xu et al. (2024), Table 9
XU = {"Atlanta": (1.06, .72), "Austin": (1.26, .82), "Boston": (1.18, .77), "Chicago": (1.65, .86),
      "Dallas": (1.30, .79), "Denver": (1.13, .78), "Honolulu": (1.48, .86), "Las_Vegas": (1.25, .83),
      "Miami": (1.07, .73), "Milwaukee": (1.21, .78), "Minneapolis": (1.05, .75), "New_Orleans": (1.19, .89),
      "New_York": (2.24, .88), "Philadelphia": (1.32, .85), "Phoenix": (1.22, .77), "Pittsburgh": (1.05, .72),
      "Portland": (1.22, .72), "San_Francisco": (1.34, .83), "Seattle": (1.23, .70), "Washington": (1.25, .75)}
T20 = pd.read_excel(os.path.join("output", "twenty_cities.xlsx"), sheet_name="summary")
T20["efficiency_gap_pct"] = (1 - 1 / T20.poa_10) * 100
T20["delay_factor"] = T20.city.map(lambda c: XU[c][0])
T20["travel_time_r"] = T20.city.map(lambda c: XU[c][1])
ben = T20[["beneficial_07", "beneficial_10", "beneficial_13"]].sum(axis=1) > 0
tests = []
for v in ["efficiency_gap_pct", "delay_factor", "travel_time_r"]:
    a, b = T20[ben][v], T20[~ben][v]
    u = mannwhitneyu(a, b, alternative="two-sided")
    tests.append(dict(variable=v, mean_with=a.mean(), mean_without=b.mean(), n_with=len(a), n_without=len(b),
                      U=u.statistic, P=u.pvalue))

os.makedirs("output", exist_ok=True)
with pd.ExcelWriter(os.path.join("output", "threshold_counts.xlsx")) as xw:
    summary.to_excel(xw, sheet_name="summary", index=False)
    R.to_excel(xw, sheet_name="by_city", index=False)
    pd.DataFrame(tests).to_excel(xw, sheet_name="group_tests", index=False)
