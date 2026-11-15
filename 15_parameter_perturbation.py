"""Sensitivity of closure effects to the link parameters of the city models.
The saving is computed at 0.6 and 1.0 times reference demand in 20 draws per city, and summed over 07:00-19:00 with return trips
(as in 12_daytime_alternatives.py, profile scaled to the busiest hour) in the first 3 draws.
Input: data/xu2024/ (Xu et al., 2024); output/work/<City>_meta.json and <City>_scan.csv
(03_city_analysis.py); output/demand_variation.xlsx (06_demand_variation.py)
Output: output/parameter_perturbation.xlsx; per-city checkpoints in output/work/<City>_perturb.csv
Usage: python 15_parameter_perturbation.py [City ...]   (re-running resumes)
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)

ARGS = sys.argv[1:]
DRAWS = 20      # draws at 0.6 and 1.0 times reference demand
DAY_DRAWS = 3   # draws over the day
CITIES = [a for a in ARGS if not a.startswith("--")] or \
    ["Honolulu", "Miami", "Minneapolis", "Atlanta", "Boston"]
TH = 2e-5  # 0.002% of total travel time
W = os.path.join("output", "work")
prof = pd.read_excel(os.path.join("output", "demand_variation.xlsx"), sheet_name="hourly_median")
HOURS = prof[prof.share_of_peak >= 0.6].set_index("hour").share_of_peak


def part(h):
    return "morning" if h <= 10 else ("midday" if h <= 14 else "afternoon")


def peak_savers(city):
    sweep = json.load(open(os.path.join(W, f"{city}_meta.json")))["sweep"]
    S = pd.read_csv(os.path.join(W, f"{city}_scan.csv"))
    s = S[(S.level == 1.0) & S.link_id.isin(sweep)].set_index("link_id").delta
    return [int(l) for l in s[s > TH].index]


def perturbed_links(city, draw):
    rng = np.random.default_rng(1000 * (CITIES_ALL.index(city) + 1) + draw)
    L = ns.load_links(city)
    L = L.astype({"capacity": float, "alpha": float, "beta": float})
    road = ~L.connector.values
    if draw > 0:  # draw 0 is the unperturbed model
        L.loc[road, "capacity"] *= np.exp(0.1 * rng.standard_normal(road.sum()))
        L.loc[road, "alpha"] *= rng.uniform(0.8, 1.2)
        L["beta"] *= rng.uniform(0.9, 1.1)
    return L


CITIES_ALL = ["Honolulu", "Miami", "Minneapolis", "Atlanta", "Boston"]


def cases(d):
    yield from (("level", m, "morning") for m in (0.6, 1.0))
    if d <= DAY_DRAWS:
        yield from (("hour", h, part(h)) for h in HOURS.index)


for city in CITIES:
    cf = os.path.join(W, f"{city}_perturb.csv")
    D = pd.read_csv(cf) if os.path.exists(cf) else pd.DataFrame(columns=["draw", "kind", "x", "link_id"])
    done = set(zip(D.draw, D.kind, D.x.astype(float), D.link_id))
    links, O = peak_savers(city), ns.load_od(city)
    for d in range(DRAWS + 1):
        pending = [c for c in cases(d) if any((d, c[0], float(c[1]), l) not in done for l in links)]
        if not pending:
            continue
        N = ns.Network(perturbed_links(city, d), O)
        od = {"morning": N.base_od, "midday": (N.base_od + N.base_od.T) / 2, "afternoon": N.base_od.T.copy()}
        for kind, x, mat in pending:
            m = x if kind == "level" else HOURS[x]
            b = N.assign(ns.OD_MULTIPLIER * m, od=od[mat], keep=True)
            rows = []
            for l in links:
                if (d, kind, float(x), l) in done:
                    continue
                r = N.assign(ns.OD_MULTIPLIER * m, od=od[mat], removed=[l], warm=b["state"])
                rows.append(dict(city=city, draw=d, kind=kind, x=float(x), link_id=l, tstt_base=b["tstt"],
                                 tstt_closed=r["tstt"], rgap=max(b["rgap"], r["rgap"])))
                done.add((d, kind, float(x), l))
            D = pd.concat([D, pd.DataFrame(rows)])
            D.to_csv(cf, index=False)
            del b
        print(city, "draw", d, "done", flush=True)

# summary: share of draws in which each closure keeps the sign class of the unperturbed model
cls = lambda v: np.where(v > TH, "save", np.where(v < -TH, "add", "within 0.002%"))
out = []
for city in CITIES_ALL:
    cf = os.path.join(W, f"{city}_perturb.csv")
    if not os.path.exists(cf):
        continue
    D = pd.read_csv(cf)
    D = D[(D.kind == "level") | (D.x.isin(HOURS.index))]
    lev = D[D.kind == "level"].assign(effect=lambda d: (d.tstt_base - d.tstt_closed) / d.tstt_base)
    day = D[D.kind == "hour"]
    nh = day.groupby(["draw", "link_id"]).x.nunique()
    full = nh[nh == len(HOURS)].index
    day = (day.set_index(["draw", "link_id"]).loc[full].reset_index()
           .assign(diff=lambda d: d.tstt_base - d.tstt_closed)
           .groupby(["draw", "link_id"]).agg(diff=("diff", "sum"), base=("tstt_base", "sum")).reset_index()
           .assign(effect=lambda d: d["diff"] / d.base, x="day"))
    allr = pd.concat([lev[["draw", "link_id", "x", "effect"]].astype({"x": str}), day[["draw", "link_id", "x", "effect"]]])
    for (l, x), g in allr.groupby(["link_id", "x"]):
        g = g.set_index("draw").effect
        if 0 not in g.index or len(g) < 2:
            continue
        c0 = cls(np.array([g[0]]))[0]
        pert = g.drop(0)
        out.append(dict(city=city, link_id=l, case=x, effect_pct=g[0] * 100, class_unperturbed=c0, draws=len(pert),
                        same_class=int((cls(pert.values) == c0).sum()),
                        same_sign=int((np.sign(pert.values) == np.sign(g[0])).sum()),
                        effect_p5_pct=np.percentile(pert, 5) * 100, effect_p95_pct=np.percentile(pert, 95) * 100))
if out:
    with pd.ExcelWriter(os.path.join("output", "parameter_perturbation.xlsx")) as xw:
        pd.DataFrame(out).to_excel(xw, sheet_name="by_closure", index=False)
