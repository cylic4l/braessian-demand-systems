"""
Summary tables: city results, efficiency-gap bound, shortfall by demand level, daytime effect,
composition outcomes.
Input: output/work/ (03_city_analysis.py), output/composition.xlsx (05_composition.py),
output/daytime.xlsx (07_daytime.py)
Output: output/summary_tables.xlsx
"""
import json
import os

import numpy as np
import pandas as pd

CITIES = ["Honolulu", "Pittsburgh", "Miami", "Minneapolis", "San_Francisco", "Boston", "New_Orleans", "Atlanta"]
TH = 2e-5
W = os.path.join("output", "work")
SCAN = [0.7, 0.8, 1.0, 1.2, 1.3]
daytime = pd.read_excel(os.path.join("output", "daytime.xlsx"), sheet_name="summary")
comp = pd.read_excel(os.path.join("output", "composition.xlsx"), sheet_name="random_draws")
obs = pd.read_excel(os.path.join("output", "composition.xlsx"), sheet_name="observed_years")
t1, sweep, short, bound, day, cmp, years = [], [], [], [], [], [], []
for c in CITIES:
    meta = json.load(open(os.path.join(W, f"{c}_meta.json")))
    B = pd.read_csv(os.path.join(W, f"{c}_base.csv")).set_index("level").sort_index()
    S = pd.read_csv(os.path.join(W, f"{c}_scan.csv"))
    S = S[S.groupby("link_id").served.transform("min") > -1e-6]
    gap = 1 - B.tstt_so / B.tstt_ue
    S["bound"] = S.level.map(gap)
    sc = S[S.level.isin(SCAN)].pivot_table(index="link_id", columns="level", values="delta")
    best = {m: (sc[m].idxmax() if sc[m].max() > TH else None) for m in SCAN}
    sw = S.pivot_table(index="link_id", columns="level", values="delta").loc[meta["sweep"]].dropna()
    nom = sw[1.0].idxmax()
    pk = sw[sw[1.0] > TH]
    t1.append(dict(city=c, road_links=meta["n_links"], zones=meta["n_zones"], trips=meta["trips"],
                   r_reference=meta["corr_reference"], poa_1=B.loc[1.0, "tstt_ue"] / B.loc[1.0, "tstt_so"],
                   candidates=sc.shape[0],
                   **{f"beneficial_{m:.1f}": int((sc[m] > TH).sum()) for m in SCAN},
                   **{f"best_{m:.1f}": best[m] for m in SCAN},
                   peak_beneficial_swept=len(pk), of_which_adding_06=int((pk[0.6] < -TH).sum()),
                   of_which_adding_07=int((pk[0.7] < -TH).sum()),
                   largest_saving_bp=sw.max().max() * 1e4,
                   largest_at_level=sw.max().idxmax(), largest_link=sw.stack().idxmax()[0],
                   tstt_vehh_1=B.loc[1.0, "tstt_ue"] / 60))
    for l in sw.index:
        sweep.append({"city": c, "link_id": l, **{f"{m:.1f}": sw.loc[l, m] * 1e4 for m in sw.columns}})
    for m in sw.columns:
        b = max(0, sw[m].max())
        short.append(dict(city=c, level=m, best_bp=b * 1e4, chosen_bp=sw.loc[nom, m] * 1e4,
                          shortfall_bp=(b - sw.loc[nom, m]) * 1e4))
    pos = S[S.delta > TH]
    bound.append(dict(city=c, evaluations=len(S), violations=int((S.delta > S.bound).sum()),
                      beneficial_cases=len(pos), share_recovered_median=(pos.delta / pos.bound).median()))
    for _, r in daytime[daytime.city == c].iterrows():
        day.append(dict(city=c, closure=r.closure, variant=r.variant,
                        peak_vehh=sw.loc[nom, 1.0] * B.loc[1.0, "tstt_ue"] / 60,
                        daytime_vehh=r.daytime_veh_h, hours_adding_time=r.hours_adding_time, hours=r.hours))
    cc = comp[comp.city == c]
    for d, g in cc.groupby("draw"):
        g = g.set_index("link_id")
        cmp.append(dict(city=c, draw=d, closure=nom, saving_ref_bp=sw.loc[nom, 1.0] * 1e4,
                        saving_draw_bp=g.delta.get(nom, np.nan) * 1e4,
                        best_draw=g.delta.idxmax() if g.delta.max() > TH else None,
                        link_flow_change=g.link_flow_change.iloc[0]))
    for y, g in obs[obs.city == c].groupby("year"):
        g = g.set_index("link_id")
        years.append(dict(city=c, year=y, od_dissim=g.od_dissim.iloc[0], link_flow_change=g.link_flow_change.iloc[0],
                          closure_2019=nom, saving_bp=g.delta.get(nom, np.nan) * 1e4,
                          best=g.delta.idxmax() if g.delta.max() > TH else None))

bd = pd.DataFrame(bound)
allpos = []
for c in CITIES:
    S = pd.read_csv(os.path.join(W, f"{c}_scan.csv"))
    B = pd.read_csv(os.path.join(W, f"{c}_base.csv")).set_index("level")
    S = S[(S.groupby("link_id").served.transform("min") > -1e-6) & (S.delta > TH)]
    allpos += list(S.delta / S.level.map(1 - B.tstt_so / B.tstt_ue))
pooled = pd.DataFrame([dict(beneficial_cases=len(allpos), median_share=np.median(allpos), max_share=np.max(allpos),
                            evaluations=bd.evaluations.sum(), violations=bd.violations.sum(),
                            max_rgap=max(pd.read_csv(os.path.join(W, f"{c}_scan.csv")).rgap.max() for c in CITIES))])

with pd.ExcelWriter(os.path.join("output", "summary_tables.xlsx")) as xw:
    pd.DataFrame(t1).to_excel(xw, sheet_name="table1", index=False)
    pd.DataFrame(sweep).to_excel(xw, sheet_name="sweep_bp", index=False)
    pd.DataFrame(short).to_excel(xw, sheet_name="shortfall", index=False)
    bd.to_excel(xw, sheet_name="bound_by_city", index=False)
    pooled.to_excel(xw, sheet_name="bound_pooled", index=False)
    pd.DataFrame(day).to_excel(xw, sheet_name="daytime", index=False)
    pd.DataFrame(years).to_excel(xw, sheet_name="observed_years", index=False)
    pd.DataFrame(cmp).to_excel(xw, sheet_name="random_draws", index=False)
    pd.concat([pd.read_csv(os.path.join(W, f"{c}_precision.csv")).assign(city=c) for c in CITIES]).to_excel(
        xw, sheet_name="precision", index=False)
