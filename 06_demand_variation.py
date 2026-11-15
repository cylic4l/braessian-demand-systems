"""
Day-to-day and within-day variation in network volume.
Input: output/utd19/utd19_detector_day.csv.gz, output/utd19/utd19_hourly_profile.csv.gz
(02_prepare_detector_data.py; source: UTD19, Loder et al., 2019
Output: output/demand_variation.xlsx
"""
import os

import pandas as pd

d = pd.read_csv(os.path.join("output", "utd19", "utd19_detector_day.csv.gz"))
d["day"] = pd.to_datetime(d.day)
d = d[d.day.dt.dayofweek < 5]
d = d[d.n_obs >= 0.9 * d.groupby(["city", "detid"]).n_obs.transform("max")]
rows = []
for city, g in d.groupby("city"):
    days = g.day.nunique()
    if days < 10:
        continue
    cnt = g.groupby("detid").day.nunique()
    g = g[g.detid.isin(cnt[cnt >= 0.9 * days].index)]
    for col in ["flow_am", "flow_pm", "flow_day"]:
        p = g.pivot_table(index="day", columns="detid", values=col)
        p = p.loc[:, p.notna().mean() > 0.95].dropna()
        if len(p) < 10 or p.shape[1] < 10:
            continue
        idx = p.sum(axis=1) / p.sum(axis=1).median()
        rows.append(dict(city=city, var=col, days=len(p), detectors=p.shape[1], cv=idx.std() / idx.mean(),
                         q05=idx.quantile(0.05), q95=idx.quantile(0.95)))
day_var = pd.DataFrame(rows)

h = pd.read_csv(os.path.join("output", "utd19", "utd19_hourly_profile.csv.gz"))
prof = h[~h.weekend].groupby(["city", "hour"]).flow_mean.sum().unstack()
rel = prof.div(prof.max(axis=1), axis=0)
median_profile = rel.median().rename("share_of_peak").reset_index()

with pd.ExcelWriter(os.path.join("output", "demand_variation.xlsx")) as xw:
    day_var.to_excel(xw, sheet_name="day_to_day", index=False)
    rel.reset_index().to_excel(xw, sheet_name="hourly_by_city", index=False)
    median_profile.to_excel(xw, sheet_name="hourly_median", index=False)
