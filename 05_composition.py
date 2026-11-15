"""
Closure savings under observed (LODES 2014-2024) and random changes in commuting composition.
Input: data/xu2024/ (Xu et al., 2024); output/work/<City>_scan.csv
output/work/<City>_zone_od.csv.gz and output/commuting_change.xlsx 
Output: output/composition.xlsx
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)

DEFAULT = ["Honolulu", "Miami", "Pittsburgh", "Minneapolis", "Atlanta", "Boston"]  # cities of Tables S4-S6
CITIES = sys.argv[1:] or DEFAULT
W = os.path.join("output", "work")
YEARS, NDRAW, NLINK = [2014, 2017, 2021, 2024], 2, 4
target = pd.read_excel(os.path.join("output", "commuting_change.xlsx"), sheet_name="consecutive_median").set_index("city").dissim


def evaluate(N, od, links, roads, ref):
    N.base_od = od
    b = N.assign(ns.OD_MULTIPLIER, keep=True)
    fc = np.abs(b["flow"].values[roads] - ref[roads]).sum() / ref[roads].sum()
    out = {}
    for l in links:
        r = N.assign(ns.OD_MULTIPLIER, removed=[l], warm=b["state"])
        assert max(b["rgap"], r["rgap"]) < ns.RGAP
        out[l] = (b["tstt"] - r["tstt"]) / b["tstt"]
    return fc, out


for city in CITIES:
    cf = os.path.join(W, f"{city}_composition.csv")
    if os.path.exists(cf):
        continue
    obs, rnd = [], []
    L, O = ns.load_links(city), ns.load_od(city)
    N = ns.Network(L, O)
    roads = ~L.connector.values
    S = pd.read_csv(os.path.join("output", "work", f"{city}_scan.csv"))
    S = S[S.groupby("link_id").served.transform("min") > -1e-6]
    links = S.groupby("link_id").delta.max().sort_values(ascending=False).head(NLINK).index.astype(int)
    base = N.base_od.copy()
    ref = N.assign(ns.OD_MULTIPLIER)["flow"].values
    zi = {z: i for i, z in enumerate(N.zones)}
    A = pd.read_csv(os.path.join("output", "work", f"{city}_zone_od.csv.gz"))

    def margins(y):
        a = A[A.year == y]
        P, Q = np.zeros(len(zi)), np.zeros(len(zi))
        np.add.at(P, a.o.map(zi).values, a.S000.values)
        np.add.at(Q, a.d.map(zi).values, a.S000.values)
        return P, Q

    P0, Q0 = margins(2019)
    for y in YEARS:
        P, Q = margins(y)
        rt = base.sum(1) * np.where(P0 > 0, P / np.maximum(P0, 1e-9), 1.0)
        ct = base.sum(0) * np.where(Q0 > 0, Q / np.maximum(Q0, 1e-9), 1.0)
        rt, ct = rt * base.sum() / rt.sum(), ct * base.sum() / ct.sum()
        M = base.copy()
        for _ in range(100):
            M *= (rt / np.maximum(M.sum(1), 1e-12))[:, None]
            M *= (ct / np.maximum(M.sum(0), 1e-12))[None, :]
        fc, sav = evaluate(N, M, links, roads, ref)
        dis = 0.5 * np.abs(M / M.sum() - base / base.sum()).sum()
        obs += [dict(city=city, year=y, od_dissim=dis, link_flow_change=fc, link_id=l, delta=d) for l, d in sav.items()]

    w = base[base > 0] / base.sum()
    for k in range(NDRAW):
        z = np.random.default_rng(100 + k).standard_normal(w.size)
        dis = lambda s: 0.5 * np.abs(w * np.exp(s * z) / (w * np.exp(s * z)).sum() - w).sum()
        sig = brentq(lambda s: dis(s) - target[city], 1e-4, 5)
        M = np.zeros_like(base)
        x = w * np.exp(sig * z)
        M[base > 0] = x / x.sum() * base.sum()
        fc, sav = evaluate(N, M, links, roads, ref)
        rnd += [dict(city=city, draw=k, sigma=sig, od_dissim=target[city], link_flow_change=fc, link_id=l, delta=d)
                for l, d in sav.items()]
    N.base_od = base
    pd.concat([pd.DataFrame(obs).assign(set="observed"), pd.DataFrame(rnd).assign(set="random")]).to_csv(cf, index=False)

if all(os.path.exists(os.path.join(W, f"{c}_composition.csv")) for c in DEFAULT):
    R = pd.concat([pd.read_csv(os.path.join(W, f"{c}_composition.csv")) for c in DEFAULT])
    with pd.ExcelWriter(os.path.join("output", "composition.xlsx")) as xw:
        R[R.set == "observed"].drop(columns=["set", "draw", "sigma"], errors="ignore").to_excel(
            xw, sheet_name="observed_years", index=False)
        R[R.set == "random"].drop(columns=["set", "year"], errors="ignore").to_excel(
            xw, sheet_name="random_draws", index=False)
