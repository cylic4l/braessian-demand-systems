"""
Equilibria, candidate closures and demand sweep for each city.
Input: data/xu2024/ (Xu et al., 2024）
Output: output/city_results.xlsx; per-city checkpoints in output/work/
Usage: python 03_city_analysis.py [City ...]   
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

FULL = ["Honolulu", "Pittsburgh", "Miami", "Minneapolis", "San_Francisco", "Boston", "New_Orleans", "Atlanta"]  # the eight smallest networks
CITIES = sys.argv[1:] or FULL
LEVELS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
SELECT = [0.7, 1.0, 1.3]          # levels at which candidates are selected
SCAN = [0.7, 0.8, 1.0, 1.2, 1.3]  # levels at which every candidate is evaluated
NTOP, NRAND, NSWEEP = 30, 10, 8
WORK = os.path.join("output", "work")
os.makedirs(WORK, exist_ok=True)


def csv(path, cols):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=cols)


def run_city(city):
    L, O = ns.load_links(city), ns.load_od(city)
    N = ns.Network(L, O)
    s0 = ns.OD_MULTIPLIER
    roads = ~L.connector.values
    mf = os.path.join(WORK, f"{city}_meta.json")
    meta = json.load(open(mf)) if os.path.exists(mf) else {}

    # baseline UE and SO at every level; UE states kept for warm starts
    bf = os.path.join(WORK, f"{city}_base.csv")
    B = csv(bf, ["level"])
    states, tb, sb = {}, {}, {}
    flows = {}
    for m in LEVELS:
        ue = N.assign(s0 * m, keep=True)
        states[m], tb[m], sb[m] = ue["state"], ue["tstt"], ue["served"]
        if m in SELECT or m not in set(B.level):
            so = N.assign(s0 * m, so=True)
            flows[m] = (ue["flow"].values, so["flow"].values)
        if m not in set(B.level):
            vc = ue["flow"].values[roads] / L.capacity.values[roads]
            B = pd.concat([B, pd.DataFrame([dict(
                level=m, tstt_ue=ue["tstt"], tstt_so=so["tstt"], rgap_ue=ue["rgap"], rgap_so=so["rgap"],
                iters_ue=ue["iters"], mean_vc=vc.mean(), share_vc_gt1=(vc > 1).mean(), served=ue["served"])])])
            B.to_csv(bf, index=False)
        if m == 1.0 and "corr_reference" not in meta:
            f = ue["flow"].values[roads]
            meta.update(corr_reference=float(np.corrcoef(f, L.ref_flow.values[roads])[0, 1]),
                        n_links=int(roads.sum()), n_zones=len(N.zones), trips=float(O.trips.sum()))
            json.dump(meta, open(mf, "w"))
    B = B.set_index("level").sort_index()

    # candidates: largest excess of UE over SO flow at each selection level, plus random links
    if "cands" not in meta:
        cand, sel = [], {}
        for m in SELECT:
            fu, fs = flows[m]
            ex = pd.Series(fu - fs, index=L.link_id)[roads & (fu > 0)]
            for l in ex.sort_values(ascending=False).head(NTOP).index.astype(int):
                sel.setdefault(l, []).append(m)
                if l not in cand:
                    cand.append(l)
        fu = flows[1.0][0]
        rest = L.link_id.values[roads & (fu > 0) & ~np.isin(L.link_id.values, cand)]
        rand = np.random.default_rng(7).choice(rest, NRAND, replace=False)
        meta["cands"] = [[int(l), "top", sel[l]] for l in cand] + [[int(l), "random", []] for l in rand]
        json.dump(meta, open(mf, "w"))
    kinds = {l: k for l, k, _ in meta["cands"]}

    sf = os.path.join(WORK, f"{city}_scan.csv")
    S = csv(sf, ["link_id", "level"])
    S = S[S.link_id.isin(list(kinds))]
    if "rgap" in S:
        S = S[S.rgap < ns.RGAP]
    done = set(zip(S.link_id.astype(int), S.level.astype(float)))

    def removal(l, m):
        nonlocal S
        r = N.assign(s0 * m, removed=[l], warm=states[m])
        S = pd.concat([S, pd.DataFrame([dict(
            link_id=l, level=m, kind=kinds[l], tstt=r["tstt"], tstt_base=tb[m], rgap=r["rgap"],
            served=r["served"] - sb[m], delta=(tb[m] - r["tstt"]) / tb[m], veh_h=(tb[m] - r["tstt"]) / 60)])])
        S.to_csv(sf, index=False)
        done.add((l, m))

    for m in SCAN:
        for l in kinds:
            if (l, m) not in done:
                removal(l, m)
    ok = S.groupby("link_id").served.min() > -1e-6
    s1 = S[S.link_id.isin(ok[ok].index) & S.level.isin(SCAN)]
    sweep = s1.groupby("link_id").delta.max().sort_values(ascending=False).head(NSWEEP).index.astype(int)
    for l in sweep:
        for m in LEVELS:
            if (l, m) not in done:
                removal(l, m)
    if meta.get("sweep") != [int(l) for l in sweep]:
        meta["sweep"] = [int(l) for l in sweep]
        json.dump(meta, open(mf, "w"))

    # solver precision: three swept links at reference demand, gap 1e-10 versus 1e-8
    pf = os.path.join(WORK, f"{city}_precision.csv")
    if not os.path.exists(pf):
        b10 = N.assign(s0, rgap=1e-10)["tstt"]
        rows = []
        for l in sweep[:3]:
            r10 = N.assign(s0, removed=[l], rgap=1e-10)["tstt"]
            d8 = float(S[(S.link_id == l) & (S.level == 1.0)].delta.iloc[0])
            rows.append(dict(link_id=l, delta_1e8_bp=d8 * 1e4, delta_1e10_bp=(b10 - r10) / b10 * 1e4))
        pd.DataFrame(rows).to_csv(pf, index=False)
    return meta, B.reset_index(), S, pd.read_csv(pf)


if __name__ == "__main__":
    out = {"cities": [], "baseline": [], "closures": [], "candidates": [], "precision": []}
    for city in CITIES:
        meta, B, S, P = run_city(city)
        out["cities"].append(pd.DataFrame([{"city": city, **{k: v for k, v in meta.items() if k != "cands"}}]))
        out["baseline"].append(B.assign(city=city))
        out["closures"].append(S.assign(city=city))
        out["candidates"].append(pd.DataFrame(
            [dict(city=city, link_id=l, kind=k, selected_at=",".join(map(str, s))) for l, k, s in meta["cands"]]))
        out["precision"].append(P.assign(city=city))
    if set(CITIES) == set(FULL):
        with pd.ExcelWriter(os.path.join("output", "city_results.xlsx")) as xw:
            for k, v in out.items():
                pd.concat(v).to_excel(xw, sheet_name=k, index=False)
