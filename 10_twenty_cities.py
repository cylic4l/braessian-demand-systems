"""
All cities in the source dataset with a common protocol: baseline UE and SO at nine demand levels;
candidates are the 10 links with the largest excess of UE over SO flow at each of 0.7, 1.0 and 1.3 times
reference demand, each evaluated at these three levels.
Input: data/xu2024/ (Xu et al., 2024）
output/work/ (03_city_analysis.py, for the eight cities of the full analysis)
Output: output/twenty_cities.xlsx; checkpoints in output/work20/
Usage: python 10_twenty_cities.py [City ...]   (re-running resumes from the checkpoints)
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
spec3 = importlib.util.spec_from_file_location("ca", "03_city_analysis.py")
ca = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(ca)

LEVELS, SELECT, NTOP, TH = ca.LEVELS, ca.SELECT, 10, 2e-5
W, W20 = os.path.join("output", "work"), os.path.join("output", "work20")
os.makedirs(W20, exist_ok=True)


def top_links(city, N, L):
    f = os.path.join(W20, f"{city}_top{NTOP}.json")
    if os.path.exists(f):
        return json.load(open(f))
    roads = ~L.connector.values
    cand = []
    for m in SELECT:
        fu = N.assign(ns.OD_MULTIPLIER * m)["flow"].values
        fs = N.assign(ns.OD_MULTIPLIER * m, so=True)["flow"].values
        ex = pd.Series(fu - fs, index=L.link_id)[roads & (fu > 0)]
        cand += [int(l) for l in ex.sort_values(ascending=False).head(NTOP).index if int(l) not in cand]
    json.dump(cand, open(f, "w"))
    return cand


def run_city(city):
    L, O = ns.load_links(city), ns.load_od(city)
    N = ns.Network(L, O)
    s0 = ns.OD_MULTIPLIER
    roads = ~L.connector.values
    mf = os.path.join(W20, f"{city}_meta.json")
    meta = json.load(open(mf)) if os.path.exists(mf) else {}
    bf = os.path.join(W20, f"{city}_base.csv")
    B = ca.csv(bf, ["level"])
    for m in LEVELS:
        if m in set(B.level):
            continue
        ue = N.assign(s0 * m)
        so = N.assign(s0 * m, so=True)
        B = pd.concat([B, pd.DataFrame([dict(level=m, tstt_ue=ue["tstt"], tstt_so=so["tstt"],
                                             rgap_ue=ue["rgap"], rgap_so=so["rgap"], served=ue["served"])])])
        B.to_csv(bf, index=False)
    if "n_links" not in meta:
        meta.update(n_links=int(roads.sum()), n_zones=len(N.zones), trips=float(O.trips.sum()))
        json.dump(meta, open(mf, "w"))
    meta["cands"] = top_links(city, N, L)
    sf = os.path.join(W20, f"{city}_scan.csv")
    S = ca.csv(sf, ["link_id", "level"])
    if "rgap" in S:
        S = S[S.rgap < ns.RGAP]
    S = S[S.link_id.isin(meta["cands"])]
    done = set(zip(S.link_id.astype(int), S.level.astype(float)))
    for m in SELECT:
        pending = [l for l in meta["cands"] if (l, m) not in done]
        if not pending:
            continue
        base = N.assign(s0 * m, keep=True)
        for l in pending:
            r = N.assign(s0 * m, removed=[l], warm=base["state"])
            S = pd.concat([S, pd.DataFrame([dict(link_id=l, level=m, tstt=r["tstt"], tstt_base=base["tstt"],
                                                 rgap=r["rgap"], served=r["served"] - base["served"],
                                                 delta=(base["tstt"] - r["tstt"]) / base["tstt"])])])
            S.to_csv(sf, index=False)
            done.add((l, m))
        del base
    return meta, B, S


FROM_FULL = ca.FULL


def from_full(city):
    """Same protocol taken from the full results of 03_city_analysis.py."""
    meta = json.load(open(os.path.join(W, f"{city}_meta.json")))
    cand = top_links(city, ns.Network(ns.load_links(city), ns.load_od(city)), ns.load_links(city))
    B = pd.read_csv(os.path.join(W, f"{city}_base.csv"))
    S = pd.read_csv(os.path.join(W, f"{city}_scan.csv"))
    return meta, B, S[S.link_id.isin(cand) & S.level.isin(SELECT)]


def summarize(city, meta, B, S):
    B = B.set_index("level").sort_index()
    ok = S.groupby("link_id").served.min() > -1e-6
    S = S[S.link_id.isin(ok[ok].index)]
    P = S.pivot_table(index="link_id", columns="level", values="delta")
    best = {m: (P[m].idxmax() if P[m].max() > TH else None) for m in SELECT}
    b10 = best[1.0]
    gap = 1 - B.tstt_so / B.tstt_ue
    return dict(city=city, road_links=meta["n_links"], zones=meta["n_zones"], candidates=len(P),
                poa_06=B.tstt_ue[0.6] / B.tstt_so[0.6], poa_10=B.tstt_ue[1.0] / B.tstt_so[1.0],
                poa_14=B.tstt_ue[1.4] / B.tstt_so[1.4],
                beneficial_07=int((P[0.7] > TH).sum()), beneficial_10=int((P[1.0] > TH).sum()),
                beneficial_13=int((P[1.3] > TH).sum()),
                best_07=best[0.7], best_10=b10, best_13=best[1.3],
                best10_saving_bp=P[1.0].max() * 1e4,
                best10_at_07_bp=P.loc[b10, 0.7] * 1e4 if b10 is not None else np.nan,
                best10_at_13_bp=P.loc[b10, 1.3] * 1e4 if b10 is not None else np.nan,
                peak_beneficial_adding_at_07=int(((P[1.0] > TH) & (P[0.7] < -TH)).sum()),
                same_best_07_13=best[0.7] == best[1.3],
                bound_violations=int((S.delta > S.level.map(gap) + 1e-9).sum()))


if __name__ == "__main__":
    cities = sys.argv[1:] or ns.all_cities()
    rows = []
    for city in cities:
        if city in FROM_FULL:
            meta, B, S = from_full(city)
        else:
            meta, B, S = run_city(city)
        rows.append(summarize(city, meta, B, S))
        print(city, "done", flush=True)
    if not sys.argv[1:]:
        with pd.ExcelWriter(os.path.join("output", "twenty_cities.xlsx")) as xw:
            pd.DataFrame(rows).to_excel(xw, sheet_name="summary", index=False)
