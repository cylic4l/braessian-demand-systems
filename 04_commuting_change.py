"""
Tract-to-zone mapping and year-to-year change in commuting.
Input: output/lodes_tracts/<City>.csv.gz and output/tract_centroids.csv (01_prepare_commuting_flows.py;

Output: output/commuting_change.xlsx; output/work/<City>_zone_od.csv.gz
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)

CITIES = ["Honolulu", "Miami", "Pittsburgh", "Minneapolis", "Atlanta", "Boston"]
cent = pd.read_csv(os.path.join("output", "tract_centroids.csv"), dtype={"tract": str})
os.makedirs(os.path.join("output", "work"), exist_ok=True)
pairs, cons, valid = [], [], []
for city in CITIES:
    D = pd.read_csv(os.path.join("output", "lodes_tracts", f"{city}.csv.gz"),
                    dtype={"h_tract": str, "w_tract": str}, usecols=["year", "h_tract", "w_tract", "S000"])
    # tract-pair share change between consecutive years
    P = D.pivot_table(index=["h_tract", "w_tract"], columns="year", values="S000", fill_value=0)
    sh = P / P.sum()
    years = sorted(sh.columns)
    cp = [dict(city=city, year0=a, year1=b, dissim=0.5 * np.abs(sh[b] - sh[a]).sum()) for a, b in zip(years[:-1], years[1:])]
    pairs += cp
    cons.append(dict(city=city, dissim=np.median([c["dissim"] for c in cp])))
    # nearest-zone assignment of tracts
    nodes = pd.read_csv([os.path.join(r, f) for r, _, fs in os.walk(ns.city_dir(city)) for f in fs if f.endswith("_node.csv")][0])
    O = ns.load_od(city)
    Z = nodes[(nodes.Tract_Node == 1) & nodes.Node_ID.isin(set(O.o) | set(O.d))]
    lat0 = np.radians(Z.Lat.mean())
    xy = lambda lon, lat: np.c_[np.radians(lon) * np.cos(lat0) * 6371, np.radians(lat) * 6371]
    tree = cKDTree(xy(Z.Lon.values, Z.Lat.values))
    thr = np.quantile(tree.query(xy(Z.Lon.values, Z.Lat.values), k=2)[0][:, 1], 0.95)
    tr = pd.DataFrame({"tract": sorted(set(D.h_tract) | set(D.w_tract))}).merge(cent, how="left")
    dist, idx = tree.query(xy(tr.lon.values, tr.lat.values))
    zone = dict(zip(tr.tract, np.where(dist <= thr, Z.Node_ID.values[idx], -1)))
    D["o"], D["d"] = D.h_tract.map(zone), D.w_tract.map(zone)
    D = D[(D.o >= 0) & (D.d >= 0) & (D.o != D.d)]
    A = D.groupby(["year", "o", "d"]).S000.sum().reset_index()
    A.to_csv(os.path.join("output", "work", f"{city}_zone_od.csv.gz"), index=False)
    x = O.set_index(["o", "d"]).trips
    a = A[A.year == 2019].set_index(["o", "d"]).S000
    j = pd.concat([x.rename("xu"), a.rename("lodes")], axis=1).fillna(0)
    valid.append(dict(city=city, tracts_mapped=int((dist <= thr).sum()), corr_2019=np.corrcoef(j.xu, j.lodes)[0, 1]))

with pd.ExcelWriter(os.path.join("output", "commuting_change.xlsx")) as xw:
    pd.DataFrame(pairs).to_excel(xw, sheet_name="consecutive_years", index=False)
    pd.DataFrame(cons).to_excel(xw, sheet_name="consecutive_median", index=False)
    pd.DataFrame(valid).to_excel(xw, sheet_name="zone_mapping", index=False)
