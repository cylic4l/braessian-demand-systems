"""
Check of the assignment code against a published best-known user equilibrium.
Input: data/tntp/SiouxFalls_net.tntp, SiouxFalls_trips.tntp, SiouxFalls_flow.tntp
(Transportation Networks for Research Core Team)
Output: output/solver_check.xlsx
"""
import importlib.util
import os
import re

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)

D = os.path.join("data", "tntp")
rows = [l.replace(";", "").split() for l in open(os.path.join(D, "SiouxFalls_net.tntp")) if re.match(r"^\s*\d", l)]
L = pd.DataFrame([[int(r[0]), int(r[1]), float(r[2]), float(r[4]), float(r[5]), float(r[6])] for r in rows],
                 columns=["a_node", "b_node", "capacity", "fftime", "alpha", "beta"])
L["link_id"] = np.arange(1, len(L) + 1)
L["connector"] = False

od = []
for blk in re.split(r"Origin\s+", open(os.path.join(D, "SiouxFalls_trips.tntp")).read())[1:]:
    o = int(blk.split()[0])
    for dz, q in re.findall(r"(\d+)\s*:\s*([\d.]+)", blk):
        if float(q) > 0 and int(dz) != o:
            od.append((o, int(dz), float(q)))
O = pd.DataFrame(od, columns=["o", "d", "trips"])

F = pd.read_csv(os.path.join(D, "SiouxFalls_flow.tntp"), sep=r"\s+")
F.columns = ["a_node", "b_node", "best_flow", "best_cost"]

N = ns.Network(L, O)
N.blocked[:] = False  # all nodes are through nodes in this network
r = N.assign(1.0, rgap=1e-12, max_iter=5000)

M = L.merge(F, on=["a_node", "b_node"], how="left")
M["flow"] = r["flow"].values
best = float((F.best_flow * F.best_cost).sum())
summary = pd.DataFrame([dict(network="Sioux Falls", links=len(L), od_pairs=len(O), relative_gap=r["rgap"],
                             iterations=r["iters"], tstt=r["tstt"], tstt_best_known=best,
                             tstt_relative_difference=(r["tstt"] - best) / best,
                             max_abs_flow_difference=float((M.flow - M.best_flow).abs().max()))])

with pd.ExcelWriter(os.path.join("output", "solver_check.xlsx")) as xw:
    summary.to_excel(xw, sheet_name="summary", index=False)
    M[["a_node", "b_node", "flow", "best_flow"]].to_excel(xw, sheet_name="link_flows", index=False)
