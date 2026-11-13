"""
Shared functions: city network loading and traffic assignment.
Input: data/xu2024/<NN_City>/ (Xu et al., 2024, Scientific Data 11, 325;
https://doi.org/10.6084/m9.figshare.24235696)
Assignment: origin-based Algorithm B (Dial, 2006, Transportation Research Part B 40, 917-936),
BPR link costs; system optimum via marginal costs; centroids are not used as through nodes.
"""
import glob
import os

import numpy as np
import pandas as pd
try:
    from numba import njit
except ImportError as e:
    raise SystemExit(f"Package numba could not be loaded ({e}). Install or update it with: python -m pip install numba")

XU = os.path.join("data", "xu2024")
ZONE0 = 10_000_000
OD_MULTIPLIER = 0.60
RGAP = 1e-8
INF = 1e300


def city_dir(city):
    return glob.glob(os.path.join(XU, f"*_{city}"))[0]


def all_cities():
    return [os.path.basename(d).split("_", 1)[1] for d in sorted(glob.glob(os.path.join(XU, "[0-9][0-9]_*")))]


def load_links(city):
    d = city_dir(city)
    ab = pd.read_csv(os.path.join(d, "03_AequilibraE_results", "network.csv"), usecols=["a", "b"])
    alpha, beta = float(ab.a.iloc[0]), float(ab.b.iloc[0])
    T = pd.read_csv(os.path.join(d, "02_TransCAD_results", f"{city}_link_LinkFlows.csv"))
    out = []
    for s in ["AB", "BA"]:
        t = T[(T[f"{s}_Link_ID"].fillna(-1) >= 0) & T[f"{s}_Free_Time"].notna()]
        out.append(pd.DataFrame({
            "a_node": t[f"{s}_From_Node_"].astype(np.int64).values,
            "b_node": t[f"{s}_To_Node_ID"].astype(np.int64).values,
            "fftime": t[f"{s}_Free_Time"].values,
            "capacity": t[f"{s}_Capacity"].values,
            "length": t[f"{s}_Length"].values,
            "link_type": t[f"{s}_Link_Type"].astype(np.int64).values,
            "ref_flow": t[f"{s}_Flow"].fillna(0).values,
        }))
    L = pd.concat(out, ignore_index=True)
    L = L[L.a_node != L.b_node].reset_index(drop=True)
    L["connector"] = L.link_type >= ZONE0
    L["alpha"] = np.where(L.connector, 0.0, alpha)
    L["beta"] = beta
    L["link_id"] = np.arange(1, len(L) + 1)
    return L


def load_od(city):
    f = glob.glob(os.path.join(city_dir(city), "01_Input_data", "*_od.csv"))[0]
    O = pd.read_csv(f)
    O = O[O.O_ID != O.D_ID]
    return O.rename(columns={"O_ID": "o", "D_ID": "d", "OD_Number": "trips"})

@njit
def _cost(x, t0, cap, al, be):
    r = x / cap
    p = r ** be if r > 0 else 0.0
    c = t0 * (1.0 + al * p)
    d = t0 * al * be * (r ** (be - 1.0)) / cap if (r > 0 and al > 0) else 0.0
    return c, d


@njit
def _update_costs(x, t0, cap, al, be, c, d):
    for a in range(x.size):
        c[a], d[a] = _cost(x[a], t0[a], cap[a], al[a], be[a])


@njit
def _dijkstra(r, N, fs, fl, head, c, alive, blocked, dist, pred):
    for i in range(N):
        dist[i] = INF
        pred[i] = -1
    hk = np.empty(fl.size + N + 1)
    hn = np.empty(fl.size + N + 1, np.int64)
    n = 0
    dist[r] = 0.0
    hk[0] = 0.0; hn[0] = r; n = 1
    done = np.zeros(N, np.bool_)
    while n > 0:
        k = hk[0]; u = hn[0]
        n -= 1
        hk[0] = hk[n]; hn[0] = hn[n]
        i = 0
        while True:
            l = 2 * i + 1
            if l >= n:
                break
            m = l
            if l + 1 < n and hk[l + 1] < hk[l]:
                m = l + 1
            if hk[m] < hk[i]:
                hk[i], hk[m] = hk[m], hk[i]
                hn[i], hn[m] = hn[m], hn[i]
                i = m
            else:
                break
        if done[u]:
            continue
        done[u] = True
        if blocked[u] and u != r:
            continue
        for e in range(fs[u], fs[u + 1]):
            a = fl[e]
            if not alive[a]:
                continue
            v = head[a]
            nd = k + c[a]
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = a
                j = n
                hk[n] = nd; hn[n] = v; n += 1
                while j > 0:
                    p = (j - 1) // 2
                    if hk[j] < hk[p]:
                        hk[j], hk[p] = hk[p], hk[j]
                        hn[j], hn[p] = hn[p], hn[j]
                        j = p
                    else:
                        break


@njit
def _topo(r, N, fs, fl, bs, bl, head, bush, order, pos):
    indeg = np.zeros(N, np.int64)
    for v in range(N):
        for e in range(bs[v], bs[v + 1]):
            if bush[bl[e]]:
                indeg[v] += 1
    for i in range(N):
        pos[i] = -1
    n = 0
    order[0] = r; pos[r] = 0; n = 1
    q = 0
    while q < n:
        u = order[q]; q += 1
        for e in range(fs[u], fs[u + 1]):
            a = fl[e]
            if bush[a]:
                v = head[a]
                indeg[v] -= 1
                if indeg[v] == 0:
                    pos[v] = n; order[n] = v; n += 1
    return n


@njit
def _labels(r, n, order, bs, bl, tail, bush, xr, c, eps, L, Lp, U, Up, use_flow):
    L[r] = 0.0; U[r] = 0.0; Lp[r] = -1; Up[r] = -1
    for q in range(1, n):
        v = order[q]
        lb = INF; lpa = -1; ub = -INF; upa = -1
        for e in range(bs[v], bs[v + 1]):
            a = bl[e]
            if not bush[a]:
                continue
            u = tail[a]
            cl = L[u] + c[a]
            if cl < lb:
                lb = cl; lpa = a
            if (not use_flow) or xr[a] > eps:
                cu = U[u] + c[a]
                if cu > ub:
                    ub = cu; upa = a
        L[v] = lb; Lp[v] = lpa
        if upa == -1:
            U[v] = lb; Up[v] = -1
        else:
            U[v] = ub; Up[v] = upa


@njit
def _shift(r, N, A, fs, fl, bs, bl, tail, head, t0, cap, al, be, x, c, d, bk, xk, eps,
           order, pos, L, Lp, U, Up, seg_min, seg_max):
    """One pass of flow shifts on the bush of one origin; returns number of shifts."""
    n = _topo(r, N, fs, fl, bs, bl, head, bk, order, pos)
    _labels(r, n, order, bs, bl, tail, bk, xk, c, eps, L, Lp, U, Up, True)
    moved = 0
    for q in range(n - 1, 0, -1):
        j = order[q]
        if Up[j] < 0 or U[j] - L[j] <= 1e-13 * (1.0 + L[j]):
            continue
        nmin = 0; nmax = 0
        a1 = j; b1 = j
        ok = True
        a = Lp[a1]; seg_min[nmin] = a; nmin += 1; a1 = tail[a]
        b = Up[b1]; seg_max[nmax] = b; nmax += 1; b1 = tail[b]
        while a1 != b1:
            if pos[a1] > pos[b1]:
                a = Lp[a1]; seg_min[nmin] = a; nmin += 1; a1 = tail[a]
            else:
                b = Up[b1]
                if b < 0:
                    ok = False
                    break
                seg_max[nmax] = b; nmax += 1; b1 = tail[b]
        if not ok:
            continue
        cmin = 0.0; cmax = 0.0; den = 0.0; fmax = INF
        for s in range(nmin):
            cmin += c[seg_min[s]]; den += d[seg_min[s]]
        for s in range(nmax):
            cmax += c[seg_max[s]]; den += d[seg_max[s]]
            if xk[seg_max[s]] < fmax:
                fmax = xk[seg_max[s]]
        diff = cmax - cmin
        if diff <= 0:
            continue
        dx = diff / den if den > 0 else fmax
        if dx > fmax:
            dx = fmax
        if dx <= 0:
            continue
        for s in range(nmax):
            e = seg_max[s]
            xk[e] -= dx
            if xk[e] < 0:
                xk[e] = 0.0
            x[e] -= dx
            if x[e] < 0:
                x[e] = 0.0
            c[e], d[e] = _cost(x[e], t0[e], cap[e], al[e], be[e])
        for s in range(nmin):
            e = seg_min[s]
            xk[e] += dx; x[e] += dx
            c[e], d[e] = _cost(x[e], t0[e], cap[e], al[e], be[e])
        moved += 1
    return moved


@njit
def _solve(N, A, fs, fl, bs, bl, tail, head, t0, cap, al, be, alive, blocked,
           orig, dnode, dem, rgap_target, max_iter, n_inner, eps, bush0, xr0, warm, gap_every, n_sweep):
    K = orig.size
    x = np.zeros(A); c = np.empty(A); d = np.empty(A)
    _update_costs(x, t0, cap, al, be, c, d)
    if warm:
        bush = bush0.copy()
        xr = xr0.copy()
    else:
        bush = np.zeros((K, A), np.bool_)
        xr = np.zeros((K, A))
    redo = np.ones(K, np.bool_)
    if warm:
        for k in range(K):
            redo[k] = False
            for a in range(A):
                if not alive[a]:
                    if bush[k, a] or xr[k, a] > 0:
                        redo[k] = True
                    bush[k, a] = False
            if redo[k]:
                for a in range(A):
                    bush[k, a] = False
                    xr[k, a] = 0.0
            else:
                for a in range(A):
                    x[a] += xr[k, a]
        _update_costs(x, t0, cap, al, be, c, d)
    dist = np.empty(N); pred = np.empty(N, np.int64)
    order = np.empty(N, np.int64); pos = np.empty(N, np.int64)
    L = np.empty(N); U = np.empty(N); Lp = np.empty(N, np.int64); Up = np.empty(N, np.int64)
    served = 0.0
    total = 0.0
    # initial all-or-nothing bushes at free flow
    for k in range(K):
        if not redo[k]:
            continue
        r = orig[k]
        _dijkstra(r, N, fs, fl, head, c, alive, blocked, dist, pred)
        for v in range(N):
            if pred[v] >= 0:
                bush[k, pred[v]] = True
        for j in range(dnode.shape[1]):
            q = dem[k, j]
            if q <= 0:
                continue
            v = dnode[k, j]
            if dist[v] >= INF:
                continue
            while v != r:
                a = pred[v]
                xr[k, a] += q
                x[a] += q
                v = tail[a]
    _update_costs(x, t0, cap, al, be, c, d)
    gap = 1.0
    it = 0
    seg_min = np.empty(N, np.int64); seg_max = np.empty(N, np.int64)
    while it < max_iter:
        it += 1
        for k in range(K):
            r = orig[k]
            bk = bush[k]; xk = xr[k]
            # 1. clear rounding residues on links leaving nodes without inflow;
            #    drop unused links not on the shortest path tree of the bush
            n = _topo(r, N, fs, fl, bs, bl, head, bk, order, pos)
            for q in range(1, n):
                v = order[q]
                fin = 0.0
                for e in range(bs[v], bs[v + 1]):
                    if bk[bl[e]]:
                        fin += xk[bl[e]]
                if fin <= eps:
                    for e in range(fs[v], fs[v + 1]):
                        a = fl[e]
                        if bk[a] and xk[a] > 0:
                            x[a] -= xk[a]
                            if x[a] < 0:
                                x[a] = 0.0
                            xk[a] = 0.0
                            c[a], d[a] = _cost(x[a], t0[a], cap[a], al[a], be[a])
            _labels(r, n, order, bs, bl, tail, bk, xk, c, eps, L, Lp, U, Up, False)
            keep = np.zeros(A, np.bool_)
            for q in range(1, n):
                keep[Lp[order[q]]] = True
            for a in range(A):
                if bk[a] and xk[a] <= eps and not keep[a]:
                    bk[a] = False
                    xk[a] = 0.0
            # 2. add shortcuts w.r.t. max-cost labels over all bush links
            n = _topo(r, N, fs, fl, bs, bl, head, bk, order, pos)
            _labels(r, n, order, bs, bl, tail, bk, xk, c, eps, L, Lp, U, Up, False)
            for v in range(N):
                if pos[v] < 0:
                    U[v] = INF
            for a in range(A):
                if bk[a] or not alive[a]:
                    continue
                u = tail[a]; v = head[a]
                if v == r or pos[u] < 0 or (blocked[u] and u != r):
                    continue
                if U[u] + c[a] < U[v] * (1.0 - 1e-14) - 1e-14:
                    bk[a] = True
            # 3. shift flow from longest to shortest used segments
            for inner in range(n_inner):
                if _shift(r, N, A, fs, fl, bs, bl, tail, head, t0, cap, al, be, x, c, d, bk, xk, eps,
                          order, pos, L, Lp, U, Up, seg_min, seg_max) == 0:
                    break
        for sweep in range(n_sweep):
            for k in range(K):
                _shift(orig[k], N, A, fs, fl, bs, bl, tail, head, t0, cap, al, be, x, c, d, bush[k], xr[k],
                       eps, order, pos, L, Lp, U, Up, seg_min, seg_max)
        # relative gap on the current cost function
        _update_costs(x, t0, cap, al, be, c, d)
        if it % gap_every != 0 and it < max_iter:
            continue
        tt = 0.0
        for a in range(A):
            tt += x[a] * c[a]
        sp = 0.0
        served = 0.0
        total = 0.0
        for k in range(K):
            _dijkstra(orig[k], N, fs, fl, head, c, alive, blocked, dist, pred)
            for j in range(dnode.shape[1]):
                q = dem[k, j]
                if q > 0:
                    total += q
                    if dist[dnode[k, j]] < INF:
                        sp += q * dist[dnode[k, j]]
                        served += q
        gap = (tt - sp) / tt
        if gap < rgap_target:
            break
    return x, gap, it, served / total, bush, xr



class Network:
    """assign() returns link flows, total travel time (TSTT), relative gap and share of trips served."""

    def __init__(self, links, od):
        self.links = links.reset_index(drop=True).copy()
        L = self.links
        nodes = np.unique(np.r_[L.a_node.values, L.b_node.values])
        nid = {v: i for i, v in enumerate(nodes)}
        self.N = len(nodes)
        self.A = len(L)
        self.tail = L.a_node.map(nid).values.astype(np.int64)
        self.head = L.b_node.map(nid).values.astype(np.int64)
        self.t0 = np.maximum(L.fftime.values.astype(float), 1e-9)
        self.cap = L.capacity.values.astype(float)
        self.al = L.alpha.values.astype(float)
        self.be = L.beta.values.astype(float)
        o1 = np.argsort(self.tail, kind="stable")
        self.fl = o1.astype(np.int64)
        self.fs = np.searchsorted(self.tail[o1], np.arange(self.N + 1)).astype(np.int64)
        o2 = np.argsort(self.head, kind="stable")
        self.bl = o2.astype(np.int64)
        self.bs = np.searchsorted(self.head[o2], np.arange(self.N + 1)).astype(np.int64)
        zones = np.array(sorted(set(od.o) | set(od.d)), dtype=np.int64)
        self.zones = zones
        self.blocked = np.zeros(self.N, np.bool_)
        for z in zones:
            if z in nid:
                self.blocked[nid[z]] = True
        zi = {z: i for i, z in enumerate(zones)}
        M = np.zeros((len(zones), len(zones)))
        M[od.o.map(zi).values, od.d.map(zi).values] = od.trips.values
        self.base_od = M
        self.zin = np.array([z in nid for z in zones])
        self.znode = np.array([nid.get(z, 0) for z in zones], np.int64)
        self.link_id = L.link_id.values
        self.gap_every = 5

    def assign(self, scale=1.0, removed=None, so=False, rgap=RGAP, max_iter=2000,
               n_inner=1, n_sweep=8, od=None, warm=None, keep=False):
        alive = np.ones(self.A, np.bool_)
        if removed is not None:
            alive[np.isin(self.link_id, np.asarray(removed))] = False
        al = self.al * ((1.0 + self.be) if so else 1.0)
        M = (self.base_od if od is None else od) * scale
        rows = np.where((M.sum(1) > 0) & self.zin)[0]
        orig = self.znode[rows]
        dnode = np.tile(self.znode, (rows.size, 1))
        dem = M[rows] * self.zin[None, :]
        if warm is None:
            b0, x0, w = np.zeros((1, 1), np.bool_), np.zeros((1, 1)), False
        else:
            b0, x0, w = warm["bush"], warm["xr"], True
        x, gap, it, served, bush, xr = _solve(self.N, self.A, self.fs, self.fl, self.bs, self.bl,
                                    self.tail, self.head, self.t0, self.cap, al, self.be,
                                    alive, self.blocked, orig, dnode, dem,
                                    rgap, max_iter, n_inner, 1e-12, b0, x0, w, self.gap_every, n_sweep)
        t = self.t0 * (1 + self.al * (x / self.cap) ** self.be)
        return {"flow": pd.Series(x, index=self.link_id), "tstt": float(np.sum(x * t)),
                "rgap": float(gap), "iters": int(it), "served": float(served),
                "state": {"bush": bush, "xr": xr} if keep else None}
