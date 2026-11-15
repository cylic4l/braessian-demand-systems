"""
Figures 1-5. 
Input: data/xu2024/ (Xu et al., 2024); output/work/ (03_city_analysis.py); output/demand_variation.xlsx;
output/summary_tables.xlsx; output/twenty_cities.xlsx; output/daytime_alternatives.xlsx and _am.xlsx (12_daytime_alternatives.py)
Output: output/fig1_twenty_cities.png, fig2_phases.png, fig3_example.png, fig4_bound.png, fig5_withinday.png (and .pdf)
"""
import glob
import importlib.util
import json
import os

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CITIES = ["Honolulu", "Pittsburgh", "Miami", "Minneapolis", "San_Francisco", "Boston", "New_Orleans", "Atlanta"]
W = os.path.join("output", "work")
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#b8b7b1", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK, "lines.linewidth": 2})
XL = "Demand (multiple of peak-hour demand)"


def scan(c):
    S = pd.read_csv(os.path.join(W, f"{c}_scan.csv"))
    return S[S.groupby("link_id").served.transform("min") > -1e-6]


def swept(c):
    sw = json.load(open(os.path.join(W, f"{c}_meta.json")))["sweep"]
    return scan(c).pivot_table(index="link_id", columns="level", values="delta").loc[sw].dropna() * 1e2


def base(c):
    return pd.read_csv(os.path.join(W, f"{c}_base.csv")).sort_values("level")


spec = importlib.util.spec_from_file_location("ns", "00_network_solver.py")
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)
figs = {}

# Fig. 2: two closures in Boston
city = "Boston"
L = ns.load_links(city)
nodes = pd.read_csv(glob.glob(os.path.join(ns.city_dir(city), "01_Input_data", "*_node.csv"))[0]).set_index("Node_ID")
sw = swept(city)
hi_link, lo_link = sw[1.0].idxmax(), sw[0.7].idxmax()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.15]})
R = L[~L.connector]
xy = lambda col: nodes.loc[R[col], ["Lon", "Lat"]].values
a, b = xy("a_node"), xy("b_node")
major = (R.link_type <= 2).values
for m, col, lw in [(~major, "#d6d5d0", 0.4), (major, "#8a8984", 0.9)]:
    for p, q in zip(a[m], b[m]):
        ax[0].plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw)
for l, col, lab in [(lo_link, ORANGE, "best at 0.6–0.7×"), (hi_link, BLUE, "best at 1.0×")]:
    r = L.set_index("link_id").loc[l]
    p, q = nodes.loc[r.a_node, ["Lon", "Lat"]].values, nodes.loc[r.b_node, ["Lon", "Lat"]].values
    ax[0].plot([p[0], q[0]], [p[1], q[1]], color=col, lw=3)
    ax[0].scatter([(p[0] + q[0]) / 2], [(p[1] + q[1]) / 2], s=90, facecolor="none", edgecolor=col, lw=1.5, label=lab)
ax[0].set_aspect(1 / np.cos(np.radians(nodes.Lat.mean())))
ax[0].set_xticks([]); ax[0].set_yticks([])
for sp in ax[0].spines.values():
    sp.set_visible(False)
ax[0].legend(frameon=False, fontsize=7, loc="lower left")
ax[0].set_title("a  Boston road network", loc="left", fontsize=9)
ax[1].axhline(0, color=INK, lw=0.6)
ax[1].plot(sw.columns, sw.loc[lo_link], color=ORANGE, marker="o", ms=3, label="primary road")
ax[1].plot(sw.columns, sw.loc[hi_link], color=BLUE, marker="o", ms=3, label="tertiary road")
ax[1].set_xlabel(XL)
ax[1].set_ylabel("Travel-time saving (% of total travel time)")
ax[1].legend(frameon=False, fontsize=7)
ax[1].set_title("b  Saving from each closure", loc="left", fontsize=9)
fig.tight_layout()
figs["fig3_example"] = fig

fig, ax = plt.subplots(2, 4, figsize=(7.2, 4.4), sharex=True)
TH_PCT = 0.002
for a, c in zip(ax.flat, CITIES):
    sw = swept(c)
    nom = sw[1.0].idxmax() if sw[1.0].max() > TH_PCT else None
    low = sw[0.7].idxmax() if sw[0.7].max() > TH_PCT else None
    for l in sw.index.difference([nom, low]):
        a.plot(sw.columns, sw.loc[l], color=GREY, lw=1)
    a.axhline(0, color=INK, lw=0.6)
    if low is not None and low != nom:
        a.plot(sw.columns, sw.loc[low], color=ORANGE, marker="o", ms=3, label="best at 0.7×")
    if nom is not None:
        a.plot(sw.columns, sw.loc[nom], color=BLUE, marker="o", ms=3, label="best at 1.0×")
    else:
        a.text(0.97, 0.06, "no beneficial closure\nat 1.0×", transform=a.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK)
    a.set_title(c.replace("_", " "), fontsize=9, loc="left")
    a.set_ylim(max(-0.15, min(-0.04, sw.values.min() * 1.05)), max(0.02, sw.values.max() * 1.15))
fig.supxlabel(XL, fontsize=9)
fig.supylabel("Travel-time saving (% of total travel time)", fontsize=9)
hl = {}
for a in ax.flat:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        hl.setdefault(ll, hh)
h, lab = list(hl.values()), list(hl.keys())
fig.legend(h, lab, loc="upper right", frameon=False, ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
figs["fig2_phases"] = fig

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3))
share, ends = [], []
for c in CITIES:
    B = base(c)
    g = (1 - B.tstt_so / B.tstt_ue) * 1e2
    ax[0].plot(B.level, g, color=GREY, lw=1.2)
    ends.append((g.iloc[-1], c.replace("_", " ")))
    S = scan(c)
    S = S[S.delta > 2e-5]
    share += list(S.delta.values / (1 - B.set_index("level").tstt_so / B.set_index("level").tstt_ue).loc[S.level].values)
ends.sort()
ypos = []
for y, _ in ends:  # spread end labels that would overlap
    ypos.append(max(y, ypos[-1] + 0.09) if ypos else y)
for (y, c), yp in zip(ends, ypos):
    ax[0].text(1.415, yp, c, fontsize=6.5, va="center", color=INK, clip_on=False)
share = np.array(share) * 100
ax[0].set_xlim(0.58, 1.42)
ax[0].set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
ax[0].set_xlabel(XL)
ax[0].set_ylabel("Efficiency gap, 1 − 1/PoA (% of total travel time)")
ax[0].set_title("a  Upper bound on the saving from any closure", loc="left", fontsize=9)
ax[1].hist(share, bins=np.logspace(np.log10(share.min() * 0.9), 2, 25), color=BLUE, edgecolor="white")
ax[1].set_xscale("log")
ax[1].axvline(np.median(share), color="#0b0b0b", lw=0.8, ls="--")
ax[1].text(np.median(share) * 1.15, ax[1].get_ylim()[1] * 0.9, f"median {np.median(share):.1f}%", fontsize=7)
ax[1].set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax[1].set_xlabel("Share of the efficiency gap recovered (%)")
ax[1].set_ylabel("Beneficial closure–demand cases")
ax[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax[1].set_title(f"b  Beneficial closures (n = {len(share)})", loc="left", fontsize=9)
fig.tight_layout()
fig.subplots_adjust(wspace=0.5)
figs["fig4_bound"] = fig

rel = pd.read_excel(os.path.join("output", "demand_variation.xlsx"), sheet_name="hourly_by_city").set_index("city")
q = rel[[c for c in rel.columns if str(c).isdigit() and int(c) < 24]]
hrs = [int(c) for c in q.columns]
day = pd.read_excel(os.path.join("output", "summary_tables.xlsx"), sheet_name="daytime")
fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.6))
# a: within-day profile
a1 = axs[0, 0]
a1.fill_between(hrs, q.quantile(0.25), q.quantile(0.75), color=BLUE, alpha=0.2, lw=0)
a1.plot(hrs, q.median(), color=BLUE)
a1.axhline(0.6, color=INK, lw=0.6, ls="--")
a1.set_xlabel("Hour of day (weekdays)")
a1.set_ylabel("Network volume (share of busiest hour)")
a1.set_title(f"a  Loop-detector volumes, {len(q)} cities", loc="left", fontsize=9)
# b: peak-hour choice applied all day, cities ordered by the effect with return trips
keep = [c for c in CITIES if (day.city == c).any() and (day[day.city == c].peak_vehh > 0).all()]
mo = day[day.variant == "morning"].set_index("city").loc[keep]
di = day[day.variant == "directional"].set_index("city").loc[keep].sort_values("daytime_vehh")
mo = mo.loc[di.index]
a2 = axs[0, 1]
y = np.arange(len(di))[::-1]
a2.barh(y + 0.27, mo.peak_vehh * mo.hours, height=0.26, color=GREY, label="peak-hour saving × 13")
a2.barh(y, mo.daytime_vehh, height=0.26, color=BLUE, label="day, morning trips only")
a2.barh(y - 0.27, di.daytime_vehh, height=0.26, color=ORANGE, label="day, with return trips")
a2.axvline(0, color=INK, lw=0.6)
a2.set_yticks(y, [c.replace("_", " ") for c in di.index])
a2.set_xlabel("Vehicle-hours per weekday")
a2.set_title("b  Peak-hour choice applied all day", loc="left", fontsize=9)
a2.legend(frameon=False, fontsize=6.5, loc="lower left", handlelength=1.2)
a2.set_ylim(-0.6, len(di) - 0.4)
# c: hour by hour, Boston's peak-hour choice with return trips
B = pd.read_csv(os.path.join(W, "Boston_daytime.csv"))
B = B[B.variant == "directional"].sort_values("hour")
a3 = axs[1, 0]
a3.bar(B.hour, B.veh_h, color=[BLUE if v > 0 else ORANGE for v in B.veh_h], width=0.7)
a3.axhline(0, color=INK, lw=0.6)
a3.set_xlabel("Hour of day (weekdays)")
a3.set_ylabel("Travel-time saving (vehicle-hours)")
a3.set_title("c  Boston, closure chosen at reference demand", loc="left", fontsize=9)
a3.text(0.98, 0.95, f"07:00–19:00 total: {B.veh_h.sum():.0f}".replace("-", "−") + " vehicle-hours", transform=a3.transAxes,
        ha="right", va="top", fontsize=7, color=INK)
# d: peak-hour saving against daytime effect for every closure evaluated at all nine levels
FA = os.path.join("output", "daytime_alternatives.xlsx")
a4 = axs[1, 1]
if os.path.exists(FA):
    A = pd.read_excel(FA)
    A[["peak_bp", "daytime_bp"]] /= 100  # to percent
    T = 0.002
    out = np.where(A.daytime_bp > T, "save", np.where(A.daytime_bp < -T, "add", "tie"))
    pk = (A.peak_bp > T).values
    col = {"save": BLUE, "tie": GREY, "add": ORANGE}
    a4.axhspan(-T, T, color=GREY, alpha=0.25, lw=0)
    a4.axvspan(-T, T, color=GREY, alpha=0.25, lw=0)
    a4.axhline(0, color=INK, lw=0.6)
    a4.axvline(0, color=INK, lw=0.6)
    a4.scatter(A.peak_bp[~pk], A.daytime_bp[~pk], s=14, facecolor="none", edgecolor=GREY, lw=0.8)
    a4.scatter(A.peak_bp[pk], A.daytime_bp[pk], s=26, c=[col[o] for o in out[pk]], edgecolor="white", lw=0.6, zorder=3)
    n = {o: int((out[pk] == o).sum()) for o in col}
    nm = None
    FM = os.path.join("output", "daytime_alternatives_am.xlsx")
    if os.path.exists(FM):  # same closures, morning peak hour = reference demand
        M = pd.read_excel(FM)
        M[["peak_bp", "daytime_bp"]] /= 100
        dm = M[M.peak_bp > T].daytime_bp
        nm = {"save": int((dm > T).sum()), "add": int((dm < -T).sum())}
    alt = (lambda o: f" ({nm[o]})") if nm else (lambda o: "")
    a4.text(0.98, 0.99, f"saves time in both: {n['save']}{alt('save')}", transform=a4.transAxes,
            ha="right", va="top", fontsize=7, color=BLUE)
    a4.text(0.98, 0.04, f"saves in the peak hour,\nadds time over the day: {n['add']}{alt('add')}", transform=a4.transAxes,
            ha="right", va="bottom", fontsize=7, color=ORANGE)
    a4.set_ylim(top=0.016)
    a4.set_xlabel("Saving in the peak hour (% of travel time)")
    a4.set_ylabel("Effect over the day (% of travel time)")
a4.set_title("d  Peak hour against the whole day", loc="left", fontsize=9)
fig.tight_layout()
figs["fig5_withinday"] = fig

# Fig. 1: common screen of 20 cities
F5 = os.path.join("output", "twenty_cities.xlsx")
if os.path.exists(F5):
    T = pd.read_excel(F5).set_index("city")
    LV = ["07", "10", "13"]
    has = T[[f"beneficial_{m}" for m in LV]].gt(0).values
    best = T[[f"best_{m}" for m in LV]]
    def group(i):
        h = has[i]
        if not h.any():
            return 0
        if not h.all():
            return 1
        return 2 if best.iloc[i].nunique() == 1 else 3
    T["group"] = [group(i) for i in range(len(T))]
    T = T.sort_values(["group", "poa_10"], ascending=[False, False])
    GN = ["No beneficial closure at any level", "Beneficial at some levels only",
          "Same best closure at all three levels", "Best closure differs between levels"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    for k, (c, r) in enumerate(T.iterrows()):
        y = len(T) - 1 - k
        ref = r.best_10
        for j, m in enumerate(LV):
            b = r[f"best_{m}"]
            if r[f"beneficial_{m}"] == 0:
                ax[0].scatter(j, y, s=60, facecolor="white", edgecolor=GREY, lw=1)
            elif pd.isna(ref) or b == ref or r.group == 2:
                ax[0].scatter(j, y, s=60, color=BLUE)
            else:
                ax[0].scatter(j, y, s=60, color=ORANGE)
        ax[1].barh(y, (1 - 1 / r.poa_10) * 1e2, color=BLUE if r.group > 0 else GREY, height=0.6)
    # beneficial closures found only in the full analysis (03)
    dag = set()
    for c in T.index:
        f = os.path.join("output", "work", f"{c}_scan.csv")
        if c in CITIES and T.loc[c, "group"] == 0 and os.path.exists(f):
            S = pd.read_csv(f)
            if (S[S.level.isin([0.7, 1.0, 1.3])].delta > 2e-5).any():
                dag.add(c)
    yl = [c.replace("_", " ") + (" †" if c in dag else "") for c in T.index][::-1]
    ax[0].set_yticks(range(len(T)), yl, fontsize=7)
    ax[0].set_xticks([0, 1, 2], ["0.7×", "1.0×", "1.3×"])
    ax[0].set_xlim(-0.6, 2.6)
    ax[0].set_title("a  Beneficial closure by demand level", loc="left", fontsize=9)
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], marker="o", ls="", color=BLUE, label="beneficial, same link as at 1.0× (or only level)"),
          Line2D([], [], marker="o", ls="", color=ORANGE, label="beneficial, different link"),
          Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="no beneficial closure"),
          Line2D([], [], ls="", label="† beneficial closure in the full analysis")]
    ax[0].legend(handles=hd, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.07))
    ax[1].set_yticks(range(len(T)), [""] * len(T))
    ax[1].set_xlabel("Efficiency gap 1 − 1/PoA at 1.0× (% of total travel time)")
    ax[1].set_title("b  Efficiency gap", loc="left", fontsize=9)
    gs = T.group.values
    for g in [3, 2, 1, 0]:
        idx = [len(T) - 1 - k for k in range(len(T)) if gs[k] == g]
        if idx:
            ax[1].text(1.02, (min(idx) + max(idx)) / 2, GN[g], transform=ax[1].get_yaxis_transform(),
                       fontsize=6.5, va="center", ha="left")
            if g:
                ax[0].axhline(min(idx) - 0.5, color=GREY, lw=0.4)
                ax[1].axhline(min(idx) - 0.5, color=GREY, lw=0.4)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    figs["fig1_twenty_cities"] = fig

for name, f in figs.items():
    f.savefig(os.path.join("output", f"{name}.png"), dpi=300)
    f.savefig(os.path.join("output", f"{name}.pdf"))
