"""Sensitivity of the main counts to the threshold for a beneficial closure.

Input:  output/summary_tables.xlsx (08_summary_tables.py; sheet sweep_bp)
        output/daytime_alternatives.xlsx and output/daytime_alternatives_am.xlsx (12_daytime_alternatives.py)
Output: output/threshold_sensitivity.xlsx
"""
import os
import pandas as pd

OUT = "output"
THRESHOLDS = [0.001, 0.002, 0.005, 0.01]  # percent of total travel time

sweep = pd.read_excel(os.path.join(OUT, "summary_tables.xlsx"), sheet_name="sweep_bp")
day = pd.read_excel(os.path.join(OUT, "daytime_alternatives.xlsx"))
day_am = pd.read_excel(os.path.join(OUT, "daytime_alternatives_am.xlsx"))
sweep = sweep[sweep.city.isin(day.city.unique())]  # same five cities as the daytime files

rows = []
for th in THRESHOLDS:
    t = th * 100  # to input units
    ref = sweep[sweep["1.0"] > t]
    pk, pk_am = day[day.peak_bp > t], day_am[day_am.peak_bp > t]
    rows.append({
        "threshold_pct": th,
        "beneficial_at_m1": len(ref),
        "of_these_adding_at_m0.6": int((ref["0.6"] < -t).sum()),
        "peak_hour_savers": len(pk),
        "save_over_day_busiest": int((pk.daytime_bp > t).sum()),
        "save_over_day_morning": int((pk_am.daytime_bp > t).sum()),
        "add_over_day_busiest": int((pk.daytime_bp < -t).sum()),
        "add_over_day_morning": int((pk_am.daytime_bp < -t).sum()),
    })

os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_excel(os.path.join(OUT, "threshold_sensitivity.xlsx"), index=False)
