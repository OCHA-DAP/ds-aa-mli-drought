"""Check that trigger.py reproduces IRI's published numbers before using it for anything else.

Compares, for the fixed / expanding / 10-year rolling sections of IRI's page:
monthly thresholds, per-rule activation years and months, and the 36-row summary
(activations, hit/false-alarm counts against CHIRPS JAS). Also checks that the Maproom
export's pnep matches the page's raw table.

Usage: uv run python scripts/check_reproduction.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import trigger as T

DATA = Path(__file__).resolve().parents[1] / "data"
IRI = DATA / "iri_page"
MAPROOM_THR = None  # set in main()
RULE_KEY = {"rule_single": "single", "rule_2consec": "2consec", "rule_2in5": "2in5",
            "rule_3in5": "3in5"}


def load_raw(name: str) -> pd.DataFrame:
    df = pd.read_csv(IRI / f"raw_{name}.csv")
    df["issue_month"] = df["issue_month"].str.lower()
    return df


def main() -> None:
    raw = load_raw("fixed")
    fc = raw[["year", "issue_month", "severity", "pnep"]]
    obs = pd.read_csv(DATA / "maproom_obs.csv").drop_duplicates("year").set_index("year")["chirps_jas"]
    ok = True
    global MAPROOM_THR
    t = pd.read_csv(DATA / "maproom_thresholds.csv")
    MAPROOM_THR = t[t.version == "prcp-v4"].set_index(["issue_month", "severity"])["maproom_threshold"]

    # Maproom export vs page raw table
    mp = pd.read_csv(DATA / "maproom_forecasts.csv")
    mp = mp[mp.version == "prcp-v4"].merge(fc, on=["year", "issue_month", "severity"],
                                            suffixes=("_api", "_page"))
    d = (mp.pnep_api - mp.pnep_page).abs().max()
    print(f"Maproom API vs IRI page pnep: {len(mp)} rows, max |diff| = {d:.2e}")
    ok &= d < 1e-6

    for name, method, col, years in [
        ("fixed", "fixed", "threshold", range(1991, 2026)),  # page column "threshold"
        ("expanding", "expanding_incl", "dynamic_threshold", None),
        ("rolling10", "rolling", "rolling_threshold", range(2001, 2026)),
    ]:
        if method == "fixed":
            # the Maproom's own fixed thresholds (its rank rule differs from round-half-up at
            # 5, 10 and 30%: k = 1, 3, 10 of 35 rather than 2, 4, 11)
            mt = T.monthly_thresholds(fc, method="fixed_maproom", maproom=MAPROOM_THR)
        else:
            mt = T.monthly_thresholds(fc, method=method, window=10)
        page = load_raw(name).rename(columns={col: "page_thr"})
        m = page.merge(mt, on=["year", "issue_month", "severity"], suffixes=("_page", ""))
        fin = np.isfinite(m["threshold"]) & m["page_thr"].notna()
        dthr = (m.loc[fin, "page_thr"] - m.loc[fin, "threshold"]).abs().max()
        dtrig = (m["trigger_m"].astype(bool) != m["exceed"]).sum()
        print(f"[{name}] thresholds: {fin.sum()} rows, max |diff| = {dthr:.2e}; "
              f"monthly trigger mismatches: {dtrig}")

        ann = T.apply_rules(mt)
        summ = pd.read_csv(IRI / f"summary_{name}.csv")
        mism = 0
        for _, r in summ.iterrows():
            rule, sev = RULE_KEY[r["Trigger Rule"]], r["Severity (%)"]
            a = ann[ann.severity == sev]
            if years is not None:
                a = a[a.year.isin(years)]
            yrs = sorted(a.year)
            act = set(a.loc[a[rule], "year"])
            s = T.score(act, yrs, obs)
            mine = (len(yrs), s["n_act"], s["worthy_action"], s["act_in_vain"], s["fail_to_act"])
            theirs = (r.get("Evaluation Years", len(yrs)), r["Activated Years"], r["Worthy Action"],
                      r["Action in Vain"], r["Failure to Act"])
            if tuple(int(x) for x in mine) != tuple(int(x) for x in theirs):
                mism += 1
                print(f"   mismatch sev {sev} {rule}: mine {mine} vs IRI {theirs}")
        print(f"[{name}] summary rows mismatching: {mism} of {len(summ)}")
        ok &= mism == 0

        # activation years + months per rule
        mm = 0
        for rule in T.RULES:
            pa = pd.read_csv(IRI / f"activations_{name}_{rule}.csv")
            pa = pa[pa["Activated"].astype(str) == "True"]
            for _, r in pa.iterrows():
                hit = ann[(ann.year == r["Year"]) & (ann.severity == r["Severity (%)"])]
                if hit.empty or not hit[rule].iloc[0] or \
                        str(hit[f"{rule}_month"].iloc[0]).lower() != str(r["Activation month"]).lower()[:3]:
                    mm += 1
            n_mine = int(ann[rule].sum()) if years is None else int(ann[ann.year.isin(years)][rule].sum())
            if n_mine != len(pa):
                mm += 1
                print(f"   {rule}: {n_mine} activations vs IRI {len(pa)}")
        print(f"[{name}] activation year/month mismatches: {mm}")
        ok &= mm == 0
    print("REPRODUCED" if ok else "NOT REPRODUCED")


if __name__ == "__main__":
    main()
