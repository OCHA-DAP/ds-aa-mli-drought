"""Check that the rolling-threshold backtest uses only past forecasts (no look-ahead).

The rule for target year X, issue month m and severity s:
  1. reference = the forecasts for month m, severity s in years X-10 .. X-1;
  2. threshold = the k-th highest of those 10, k = round_half_up(s/100 * 10);
  3. month m counts if forecast(X) >= threshold;
  4. evaluated from 2001 (first year with 10 earlier years, data start 1991).

Four independent checks:
  A. A from-scratch implementation (plain loops, nothing imported from trigger.py or
     analysis.py) must give the same activations as analysis.py (data/results.json) and
     as IRI's published rolling-10 tables, for every severity, rule and year.
  B. Leakage test: for each year X, replace every forecast after X with random numbers and
     recompute. Year X's thresholds, monthly flags and activations must not change. Run on
     trigger.py, on analysis.py and on the from-scratch version. The same test run on the
     fixed full-record method must FAIL, which shows the test can detect look-ahead.
  C. Which window definition did IRI use? Candidate definitions are compared with every one
     of the 1,125 rolling thresholds on IRI's page; only a definition that reproduces all
     of them can be what IRI computed.
  D. The 2026 thresholds and activations come from 2016-2025 only.

Usage: cd scripts && uv run python check_no_lookahead.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
MONTHS = ["feb", "mar", "apr", "may", "jun"]
SEVS = list(range(5, 50, 5))
RULES = ["single", "2consec", "2in5", "3in5"]
TOL = 1e-9


# ---------------------------------------------------------------- A. from scratch
def k_round_half_up(s, n):
    return int(math.floor(s * n / 100 + 0.5))


def scratch_rolling(fc: dict, years: list[int], window: int = 10):
    """fc[(year, month, sev)] -> pnep. Returns {(year, sev): {rule: month or None}} and thresholds."""
    thr, out = {}, {}
    for X in years:
        past = [y for y in years if X - window <= y <= X - 1]
        if len(past) < window:
            continue
        for s in SEVS:
            k = k_round_half_up(s, window)
            flags = []
            for m in MONTHS:
                ref = sorted((fc[(y, m, s)] for y in past), reverse=True)
                t = ref[k - 1] if k >= 1 else math.inf
                thr[(X, m, s)] = t
                flags.append(fc[(X, m, s)] >= t - TOL)
            res = {}
            for rule in RULES:
                hit = None
                for i in range(len(MONTHS)):
                    if rule == "single" and sum(flags[: i + 1]) >= 1 or \
                       rule == "2in5" and sum(flags[: i + 1]) >= 2 or \
                       rule == "3in5" and sum(flags[: i + 1]) >= 3 or \
                       rule == "2consec" and i >= 1 and flags[i] and flags[i - 1]:
                        hit = MONTHS[i]
                        break
                res[rule] = hit
            out[(X, s)] = res
    return out, thr


def load_fc() -> tuple[dict, list[int]]:
    df = pd.read_csv(DATA / "maproom_forecasts.csv")
    df = df[(df.version == "prcp-v4") & df.issue_month.isin(MONTHS) & df.severity.isin(SEVS)]
    fc = {(int(r.year), r.issue_month, int(r.severity)): float(r.pnep) for r in df.itertuples()}
    return fc, sorted(df.year.unique().tolist())


def main() -> None:
    import analysis as A
    import trigger as T

    fc, years = load_fc()
    ok = True
    scratch, sthr = scratch_rolling(fc, years)
    print(f"[A] from-scratch rolling-10: {len(scratch)} year x severity cells, years "
          f"{min(x for x, _ in scratch)}-{max(x for x, _ in scratch)}")

    # A1. vs analysis.py (results.json)
    R = json.loads((DATA / "results.json").read_text())
    mism = 0
    for s in SEVS:
        for rule in RULES:
            mine = R["combos"][f"{s}_{rule}"]["rolling10"]["act"]
            ref = {str(X): v[rule] for (X, ss), v in scratch.items() if ss == s and v[rule]}
            if mine != ref:
                mism += 1
                print("   mismatch vs analysis.py", s, rule, mine, ref)
    print(f"    vs analysis.py (2001-2026, 36 combos): {mism} mismatches")
    ok &= mism == 0

    # A2. vs IRI's published rolling-10 tables (thresholds + activations, 2001-2025)
    page = pd.read_csv(DATA / "iri_page" / "raw_rolling10.csv")
    page["issue_month"] = page.issue_month.str.lower()
    d = max(abs(r.rolling_threshold - sthr[(int(r.year), r.issue_month, int(r.severity))])
            for r in page.itertuples())
    print(f"    vs IRI thresholds: {len(page)} rows, max |diff| = {d:.1e}")
    ok &= d < 1e-9
    mism = 0
    for rule in RULES:
        t = pd.read_csv(DATA / "iri_page" / f"activations_rolling10_{rule}.csv")
        for r in t.itertuples(index=False):
            X, s, act, mon = int(r[1]), int(r[0]), str(r[2]) == "True", r[3]
            mine = scratch[(X, s)][rule]
            if (mine is not None) != act or (act and mine != str(mon).lower()[:3]):
                mism += 1
    print(f"    vs IRI activation tables: {mism} mismatches")
    ok &= mism == 0

    # B. leakage test
    rng = np.random.default_rng(1)

    def scramble_after(X):
        return {k: (v if k[0] <= X else float(rng.uniform(0, 100))) for k, v in fc.items()}

    def df_of(f):
        return pd.DataFrame([{"year": k[0], "issue_month": k[1], "severity": k[2], "pnep": v}
                             for k, v in f.items()])

    base_t = T.apply_rules(T.monthly_thresholds(df_of(fc), "rolling", 10)).set_index(["year", "severity"])
    base_a = A.run(A.cube(df_of(fc), years), years, "rolling", window=10)
    fixed_base = T.apply_rules(T.monthly_thresholds(df_of(fc), "fixed")).set_index(["year", "severity"])
    leaks = {"scratch": 0, "trigger.py": 0, "analysis.py": 0, "expanding (IRI dynamic)": 0}
    fixed_changes = 0
    exp_base = T.apply_rules(T.monthly_thresholds(df_of(fc), "expanding_incl")).set_index(
        ["year", "severity"])
    cols = [f"{r}_month" for r in RULES]
    for X in range(2001, max(years)):  # nothing comes after the last year
        f2 = scramble_after(X)
        s2, t2 = scratch_rolling(f2, years)
        for s in SEVS:
            if s2[(X, s)] != scratch[(X, s)] or any(t2[(X, m, s)] != sthr[(X, m, s)] for m in MONTHS):
                leaks["scratch"] += 1
        tt = T.apply_rules(T.monthly_thresholds(df_of(f2), "rolling", 10)).set_index(["year", "severity"])
        if not tt.loc[X].fillna("-").equals(base_t.loc[X].fillna("-")):
            leaks["trigger.py"] += 1
        aa = A.run(A.cube(df_of(f2), years), years, "rolling", window=10)
        i = years.index(X)
        if not np.array_equal(aa["thr"][i], base_a["thr"][i]) or any(
                aa[r][s].get(X) != base_a[r][s].get(X) for r in RULES for s in SEVS):
            leaks["analysis.py"] += 1
        ee = T.apply_rules(T.monthly_thresholds(df_of(f2), "expanding_incl")).set_index(["year", "severity"])
        if not ee.loc[X][cols].fillna("-").equals(exp_base.loc[X][cols].fillna("-")):
            leaks["expanding (IRI dynamic)"] += 1
        ff = T.apply_rules(T.monthly_thresholds(df_of(f2), "fixed")).set_index(["year", "severity"])
        if not ff.loc[X][cols].fillna("-").equals(fixed_base.loc[X][cols].fillna("-")):
            fixed_changes += 1
    n = max(years) - 2001
    for k, v in leaks.items():
        print(f"[B] leakage, {k}: year X changed when later years were scrambled in {v} of {n} years")
        ok &= v == 0
    print(f"[B] control, fixed full-record threshold: year X changed in {fixed_changes} of {n} years "
          f"(should be > 0: the test detects look-ahead)")
    ok &= fixed_changes > 0

    # C. which definition reproduces IRI's rolling thresholds?
    by = {}
    for (y, m, s), v in fc.items():
        by.setdefault((m, s), {})[y] = v

    def cand(name, X, m, s):
        ser = by[(m, s)]
        k10 = k_round_half_up(s, 10)
        if name == "prior 10 (X-10..X-1)":
            ref = [ser[y] for y in range(X - 10, X)]
        elif name == "incl. current (X-9..X)":
            ref = [ser[y] for y in range(X - 9, X + 1)]
        elif name == "prior 10 + current (X-10..X), k of 11":
            ref = [ser[y] for y in range(X - 10, X + 1)]
            k10 = k_round_half_up(s, 11)
        elif name == "full record 1991-2025":
            ref = [ser[y] for y in range(1991, 2026)]
            k10 = k_round_half_up(s, 35)
        elif name == "centred (X-5..X+4)":
            ref = [ser[y] for y in range(X - 5, X + 5) if y in ser]
        elif name == "next 10 (X+1..X+10)":
            ref = [ser[y] for y in range(X + 1, X + 11) if y in ser]
        if len(ref) < k10 or k10 < 1:
            return math.nan
        return sorted(ref, reverse=True)[k10 - 1]

    names = ["prior 10 (X-10..X-1)", "incl. current (X-9..X)", "prior 10 + current (X-10..X), k of 11",
             "full record 1991-2025", "centred (X-5..X+4)", "next 10 (X+1..X+10)"]
    print("[C] IRI's 1,125 published rolling thresholds reproduced by each candidate definition:")
    for nm in names:
        hits = sum(1 for r in page.itertuples()
                   if abs(cand(nm, int(r.year), r.issue_month, int(r.severity)) - r.rolling_threshold) < 1e-9)
        print(f"    {nm:42s} {hits:5d} of {len(page)}")
        if nm.startswith("prior 10 (X-10"):
            ok &= hits == len(page)
    # IRI's year_position column: 2001 -> 11, i.e. the 11th year, after 10 reference years
    yp = page.groupby("year").year_position.first()
    print(f"    IRI year_position: 2001 -> {yp[2001]}, 2025 -> {yp[2025]}")

    # D. 2026 uses 2016-2025 only
    t26 = {(m, s): sthr[(2026, m, s)] for m in MONTHS for s in SEVS}
    direct = {(m, s): sorted((fc[(y, m, s)] for y in range(2016, 2026)), reverse=True)[k_round_half_up(s, 10) - 1]
              for m in MONTHS for s in SEVS}
    print(f"[D] 2026 thresholds equal the k-th highest of 2016-2025: {t26 == direct}")
    ok &= t26 == direct
    print("NO LOOK-AHEAD: ALL CHECKS PASSED" if ok else "CHECKS FAILED")


if __name__ == "__main__":
    main()
