"""Is there a trend in the NextGen forecast probabilities, or level shifts? And is it real?

For the 30% forecast (probability of JAS rainfall in the driest 30%), Feb-Jun issues:
  1. Straight line (1991-2025) vs step models (best one or two break years, searched over
     all splits with at least 4 years per segment), compared by BIC; slope within each
     segment; rank tests between periods.
  2. Observed rainfall by period (CHIRPS, ERA5): mean and share of seasons below the
     1991-2020 30th percentile, i.e. the event the forecast gives a probability for.
  3. ECMWF SEAS5 (team raster-stats DB, Mali national mean; Mar-Jun issues, the earliest that
     cover Jul-Sep at lead <= 6): does an independent forecast show the same shift?
  4. ENSO: JAS ONI (NOAA CPC) against the forecast, and the period shift after removing it.

Writes data/trend_check.json. Needs data/ from the other scripts; SEAS5 is read from the
prod DB once (tunnel: DSCI_AZ_DB_PROD_HOST=127.0.0.1:15433) and cached in data/seas5_mli_jas.csv.

Usage: uv run --with ocha-stratus --with psycopg2-binary python scripts/trend_check.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

DATA = Path(__file__).resolve().parents[1] / "data"
MONTHS = ["feb", "mar", "apr", "may", "jun"]
PER = {"1991-2000": (1991, 2000), "2001-2019": (2001, 2019), "2020-2025": (2020, 2025)}


def forecast(version="prcp-v4", sev=30) -> pd.DataFrame:
    fc = pd.read_csv(DATA / "maproom_forecasts.csv")
    fc = fc[(fc.version == version) & (fc.severity == sev) & fc.issue_month.isin(MONTHS)]
    return fc.pivot_table(index="year", columns="issue_month", values="pnep")[MONTHS]


def bic(y, fitted, k):
    n = len(y)
    rss = float(((y - fitted) ** 2).sum())
    return n * np.log(rss / n) + k * np.log(n), rss


def step_fit(y: np.ndarray, breaks: tuple) -> np.ndarray:
    idx = [0, *breaks, len(y)]
    out = np.empty_like(y, dtype=float)
    for a, b in zip(idx[:-1], idx[1:]):
        out[a:b] = y[a:b].mean()
    return out


def model_compare(s: pd.Series, min_seg=4) -> dict:
    yrs, y = s.index.to_numpy(), s.to_numpy(dtype=float)
    lin = stats.linregress(yrs, y)
    b_lin, rss_lin = bic(y, lin.intercept + lin.slope * yrs, 3)
    # parameters: line = intercept, slope, variance (3); one searched break = two means, break
    # year, variance (4); two breaks = three means, two break years, variance (6)
    best1 = min((bic(y, step_fit(y, (i,)), 4)[0], i) for i in range(min_seg, len(y) - min_seg + 1))
    best2 = min((bic(y, step_fit(y, (i, j)), 6)[0], i, j)
                for i, j in itertools.combinations(range(min_seg, len(y) - min_seg + 1), 2)
                if j - i >= min_seg)
    b_fix, rss_fix = bic(y, step_fit(y, (10, 29)), 6)  # breaks at 2001 and 2020, chosen from these data
    # line through 1991-2019 plus a step at 2020 (does a trend remain once the 2020 drop is allowed?)
    X = np.column_stack([np.ones_like(yrs, dtype=float), yrs - 2005.0, (yrs >= 2020).astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    resid = y - fit
    dof = len(y) - 3
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    tval = beta / np.sqrt(np.diag(cov))
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    b_ls, _ = bic(y, fit, 5)  # intercept, slope, step size, break year, variance
    pre = s.loc[1991:2019]
    lr_pre = stats.linregress(pre.index, pre.values)
    mid = s.loc[2001:2019]
    lr_mid = stats.linregress(mid.index, mid.values)
    tq = stats.t.ppf(0.975, len(mid) - 2)
    # two steps plus a slope in each segment (does any segment trend?)
    seg = {}
    for p, (a, b) in PER.items():
        ss = s.loc[a:b]
        sl = stats.linregress(ss.index, ss.values)
        seg[p] = {"mean": round(float(ss.mean()), 1), "slope_dec": round(sl.slope * 10, 2), "p": float(sl.pvalue)}
    mw_late = stats.mannwhitneyu(s.loc[2001:2019], s.loc[2020:2025], alternative="greater")
    mw_early = stats.mannwhitneyu(s.loc[1991:2000], s.loc[2001:2019], alternative="greater")
    return {
        "linear": {"slope_dec": round(lin.slope * 10, 2), "p": float(lin.pvalue), "bic": round(b_lin, 1)},
        "best_one_break": {"first_year_after": int(yrs[best1[1]]), "bic": round(best1[0], 1)},
        "best_two_breaks": {"first_years_after": [int(yrs[best2[1]]), int(yrs[best2[2]])], "bic": round(best2[0], 1)},
        "steps_2001_2020": {"bic": round(b_fix, 1)},
        "line_plus_step_2020": {"bic": round(b_ls, 1), "slope_dec": round(beta[1] * 10, 2), "slope_p": float(pval[1]),
                                "step": round(beta[2], 1), "step_p": float(pval[2])},
        "slope_1991_2019": {"slope_dec": round(lr_pre.slope * 10, 2), "p": float(lr_pre.pvalue)},
        "slope_2001_2019_ci95": [round((lr_mid.slope - tq * lr_mid.stderr) * 10, 1),
                                 round((lr_mid.slope + tq * lr_mid.stderr) * 10, 1)],
        "mean_2010_2019": round(float(s.loc[2010:2019].mean()), 1),
        "segments": seg,
        "mw_2001_2019_gt_2020_2025_p": float(mw_late.pvalue),
        "mw_1991_2000_gt_2001_2019_p": float(mw_early.pvalue),
        "n_2020_2025_below_2001_2019_min": int((s.loc[2020:2025] < s.loc[2001:2019].min()).sum()),
    }


def seas5() -> pd.DataFrame:
    p = DATA / "seas5_mli_jas.csv"
    if not p.exists():
        import ocha_stratus as stratus
        from sqlalchemy import text
        sql = text("""select issued_date, valid_date, mean from public.seas5
                      where iso3 = 'MLI' and adm_level = 0
                        and extract(month from valid_date) in (7, 8, 9)
                        and extract(month from issued_date) in (3, 4, 5, 6)
                        and extract(year from valid_date) = extract(year from issued_date)""")
        with stratus.get_engine("prod").connect() as c:
            df = pd.read_sql(sql, c, parse_dates=["issued_date", "valid_date"])
        df["mm"] = df["mean"] * df.valid_date.dt.days_in_month
        df["year"] = df.issued_date.dt.year
        df["issue_month"] = df.issued_date.dt.strftime("%b").str.lower()
        g = df.groupby(["year", "issue_month"]).agg(jas=("mm", "sum"), n=("mm", "size")).reset_index()
        g[g.n == 3].drop(columns="n").to_csv(p, index=False)
    return pd.read_csv(p).pivot_table(index="year", columns="issue_month", values="jas")[["mar", "apr", "may", "jun"]]


def oni_jas() -> pd.Series:
    p = DATA / "oni.txt"
    if not p.exists():
        p.write_text(requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=60).text)
    rows = [ln.split() for ln in p.read_text().splitlines()[1:] if ln.strip()]
    df = pd.DataFrame(rows, columns=["seas", "yr", "total", "anom"])
    df = df[df.seas == "JAS"]
    return pd.Series(df.anom.astype(float).values, index=df.yr.astype(int).values)


def main() -> None:
    out = {"nextgen_v4": {}, "nextgen_v1": {}}
    v4, v1 = forecast("prcp-v4"), forecast("prcp-v1")
    for m in MONTHS:
        out["nextgen_v4"][m] = model_compare(v4[m].loc[1991:2025])
        out["nextgen_v4"][m]["y2026"] = round(float(v4.loc[2026, m]), 1)
        out["nextgen_v1"][m] = model_compare(v1[m].loc[1991:2025])
        sv = v4[m].loc[1991:2025]
        out["nextgen_v4"][m]["shift_sd_units"] = round(float((sv.loc[2020:2025].mean() - sv.loc[2001:2019].mean())
                                                             / sv.loc[1991:2020].std()), 2)

    # observed: share of seasons below the 1991-2020 30th percentile
    R = json.loads((DATA / "results.json").read_text())
    obs = pd.DataFrame({k: R["obs"][k] for k in ("chirps", "era5")}, index=R["obs"]["years"]).astype(float)
    ob = {}
    for k in ("chirps", "era5"):
        s = obs[k].loc[1991:2026]
        q30 = s.loc[1991:2020].quantile(0.30)
        ob[k] = {p: {"mean": round(float(s.loc[a:b].mean()), 1), "share_below_p30": round(float((s.loc[a:b] < q30).mean()), 2),
                     "n_below": int((s.loc[a:b] < q30).sum()), "n": int(s.loc[a:b].notna().sum())}
                 for p, (a, b) in {**PER, "2026": (2026, 2026)}.items()}
    for k in ("chirps", "era5"):
        a_, b_ = obs[k].loc[2001:2019], obs[k].loc[2020:2025]
        ob[k]["shift_2020_2025"] = {"pct": round(float(100 * (b_.mean() / a_.mean() - 1)), 1),
                                    "mw_p": float(stats.mannwhitneyu(b_, a_, alternative="greater").pvalue),
                                    "sd_units": round(float((b_.mean() - a_.mean()) / obs[k].loc[1991:2020].std()), 2)}
        c_ = obs[k].loc[2010:2019]
        ob[k]["shift_vs_2010_2019"] = {"mean_2010_2019": round(float(c_.mean()), 1),
                                       "pct": round(float(100 * (b_.mean() / c_.mean() - 1)), 1)}
    out["observed"] = ob
    out["forecast_period_mean_30"] = {p: round(float(v4.loc[a:b].mean().mean()), 1) for p, (a, b) in PER.items()}

    # SEAS5: JAS total forecast, % of its own 1991-2020 mean, by period; and share of years
    # it forecasts below the 30th percentile of its own 1991-2020 forecasts
    s5 = seas5()
    s5o = {}
    for m in s5.columns:
        s = s5[m].loc[1991:2026]
        clim = s.loc[1991:2020].mean()
        q30 = s.loc[1991:2020].quantile(0.30)
        r = stats.spearmanr(s.loc[1991:2025], v4[m].loc[1991:2025])
        s5o[m] = {p: {"pct_of_clim": round(float(100 * s.loc[a:b].mean() / clim), 1),
                      "share_below_own_p30": round(float((s.loc[a:b] < q30).mean()), 2)}
                  for p, (a, b) in {**PER, "2026": (2026, 2026)}.items()}
        s5o[m]["spearman_with_nextgen_30"] = round(float(r.statistic), 2)
        s5o[m]["model"] = model_compare(-s.loc[1991:2025])  # sign flipped: dry = high, like pnep
        s5o[m]["y2017_2019_pct"] = round(float(100 * s.loc[2017:2019].mean() / clim), 1)
        s5o[m]["pct_2020_2025_vs_2010_2019"] = round(float(100 * (s.loc[2020:2025].mean() / s.loc[2010:2019].mean() - 1)), 1)
        s5o[m]["mw_vs_2010_2019_p"] = float(stats.mannwhitneyu(s.loc[2020:2025], s.loc[2010:2019], alternative="greater").pvalue)
        s5o[m]["shift_sd_units"] = round(float((s.loc[2020:2025].mean() - s.loc[2001:2019].mean()) / s.loc[1991:2020].std()), 2)
        s5o[m]["spearman_with_chirps"] = round(float(stats.spearmanr(s.loc[1991:2025], obs["chirps"].loc[1991:2025]).statistic), 2)
    out["seas5"] = s5o

    # ENSO
    oni = oni_jas()
    en = {}
    for m in MONTHS:
        f = v4[m].loc[1991:2026]
        o = oni.reindex(f.index)
        r = stats.spearmanr(o.loc[1991:2025], f.loc[1991:2025])
        lr = stats.linregress(o.loc[1991:2025], f.loc[1991:2025])
        resid = f - (lr.intercept + lr.slope * o)
        en[m] = {"spearman_oni": round(float(r.statistic), 2), "p": float(r.pvalue),
                 "resid_mean": {p: round(float(resid.loc[a:b].mean()), 1) for p, (a, b) in PER.items()},
                 "mw_resid_p": float(stats.mannwhitneyu(resid.loc[2001:2019], resid.loc[2020:2025],
                                                        alternative="greater").pvalue)}
    out["enso"] = en
    out["oni_jas"] = {int(y): float(v) for y, v in oni.loc[2015:2026].items()}
    (DATA / "trend_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"v4_jun": out["nextgen_v4"]["jun"], "v4_mar": out["nextgen_v4"]["mar"]}, indent=1))


if __name__ == "__main__":
    main()
