"""Evaluation of the IRI / Mali-Météo candidate drought triggers (NextGen JAS forecast, prcp-v4).

Runs on the Maproom pull (fetch_maproom.py), the CHIRPS / ERA5 national JAS series
(chirps_national.py, fetch_era5.py) and writes data/results.json for the page builder.
All 36 severity x rule combinations are evaluated; nothing here picks a "recommended"
option.

Usage: uv run python scripts/analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

import trigger as T

DATA = Path(__file__).resolve().parents[1] / "data"
MONTHS = T.MONITOR_MONTHS
SEVS = list(range(5, 50, 5))  # IRI's grid: 5..45
Y0, Y_LAST_OBS = 1991, 2025
EVAL = list(range(2001, 2026))
RNG = np.random.default_rng(20261119)


# ---------------------------------------------------------------- data
def load_forecasts(version: str = "prcp-v4") -> pd.DataFrame:
    fc = pd.read_csv(DATA / "maproom_forecasts.csv")
    fc = fc[(fc.version == version) & fc.issue_month.isin(MONTHS) & fc.severity.isin(SEVS)]
    return fc[["year", "issue_month", "severity", "pnep"]].dropna()


def cube(fc: pd.DataFrame, years: list[int]) -> np.ndarray:
    """pnep as array [year, month, severity]."""
    p = fc.pivot_table(index="year", columns=["issue_month", "severity"], values="pnep")
    p = p.reindex(years)
    return np.stack([np.stack([p[(m, s)].to_numpy() for s in SEVS], -1) for m in MONTHS], 1)


def load_obs() -> pd.DataFrame:
    mp = (pd.read_csv(DATA / "maproom_obs.csv").drop_duplicates("year")
          .set_index("year")["chirps_jas"].sort_index())
    nat = pd.read_csv(DATA / "chirps_jas_national.csv").set_index("year")
    # own recomputation (geoBoundaries outline) runs ~2.4 mm under the Maproom series;
    # the 2026 value is placed on the Maproom scale with the mean offset over 1991-2025.
    both = nat[["national_cos"]].join(mp.rename("mp"), how="inner")
    off = float((both["mp"] - both["national_cos"]).mean())
    off_sd = float((both["mp"] - both["national_cos"]).std())
    chirps = mp.copy()
    chirps.loc[2026] = nat.loc[2026, "national_cos"] + off
    era5 = pd.read_csv(DATA / "era5_jas_national.csv").set_index("year")["era5_jas"]
    df = pd.DataFrame({"chirps": chirps, "chirps_own": nat["national_cos"], "era5": era5})
    df.attrs["chirps_offset"] = (off, off_sd, len(both))
    return df


def load_bad_years() -> pd.Series:
    return pd.read_csv(DATA / "maproom_bad_years.csv").set_index("year")["bad_year_rank"]


# ---------------------------------------------------------------- vectorised trigger core
def k_of(n: int) -> np.ndarray:
    return np.array([T.top_k(s, n) for s in SEVS])


def thresholds(arr: np.ndarray, method: str, window: int = 10, base: slice | None = None):
    """Thresholds for every row of arr [year, month, sev]; NaN where none defined."""
    ny = arr.shape[0]
    thr = np.full(arr.shape, np.nan)
    if method == "rolling":
        win = np.sort(sliding_window_view(arr, window, axis=0), axis=-1)[..., ::-1]
        k = k_of(window)
        sel = np.take_along_axis(win, (k - 1)[None, None, :, None], -1)[..., 0]
        thr[window:] = sel[: ny - window]
    elif method == "expanding":  # prior years only (no look-ahead)
        for i in range(1, ny):
            ref = np.sort(arr[:i], axis=0)[::-1]
            k = k_of(i)
            for j, kk in enumerate(k):
                if kk >= 1:
                    thr[i, :, j] = ref[kk - 1, :, j]
    elif method == "fixed_maproom":  # the Maproom's own full-record thresholds
        t = pd.read_csv(DATA / "maproom_thresholds.csv")
        t = t[t.version == "prcp-v4"].set_index(["issue_month", "severity"])["maproom_threshold"]
        for i, m in enumerate(MONTHS):
            for j, s in enumerate(SEVS):
                thr[:, i, j] = t.loc[(m, s)]
    elif method == "fixed":
        ref = arr[base] if base is not None else arr
        n = ref.shape[0]
        srt = np.sort(ref, axis=0)[::-1]
        for j, kk in enumerate(k_of(n)):
            thr[:, :, j] = srt[kk - 1, :, j]
    return thr


def rules(exceed: np.ndarray) -> dict[str, np.ndarray]:
    """exceed [year, month, sev] -> activation [year, sev] per rule."""
    n = exceed.sum(1)
    return {
        "single": n >= 1,
        "2in5": n >= 2,
        "3in5": n >= 3,
        "2consec": (exceed[:, 1:] & exceed[:, :-1]).any(1),
    }


def first_month(exceed: np.ndarray, rule: str) -> np.ndarray:
    need = {"single": 1, "2in5": 2, "3in5": 3}
    if rule in need:
        c = np.cumsum(exceed, 1)
        hit = c >= need[rule]
    else:
        hit = np.zeros_like(exceed)
        hit[:, 1:] = exceed[:, 1:] & exceed[:, :-1]
        hit = np.cumsum(hit, 1) >= 1
    idx = np.where(hit.any(1), hit.argmax(1), -1)
    return idx


def run(arr: np.ndarray, years: list[int], method: str, floor: bool = False, **kw) -> dict:
    thr = thresholds(arr, method, **kw)
    ok = ~np.isnan(thr).any(axis=(1, 2))
    exc = np.where(np.isnan(thr), False, arr >= thr - T.TOL)
    if floor:  # a month only counts if the forecast is above the event's base rate
        exc &= arr > np.array(SEVS)[None, None, :]
    act = rules(exc)
    out = {"years": [y for y, o in zip(years, ok) if o], "thr": thr, "exceed": exc, "ok": ok}
    for r, a in act.items():
        fm = first_month(exc, r)
        out[r] = {s: {y: MONTHS[fm[i, j]] for i, y in enumerate(years)
                      if ok[i] and a[i, j]} for j, s in enumerate(SEVS)}
    return out


# ---------------------------------------------------------------- helpers
def rp(n_years: int, n_act: int) -> float | None:
    return round((n_years + 1) / n_act, 2) if n_act else None


def spearman_ci(x, y, alpha=0.10):
    r = stats.spearmanr(x, y).statistic
    n = len(x)
    z, se = np.arctanh(r), 1.06 / np.sqrt(n - 3)  # Fieller et al. SE for Spearman
    q = stats.norm.ppf(1 - alpha / 2)
    return float(r), float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def detrend(s: pd.Series) -> pd.Series:
    sl = stats.linregress(s.index, s.values)
    return s - (sl.intercept + sl.slope * s.index)


def weibull_rank_rp(series: pd.Series, year: int, dry_low: bool = True) -> dict:
    s = series.dropna()
    rank = int((s < s[year]).sum() + 1) if dry_low else int((s > s[year]).sum() + 1)
    return {"rank": rank, "n": len(s), "rp": round((len(s) + 1) / rank, 1),
            "first": int(s.index.min()), "last": int(s.index.max())}


# ---------------------------------------------------------------- main
def main() -> None:
    fc = load_forecasts()
    years_all = list(range(Y0, 2027))  # 2026 forecasts exist (include_upcoming)
    arr = cube(fc, years_all)
    assert not np.isnan(arr).any()
    obs = load_obs()
    chirps = obs["chirps"]
    bad = load_bad_years()
    res: dict = {"meta": {"months": MONTHS, "severities": SEVS, "eval": [EVAL[0], EVAL[-1]],
                          "chirps_offset": obs.attrs["chirps_offset"]}}

    # 1. forecast series + trend per month and severity (1991-2025)
    ser, trend = {}, []
    for j, s in enumerate(SEVS):
        for i, m in enumerate(MONTHS):
            v = pd.Series(arr[:, i, j], index=years_all)
            ser[f"{m}_{s}"] = [round(x, 2) for x in v.tolist()]
            h = v.loc[Y0:Y_LAST_OBS]
            sl = stats.linregress(h.index, h.values)
            trend.append({"month": m, "sev": s, "slope_dec": round(sl.slope * 10, 2),
                          "p": float(sl.pvalue),
                          "mean_1991_2000": round(h.loc[1991:2000].mean(), 1),
                          "mean_2016_2025": round(h.loc[2016:2025].mean(), 1),
                          "share_below_base_2016_2025": float((h.loc[2016:2025] < s).mean())})
    res["forecast"] = {"years": years_all, "series": ser}
    res["trend"] = trend
    o = chirps.loc[Y0:Y_LAST_OBS]
    sl = stats.linregress(o.index, o.values)
    e5 = obs["era5"].loc[Y0:Y_LAST_OBS]
    sle = stats.linregress(e5.index, e5.values)
    res["obs_trend"] = {
        "chirps_slope_dec": round(sl.slope * 10, 1), "chirps_p": float(sl.pvalue),
        "chirps_mean_1991_2000": round(o.loc[1991:2000].mean(), 1),
        "chirps_mean_2016_2025": round(o.loc[2016:2025].mean(), 1),
        "era5_slope_dec": round(sle.slope * 10, 1), "era5_p": float(sle.pvalue)}
    # share of years below the 1991-2020 30th percentile, by decade (what the forecast is about)
    q30 = o.loc[1991:2020].quantile(0.30)
    res["obs_below_p30"] = {d: float((o.loc[d:d + 9] < q30).mean()) for d in (1991, 2001, 2011, 2016)}

    # 2. skill (Spearman, 90% CI), raw and with linear trends removed from both series
    skill = []
    for i, m in enumerate(MONTHS):
        j = SEVS.index(30)
        f = pd.Series(arr[:, i, j], index=years_all).loc[Y0:Y_LAST_OBS]
        for lab, yrs in [("1991-2025", (1991, 2025)), ("2001-2025", (2001, 2025))]:
            ff, oo = f.loc[yrs[0]:yrs[1]], o.loc[yrs[0]:yrs[1]]
            r, lo, hi = spearman_ci(ff.values, oo.values)
            rd, lod, hid = spearman_ci(detrend(ff).values, detrend(oo).values)
            skill.append({"month": m, "period": lab, "r": r, "lo": lo, "hi": hi,
                          "r_dt": rd, "lo_dt": lod, "hi_dt": hid})
    res["skill"] = skill

    # 3. thresholds through time (rolling 10 / fixed full record) for the chart
    roll = run(arr, years_all, "rolling", window=10)
    fixed = run(arr, years_all, "fixed_maproom")
    thr_out = {}
    for j, s in enumerate(SEVS):
        for i, m in enumerate(MONTHS):
            thr_out[f"{m}_{s}"] = {
                "rolling": [None if np.isnan(x) else round(x, 2) for x in roll["thr"][:, i, j]],
                "fixed": round(float(fixed["thr"][0, i, j]), 2)}
    # 2027 threshold (window 2017-2026) for the record
    thr27 = {}
    for j, s in enumerate(SEVS):
        k = T.top_k(s, 10)
        w = arr[years_all.index(2017):, :, j]
        thr27[s] = {m: round(float(np.sort(w[:, i])[::-1][k - 1]), 2) for i, m in enumerate(MONTHS)}
    res["thresholds"] = {"years": years_all, "series": thr_out, "y2027": thr27}

    # below-base-rate exceedances: monthly "trigger" months whose forecast is at or below
    # the climatological probability of the event (the severity itself)
    bb = []
    for j, s in enumerate(SEVS):
        rows = [(y, m) for i_y, y in enumerate(years_all) for i, m in enumerate(MONTHS)
                if y >= 2001 and roll["exceed"][i_y, i, j] and arr[i_y, i, j] <= s]
        thr_below = sum(1 for i_y, y in enumerate(years_all) for i in range(len(MONTHS))
                        if y >= 2001 and roll["thr"][i_y, i, j] < s)
        bb.append({"sev": s, "n_cells": len(years_all[10:]) * len(MONTHS),
                   "thr_below_base": thr_below, "exceed_at_or_below_base": rows})
    res["below_base"] = bb

    # 4. every combination under every threshold method
    methods = {
        "rolling10": ("rolling", {"window": 10}),
        "rolling8": ("rolling", {"window": 8}),
        "rolling12": ("rolling", {"window": 12}),
        "rolling15": ("rolling", {"window": 15}),
        "rolling20": ("rolling", {"window": 20}),
        "expanding": ("expanding", {}),
    }
    combos = {}
    runs = {name: run(arr, years_all, meth, **kw) for name, (meth, kw) in methods.items()}
    runs["rolling10_floor"] = run(arr, years_all, "rolling", window=10, floor=True)
    # fixed thresholds over the full 1991-2025 record and over 2001-2025 only
    last = years_all.index(Y_LAST_OBS) + 1
    runs["fixed_1991"] = run(arr, years_all, "fixed_maproom")  # = fixed over 1991-2025
    runs["fixed_2001"] = run(arr, years_all, "fixed", base=slice(10, last))
    for name, r_ in runs.items():
        for rule in T.RULES:
            for s in SEVS:
                acts = r_[rule][s]
                ev = [y for y in r_["years"] if 2001 <= y <= 2025]
                a = sorted(y for y in acts if 2001 <= y <= 2025)
                combos.setdefault(f"{s}_{rule}", {})[name] = {
                    "act": {str(y): acts[y] for y in sorted(acts)},
                    "n_eval": len(ev), "n_act": len(a), "rp": rp(len(ev), len(a)),
                }

    # scoring for the rolling10 combos against several benchmarks (2001-2025)
    ev_years = EVAL
    o_ev = chirps.loc[ev_years]
    # observed dryness relative to the 10 preceding years (what a rolling forecast threshold
    # implicitly targets): z-score of JAS rain against the previous 10 seasons
    rel = pd.Series({y: (chirps.loc[y] - chirps.loc[y - 10:y - 1].mean())
                     / chirps.loc[y - 10:y - 1].std() for y in ev_years})
    bad_years = {y for y, r in bad.items() if 2001 <= y <= 2025 and r < 15}
    res["benchmarks"] = {
        "chirps_driest": {str(y): round(float(o_ev[y]), 1) for y in o_ev.sort_values().index},
        "chirps_rel10": {str(y): round(float(rel[y]), 2) for y in rel.sort_values().index},
        "bad_years": sorted(bad_years),
        "bad_rank": {str(y): float(r) for y, r in bad.items()},
    }
    for key, c in combos.items():
        s, rule = key.split("_", 1)
        act = {int(y) for y in c["rolling10"]["act"] if 2001 <= int(y) <= 2025}
        n = len(ev_years)
        sc = T.score(act, ev_years, o_ev)
        sc_rel = T.score(act, ev_years, rel.rename(None))
        hits_bad = len(act & bad_years)
        c["score"] = {
            "chirps": {k: sc[k] for k in ("worthy_action", "act_in_vain", "fail_to_act",
                                          "worthy_inaction", "hit_rate", "far", "drought_years")},
            "p_chance": T.p_hits_by_chance(n, len(act), len(act), sc["worthy_action"])
            if act else None,
            "rel10": {"hits": sc_rel["worthy_action"], "drought_years": sc_rel["drought_years"]},
            "p_chance_rel10": T.p_hits_by_chance(n, len(act), len(act), sc_rel["worthy_action"])
            if act else None,
            "bad": {"hits": hits_bad, "n_bad": len(bad_years),
                    "p_chance": T.p_hits_by_chance(n, len(bad_years), len(act), hits_bad)
                    if act else None},
            "by_period": {p: sum(1 for y in act if lo <= y <= hi)
                          for p, (lo, hi) in {"2001-2008": (2001, 2008), "2009-2017": (2009, 2017),
                                              "2018-2025": (2018, 2025)}.items()},
        }
        # v1 / v4 comparison filled below
    # 5. stationary activation rate: shuffle whole years (keeps each year's 5 months and
    # 9 severities together, removes the trend) and re-run the rolling-10 rule.
    # Two versions: raw years shuffled, and years shuffled after removing a linear trend
    # from each month x severity series (keeps only the year-to-year variation).
    n_perm = 2000
    hist = arr[: years_all.index(Y_LAST_OBS) + 1]
    t = np.arange(hist.shape[0])
    X = np.column_stack([np.ones_like(t), t])
    flat = hist.reshape(hist.shape[0], -1)
    beta = np.linalg.lstsq(X, flat, rcond=None)[0]
    hist_dt = (flat - X @ beta + flat.mean(0)).reshape(hist.shape)
    for label, base_arr in [("stationary", hist_dt), ("stationary_raw", hist)]:
        tot = {r: np.zeros(len(SEVS)) for r in T.RULES}
        cnt = 0
        per_rate = {r: [] for r in T.RULES}
        for _ in range(n_perm):
            perm = base_arr[RNG.permutation(base_arr.shape[0])]
            thr = thresholds(perm, "rolling", window=10)
            exc = np.where(np.isnan(thr), False, perm >= thr - T.TOL)[10:]
            act = rules(exc)
            for r in T.RULES:
                tot[r] += act[r].sum(0)
                per_rate[r].append(act[r].mean(0))
            cnt += exc.shape[0]
        for r in T.RULES:
            rates = np.array(per_rate[r])
            for j, s in enumerate(SEVS):
                rate = float(tot[r][j] / cnt)
                combos[f"{s}_{r}"][label] = {
                    "rate": round(rate, 3),
                    "rp": round(1 / rate, 2) if rate else None,
                    "p05": round(float(np.quantile(rates[:, j], 0.05)), 3),
                    "p95": round(float(np.quantile(rates[:, j], 0.95)), 3),
                    # chance of no activation in 8 consecutive years at that rate
                    "p_zero_8yr": round((1 - rate) ** 8, 3)}

    # 6. model version: prcp-v1 under the same rolling-10 rules (Feb-Jun issues)
    fc1 = load_forecasts("prcp-v1")
    yrs1 = sorted(fc1.year.unique())
    yrs1 = [y for y in yrs1 if y >= Y0]
    arr1 = cube(fc1, yrs1)
    keep = ~np.isnan(arr1).any(axis=(1, 2))
    yrs1 = [y for y, k_ in zip(yrs1, keep) if k_]
    arr1 = arr1[keep]
    r1 = run(arr1, yrs1, "rolling", window=10)
    res["v1_years"] = [int(yrs1[0]), int(yrs1[-1])]
    res["meta"]["n_perm"] = n_perm
    for key, c in combos.items():
        s, rule = key.split("_", 1)
        c["v1_rolling10"] = {str(y): m for y, m in sorted(r1[rule][int(s)].items())}

    # cross-check: the vectorised core must give the same rolling-10 activations as trigger.py
    # (which reproduces IRI's published tables, see check_reproduction.py)
    mt = T.monthly_thresholds(fc[fc.year <= 2025], method="rolling", window=10)
    ann = T.apply_rules(mt)
    for key, c in combos.items():
        s, rule = key.split("_", 1)
        a = ann[(ann.severity == int(s)) & ann[rule]]
        ref = {str(y): m for y, m in zip(a.year, a[f"{rule}_month"])}
        mine = {y: m for y, m in c["rolling10"]["act"].items() if int(y) <= 2025}
        assert ref == mine, (key, ref, mine)

    res["combos"] = combos

    # 7. 2026 observed season
    res["y2026"] = {
        "chirps": {"value": round(float(chirps[2026]), 1),
                   "own": round(float(obs.loc[2026, "chirps_own"]), 1),
                   **weibull_rank_rp(chirps.loc[1991:2026], 2026)},
        "chirps_own_series": weibull_rank_rp(obs["chirps_own"].dropna(), 2026),
        "era5": {"value": round(float(obs.loc[2026, "era5"]), 1),
                 **weibull_rank_rp(obs["era5"].loc[1991:2026], 2026),
                 "full": weibull_rank_rp(obs["era5"].dropna(), 2026)},
        "forecast_30": {m: round(float(arr[-1, i, SEVS.index(30)]), 1) for i, m in enumerate(MONTHS)},
    }
    res["obs"] = {"years": [int(y) for y in obs.index],
                  "chirps": [None if np.isnan(x) else round(x, 1) for x in obs["chirps"]],
                  "chirps_own": [None if np.isnan(x) else round(x, 1) for x in obs["chirps_own"]],
                  "era5": [None if np.isnan(x) else round(x, 1) for x in obs["era5"]]}
    (DATA / "results.json").write_text(json.dumps(res, default=lambda x: x.item()
                                                  if hasattr(x, "item") else str(x)))
    print("wrote", DATA / "results.json")


if __name__ == "__main__":
    main()
