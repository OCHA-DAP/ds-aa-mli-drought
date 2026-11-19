"""Season-by-season impact record for Mali, 2001-2026, and how each trigger combination lines up.

Rows are rainy seasons (JAS of year Y). Impact records are placed on the season that
caused them: a lean-season crisis in mid-Y+1 sits on season Y.

- CERF drought: allocations typed "Drought", placed on the season in the team's
  aa.cerf_supplement (valid_year_start). CERF began in 2006, so the record covers seasons
  2005-2025. Mali's other CERF allocations are typed conflict / displacement; the four
  titled after food insecurity (Mar 2016, Jun 2022, May 2023, May 2024) were read by hand
  and attribute it to conflict, displacement, sanctions and funding gaps, not to rainfall.
  Allocations before 2016 carry no narrative text in the feed.
- EM-DAT drought: placed by the judgements in EMDAT_SEASON (EM-DAT dates are coarse).
  The snapshot (fetch_impacts.py) cannot yet hold lean-season impacts of the 2024 season,
  so the record covers seasons 2001-2023.
- Humanitarian bad years: the Maproom's bad-years-humanitarian list (rank < 15), by season,
  2001-2024.

Each record is scored over its own window. Writes data/impacts.json.
Usage: uv run python scripts/impacts.py
"""

import json
from pathlib import Path

import pandas as pd

import trigger as T

DATA = Path(__file__).resolve().parents[1] / "data"
YEARS = list(range(2001, 2027))

# EM-DAT record id -> rainy season it refers to (lean-season events go on the season before)
EMDAT_SEASON = {
    "2001-9044-MLI": 2000,
    "2005-9142-MLI": 2004,
    "2006-9425-MLI": 2006,
    "2010-9249-MLI": 2009,
    "2011-9661-MLI": 2011,
    "2020-9569-MLI": 2020,
    "2022-9786-MLI": 2021,
}
# shown but not counted: its figure is the 2020 humanitarian plan's people in need, and
# neither the 2019 nor the 2020 season had a national rainfall deficit
EMDAT_NOT_COUNTED = {"2020-9569-MLI"}

WINDOWS = {"cerf": (2005, 2025), "emdat": (2001, 2023), "bad": (2001, 2024), "two_plus": (2001, 2023)}


def main() -> None:
    R = json.loads((DATA / "results.json").read_text())
    cerf = pd.read_csv(DATA / "cerf_mli.csv", parse_dates=["first_project_approved_date"])
    em = pd.read_csv(DATA / "emdat_mli_drought.csv")
    meta = json.loads((DATA / "emdat_meta.json").read_text())
    bad = pd.read_csv(DATA / "maproom_bad_years.csv").set_index("year")["bad_year_rank"]

    cd = cerf[cerf.emergency_type == "Drought"].copy()
    assert cd.valid_year_start.notna().all()
    assert cerf.first_project_approved_date.min().year >= 2006
    cerf_by = {}
    for r in cd.itertuples():
        cerf_by.setdefault(int(r.valid_year_start), []).append(
            {"code": r.application_code, "date": r.first_project_approved_date.strftime("%b %Y"),
             "usd_m": round(r.amount_approved / 1e6, 1), "title": r.title})
    other = cerf[(cerf.emergency_type != "Drought")
                 & cerf.title.str.contains("food|famine", case=False)]

    em_by = {}
    for r in em.itertuples():
        key = r[1]
        assert key in EMDAT_SEASON or int(key[:4]) < 2000, key
        if key not in EMDAT_SEASON or EMDAT_SEASON[key] < 2001:
            continue
        em_by.setdefault(EMDAT_SEASON[key], []).append(
            {"id": key, "affected_m": None if pd.isna(r[6]) else round(r[6] / 1e6, 2),
             "counted": key not in EMDAT_NOT_COUNTED})

    bad_by = {int(y): int(r) for y, r in bad.items() if r < 15 and 2001 <= y <= 2026}

    obs = R["obs"]

    def rank(name, y, lo=1991, hi=2026):
        vals = {yy: v for yy, v in zip(obs["years"], obs[name]) if v is not None and lo <= yy <= hi}
        return 1 + sum(1 for v in vals.values() if v < vals[y]) if y in vals else None

    def inside(k, y):
        return WINDOWS[k][0] <= y <= WINDOWS[k][1]

    rows = []
    for y in sorted(YEARS, reverse=True):
        srcs = [k for k, hit in [("cerf", y in cerf_by),
                                 ("emdat", any(e["counted"] for e in em_by.get(y, []))),
                                 ("bad", y in bad_by)] if hit and inside(k, y)]
        rows.append({"year": y, "cerf": cerf_by.get(y, []), "emdat": em_by.get(y, []),
                     "bad": bad_by.get(y),
                     "known": {k: inside(k, y) for k in ("cerf", "emdat", "bad")},
                     "n_src": len(srcs), "srcs": srcs,
                     "chirps_rank": rank("chirps", y), "era5_rank": rank("era5", y)})

    sets = {k: {r["year"] for r in rows if k in r["srcs"]} for k in ("cerf", "emdat", "bad")}
    sets["two_plus"] = {r["year"] for r in rows if r["n_src"] >= 2 and inside("two_plus", r["year"])}
    n_years = {k: b - a + 1 for k, (a, b) in WINDOWS.items()}

    combos = {}
    for key, c in R["combos"].items():
        act_all = {int(y) for y in c["rolling10"]["act"]}
        sc = {}
        for k, s in sets.items():
            a = {y for y in act_all if inside(k, y)}
            hits = len(a & s)
            sc[k] = {"hits": hits, "n_events": len(s), "n_act": len(a),
                     "p_chance": T.p_hits_by_chance(n_years[k], len(a), len(s), hits) if a else None}
        combos[key] = sc
    out = {"rows": rows, "sets": {k: sorted(v) for k, v in sets.items()}, "windows": WINDOWS,
           "rp": {k: round((n_years[k] + 1) / len(v), 1) for k, v in sets.items()},
           "n_years": n_years, "combos": combos,
           "other_cerf": [{"date": r.first_project_approved_date.strftime("%b %Y"), "title": r.title,
                           "usd_m": round(r.amount_approved / 1e6, 1)} for r in other.itertuples()],
           "emdat_snapshot": meta["snapshot_last_update"]}
    (DATA / "impacts.json").write_text(json.dumps(out, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    print("impact seasons:", out["sets"])
    print("RP:", out["rp"], "n:", n_years)
    for k in ["30_2consec", "40_3in5", "30_2in5", "45_3in5", "20_single"]:
        print(k, {s: (v["hits"], v["n_events"], round(v["p_chance"], 2) if v["p_chance"] is not None else None)
                  for s, v in combos[k].items()})


if __name__ == "__main__":
    main()
