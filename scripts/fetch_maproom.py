"""Pull the Mali national JAS forecast and observation series from the IRI FbF Maproom.

The Maproom (https://iridl.ldeo.columbia.edu/fbfmaproom2/mali) has a public export
endpoint. One call returns, for one forecast issue month and one frequency ("severity"):
the national-average forecast probability for every year, the Maproom's fixed
full-record threshold, and the humanitarian bad-year ranks.

Writes data/maproom_forecasts.csv (long: version x issue month x severity x year),
data/maproom_thresholds.csv, data/maproom_obs.csv (CHIRPS JAS) and
data/maproom_bad_years.csv.

Usage: uv run python scripts/fetch_maproom.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://iridl.ldeo.columbia.edu/fbfmaproom2/mali/export"
OUT = Path(__file__).resolve().parents[1] / "data"
VERSIONS = ["prcp-v4", "prcp-v1"]
ISSUE_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun"]
SEVERITIES = list(range(5, 55, 5))
PREDICTAND = "bad-years-humanitarian"


def export(predictor: str, issue_month: str, freq: int) -> dict:
    params = {
        "mode": 0,
        "season": "season1",
        "issue_month": issue_month,
        "freq": freq,
        "region": "ML",
        "predictor": predictor,
        "predictand": PREDICTAND,
        "include_upcoming": "true",
    }
    for attempt in range(3):
        r = requests.get(BASE, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 500:
            return {}  # e.g. prcp-v4 has no January issue
        time.sleep(5 * (attempt + 1))
    return {}  # e.g. prcp-v4 has no January issue (HTTP 500)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows, thresholds, bad = [], [], {}
    for version in VERSIONS:
        for im in ISSUE_MONTHS:
            for sev in SEVERITIES:
                d = export(version, im, sev)
                if not d:
                    print(version, im, sev, "not available")
                    continue
                thresholds.append(
                    {"version": version, "issue_month": im, "severity": sev,
                     "maproom_threshold": d["threshold"]}
                )
                for h in d["history"]:
                    rows.append(
                        {"version": version, "issue_month": im, "severity": sev,
                         "year": h["year"], "pnep": h.get(version)}
                    )
                    if h.get(PREDICTAND) == h.get(PREDICTAND):  # not NaN
                        bad[h["year"]] = h[PREDICTAND]
                time.sleep(0.5)
            print(version, im, "done")
    pd.DataFrame(rows).to_csv(OUT / "maproom_forecasts.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "maproom_thresholds.csv", index=False)
    pd.Series(bad, name="bad_year_rank").rename_axis("year").sort_index().to_csv(
        OUT / "maproom_bad_years.csv"
    )

    # Observed CHIRPS JAS precipitation (national mean); independent of issue month.
    obs = []
    for sev in SEVERITIES:
        d = export("chirps-precip-jas", "jun", sev)
        for h in d["history"]:
            obs.append({"severity": sev, "year": h["year"],
                        "chirps_jas": h.get("chirps-precip-jas"),
                        "maproom_threshold": d["threshold"]})
        time.sleep(0.5)
    pd.DataFrame(obs).to_csv(OUT / "maproom_obs.csv", index=False)
    print("wrote", sorted(p.name for p in OUT.glob("maproom_*.csv")))


if __name__ == "__main__":
    main()
