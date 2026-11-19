"""Pull Mali impact records: CERF allocations (team mirror) and EM-DAT droughts (team blob).

- CERF: aa.cerf_allocation (daily OneGMS mirror) joined to aa.cerf_supplement, whose
  valid_year_* fields give each drought allocation's rainfall season (team-curated).
  Dev DB, read-only; needs the DB tunnel (DSCI_AZ_DB_DEV_HOST=127.0.0.1:15432).
- EM-DAT: ocha_stratus.emdat.load_emdat_from_blob("MLI") (manual snapshot; check
  "Last Update"). EM-DAT terms: non-commercial, no redistribution of raw records, so
  data/ stays out of git.

Writes data/cerf_mli.csv, data/emdat_mli_drought.csv and data/emdat_meta.json.

Usage: DSCI_AZ_DB_DEV_HOST=127.0.0.1:15432 uv run --with ocha-stratus --with psycopg2-binary \
           python scripts/fetch_impacts.py
"""

import json
from pathlib import Path

import ocha_stratus as stratus
import pandas as pd
from ocha_stratus import emdat
from sqlalchemy import text

DATA = Path(__file__).resolve().parents[1] / "data"

SQL = """
select a.application_code, a.year, a.window_name, a.emergency_type, a.title,
       a.amount_approved, a.first_project_approved_date,
       s.valid_year_start, s.valid_month_start, s.valid_year_end, s.valid_month_end,
       s.confidence, s.not_drought,
       (coalesce(a.summary, '') || ' ' || coalesce(a.humanitarian_overview, '') || ' '
        || coalesce(a.allocation_rationale, '')) ~* '(drought|rainfall|poor rains|dry spell|harvest)'
           as text_mentions_rain
from aa.cerf_allocation a
left join aa.cerf_supplement s using (application_code)
where a.country_iso3 = 'MLI' and a.year > 0
order by a.first_project_approved_date
"""


def main() -> None:
    with stratus.get_engine("dev").connect() as c:
        cerf = pd.read_sql(text(SQL), c)
    cerf.to_csv(DATA / "cerf_mli.csv", index=False)
    print(f"CERF: {len(cerf)} Mali allocations, {int((cerf.emergency_type == 'Drought').sum())} drought-typed")

    em = emdat.load_emdat_from_blob("MLI", include_historic=True)
    dr = em[em["Disaster Type"] == "Drought"][
        ["DisNo.", "Start Year", "Start Month", "End Year", "End Month", "Total Affected",
         "Total Deaths", "Location", "Last Update"]]
    dr.to_csv(DATA / "emdat_mli_drought.csv", index=False)
    (DATA / "emdat_meta.json").write_text(
        json.dumps({"snapshot_last_update": str(em["Last Update"].max())}))
    print(f"EM-DAT: {len(dr)} Mali drought records, snapshot last update {em['Last Update'].max()}")


if __name__ == "__main__":
    main()
