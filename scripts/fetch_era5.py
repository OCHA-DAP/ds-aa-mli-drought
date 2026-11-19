"""ERA5 Mali national JAS rainfall (team raster-stats DB), as a second check on 2026.

Reads public.era5 (monthly mean precipitation rate, mm/day, national mean over the Mali
admin-0 outline) and writes data/era5_jas_national.csv (JAS total, mm).
Needs the DB tunnel (DSCI_AZ_DB_PROD_HOST=127.0.0.1:15433). Read-only.

Usage: uv run --with ocha-stratus --with psycopg2-binary python scripts/fetch_era5.py
"""

from pathlib import Path

import ocha_stratus as stratus
import pandas as pd
from sqlalchemy import text

OUT = Path(__file__).resolve().parents[1] / "data" / "era5_jas_national.csv"
SQL = """
select valid_date, mean from public.era5
where iso3 = 'MLI' and adm_level = 0
  and extract(month from valid_date) in (7, 8, 9)
order by valid_date
"""

with stratus.get_engine("prod").connect() as c:
    df = pd.read_sql(text(SQL), c, parse_dates=["valid_date"])
df["year"] = df.valid_date.dt.year
df["days"] = df.valid_date.dt.days_in_month
df["mm"] = df["mean"] * df["days"]
jas = df.groupby("year").agg(era5_jas=("mm", "sum"), n=("mm", "size"))
jas = jas[jas.n == 3].drop(columns="n")
jas.to_csv(OUT)
print(jas.tail(8).round(1), len(jas), "years")
