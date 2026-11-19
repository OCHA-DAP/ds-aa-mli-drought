"""National-mean CHIRPS JAS rainfall for Mali, to extend the Maproom series to 2026.

The Maproom's CHIRPS JAS series ends in 2025. This recomputes the national mean from
IRI Data Library CHIRPS v2.0 monthly grids (0.05 deg, Jul-Sep summed) inside the Mali
outline (geoBoundaries ADM0), checks it against the Maproom values for the years it has,
and writes data/chirps_jas_national.csv.

Usage: uv run python scripts/chirps_national.py 2002 2011 2014 2024 2025 2026
(downloads any missing year to data/chirps/chirps_jas_<year>.nc first)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
import xarray as xr
from shapely.geometry import shape
from constants import data_dir  # noqa: E402

DATA = data_dir()
DL = ("https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/"
      "X/(-12.5)/(5)/RANGE/Y/(9)/(25)/RANGE/T/(Jul%20{y})/(Sep%20{y})/RANGE/data.nc")


def get(year: int) -> Path:
    p = DATA / "chirps" / f"chirps_jas_{year}.nc"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        r = requests.get(DL.format(y=year), timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def national_mean(year: int, geom, weights: str = "cos") -> float:
    ds = xr.open_dataset(get(year), decode_times=False)
    pr = ds["precipitation"]  # mm/day? (monthly mean rate) -> check units
    days = np.array([31, 31, 30])
    units = pr.attrs.get("units", "")
    tot = (pr * xr.DataArray(days, dims="T")).sum("T") if "day" in units else pr.sum("T")
    X, Y = np.meshgrid(ds["X"].values, ds["Y"].values)
    inside = shapely.contains_xy(geom, X, Y)
    w = np.cos(np.deg2rad(Y)) if weights == "cos" else np.ones_like(Y)
    v = tot.values
    m = inside & np.isfinite(v)
    return float((v[m] * w[m]).sum() / w[m].sum())


def main() -> None:
    years = [int(a) for a in sys.argv[1:]] or [2026]
    gj = json.loads((DATA / "mli_adm0.geojson").read_text())
    geom = shape(gj["features"][0]["geometry"])
    shapely.prepare(geom)
    mp = (pd.read_csv(DATA / "maproom_obs.csv").drop_duplicates("year")
          .set_index("year")["chirps_jas"])
    rows = []
    for y in years:
        ds = xr.open_dataset(get(y), decode_times=False)
        if y == years[0]:
            print("units:", ds["precipitation"].attrs.get("units"))
        rows.append({"year": y, "national_cos": national_mean(y, geom, "cos"),
                     "national_flat": national_mean(y, geom, "flat"),
                     "maproom": mp.get(y, np.nan)})
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    df.to_csv(DATA / "chirps_jas_national.csv", index=False)


if __name__ == "__main__":
    main()
