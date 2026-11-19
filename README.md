# Mali drought anticipatory action

Analysis for the Mali drought anticipatory-action framework (OCHA Centre for Humanitarian Data).

Site: <https://ocha-dap.github.io/ds-aa-mli-drought/> (landing page; restricted pages are
password-protected).

## Review of the IRI / Mali-Météo candidate forecast triggers (Nov 2026)

IRI proposed seasonal-forecast trigger options for Mali based on the NextGen July–September
rainfall forecast (Maproom `prcp-v4`, national average, issues February–June) with a 10-year
rolling threshold. This repo reproduces IRI's published analysis
(<https://fist.iri.columbia.edu/publications/docs/Mali_AA_OCHA_Framework_candidates/>) from the
public Maproom data and tests the method: trend in the forecast series, sensitivity to the
threshold window, expected activation frequency without the trend, skill with the trend
removed, and the 2026 season.

```bash
uv sync
uv run python scripts/fetch_maproom.py        # Maproom export API -> data/maproom_*.csv
uv run python scripts/fetch_iri_tables.py     # IRI analysis page tables -> data/iri_page/
uv run python scripts/chirps_national.py $(seq 1991 2026)   # CHIRPS JAS national mean (IRI DL)
DSCI_AZ_DB_PROD_HOST=127.0.0.1:15433 uv run --with ocha-stratus --with psycopg2-binary \
    python scripts/fetch_era5.py              # ERA5 JAS national mean (team DB, read-only)
cd scripts && uv run python check_reproduction.py   # must print REPRODUCED
uv run python analysis.py                     # -> data/results.json
uv run python check_no_lookahead.py           # must print NO LOOK-AHEAD: ALL CHECKS PASSED
cd .. && DSCI_AZ_DB_DEV_HOST=127.0.0.1:15432 uv run --with ocha-stratus --with psycopg2-binary \
    python scripts/fetch_impacts.py           # CERF (team mirror) + EM-DAT (team blob), read-only
cd scripts && uv run python impacts.py        # season-by-season impact table -> data/impacts.json
```

| file | what it does |
|---|---|
| `scripts/trigger.py` | IRI's threshold and rule logic (fixed / expanding / rolling, four rules, scoring) |
| `scripts/check_reproduction.py` | compares against every table on IRI's page |
| `scripts/analysis.py` | trend, skill, window sensitivity, shuffled-year expected frequency, 2026 |
| `scripts/impacts.py` | CERF drought allocations, EM-DAT droughts and the Maproom bad-year list by season; hits and RPs per option |
| `scripts/check_no_lookahead.py` | from-scratch rolling rule, leakage test (scramble later years), which window IRI used |

`data/` and `build/` are gitignored. The restricted page is built by a gitignored generator
(`scripts/page/gen_page.py`, `template.html`) because it summarises a partner document that is
not public; only the encrypted `pages/trigger-review/page.enc` is committed
(`scripts/page/encrypt_page.py`, password from `PAGE_PASSWORD`).
