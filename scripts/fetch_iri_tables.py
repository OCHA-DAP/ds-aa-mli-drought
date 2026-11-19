"""Extract the tables embedded in IRI's public Mali candidate-trigger analysis page.

https://fist.iri.columbia.edu/publications/docs/Mali_AA_OCHA_Framework_candidates/ is an
R Markdown page with DataTables widgets. The three sections (fixed threshold, dynamic
(expanding) threshold, 10-year rolling threshold) each carry a raw forecast table and a
36-row summary (9 severities x 4 rules). This saves those six tables as CSV so the
reproduction check in analysis.py can compare against them.

Usage: uv run python scripts/fetch_iri_tables.py
"""

import html as htmllib
import json
import re
from pathlib import Path

import pandas as pd
import requests

URL = "https://fist.iri.columbia.edu/publications/docs/Mali_AA_OCHA_Framework_candidates/"
OUT = Path(__file__).resolve().parents[1] / "data" / "iri_page"


def widgets(page: str) -> list[pd.DataFrame]:
    out = []
    for _, payload in re.findall(
        r'<script type="application/json" data-for="([^"]+)">(.*?)</script>', page, flags=re.S
    ):
        x = json.loads(payload)["x"]
        cols = [htmllib.unescape(re.sub("<[^>]+>", "", c))
                for c in re.findall(r"<th[^>]*>(.*?)</th>", x["container"])]
        data = x["data"]
        if len(cols) != len(data):
            cols = cols[-len(data):] if len(cols) > len(data) else (
                [f"_c{i}" for i in range(len(data) - len(cols))] + cols)
        out.append(pd.DataFrame(dict(zip(cols, data))))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page = requests.get(URL, timeout=120).text
    tabs = widgets(page)
    raw = [t for t in tabs if {"pnep", "issue_month", "severity"} <= set(t.columns)]
    summ = [t for t in tabs if {"Trigger Rule", "Activation Rate (%)", "Hit Rate"} <= set(t.columns)
            and "Distance from 1/4" not in t.columns]
    acts = [t for t in tabs if {"Severity (%)", "Year", "Activated", "Activation month"}
            <= set(t.columns)]
    names = ["fixed", "expanding", "rolling10"]
    assert len(raw) == 3 and len(summ) == 3, (len(raw), len(summ))
    for n, t in zip(names, raw):
        t.to_csv(OUT / f"raw_{n}.csv", index=False)
    for n, t in zip(names, summ):
        t.to_csv(OUT / f"summary_{n}.csv", index=False)
    # four activation tables per section, one per rule, in page order
    rules = ["single", "2in5", "2consec", "3in5"]
    for i, t in enumerate(acts):
        t.to_csv(OUT / f"activations_{names[i // 4]}_{rules[i % 4]}.csv", index=False)
    print("wrote", len(list(OUT.glob("*.csv"))), "tables to", OUT)


if __name__ == "__main__":
    main()
