"""Sync the analysis data between the team's dev blob and a temporary local folder.

The data never stays in the repo. Typical run:

    export MLI_DATA_DIR=$(uv run --with ocha-stratus python scripts/blob_data.py pull)
    uv run python scripts/analysis.py            # any script; reads and writes $MLI_DATA_DIR
    uv run --with ocha-stratus python scripts/blob_data.py push
    rm -rf "$MLI_DATA_DIR"

pull [--to DIR] [--only PREFIX ...]   download into DIR (default: a new temp folder; its path is printed)
push [--from DIR]                     upload every file in DIR (default $MLI_DATA_DIR), overwriting
list                                  list the project's blobs with sizes

Blob layout follows the team convention {PROJECT_PREFIX}/{raw|processed}/{datasource}/{file}: source
downloads go under raw/, everything derived under processed/. RULES maps local paths to blob paths.
Container: `projects` on the dev storage account (private).
"""

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import ocha_stratus as stratus

sys.path.insert(0, str(Path(__file__).parent))
from constants import PROJECT_PREFIX  # noqa: E402

# (local path prefix under the data folder, blob path prefix under PROJECT_PREFIX); first match wins
RULES = [
    ("maproom_", "raw/iri_maproom/maproom_"),
    ("iri_page/", "raw/iri_page/"),
    ("chirps/", "raw/chirps/"),
    ("chirp/", "raw/chirp/"),
    ("cerf_mli.csv", "raw/cerf/cerf_mli.csv"),
    ("emdat_", "raw/emdat/emdat_"),
    ("oni.txt", "raw/oni/oni.txt"),
    ("mli_adm0.geojson", "raw/geoboundaries/mli_adm0.geojson"),
    ("nextgen/pycpt_v4/", "raw/iri_pycpt/prcp-jas-v4/"),
    ("nextgen/nmme/", "raw/nmme/"),
    ("nextgen/ersst5.nc", "raw/ersst/ersst5.nc"),
    ("nextgen/xcountry_pnep30.csv", "raw/iri_maproom/xcountry_pnep30.csv"),
    ("nextgen/xcountry_chirps_jas.csv", "raw/iri_maproom/xcountry_chirps_jas.csv"),
    ("", "processed/"),
]
SKIP = (".DS_Store", ".log")


def to_blob(rel: str) -> str:
    for loc, blob in RULES:
        if rel.startswith(loc):
            return f"{PROJECT_PREFIX}/{blob}{rel[len(loc):]}"
    raise ValueError(rel)


def to_local(name: str) -> str:
    sub = name[len(PROJECT_PREFIX) + 1:]
    for loc, blob in RULES:
        if sub.startswith(blob):
            return loc + sub[len(blob):]
    raise ValueError(name)


def md5(path: Path) -> bytes:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def push(src: Path) -> None:
    cc = stratus.get_container_client("projects", "dev", write=True)
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and not p.name.endswith(SKIP)
             and not (p.name.startswith("log") and p.suffix == ".txt")]
    for p in files:
        rel = p.relative_to(src).as_posix()
        name = to_blob(rel)
        assert to_local(name) == rel, (rel, name)
        with open(p, "rb") as f:
            cc.upload_blob(name, f, overwrite=True, max_concurrency=4)
        props = cc.get_blob_client(name).get_blob_properties()
        local_md5 = md5(p)
        remote_md5 = props.content_settings.content_md5
        ok = props.size == p.stat().st_size and (remote_md5 is None or bytes(remote_md5) == local_md5)
        print(f"{'ok ' if ok else 'BAD'} {props.size:>11,}  {name}", flush=True)
        if not ok:
            raise SystemExit(f"verification failed for {name}")
    print(f"pushed {len(files)} files to projects/{PROJECT_PREFIX}/ (dev)", file=sys.stderr)


def blobs(cc):
    """Project blobs, skipping the folder entries of the hierarchical namespace."""
    for b in cc.list_blobs(name_starts_with=f"{PROJECT_PREFIX}/", include=["metadata"]):
        if (b.metadata or {}).get("hdi_isfolder") == "true":
            continue
        yield b


def pull(dst: Path, only: list[str]) -> None:
    cc = stratus.get_container_client("projects", "dev")
    n = 0
    for b in blobs(cc):
        rel = to_local(b.name)
        if only and not any(rel.startswith(o) for o in only):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            cc.download_blob(b.name, max_concurrency=4).readinto(f)
        n += 1
    print(f"pulled {n} files into {dst}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pull"); p1.add_argument("--to"); p1.add_argument("--only", nargs="*", default=[])
    p2 = sub.add_parser("push"); p2.add_argument("--from", dest="src")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "pull":
        dst = Path(a.to) if a.to else Path(tempfile.mkdtemp(prefix="mli-data-"))
        pull(dst, a.only)
        print(dst)
    elif a.cmd == "push":
        src = a.src or os.environ.get("MLI_DATA_DIR")
        if not src:
            raise SystemExit("give --from DIR or set MLI_DATA_DIR")
        push(Path(src))
    else:
        cc = stratus.get_container_client("projects", "dev")
        tot = 0
        for b in blobs(cc):
            tot += b.size
            print(f"{b.size:>11,}  {b.name}")
        print(f"total {tot / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
