"""
Download every CALCE CX2 cell (prismatic, 1.35 Ah, LiCoO2) from the
University of Maryland battery data repository.

Usage
-----
    python download_cx2.py                      # all 12 cells: download, verify, unzip
    python download_cx2.py --only CX2_34 CX2_36 # just these cells
    python download_cx2.py --group constant     # only the 0.5C constant-current cells
    python download_cx2.py --no-unzip           # keep the .zip archives packed
    python download_cx2.py --out ~/data/calce   # choose the output folder

Notes
-----
* Re-running skips archives that are already present and valid, so a failed
  run can simply be repeated.
* Each archive holds Arbin .xlsx exports, one file per test segment.
* Rated capacity for CX2 is 1.35 Ah; end of life is usually 80% of that
  (1.08 Ah).

Source: https://calce.umd.edu/battery-data
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://web.calce.umd.edu/batteries/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic data download)"}

# cell -> (group key, test condition described on the CALCE page)
CX2_CELLS: dict[str, tuple[str, str]] = {
    "CX2_16": ("constant", "0.5C constant-current cycling"),
    "CX2_31": ("constant", "0.5C constant-current cycling"),
    "CX2_33": ("constant", "0.5C constant-current cycling"),
    "CX2_35": ("constant", "0.5C constant-current cycling"),
    "CX2_34": ("constant", "0.5C constant-current cycling (second batch)"),
    "CX2_36": ("constant", "0.5C constant-current cycling (second batch)"),
    "CX2_37": ("constant", "0.5C constant-current cycling (second batch)"),
    "CX2_38": ("constant", "0.5C constant-current cycling (second batch)"),
    "CX2_8":  ("varied", "3C constant-current discharge"),
    "CX2_3":  ("varied", "pulsed discharge 0.5C/1C"),
    "CX2_4":  ("varied", "temperature cycling 25-55 C at 1C"),
    "CX2_32": ("varied", "pulsed loading 0.5C/1C/2C"),
}

CHUNK = 1 << 20  # 1 MB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except zipfile.BadZipFile:
        return False


def download(cell: str, dest_dir: Path, retries: int = 3, timeout: int = 180) -> Path | None:
    """Fetch one cell's archive. Returns the path, or None if it failed."""
    url = f"{BASE_URL}/{cell}.zip"
    dest = dest_dir / f"{cell}.zip"
    part = dest.with_suffix(".zip.part")

    if is_valid_zip(dest):
        print(f"    already downloaded ({human(dest.stat().st_size)}) - skipping")
        return dest

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                start = time.time()
                while chunk := resp.read(CHUNK):
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100 * got / total
                        print(f"\r    {human(got)} / {human(total)} ({pct:5.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r    {human(got)}", end="", flush=True)
                elapsed = time.time() - start
            print(f"\r    downloaded {human(got)} in {elapsed:.0f}s" + " " * 20)

            part.replace(dest)
            if not is_valid_zip(dest):
                raise zipfile.BadZipFile("archive failed integrity check")
            return dest

        except (HTTPError, URLError, TimeoutError, zipfile.BadZipFile, OSError) as err:
            print(f"\r    attempt {attempt}/{retries} failed: {err}" + " " * 20)
            part.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            if attempt < retries:
                wait = 5 * attempt
                print(f"    retrying in {wait}s ...")
                time.sleep(wait)

    print(f"    GAVE UP - download by hand: {url}")
    return None


def unzip(archive: Path, out_root: Path) -> int:
    """Extract one archive into out_root/<cell>/. Returns number of files."""
    target = out_root / archive.stem
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        z.extractall(target)
    xlsx = sum(1 for m in members if m.lower().endswith((".xls", ".xlsx")))
    print(f"    extracted {len(members)} files ({xlsx} Excel) -> {target}")
    return len(members)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the CALCE CX2 battery datasets.")
    ap.add_argument("--out", default="calce_cx2", help="output folder (default: calce_cx2)")
    ap.add_argument("--only", nargs="+", metavar="CELL",
                    help="download only these cells, e.g. --only CX2_34 CX2_36")
    ap.add_argument("--group", choices=["all", "constant", "varied"], default="all",
                    help="'constant' = the eight 0.5C cells, 'varied' = the four "
                         "special-condition cells")
    ap.add_argument("--no-unzip", action="store_true", help="keep archives packed")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    cells = dict(CX2_CELLS)
    if args.group != "all":
        cells = {k: v for k, v in cells.items() if v[0] == args.group}
    if args.only:
        wanted = {c.upper().replace("-", "_") for c in args.only}
        unknown = wanted - set(CX2_CELLS)
        if unknown:
            print(f"Unknown cell(s): {', '.join(sorted(unknown))}")
            print("Available:", ", ".join(CX2_CELLS))
            return 2
        cells = {k: v for k, v in CX2_CELLS.items() if k in wanted}

    out_root = Path(args.out).expanduser()
    raw_dir = out_root / "raw"
    ext_dir = out_root / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(cells)} CX2 cell(s) into {out_root.resolve()}\n")

    ok: list[str] = []
    failed: list[str] = []
    for i, (cell, (_group, condition)) in enumerate(cells.items(), start=1):
        print(f"[{i}/{len(cells)}] {cell} - {condition}")
        archive = download(cell, raw_dir, retries=args.retries)
        if archive is None:
            failed.append(cell)
            continue
        if not args.no_unzip:
            try:
                unzip(archive, ext_dir)
            except zipfile.BadZipFile as err:
                print(f"    could not extract: {err}")
                failed.append(cell)
                continue
        ok.append(cell)

    total_bytes = sum(p.stat().st_size for p in raw_dir.glob("*.zip"))
    print("\n" + "-" * 60)
    print(f"Downloaded {len(ok)}/{len(cells)} cells, {human(total_bytes)} on disk")
    print(f"Archives : {raw_dir.resolve()}")
    if not args.no_unzip:
        print(f"Extracted: {ext_dir.resolve()}")
    if failed:
        print("Failed   :", ", ".join(failed), "(re-run this script to retry)")
    else:
        print("\nNext step:")
        print(f"  python extract_capacity.py --in {ext_dir} --out calce_cx2_capacity")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
