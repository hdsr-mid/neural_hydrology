#!/usr/bin/env python3
"""Create a minimal ensemble_forecast.pbix page title (stub). Use PBIP + build_pbix.ps1 for full report."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLANK_URL = (
    "https://media.githubusercontent.com/media/pbi-tools/resources/main/Blank/2.100.1401.pbix"
)
BLANK_CACHE = ROOT / "_build" / "blank2.pbix"
OUT = ROOT / "ensemble_forecast.pbix"


def main() -> None:
    BLANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not BLANK_CACHE.is_file() or BLANK_CACHE.stat().st_size < 1000:
        raise SystemExit(
            f"Download blank template first:\n  curl -L '{BLANK_URL}' -o '{BLANK_CACHE}'"
        )

    shutil.copy2(BLANK_CACHE, OUT)
    with zipfile.ZipFile(OUT, "r") as zin:
        layout = json.loads(zin.read("Report/Layout"))
    layout["sections"][0]["displayName"] = "Ensemble-afvoer"
    layout["sections"][0]["name"] = "EnsembleAfvoer"
    config = json.loads(layout["config"])
    config["reportName"] = "HDSR ensemble-afvoer (1 uur)"
    layout["config"] = json.dumps(config)

    entries: list[tuple[zipfile.ZipInfo, bytes]] = []
    with zipfile.ZipFile(OUT, "r") as zin:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "Report/Layout":
                data = json.dumps(layout).encode("utf-8")
            entries.append((item, data))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for item, data in entries:
            zout.writestr(item, data)

    print(f"Stub PBIX written: {OUT}")
    print("For full model + data: open ensemble_forecast.pbip in Power BI Desktop, then Save As PBIX.")
    print("Windows: .\\scripts\\build_pbix.ps1")


if __name__ == "__main__":
    main()
