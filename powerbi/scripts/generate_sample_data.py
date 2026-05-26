#!/usr/bin/env python3
"""Generate sample CSV data matching output_forecast schema (for offline PBIX build)."""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
BASINS = ["AFVG16", "AFVG1", "AFVG13"]
N_ENSEMBLES = 30
N_HOURS = 72


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    out_path = SAMPLE_DIR / "output_forecast_sample.csv"
    start = datetime(2026, 5, 1, 0, 0, 0)
    rows_written = 0

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["datetime", "ensemble_id", "afvoergeb_id", "value"],
        )
        writer.writeheader()
        for basin_idx, basin in enumerate(BASINS):
            base = 2.0 + 0.5 * basin_idx
            for ens in range(1, N_ENSEMBLES + 1):
                phase = ens * 0.07
                for h in range(N_HOURS):
                    t = start + timedelta(hours=h)
                    seasonal = math.sin(h / 12.0 + phase)
                    noise = rng.gauss(0, 0.15)
                    writer.writerow(
                        {
                            "datetime": t.strftime("%Y-%m-%d %H:%M:%S"),
                            "ensemble_id": ens,
                            "afvoergeb_id": basin,
                            "value": round(max(0.0, base + seasonal + noise), 4),
                        }
                    )
                    rows_written += 1

    print(f"Wrote {rows_written} rows to {out_path}")


if __name__ == "__main__":
    main()
