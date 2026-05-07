from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import netCDF4


DEFAULT_NETCDF_PATH = (Path(__file__).resolve().parents[2] / "inference_runs" / "polders_hdsr_1h.nc").resolve()
DEFAULT_TARGET = "afvoer"
DEFAULT_INTERVALS = (
    (0.05, 0.95),
    (0.25, 0.75),
)


def main() -> None:
    nc_path = Path(DEFAULT_NETCDF_PATH).resolve()
    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    out_dir = nc_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_prefix = f"{DEFAULT_TARGET}_sim_"

    with netCDF4.Dataset(nc_path, mode="r") as root:
        basins = sorted(list(root.groups.keys()))

    for basin in sorted(basins):
        ds = xr.open_dataset(nc_path, group=basin)
        try:
            if "datetime" not in ds:
                raise KeyError(f"'datetime' not found in group {basin}. vars={list(ds.variables)}")

            sim_vars = sorted(
                [v for v in ds.data_vars if v.startswith(sim_prefix)],
                key=lambda s: int(s.split("_")[-1]),
            )
            if not sim_vars:
                raise KeyError(f"No variables found with prefix {sim_prefix!r} in group {basin}.")

            t = ds["datetime"].values
            y = np.vstack([ds[v].values for v in sim_vars])  # [n_ens, n_time]

            out_path = out_dir / f"{Path(nc_path).stem}_{basin}.png"

            med = np.nanmedian(y, axis=0)

            plt.figure(figsize=(14, 5))
            for i in range(y.shape[0]):
                plt.plot(t, y[i, :], color="tab:blue", alpha=0.18, linewidth=1.0)

            # Confidence intervals (equal weights per ensemble member)
            # Wider interval behind, narrower interval on top.
            for (lo_q, hi_q), color, alpha, label in [
                (DEFAULT_INTERVALS[0], "tab:orange", 0.18, "5–95%"),
                (DEFAULT_INTERVALS[1], "tab:orange", 0.30, "25–75%"),
            ]:
                q_lo = np.nanquantile(y, lo_q, axis=0)
                q_hi = np.nanquantile(y, hi_q, axis=0)
                plt.fill_between(t, q_lo, q_hi, color=color, alpha=alpha, linewidth=0.0, label=label)

            plt.plot(t, med, color="black", linewidth=2.2, label="median", linestyle="--")
            plt.title(f"{basin}: {DEFAULT_TARGET} ({len(sim_vars)} ensembles)")
            plt.xlabel("DatumTijd")
            plt.ylabel("Afvoer (mm)")
            plt.legend(loc="best", frameon=False)
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
        finally:
            ds.close()


if __name__ == "__main__":
    main()

