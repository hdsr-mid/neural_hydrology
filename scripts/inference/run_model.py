import argparse
import logging
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import yaml
from dotenv import dotenv_values

from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble inference using a trained NeuralHydrology run.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to trained run directory (contains config.yml).")
    parser.add_argument("--data_dir", type=str,
                        default=str((Path(__file__).parent.parent.parent / "data_ens").resolve()),
                        help="Ensemble data directory containing time_series/ and attributes/.")
    parser.add_argument("--out_dir", type=str,
                        default=str((Path(__file__).parent.parent.parent / "inference_runs").resolve()),
                        help="Output directory.")
    parser.add_argument("--n_ensembles", type=int, default=30, help="Number of ensemble members.")
    parser.add_argument("--basin_file", type=str, default=None,
                        help="Optional basin file. Default: <data_dir>/hdsr_polders.txt")
    return parser.parse_args()


def _parse_ensemble_starttime(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    return pd.to_datetime(value, format="%Y%m%d%H", utc=True).tz_convert(None)


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------
def _to_timedelta(freq: str) -> pd.Timedelta:
    if freq.endswith("h"):
        return pd.Timedelta(hours=int(freq[:-1] or "1"))
    if freq.endswith("D"):
        return pd.Timedelta(days=int(freq[:-1] or "1"))
    if freq.endswith("min"):
        return pd.Timedelta(minutes=int(freq[:-3] or "1"))
    raise ValueError(f"Unsupported frequency '{freq}'. Expected e.g. '1h' or '1D'.")


def _ceil_to_offset(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    off = pd.tseries.frequencies.to_offset(freq)
    if off.is_on_offset(ts):
        return ts
    try:
        return ts.ceil(freq)
    except ValueError:
        cur = ts
        for _ in range(10000):
            cur = cur + off
            if off.is_on_offset(cur):
                return cur
        raise RuntimeError(f"Could not ceil timestamp {ts} to frequency {freq}.")


def _align_to_frequencies(ts: pd.Timestamp, frequencies: Iterable[str]) -> pd.Timestamp:
    for f in frequencies:
        ts = _ceil_to_offset(ts, f)
    return ts


# ---------------------------------------------------------------------------
# Period inference
# ---------------------------------------------------------------------------
def _read_netcdf_date_bounds(time_series_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    nc_files = sorted(list(time_series_dir.glob("*.nc")) + list(time_series_dir.glob("*.nc4")))
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {time_series_dir}")

    with xr.open_dataset(nc_files[0]) as ds:
        if "date" not in ds.coords and "date" not in ds.variables:
            raise ValueError(f"NetCDF {nc_files[0]} has no 'date' coordinate/variable.")
        dates = pd.to_datetime(ds["date"].values)
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _compute_warmup_span(
    frequencies: Iterable[str],
    seq_length: dict,
    predict_last_n: dict,
) -> pd.Timedelta:
    return max(
        (int(seq_length[f]) - int(predict_last_n[f])) * _to_timedelta(f)
        for f in frequencies
    )


def _infer_period(
    time_series_dir: Path,
    frequencies: Iterable[str],
    seq_length: dict,
    predict_last_n: dict,
    ensemble_starttime: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    netcdf_first, last = _read_netcdf_date_bounds(time_series_dir)

    freqs = list(frequencies)
    if not freqs:
        raise ValueError("No frequencies configured in use_frequencies.")
    lowest = sort_frequencies(freqs)[0]
    warmup_span = _compute_warmup_span(freqs, seq_length, predict_last_n)

    # ENSEMBLE_STARTTIME marks where predictions should start; fall back to data start if missing.
    start = _align_to_frequencies(pd.Timestamp(ensemble_starttime or netcdf_first), freqs)
    warmup_start = start - warmup_span

    # Push start forward if warmup would predate available NetCDF data.
    if warmup_start < netcdf_first:
        start = _align_to_frequencies(netcdf_first + warmup_span, freqs)
        warmup_start = start - warmup_span

    if start > last:
        raise RuntimeError("Not enough data to start inference within available NetCDF period after warmup.")

    logger.info(
        "NetCDF period | first=%s | last=%s | ENSEMBLE_STARTTIME=%s | start=%s | warmup_start=%s | freqs=%s",
        netcdf_first, last, ensemble_starttime, start, warmup_start, freqs,
    )

    # End at the last complete day (23:00) so hourly length is divisible by 24 in inclusive ranges.
    last_day = last.floor("D")
    end = last_day + pd.Timedelta(hours=23)
    if end > last:
        end = (last_day - pd.Timedelta(days=1)) + pd.Timedelta(hours=23)

    for _ in range(20000):
        if all(
            len(pd.date_range(start=warmup_start, end=end, freq=f))
            % max(int(get_frequency_factor(lowest, f)), 1) == 0
            for f in freqs
        ):
            return start, end
        end = end - _to_timedelta(lowest)
        if end <= start:
            logger.error(
                "Failed to find divisibility-compatible end | last_tried_end=%s | warmup_start=%s | lowest=%s | freqs=%s",
                end, warmup_start, lowest, freqs,
            )
            raise RuntimeError("Could not find an end date satisfying frequency divisibility.")

    raise RuntimeError("Could not infer a suitable period within iteration budget.")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EnsembleGenericDataset(GenericDataset):
    """`GenericDataset` selecting a single ensemble member via `<var>_<k>` columns."""

    def __init__(self, *args, ensemble: int, **kwargs):
        # BaseDataset loads data in __init__ (calls _load_basin_data); set ensemble first.
        self._ensemble = int(ensemble)
        super().__init__(*args, **kwargs)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        files_dir = Path(self.cfg.data_dir) / "time_series"
        nc_files = list(files_dir.glob("*.nc4")) + list(files_dir.glob("*.nc"))
        matches = [f for f in nc_files if f.stem == basin]
        if not matches:
            raise FileNotFoundError(f"No NetCDF file found for basin {basin} in {files_dir}")
        if len(matches) > 1:
            raise ValueError(f"Multiple NetCDF files found for basin {basin} in {files_dir}")

        with xr.open_dataset(matches[0]) as ds:
            if "date" not in ds.dims:
                raise ValueError(f"Expected dimension 'date' in {matches[0]}, got {list(ds.dims)}")
            df = ds.to_dataframe()

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        if isinstance(self.cfg.dynamic_inputs, dict):
            needed = sorted({v for vars_ in self.cfg.dynamic_inputs.values() for v in vars_})
        else:
            needed = list(self.cfg.dynamic_inputs)

        for var in needed:
            ens_col = f"{var}_{self._ensemble}"
            if ens_col in df.columns:
                df[var] = df[ens_col]
            elif var not in df.columns:
                raise KeyError(f"Missing required input column '{ens_col}' (or '{var}') for basin {basin}")
        return df


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _build_test_config(base_config_path: Path, overrides: dict) -> Config:
    with base_config_path.open() as fp:
        cfg_yaml = yaml.safe_load(fp)
    cfg_yaml.update(overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp:
        yaml.safe_dump(cfg_yaml, temp, sort_keys=False)
        temp_path = Path(temp.name)
    try:
        return Config(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Result extraction & output
# ---------------------------------------------------------------------------
def _extract_simulation_series(xr_ds: xr.Dataset, freq: str, target: str) -> xr.DataArray:
    var = f"{target}_sim"
    if var not in xr_ds:
        raise KeyError(f"Expected variable '{var}' in results dataset.")
    stacked = xr_ds[var].stack(datetime=["date", "time_step"])
    dt = _to_timedelta(freq)
    dates = pd.to_datetime(stacked["date"].values)
    steps = stacked["time_step"].values.astype(int)
    datetimes = dates + steps * dt
    stacked = stacked.reset_index("datetime").drop_vars(["date", "time_step"])
    stacked = stacked.assign_coords(datetime=("datetime", datetimes))
    return stacked.set_index(datetime="datetime")


def _write_grouped_netcdf(
    out_path: Path,
    *,
    by_basin: dict[str, dict[str, np.ndarray]],
    datetime_index: pd.DatetimeIndex,
    global_attrs: dict[str, str],
) -> None:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    units = "seconds since 1970-01-01 00:00:00"
    calendar = "standard"
    times_num = netCDF4.date2num(datetime_index.to_pydatetime(), units=units, calendar=calendar)

    with netCDF4.Dataset(out_path, mode="w", format="NETCDF4") as root:
        for k, v in global_attrs.items():
            setattr(root, k, v)
        for basin_id, vars_map in by_basin.items():
            grp = root.createGroup(str(basin_id))
            grp.createDimension("datetime", size=len(datetime_index))
            tvar = grp.createVariable("datetime", "f8", ("datetime",))
            tvar.units = units
            tvar.calendar = calendar
            tvar[:] = times_num
            for var_name, values in vars_map.items():
                v = grp.createVariable(var_name, "f4", ("datetime",), fill_value=np.nan)
                v[:] = values.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Ensemble runner
# ---------------------------------------------------------------------------
def _make_ensemble_dataset_factory(ensemble: int):
    def _get_dataset(self, basin_id: str):
        return EnsembleGenericDataset(
            cfg=self.cfg,
            is_train=False,
            period=self.period,
            basin=basin_id,
            additional_features=self.additional_features,
            id_to_int=self.id_to_int,
            scaler=self.scaler,
            ensemble=ensemble,
        )
    return _get_dataset


def _run_ensemble_member(
    tester,
    *,
    ensemble: int,
    targets: list[str],
    frequencies: list[str],
    per_freq_by_basin: dict[str, dict[str, dict[str, np.ndarray]]],
    per_freq_datetime: dict[str, pd.DatetimeIndex],
) -> None:
    factory = _make_ensemble_dataset_factory(ensemble)
    tester._get_dataset = factory.__get__(tester, tester.__class__)  # type: ignore[attr-defined]
    results = tester.evaluate(save_results=False, metrics=[])

    for freq in frequencies:
        for target in targets:
            for basin in tester.basins:
                if basin not in results or freq not in results[basin]:
                    continue
                da = _extract_simulation_series(results[basin][freq]["xr"], freq=freq, target=target)
                dt_idx = pd.DatetimeIndex(pd.to_datetime(da["datetime"].values), name="datetime")
                if freq not in per_freq_datetime:
                    per_freq_datetime[freq] = dt_idx
                elif not per_freq_datetime[freq].equals(dt_idx):
                    per_freq_datetime[freq] = per_freq_datetime[freq].intersection(dt_idx)
                per_freq_by_basin[freq].setdefault(basin, {})[f"{target}_sim_{ensemble}"] = np.asarray(da.values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _resolve_seq_dict(value, frequencies: list[str]) -> dict:
    return value if isinstance(value, dict) else {frequencies[0]: value}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    basin_file = Path(args.basin_file).resolve() if args.basin_file else (data_dir / "hdsr_polders.txt")

    env = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    ensemble_starttime = _parse_ensemble_starttime(env.get("ENSEMBLE_STARTTIME"))

    time_series_dir = data_dir / "time_series"
    if not time_series_dir.exists():
        raise FileNotFoundError(f"Expected ensemble NetCDFs at {time_series_dir}, but directory does not exist.")

    cfg_base_path = run_dir / "config.yml"
    if not cfg_base_path.exists():
        raise FileNotFoundError(f"Run config not found at {cfg_base_path}")

    base_cfg = Config(cfg_base_path)
    frequencies = list(base_cfg.use_frequencies)
    seq_length = _resolve_seq_dict(base_cfg.seq_length, frequencies)
    predict_last_n = _resolve_seq_dict(base_cfg.predict_last_n, frequencies)

    start, end = _infer_period(
        time_series_dir=time_series_dir,
        frequencies=frequencies,
        seq_length=seq_length,
        predict_last_n=predict_last_n,
        ensemble_starttime=ensemble_starttime,
    )

    cfg = _build_test_config(cfg_base_path, {
        "data_dir": str(data_dir),
        "test_basin_file": str(basin_file),
        "train_basin_file": str(basin_file),
        "validation_basin_file": str(basin_file),
        "metrics": [],
        "test_start_date": start.strftime("%d/%m/%Y"),
        "test_end_date": end.strftime("%d/%m/%Y"),
    })

    tester = get_tester(cfg=cfg, run_dir=run_dir, period="test", init_model=True)
    targets = list(cfg.target_variables)
    frequencies = list(cfg.use_frequencies)

    out_dir.mkdir(parents=True, exist_ok=True)

    # per_freq_by_basin[freq][basin_id][var_name] -> ndarray over datetime
    per_freq_by_basin: dict[str, dict[str, dict[str, np.ndarray]]] = {f: {} for f in frequencies}
    per_freq_datetime: dict[str, pd.DatetimeIndex] = {}

    for k in range(1, int(args.n_ensembles) + 1):
        _run_ensemble_member(
            tester,
            ensemble=k,
            targets=targets,
            frequencies=frequencies,
            per_freq_by_basin=per_freq_by_basin,
            per_freq_datetime=per_freq_datetime,
        )

    for freq in frequencies:
        by_basin = per_freq_by_basin.get(freq, {})
        dt_idx = per_freq_datetime.get(freq)
        if not by_basin or dt_idx is None or len(dt_idx) == 0:
            continue
        _write_grouped_netcdf(
            out_dir / f"polders_hdsr_{freq}.nc",
            by_basin=by_basin,
            datetime_index=dt_idx,
            global_attrs={
                "run_dir": str(run_dir),
                "n_ensembles": str(int(args.n_ensembles)),
                "frequency": str(freq),
                "start_date": str(start),
                "end_date": str(end),
            },
        )


if __name__ == "__main__":
    main()
