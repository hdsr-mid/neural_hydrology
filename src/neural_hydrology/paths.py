"""Project root and configurable paths for HDSR scripts."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# Databricks/Spark: suppress noisy Py4J bridge logs when scripts set root logging to INFO.
_PY4J_LOGGERS = ("py4j", "py4j.java_gateway", "py4j.clientserver")

# cfgrib: stale .idx sidecar warnings on every GRIB open (harmless; index is rebuilt).
_CFGRIB_LOGGERS = ("cfgrib", "cfgrib.messages")


def _suppress_noisy_third_party_logging() -> None:
    for name in (*_PY4J_LOGGERS, *_CFGRIB_LOGGERS):
        logging.getLogger(name).setLevel(logging.ERROR)


_suppress_noisy_third_party_logging()

# Env key -> default path relative to project root (or special handling).
_PATH_DEFAULTS: dict[str, str] = {
    "DATA_DIR": "data",
    "DATA_ENS_DIR": "data_ens",
    "INFERENCE_RUNS_DIR": "inference_runs",
    "CONFIG_PATH": "config.yml",
    "RUNS_DIR": "runs",
    "OUTPUT_DIR": "runs",
}

# Keys that default to another env-backed path when unset.
_PATH_FALLBACKS: dict[str, str] = {
    "BASE_CONFIG": "CONFIG_PATH",
    "HPO_OUTPUT_DIR": "OUTPUT_DIR",
    "RETRAIN_BASE_DIR": "OUTPUT_DIR",
}

_PROJECT_MARKERS = ("config.yml", "data_ens")


def _project_root_from_package() -> Path:
    """Git repo root (config.yml, data_ens/): two levels above src/neural_hydrology/paths.py."""
    return Path(__file__).resolve().parents[2]


def _discover_project_root(start: Path) -> Path | None:
    current = start.resolve()
    for _ in range(12):
        if any((current / marker).exists() for marker in _PROJECT_MARKERS):
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Return the HDSR project root directory."""
    env_root = os.environ.get("NEURAL_HYDROLOGY_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    pkg_root = _project_root_from_package()
    if any((pkg_root / marker).exists() for marker in _PROJECT_MARKERS):
        return pkg_root

    discovered = _discover_project_root(Path.cwd())
    if discovered is not None:
        return discovered

    return pkg_root


@lru_cache(maxsize=1)
def load_env() -> dict[str, str | None]:
    """
    Load variables from ``<project_root>/.env``.

    Process environment variables take precedence over ``.env`` values.
    """
    env_path = get_project_root() / ".env"
    file_values: dict[str, str | None] = {}
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        file_values = dotenv_values(env_path)

    merged: dict[str, str | None] = dict(file_values)
    for key, value in os.environ.items():
        merged[key] = value
    return merged


def _resolve_path_value(key: str, env: dict[str, str | None], *, _stack: set[str] | None = None) -> Path:
    stack = set() if _stack is None else _stack
    if key in stack:
        raise RuntimeError(f"Circular path fallback for {key!r}")
    stack.add(key)

    raw = env.get(key)
    if raw is not None and str(raw).strip():
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = get_project_root() / path
        return path.resolve()

    if key in _PATH_FALLBACKS:
        fallback_key = _PATH_FALLBACKS[key]
        base = _resolve_path_value(fallback_key, env, _stack=stack)
        if key == "HPO_OUTPUT_DIR":
            return (base / "HPO").resolve()
        if key == "RETRAIN_BASE_DIR":
            return (base / "BATCH_RETRAIN").resolve()
        return base

    if key in _PATH_DEFAULTS:
        return (get_project_root() / _PATH_DEFAULTS[key]).resolve()

    raise KeyError(f"Unknown path key: {key!r}")


def get_path(key: str) -> Path:
    """Resolve a configured path (env > .env > default under project root)."""
    return _resolve_path_value(key, load_env())


def get_env(key: str, default: str | None = None) -> str | None:
    """Read a single env var with .env fallback."""
    value = load_env().get(key)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


BAGGING_MODEL_DIR_KEYS: tuple[str, ...] = tuple(f"BEST_MODEL_DIR_{i}" for i in range(1, 6))


def resolve_bagging_model_dirs() -> list[Path]:
    """
    Resolve the five trained run directories for model-bagging inference.

    Each path must exist and contain ``config.yml``. All ``BEST_MODEL_DIR_1`` …
    ``BEST_MODEL_DIR_5`` must be set in ``.env`` or the process environment.
    """
    env = load_env()
    root = get_project_root()
    dirs: list[Path] = []
    missing: list[str] = []

    for key in BAGGING_MODEL_DIR_KEYS:
        raw = env.get(key)
        if raw is None or not str(raw).strip():
            missing.append(key)
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_dir():
            raise RuntimeError(f"{key} is not a directory: {path}")
        cfg = path / "config.yml"
        if not cfg.is_file():
            raise RuntimeError(
                f"{key} must point to a run folder containing config.yml: {path}"
            )
        dirs.append(path)

    if missing:
        raise RuntimeError(
            "All five model directories are required for ensemble bagging. "
            f"Set in .env: {', '.join(missing)}"
        )
    return dirs
