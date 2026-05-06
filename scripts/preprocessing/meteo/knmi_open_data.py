from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_REQUESTS_PER_RUN = 100

# Add to the nominal historical fetch end (UTC) so Cabauw/radar series extend through the
# HARMONIE lagged-ensemble composition window (~6 h) tied to ENSEMBLE_STARTTIME.
HISTORICAL_FETCH_END_OFFSET_HOURS = 6


class KnmiRequestBudgetExceeded(RuntimeError):
    def __init__(self, *, max_requests: int, attempted_request_no: int) -> None:
        super().__init__(f"KNMI Open Data request budget exceeded: max={max_requests}, attempted={attempted_request_no}")
        self.max_requests = int(max_requests)
        self.attempted_request_no = int(attempted_request_no)


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


@dataclass(frozen=True)
class KnmiOpenDataClient:
    base_url: str
    api_key: str
    max_requests: int = DEFAULT_MAX_REQUESTS_PER_RUN
    _request_count: int = 0

    @staticmethod
    def from_neural_hydrology_env(*, max_requests: int = DEFAULT_MAX_REQUESTS_PER_RUN) -> "KnmiOpenDataClient":
        # File is in neural_hydrology/scripts/preprocessing/meteo/, so parents[3] is neural_hydrology/
        root = Path(__file__).resolve().parents[3]
        env = load_env_file(root / ".env")
        base_url = env.get("KNMI_API_URL", "https://api.dataplatform.knmi.nl/open-data").rstrip("/")
        api_key = env.get("KNMI_API_KEY")
        if not api_key:
            raise RuntimeError("KNMI_API_KEY not set in neural_hydrology/.env")
        return KnmiOpenDataClient(base_url=base_url, api_key=api_key, max_requests=int(max_requests), _request_count=0)

    def with_request_budget(self, *, max_requests: int = DEFAULT_MAX_REQUESTS_PER_RUN) -> "KnmiOpenDataClient":
        """
        Return a copy of this client with a fresh request budget.

        We count each outgoing HTTP request (including file downloads) and raise
        `KnmiRequestBudgetExceeded` once the budget is exhausted.
        """
        return KnmiOpenDataClient(base_url=self.base_url, api_key=self.api_key, max_requests=int(max_requests), _request_count=0)

    def _count_request_or_raise(self) -> None:
        next_no = self._request_count + 1
        if next_no > self.max_requests:
            raise KnmiRequestBudgetExceeded(max_requests=self.max_requests, attempted_request_no=next_no)
        object.__setattr__(self, "_request_count", next_no)

    @property
    def request_count(self) -> int:
        return int(self._request_count)

    def _get_json(self, url: str, *, timeout_s: int = 60) -> dict[str, Any]:
        self._count_request_or_raise()
        req = urllib.request.Request(url, headers={"Authorization": self.api_key})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # nosec - URL is constructed from fixed base + path
            data = resp.read()
        return json.loads(data.decode("utf-8"))

    def list_files(
        self,
        *,
        dataset: str,
        version: str,
        max_keys: int = 1000,
        order_by: str = "created",
        sorting: str = "asc",
        start_after_filename: str | None = None,
    ) -> list[dict[str, Any]]:
        # KNMI Open Data API constraint: when using `startAfterFilename`, we must not
        # include `orderBy` or `sorting` (or other paging/query params).
        query: dict[str, str] = {"maxKeys": str(max_keys)}
        if start_after_filename is None:
            query["orderBy"] = order_by
            query["sorting"] = sorting
        else:
            query["startAfterFilename"] = start_after_filename
        url = f"{self.base_url}/v1/datasets/{dataset}/versions/{version}/files?{urllib.parse.urlencode(query)}"
        payload = self._get_json(url, timeout_s=60)
        return list(payload.get("files", []))

    def get_temporary_download_url(self, *, dataset: str, version: str, filename: str) -> str | None:
        url = f"{self.base_url}/v1/datasets/{dataset}/versions/{version}/files/{filename}/url"
        try:
            payload = self._get_json(url, timeout_s=60)
        except urllib.error.HTTPError as e:
            if e.code in {400, 403, 404}:
                return None
            raise
        tmp = payload.get("temporaryDownloadUrl")
        if not tmp:
            return None
        return str(tmp)

    def download_file(
        self,
        *,
        dataset: str,
        version: str,
        filename: str,
        out_dir: Path,
        retries: int = 5,
        timeout_s: int = 600,
        log_each_download: bool = False,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        if out_path.exists():
            return out_path

        tmp_url = self.get_temporary_download_url(dataset=dataset, version=version, filename=filename)
        if not tmp_url:
            raise FileNotFoundError(f"KNMI file not available: {dataset}/{version}/{filename}")
        part_path = out_path.with_suffix(out_path.suffix + ".part")

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if part_path.exists():
                    part_path.unlink()
                if out_path.exists():
                    out_path.unlink()

                if log_each_download:
                    LOGGER.info("Downloading %s (%s) attempt %d/%d", filename, dataset, attempt, retries)
                self._count_request_or_raise()
                req = urllib.request.Request(tmp_url)
                with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(part_path, "wb") as f:  # nosec
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                part_path.replace(out_path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                try:
                    if part_path.exists():
                        part_path.unlink()
                except Exception:
                    pass
                time.sleep(2 * attempt)

        if last_err is not None:
            raise last_err
        return out_path

