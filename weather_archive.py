"""Download and version archived weather forecast runs for the two turbines.

The module uses Open-Meteo Single Runs API.  A run is selected from the
forecast initialisation cycles that were already available at the requested
historical ``as_of`` moment.  Each download creates an immutable snapshot:
raw JSON, normalized CSV and one manifest record containing all provenance.

The code uses only Python's standard library.  It can be copied to the root of
the hackathon repository and run independently of ``ml_baseline.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


API_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
DEFAULT_MODEL = "ecmwf_ifs"
DEFAULT_TIMEZONE = "Asia/Almaty"
DEFAULT_AVAILABILITY_LAG_HOURS = 6.0
RUN_CYCLES_UTC = (0, 6, 12, 18)

TURBINES = {
    "turbine_1": {"latitude": 43.645150, "longitude": 78.535604},
    "turbine_2": {"latitude": 43.643194, "longitude": 78.538833},
}


def parse_as_of(value: str, timezone_name: str) -> datetime:
    """Parse a local historical as-of time and return an aware UTC datetime."""
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc

    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError("--as-of must be aligned to an exact hour, for example 2026-01-31T00:00")
    return parsed.astimezone(timezone.utc)


def select_latest_available_run(
    as_of_utc: datetime,
    availability_lag_hours: float = DEFAULT_AVAILABILITY_LAG_HOURS,
) -> datetime:
    """Select the latest 00/06/12/18 UTC run available at ``as_of_utc``.

    Open-Meteo documents the ``run`` parameter as model initialisation time,
    not publication time.  For ECMWF global runs the default six-hour lag is a
    conservative reproducibility assumption; it is stored in the manifest and
    can be overridden from the CLI.
    """
    if as_of_utc.tzinfo is None:
        raise ValueError("as_of_utc must be timezone-aware")
    latest_allowed = as_of_utc - timedelta(hours=availability_lag_hours)
    day = latest_allowed.date()
    candidates = []
    for day_offset in (-1, 0):
        current_day = day + timedelta(days=day_offset)
        for hour in RUN_CYCLES_UTC:
            candidates.append(datetime(current_day.year, current_day.month, current_day.day, hour, tzinfo=timezone.utc))
    available = [candidate for candidate in candidates if candidate <= latest_allowed]
    if not available:
        raise ValueError(f"Could not select a model run for as_of={as_of_utc.isoformat()}")
    return max(available)


def build_request_url(
    latitude: float,
    longitude: float,
    run_utc: datetime,
    request_hours_from_run: int,
    model: str = DEFAULT_MODEL,
) -> str:
    if run_utc.tzinfo is None:
        raise ValueError("run_utc must be timezone-aware")
    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "models": model,
        "run": run_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": str(int(request_hours_from_run)),
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_80m",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "timeformat": "iso8601",
        "cell_selection": "land",
    }
    return f"{API_URL}?{urlencode(params)}"


def fetch_json(url: str, timeout_seconds: int = 90) -> tuple[bytes, dict[str, Any]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "HackAlemAI-WindPower/0.1",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Open-Meteo HTTP {exc.code}: {body[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Open-Meteo: {exc.reason}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Open-Meteo returned invalid JSON") from exc
    if payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', 'unknown error')}")
    return raw, payload


def parse_api_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_hourly(
    payload: dict[str, Any],
    turbine_name: str,
    as_of_utc: datetime,
    horizon_hours: int,
    local_timezone: str,
) -> list[dict[str, Any]]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    variables = {
        "temperature_2m_c": hourly.get("temperature_2m", []),
        "wind_speed_10m_ms": hourly.get("wind_speed_10m", []),
        "wind_speed_80m_ms": hourly.get("wind_speed_80m", []),
    }
    lengths = [len(times)] + [len(values) for values in variables.values()]
    if len(set(lengths)) != 1:
        raise RuntimeError(f"Open-Meteo hourly arrays have different lengths: {lengths}")

    zone = ZoneInfo(local_timezone)
    window_end = as_of_utc + timedelta(hours=horizon_hours)
    rows = []
    for index, value in enumerate(times):
        valid_utc = parse_api_time(value)
        if not (as_of_utc <= valid_utc < window_end):
            continue
        rows.append(
            {
                "turbine": turbine_name,
                "valid_time_utc": valid_utc.isoformat().replace("+00:00", "Z"),
                "valid_time_local": valid_utc.astimezone(zone).isoformat(),
                "temperature_2m_c": variables["temperature_2m_c"][index],
                "wind_speed_10m_ms": variables["wind_speed_10m_ms"][index],
                "wind_speed_80m_ms": variables["wind_speed_80m_ms"][index],
            }
        )
    return rows


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "turbine",
        "valid_time_utc",
        "valid_time_local",
        "temperature_2m_c",
        "wind_speed_10m_ms",
        "wind_speed_80m_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_and_store(
    turbine_name: str,
    as_of_utc: datetime,
    horizon_hours: int,
    output_dir: Path,
    model: str,
    local_timezone: str,
    availability_lag_hours: float,
) -> dict[str, Any]:
    coordinates = TURBINES[turbine_name]
    run_utc = select_latest_available_run(as_of_utc, availability_lag_hours)
    lead_hours = math.ceil((as_of_utc - run_utc).total_seconds() / 3600.0)
    request_hours_from_run = lead_hours + horizon_hours
    url = build_request_url(
        latitude=coordinates["latitude"],
        longitude=coordinates["longitude"],
        run_utc=run_utc,
        request_hours_from_run=request_hours_from_run,
        model=model,
    )
    raw, payload = fetch_json(url)
    rows = normalize_hourly(payload, turbine_name, as_of_utc, horizon_hours, local_timezone)
    retrieved_at = datetime.now(timezone.utc)
    snapshot_id = (
        f"{turbine_name}__asof-{as_of_utc:%Y%m%dT%H%MZ}"
        f"__run-{run_utc:%Y%m%dT%H%MZ}"
        f"__retrieved-{retrieved_at:%Y%m%dT%H%M%SZ}"
        f"__{uuid.uuid4().hex[:8]}"
    )
    snapshot_dir = output_dir / "runs" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    raw_path = snapshot_dir / "response.json"
    csv_path = snapshot_dir / "forecast.csv"
    raw_path.write_bytes(raw)
    write_csv(csv_path, rows)

    manifest_entry = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "source": "Open-Meteo Single Runs API",
        "source_url": url,
        "retrieved_at_utc": retrieved_at.isoformat().replace("+00:00", "Z"),
        "as_of_utc": as_of_utc.isoformat().replace("+00:00", "Z"),
        "selected_run_utc": run_utc.isoformat().replace("+00:00", "Z"),
        "availability_lag_hours": availability_lag_hours,
        "model": model,
        "timezone": local_timezone,
        "horizon_hours": horizon_hours,
        "request_hours_from_run": request_hours_from_run,
        "lead_hours_from_run_to_as_of": lead_hours,
        "turbine": turbine_name,
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "sha256_response": sha256_bytes(raw),
        "raw_response_path": str(raw_path.relative_to(output_dir)),
        "forecast_csv_path": str(csv_path.relative_to(output_dir)),
        "response_hour_count": len(payload.get("hourly", {}).get("time", [])),
        "forecast_rows_saved": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    return manifest_entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and version archived weather forecasts")
    parser.add_argument("--as-of", required=True, help="Historical local launch time, e.g. 2026-01-31T00:00")
    parser.add_argument("--horizon-hours", type=int, default=48)
    parser.add_argument("--output-dir", default="outputs/weather_archive")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--availability-lag-hours", type=float, default=DEFAULT_AVAILABILITY_LAG_HOURS)
    parser.add_argument("--turbine", choices=sorted(TURBINES), action="append")
    parser.add_argument("--dry-run", action="store_true", help="Print selected runs and URLs without downloading")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of_utc = parse_as_of(args.as_of, args.timezone)
    turbine_names = args.turbine or list(TURBINES)
    run_utc = select_latest_available_run(as_of_utc, args.availability_lag_hours)

    if args.dry_run:
        print("as_of_utc:", as_of_utc.isoformat())
        print("selected_run_utc:", run_utc.isoformat())
        for turbine_name in turbine_names:
            coordinates = TURBINES[turbine_name]
            print(
                turbine_name,
                build_request_url(
                    coordinates["latitude"],
                    coordinates["longitude"],
                    run_utc,
                    math.ceil((as_of_utc - run_utc).total_seconds() / 3600.0) + args.horizon_hours,
                    args.model,
                ),
            )
        return

    output_dir = Path(args.output_dir)
    for turbine_name in turbine_names:
        entry = fetch_and_store(
            turbine_name=turbine_name,
            as_of_utc=as_of_utc,
            horizon_hours=args.horizon_hours,
            output_dir=output_dir,
            model=args.model,
            local_timezone=args.timezone,
            availability_lag_hours=args.availability_lag_hours,
        )
        print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
