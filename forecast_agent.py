"""One-command weather -> ML forecast -> uncertainty -> energy pipeline.

Run this file from the hackathon repository root, where ``ml_baseline.py`` and
``weather_archive.py`` are present.  LightGBM is used when installed.  Without
it, the script remains runnable and marks the fallback in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_baseline import (
    COUNT,
    FEATURE_NAMES,
    TARGET,
    TEMP,
    WIND,
    PowerCurveRegressor,
    RidgePowerRegressor,
    _feature_row,
    _training_rows,
    make_features,
    read_turbine_csv,
    to_hourly,
)
from weather_archive import TURBINES, fetch_and_store


try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - depends on the user's environment
    lgb = None


DEFAULT_WEIGHTS = {
    "power_curve": 0.35,
    "ridge": 0.20,
    "lightgbm": 0.45,
}


class LightGBMAdapter:
    """Small adapter for point or quantile LightGBM models."""

    def __init__(self, objective: str = "regression_l1", alpha: float | None = None):
        self.objective = objective
        self.alpha = alpha
        self.model = None

    @property
    def available(self) -> bool:
        return lgb is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LightGBMAdapter":
        if lgb is None:
            raise RuntimeError("LightGBM is not installed. Run: python -m pip install lightgbm")
        params: dict[str, Any] = {
            "objective": self.objective,
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
            "random_state": 42,
            "verbosity": -1,
        }
        if self.objective == "quantile":
            params["alpha"] = self.alpha
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM model is not fitted")
        return np.clip(np.asarray(self.model.predict(x), dtype=float), 0.0, 1.0)


class QuantileBundle:
    """P10/P50/P90 models with a deterministic residual fallback."""

    def __init__(self):
        self.models: dict[float, LightGBMAdapter] = {}
        self.fallback_model: RidgePowerRegressor | None = None
        self.residual_quantiles: dict[float, float] = {}
        self.backend = "lightgbm_quantile"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuantileBundle":
        if lgb is not None:
            for alpha in (0.1, 0.5, 0.9):
                self.models[alpha] = LightGBMAdapter("quantile", alpha).fit(x, y)
            return self

        # Dependency-light fallback.  It is an empirical residual band around
        # the point model, not a substitute for conditional LightGBM quantiles.
        self.backend = "ridge_residual_band_fallback"
        self.fallback_model = RidgePowerRegressor(alpha=5.0).fit(x, y)
        residuals = y - self.fallback_model.predict(x)
        self.residual_quantiles = {
            alpha: float(np.quantile(residuals, alpha)) for alpha in (0.1, 0.5, 0.9)
        }
        return self

    def predict(self, x: np.ndarray) -> tuple[float, float, float]:
        if self.models:
            values = [float(self.models[alpha].predict(x.reshape(1, -1))[0]) for alpha in (0.1, 0.5, 0.9)]
        else:
            if self.fallback_model is None:
                raise RuntimeError("QuantileBundle is not fitted")
            p50 = float(self.fallback_model.predict(x.reshape(1, -1))[0])
            values = [
                p50 + self.residual_quantiles[0.1],
                p50 + self.residual_quantiles[0.5],
                p50 + self.residual_quantiles[0.9],
            ]
        values = np.clip(np.sort(np.asarray(values, dtype=float)), 0.0, 1.0)
        return float(values[0]), float(values[1]), float(values[2])


@dataclass
class ModelBundle:
    power_curve: PowerCurveRegressor
    ridge: RidgePowerRegressor
    lightgbm: LightGBMAdapter | None
    quantiles: QuantileBundle
    weights: dict[str, float]

    @property
    def backend(self) -> str:
        return "lightgbm" if self.lightgbm is not None else "ridge_fallback_no_lightgbm"


def fit_bundle(hourly: pd.DataFrame, cutoff: pd.Timestamp) -> ModelBundle:
    train = hourly.loc[hourly.index < cutoff].copy()
    features = make_features(train, use_weather=True)
    valid = _training_rows(train, features)
    x = features.loc[valid, FEATURE_NAMES].to_numpy(dtype=float)
    y = train.loc[valid, TARGET].to_numpy(dtype=float)

    power_curve = PowerCurveRegressor().fit(train)
    ridge = RidgePowerRegressor(alpha=5.0).fit(x, y)
    point_lightgbm = None
    if lgb is not None:
        point_lightgbm = LightGBMAdapter("regression_l1").fit(x, y)
    quantiles = QuantileBundle().fit(x, y)
    return ModelBundle(power_curve, ridge, point_lightgbm, quantiles, dict(DEFAULT_WEIGHTS))


def parse_local_as_of(value: str, timezone_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(timezone_name).tz_localize(None)
    return timestamp


def load_weather_csv(path: Path, wind_column: str = "wind_speed_10m_ms") -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"valid_time_local", "temperature_2m_c", wind_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Weather file {path} is missing columns: {sorted(missing)}")
    local_times = pd.to_datetime(frame["valid_time_local"])
    if getattr(local_times.dt, "tz", None) is not None:
        local_times = local_times.dt.tz_localize(None)
    result = pd.DataFrame(index=pd.DatetimeIndex(local_times))
    result[WIND] = pd.to_numeric(frame[wind_column], errors="coerce").to_numpy()
    result[TEMP] = pd.to_numeric(frame["temperature_2m_c"], errors="coerce").to_numpy()
    result = result[~result.index.duplicated(keep="last")].sort_index()
    if result[[WIND, TEMP]].isna().any().any():
        raise ValueError(f"Weather file {path} contains missing forecast values")
    return result


def recursive_ensemble_forecast(
    bundle: ModelBundle,
    history: pd.DataFrame,
    future_weather: pd.DataFrame,
    capacity_kw: float | None,
    turbine_name: str,
    weather_snapshot_id: str,
) -> pd.DataFrame:
    y_state = history[TARGET].ffill().copy()
    rows: list[dict[str, Any]] = []
    for timestamp, weather in future_weather.iterrows():
        wind = float(weather[WIND])
        temperature = float(weather[TEMP])
        x = _feature_row(timestamp, y_state, wind, temperature)
        power_curve = float(bundle.power_curve.predict([wind])[0])
        ridge = float(bundle.ridge.predict(x.reshape(1, -1))[0])
        lightgbm = float(bundle.lightgbm.predict(x.reshape(1, -1))[0]) if bundle.lightgbm else ridge
        weights = bundle.weights
        ensemble = (
            weights["power_curve"] * power_curve
            + weights["ridge"] * ridge
            + weights["lightgbm"] * lightgbm
        )
        q10, q50, q90 = bundle.quantiles.predict(x)
        # Keep the uncertainty band centred on the final ensemble prediction.
        shift = ensemble - q50
        p10, p50, p90 = np.sort(np.clip([q10 + shift, ensemble, q90 + shift], 0.0, 1.0))
        row: dict[str, Any] = {
            "valid_time": timestamp.isoformat(),
            "turbine": turbine_name,
            "wind_ms": wind,
            "temperature_c": temperature,
            "power_curve_norm": power_curve,
            "ridge_norm": ridge,
            "lightgbm_norm": lightgbm,
            "power_p10_norm": float(p10),
            "power_p50_norm": float(p50),
            "power_p90_norm": float(p90),
            "ensemble_norm": float(p50),
            "weather_snapshot_id": weather_snapshot_id,
            "quantile_backend": bundle.quantiles.backend,
            "model_backend": bundle.backend,
        }
        if capacity_kw is not None:
            row["capacity_kw"] = float(capacity_kw)
            row["energy_p10_kwh"] = float(p10 * capacity_kw)
            row["energy_p50_kwh"] = float(p50 * capacity_kw)
            row["energy_p90_kwh"] = float(p90 * capacity_kw)
            row["energy_p50_mwh"] = float(p50 * capacity_kw / 1000.0)
        else:
            row["capacity_kw"] = None
            row["energy_p10_kwh"] = None
            row["energy_p50_kwh"] = None
            row["energy_p90_kwh"] = None
            row["energy_p50_mwh"] = None
        rows.append(row)
        y_state.loc[timestamp] = float(p50)
    return pd.DataFrame(rows)


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_turbine(
    turbine_name: str,
    historical_csv: Path,
    as_of_local: str,
    horizon_hours: int,
    weather_dir: Path,
    output_dir: Path,
    timezone_name: str,
    capacity_kw: float | None,
    model: str,
    weather_csv_override: Path | None = None,
) -> dict[str, Any]:
    if model != "ensemble":
        raise ValueError("The integrated runner currently exposes the ensemble as the final model")
    cutoff = parse_local_as_of(as_of_local, timezone_name)
    as_of_utc = pd.Timestamp(as_of_local, tz=timezone_name).to_pydatetime().astimezone(timezone.utc)
    coordinates = TURBINES[turbine_name]
    if weather_csv_override is None:
        weather_entry = fetch_and_store(
            turbine_name=turbine_name,
            as_of_utc=as_of_utc,
            horizon_hours=horizon_hours,
            output_dir=weather_dir,
            model="ecmwf_ifs",
            local_timezone=timezone_name,
            availability_lag_hours=6.0,
        )
        weather_path = weather_dir / weather_entry["forecast_csv_path"]
    else:
        weather_path = weather_csv_override
        weather_entry = {
            "source": "local_weather_snapshot",
            "snapshot_id": weather_path.parent.name,
            "forecast_csv_path": str(weather_path),
        }
    weather = load_weather_csv(weather_path)
    weather = weather.loc[weather.index >= cutoff].iloc[:horizon_hours]
    if len(weather) != horizon_hours:
        raise RuntimeError(f"Expected {horizon_hours} weather rows, got {len(weather)} from {weather_path}")

    hourly = to_hourly(read_turbine_csv(historical_csv))
    bundle = fit_bundle(hourly, cutoff)
    forecast = recursive_ensemble_forecast(
        bundle=bundle,
        history=hourly.loc[hourly.index < cutoff],
        future_weather=weather,
        capacity_kw=capacity_kw,
        turbine_name=turbine_name,
        weather_snapshot_id=weather_entry["snapshot_id"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_as_of = cutoff.strftime("%Y%m%dT%H%M")
    forecast_path = output_dir / f"forecast_{turbine_name}_{safe_as_of}.csv"
    forecast.to_csv(forecast_path, index=False, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "as_of_local": cutoff.isoformat(),
        "as_of_utc": as_of_utc.isoformat().replace("+00:00", "Z"),
        "turbine": turbine_name,
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "historical_csv": str(historical_csv),
        "weather_snapshot_id": weather_entry["snapshot_id"],
        "weather_manifest_record": weather_entry,
        "forecast_path": str(forecast_path),
        "horizon_hours": horizon_hours,
        "capacity_kw": capacity_kw,
        "energy_units": "kWh/MWh" if capacity_kw is not None else None,
        "model": "weighted_ensemble",
        "model_backend": bundle.backend,
        "quantile_backend": bundle.quantiles.backend,
        "weights": bundle.weights,
        "rows": len(forecast),
    }
    append_jsonl(output_dir / "forecast_manifest.jsonl", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weather archive -> ensemble forecast -> energy")
    parser.add_argument("--as-of", required=True, help="Local historical launch time, e.g. 2026-01-31T00:00")
    parser.add_argument("--horizon-hours", type=int, choices=[24, 48], default=48)
    parser.add_argument("--timezone", default="Asia/Almaty")
    parser.add_argument("--weather-dir", default="outputs/weather_archive")
    parser.add_argument("--output-dir", default="outputs/forecast")
    parser.add_argument("--turbine1-csv", default="datasets/Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 1.csv")
    parser.add_argument("--turbine2-csv", default="datasets/Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 2.csv")
    parser.add_argument("--capacity-kw-turbine1", type=float)
    parser.add_argument("--capacity-kw-turbine2", type=float)
    parser.add_argument("--weather-csv-turbine1", help="Use a previously saved forecast.csv instead of downloading")
    parser.add_argument("--weather-csv-turbine2", help="Use a previously saved forecast.csv instead of downloading")
    parser.add_argument("--model", choices=["ensemble"], default="ensemble")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if lgb is None:
        print("WARNING: lightgbm is not installed; using ridge fallback for the LightGBM component and residual quantile bands.")
        print("Install the full model stack with: python -m pip install -r requirements.txt")
    configs = [
        ("turbine_1", Path(args.turbine1_csv), args.capacity_kw_turbine1, Path(args.weather_csv_turbine1) if args.weather_csv_turbine1 else None),
        ("turbine_2", Path(args.turbine2_csv), args.capacity_kw_turbine2, Path(args.weather_csv_turbine2) if args.weather_csv_turbine2 else None),
    ]
    for turbine_name, csv_path, capacity_kw, weather_csv_override in configs:
        manifest = run_turbine(
            turbine_name=turbine_name,
            historical_csv=csv_path,
            as_of_local=args.as_of,
            horizon_hours=args.horizon_hours,
            weather_dir=Path(args.weather_dir),
            output_dir=Path(args.output_dir),
            timezone_name=args.timezone,
            capacity_kw=capacity_kw,
            model=args.model,
            weather_csv_override=weather_csv_override,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
