"""First ML layer for the HackAlemAI wind-power forecasting case.

The pipeline is intentionally dependency-light: pandas and numpy are enough to
run the first benchmark.  It compares a wind power-curve baseline with a
weather-aware ridge model and evaluates both using rolling 24/48-hour origins.

The weather-aware validation is an oracle-style benchmark: historical measured
wind and temperature are supplied for the future validation horizon.  In the
final agent these two columns are replaced by archived weather forecasts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET = "power"
WIND = "wind_ms"
TEMP = "temperature_c"
COUNT = "samples_in_hour"


def read_turbine_csv(path: str | Path) -> pd.DataFrame:
    """Read one supplied CSV and return a normalized 10-minute frame."""
    raw = pd.read_csv(path, encoding="utf-8")
    if raw.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns in {path}, got {raw.shape[1]}")

    frame = raw.iloc[:, :5].copy()
    frame.columns = ["id", "timestamp", WIND, TARGET, TEMP]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in [WIND, TARGET, TEMP]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates("timestamp", keep="last")
    return frame.set_index("timestamp")[[WIND, TARGET, TEMP]]


def to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 10-minute data to a fixed hourly grid.

    Empty hours remain in the index.  The target is not imputed.  ``samples_in_hour``
    lets the training/evaluation code exclude low-quality partial hours.
    """
    hourly = frame.resample("1h").agg({WIND: "mean", TARGET: "mean", TEMP: "mean"})
    hourly[COUNT] = frame[TARGET].resample("1h").count().astype("int16")
    return hourly


def _safe_sin(values: np.ndarray, period: float) -> np.ndarray:
    return np.sin(2.0 * np.pi * values / period)


def _safe_cos(values: np.ndarray, period: float) -> np.ndarray:
    return np.cos(2.0 * np.pi * values / period)


FEATURE_NAMES = [
    WIND,
    "wind_sq",
    "wind_cube",
    "wind_fourth",
    "wind_fifth",
    TEMP,
    "temp_sq",
    "wind_x_temp",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "target_lag_1",
    "target_lag_2",
    "target_lag_3",
    "target_lag_6",
    "target_lag_12",
    "target_lag_24",
    "target_lag_48",
    "target_roll_3_mean",
    "target_roll_6_mean",
    "target_roll_24_mean",
    "target_roll_24_std",
]


def make_features(hourly: pd.DataFrame, use_weather: bool = True) -> pd.DataFrame:
    """Create causal features for every hourly timestamp.

    Current-hour weather is exogenous and is therefore intentionally not shifted.
    It represents a weather forecast in production.  Target-derived features are
    shifted first, so they never use the target from the row being predicted.
    """
    idx = hourly.index
    y_state = hourly[TARGET].ffill()
    hour = idx.hour.to_numpy(dtype=float)
    dow = idx.dayofweek.to_numpy(dtype=float)
    doy = idx.dayofyear.to_numpy(dtype=float)

    wind = hourly[WIND].to_numpy(dtype=float) if use_weather else np.full(len(idx), np.nan)
    temp = hourly[TEMP].to_numpy(dtype=float) if use_weather else np.full(len(idx), np.nan)
    features = pd.DataFrame(index=idx)
    features[WIND] = wind
    features["wind_sq"] = wind**2
    features["wind_cube"] = wind**3
    features["wind_fourth"] = wind**4
    features["wind_fifth"] = wind**5
    features[TEMP] = temp
    features["temp_sq"] = temp**2
    features["wind_x_temp"] = wind * temp
    features["hour_sin"] = _safe_sin(hour, 24.0)
    features["hour_cos"] = _safe_cos(hour, 24.0)
    features["dow_sin"] = _safe_sin(dow, 7.0)
    features["dow_cos"] = _safe_cos(dow, 7.0)
    features["doy_sin"] = _safe_sin(doy, 365.25)
    features["doy_cos"] = _safe_cos(doy, 365.25)
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        features[f"target_lag_{lag}"] = y_state.shift(lag)
    shifted = y_state.shift(1)
    features["target_roll_3_mean"] = shifted.rolling(3, min_periods=1).mean()
    features["target_roll_6_mean"] = shifted.rolling(6, min_periods=1).mean()
    features["target_roll_24_mean"] = shifted.rolling(24, min_periods=1).mean()
    features["target_roll_24_std"] = shifted.rolling(24, min_periods=2).std().fillna(0.0)
    return features[FEATURE_NAMES]


def _training_rows(hourly: pd.DataFrame, features: pd.DataFrame, min_samples: int = 6) -> pd.Series:
    valid = hourly[TARGET].notna() & (hourly[COUNT] >= min_samples)
    valid &= features.notna().all(axis=1)
    return valid


def _clip_power(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


class PowerCurveRegressor:
    """One-metre wind-bin historical power curve baseline."""

    def __init__(self, bin_width: float = 1.0):
        self.bin_width = float(bin_width)
        self.centers: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.fallback = 0.0

    def fit(self, hourly: pd.DataFrame) -> "PowerCurveRegressor":
        valid = hourly[TARGET].notna() & hourly[WIND].notna() & (hourly[COUNT] >= 6)
        wind = hourly.loc[valid, WIND].to_numpy(dtype=float)
        power = hourly.loc[valid, TARGET].to_numpy(dtype=float)
        self.fallback = float(np.mean(power)) if len(power) else 0.0
        if not len(power):
            self.centers = np.array([0.0])
            self.values = np.array([self.fallback])
            return self
        bins = np.floor(wind / self.bin_width).astype(int)
        grouped = pd.DataFrame({"bin": bins, "power": power}).groupby("bin")["power"].mean()
        self.centers = grouped.index.to_numpy(dtype=float) * self.bin_width + self.bin_width / 2.0
        self.values = grouped.to_numpy(dtype=float)
        return self

    def predict(self, wind: Iterable[float]) -> np.ndarray:
        if self.centers is None or self.values is None:
            raise RuntimeError("PowerCurveRegressor must be fitted before predict")
        wind_arr = np.asarray(list(wind), dtype=float)
        valid = np.isfinite(wind_arr)
        pred = np.full(len(wind_arr), self.fallback, dtype=float)
        if valid.any():
            pred[valid] = np.interp(wind_arr[valid], self.centers, self.values)
        return _clip_power(pred)


class RidgePowerRegressor:
    """Standardized ridge regression implemented with numpy only."""

    def __init__(self, alpha: float = 5.0):
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgePowerRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.scale_
        z1 = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(z1.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        gram = z1.T @ z1 + penalty
        rhs = z1.T @ y
        beta = np.linalg.solve(gram, rhs)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("RidgePowerRegressor must be fitted before predict")
        x = np.asarray(x, dtype=float)
        z = (x - self.mean_) / self.scale_
        return _clip_power(self.intercept_ + z @ self.coef_)


def fit_ridge(hourly: pd.DataFrame, cutoff: pd.Timestamp, alpha: float = 5.0) -> RidgePowerRegressor:
    train = hourly.loc[hourly.index < cutoff]
    features = make_features(train, use_weather=True)
    valid = _training_rows(train, features)
    model = RidgePowerRegressor(alpha=alpha)
    model.fit(features.loc[valid, FEATURE_NAMES].to_numpy(), train.loc[valid, TARGET].to_numpy())
    return model


def _feature_row(ts: pd.Timestamp, y_state: pd.Series, wind: float, temp: float) -> np.ndarray:
    idx = pd.DatetimeIndex([ts])
    scratch = pd.DataFrame(index=y_state.index.union(idx))
    scratch[TARGET] = y_state.reindex(scratch.index)
    scratch[WIND] = np.nan
    scratch[TEMP] = np.nan
    scratch.loc[ts, WIND] = wind
    scratch.loc[ts, TEMP] = temp
    row = make_features(scratch, use_weather=True).loc[[ts]]
    return row[FEATURE_NAMES].to_numpy(dtype=float)[0]


def recursive_forecast(
    model: RidgePowerRegressor,
    history: pd.DataFrame,
    future_weather: pd.DataFrame,
) -> pd.Series:
    """Forecast recursively, using future wind/temp as exogenous inputs."""
    y_state = history[TARGET].ffill().copy()
    preds: list[float] = []
    for ts, row in future_weather.iterrows():
        wind = float(row[WIND])
        temp = float(row[TEMP])
        x = _feature_row(ts, y_state, wind, temp)
        pred = float(model.predict(x.reshape(1, -1))[0])
        preds.append(pred)
        y_state.loc[ts] = pred
    return pd.Series(preds, index=future_weather.index, name="prediction")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom else float("nan")


def evaluate_origins(
    hourly: pd.DataFrame,
    validation_start: pd.Timestamp,
    horizon: int,
    model_name: str,
    origin_step_hours: int = 24,
) -> dict:
    """Evaluate 24/48-hour recursive forecasts at daily rolling origins."""
    train = hourly.loc[hourly.index < validation_start].copy()
    origin_start = validation_start
    last_origin = hourly.index.max() - pd.Timedelta(hours=horizon - 1)
    validation_end = validation_start + pd.Timedelta(days=28)
    last_origin = min(last_origin, validation_end - pd.Timedelta(hours=1))
    origins = pd.date_range(origin_start, last_origin, freq=f"{origin_step_hours}h")
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    per_origin = []

    if model_name == "power_curve":
        fitted_model = PowerCurveRegressor().fit(train)
    elif model_name == "ridge":
        fitted_model = fit_ridge(train, cutoff=validation_start)
    else:
        raise ValueError(model_name)

    for origin in origins:
        history = hourly.loc[hourly.index < origin].copy()
        future = hourly.loc[(hourly.index >= origin) & (hourly.index < origin + pd.Timedelta(hours=horizon))].copy()
        if len(future) != horizon or future[[WIND, TEMP, TARGET]].isna().any().any() or (future[COUNT] < 6).any():
            continue
        if model_name == "power_curve":
            pred = pd.Series(fitted_model.predict(future[WIND].to_numpy()), index=future.index)
        elif model_name == "ridge":
            pred = recursive_forecast(fitted_model, history, future[[WIND, TEMP]])
        y = future[TARGET].to_numpy(dtype=float)
        p = pred.to_numpy(dtype=float)
        ys.append(y)
        ps.append(p)
        per_origin.append({"origin": origin.isoformat(), "mae": mae(y, p), "rmse": rmse(y, p)})

    if not ys:
        raise RuntimeError(f"No complete validation origins for {model_name}, horizon={horizon}")
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    return {
        "model": model_name,
        "horizon_hours": horizon,
        "origins_used": len(ys),
        "mae": mae(y_all, p_all),
        "rmse": rmse(y_all, p_all),
        "wmape": wmape(y_all, p_all),
        "per_origin": per_origin,
    }


def run_benchmark(path: str | Path, cutoff: str = "2026-01-04 00:00:00") -> dict:
    frame = to_hourly(read_turbine_csv(path))
    validation_start = pd.Timestamp(cutoff)
    result = {
        "file": str(path),
        "hourly_rows": int(len(frame)),
        "complete_hours": int((frame[COUNT] >= 6).sum()),
        "validation_start": validation_start.isoformat(),
        "validation": [],
    }
    for horizon in [24, 48]:
        for model_name in ["power_curve", "ridge"]:
            result["validation"].append(evaluate_origins(frame, validation_start, horizon, model_name))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbine1", required=True)
    parser.add_argument("--turbine2", required=True)
    parser.add_argument("--output", default="outputs/ml_benchmark.json")
    parser.add_argument("--cutoff", default="2026-01-04 00:00:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        "turbine_1": run_benchmark(args.turbine1, args.cutoff),
        "turbine_2": run_benchmark(args.turbine2, args.cutoff),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for turbine, result in results.items():
        print(f"\n{turbine}")
        for row in result["validation"]:
            print(row["model"], row["horizon_hours"], "MAE=", round(row["mae"], 5), "RMSE=", round(row["rmse"], 5), "WMAPE=", round(row["wmape"], 5), "origins=", row["origins_used"])


if __name__ == "__main__":
    main()
