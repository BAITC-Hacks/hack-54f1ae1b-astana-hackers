"""Empirical wind-only curve and LightGBM forecasting model."""
from __future__ import annotations
from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .config import ARTIFACTS, CSV_UTC_OFFSET_HOURS

FEATURES = ["wind_ms", "temperature_c", "power_curve", "scada_curve", "wind_sq", "wind_cube",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lag_7d", "lead_hours",
            "recent_power_24h", "recent_wind_24h"]


@dataclass
class ModelBundle:
    site: str
    cutoff: str
    curve_x: np.ndarray
    curve_y: np.ndarray
    scada_curve_x: np.ndarray
    scada_curve_y: np.ndarray
    model: LGBMRegressor

    def curve(self, wind: pd.Series | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(wind, dtype=float), self.curve_x, self.curve_y,
                         left=self.curve_y[0], right=self.curve_y[-1])

    def scada_curve(self, wind: pd.Series | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(wind, dtype=float), self.scada_curve_x, self.scada_curve_y,
                         left=self.scada_curve_y[0], right=self.scada_curve_y[-1])


def fit_curve(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Passport curve was not supplied: derive a transparent wind-only baseline.
    bins = np.arange(0, 26.5, 0.5)
    grouped = train.groupby(pd.cut(train.wind_ms, bins=bins, labels=False, include_lowest=True), observed=True).power.median()
    x = bins[:-1] + 0.25
    y = np.full(len(x), np.nan)
    y[grouped.index.astype(int)] = grouped.values
    y = pd.Series(y).interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).mean().to_numpy()
    return x, np.clip(y, 0, 1)


def prepare_features(df: pd.DataFrame, bundle: ModelBundle, historical: pd.DataFrame | None = None,
                     as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    times = pd.to_datetime(out.valid_time, utc=True)
    local = times + pd.Timedelta(hours=CSV_UTC_OFFSET_HOURS)
    out["power_curve"] = bundle.curve(out.wind_ms)
    out["scada_curve"] = bundle.scada_curve(out.wind_ms)
    out["wind_sq"] = out.wind_ms ** 2
    out["wind_cube"] = out.wind_ms ** 3
    out["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * local.dt.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * local.dt.dayofyear / 365.25)
    if "lead_hours" not in out:
        out["lead_hours"] = 30.0  # training placeholder; constant carries no target signal
    if historical is not None:
        h = historical.set_index("valid_time").power
        lag_time = times - pd.Timedelta(days=7)
        lag = h.reindex(pd.DatetimeIndex(lag_time)).to_numpy()
        if as_of is not None:
            lag = np.where(lag_time <= pd.Timestamp(as_of), lag, np.nan)
        elif "as_of" in out:
            lag = np.where(lag_time <= pd.to_datetime(out.as_of, utc=True), lag, np.nan)
        out["lag_7d"] = lag
        issue_times = (pd.Series(pd.Timestamp(as_of), index=out.index) if as_of is not None
                       else pd.to_datetime(out.as_of, utc=True))
        recents = {}
        for issue in issue_times.unique():
            issue = pd.Timestamp(issue)
            prior = historical[(historical.valid_time < issue) &
                               (historical.valid_time >= issue - pd.Timedelta(hours=24))]
            recents[issue] = (float(prior.power.mean()) if prior.power.notna().sum() >= 18 else np.nan,
                              float(prior.wind_ms.mean()) if prior.wind_ms.notna().sum() >= 18 else np.nan)
        out["recent_power_24h"] = [recents[pd.Timestamp(t)][0] for t in issue_times]
        out["recent_wind_24h"] = [recents[pd.Timestamp(t)][1] for t in issue_times]
    elif "lag_7d" not in out:
        out["lag_7d"] = np.nan
        out["recent_power_24h"] = np.nan
        out["recent_wind_24h"] = np.nan
    return out


def train_model(site: str, hourly: pd.DataFrame, cutoff: str, tag: str,
                weather_training: pd.DataFrame) -> ModelBundle:
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train = weather_training[weather_training.valid_time < cutoff_ts].dropna(
        subset=["wind_ms", "power", "temperature_c"]).copy()
    if train.empty:
        raise ValueError(f"No training rows for {site}")
    x, y = fit_curve(train)
    full_history = hourly[(hourly.valid_time < cutoff_ts) & ~hourly.is_outage].dropna(subset=["wind_ms", "power"])
    hx, hy = fit_curve(full_history)
    model = LGBMRegressor(n_estimators=220, learning_rate=0.05, num_leaves=20,
                          max_depth=6, min_child_samples=40, verbosity=-1,
                          random_state=42, n_jobs=4)
    bundle = ModelBundle(site, cutoff, x, y, hx, hy, model)
    features = prepare_features(train, bundle, historical=hourly)
    # Missing lag examples teach the model the no-new-SCADA production path.
    features.loc[features.index % 3 == 0, "lag_7d"] = np.nan
    features.loc[features.index % 4 == 0, ["recent_power_24h", "recent_wind_24h"]] = np.nan
    model.fit(features[FEATURES], train.power)
    joblib.dump(bundle, ARTIFACTS / f"model_{site}_{tag}.joblib")
    return bundle


def run_prediction(features: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    out = features[["valid_time", "run_time", "lead_hours", "wind_ms", "temperature_c"]].copy()
    out["baseline"] = np.clip(features.power_curve.to_numpy(), 0, 1)
    out["prediction"] = np.clip(bundle.model.predict(features[FEATURES]), 0, 1)
    return out
