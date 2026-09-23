"""Callable tools shared by the FSM, API, and chat assistant."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ARTIFACTS, SITES
from .data import load_scada
from .model import ModelBundle, prepare_features, run_prediction as predict_model
from .weather_agent import fetch_single_run_forecast, assert_no_leakage


def _jsonable(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class AgentTools:
    def __init__(self, mode: str, bundles: dict[str, ModelBundle], data: dict[str, pd.DataFrame],
                 champions: dict[str, str] | None = None):
        self.mode, self.bundles, self.data = mode, bundles, data
        self.champions = champions or {site: "ml" for site in bundles}
        self.log_path = ARTIFACTS / f"agent_log_{mode}.jsonl"

    def log(self, site: str, as_of: datetime, tool: str, received: dict, decision: str, result: dict):
        record = {"at_utc": datetime.now(timezone.utc).isoformat(), "mode": self.mode,
                  "site": site, "as_of": as_of.isoformat(), "tool": tool,
                  "input": received, "decision": decision, "output": result}
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=_jsonable, ensure_ascii=False) + "\n")

    def get_weather_forecast(self, site: str, as_of: datetime, horizon_hours: int = 48) -> pd.DataFrame:
        try:
            df = fetch_single_run_forecast(SITES[site], as_of, horizon_hours=horizon_hours)
            assert_no_leakage(df, as_of)
            if len(df) != horizon_hours:
                raise ValueError(f"Expected {horizon_hours} hourly rows, got {len(df)}")
            self.log(site, as_of, "get_weather_forecast", {"horizon_hours": horizon_hours},
                     "use_latest_published_single_run", {"rows": len(df), "run_time": df.run_time.iloc[0]})
            return df
        except Exception as exc:
            self.log(site, as_of, "get_weather_forecast", {"horizon_hours": horizon_hours},
                     "error", {"error": repr(exc)})
            raise

    def prepare_features(self, site: str, weather_df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
        df = weather_df.rename(columns={"wind_speed_100m": "wind_ms", "temperature_2m": "temperature_c"})
        out = prepare_features(df, self.bundles[site], self.data[site], pd.Timestamp(as_of))
        self.log(site, as_of, "prepare_features", {"weather_rows": len(df)},
                 "calendar_curve_and_available_lags", {"rows": len(out), "lag_available": int(out.lag_7d.notna().sum())})
        return out

    def run_prediction(self, site: str, features_df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
        out = predict_model(features_df, self.bundles[site])
        champion = self.champions[site]
        out["operational_prediction"] = out["baseline"] if champion == "baseline" else out["prediction"]
        out["chosen_model"] = champion
        out["site"] = site
        out["as_of"] = pd.Timestamp(as_of)
        self.log(site, as_of, "run_prediction", {"feature_rows": len(features_df)},
                 "predict_ml_and_wind_only_baseline", {"rows": len(out),
                     "mean_prediction": float(out.operational_prediction.mean()), "chosen_model": champion})
        return out

    def get_actual_generation(self, site: str, date_from: str, date_to: str, as_of: datetime | None = None) -> pd.DataFrame:
        start, end = pd.Timestamp(date_from, tz="UTC"), pd.Timestamp(date_to, tz="UTC")
        out = self.data[site]
        out = out[(out.valid_time >= start) & (out.valid_time < end)][["valid_time", "power", "wind_ms"]].dropna(subset=["power"])
        out = out.rename(columns={"wind_ms": "observed_wind_ms"})
        if as_of is not None:
            out = out[out.valid_time < pd.Timestamp(as_of)]
            self.log(site, as_of, "get_actual_generation", {"date_from": date_from, "date_to": date_to},
                     "return_only_known_actuals", {"rows": len(out)})
        return out

    def compute_metrics(self, site: str, pred_df: pd.DataFrame, actual_df: pd.DataFrame, as_of: datetime) -> dict:
        joined = pred_df.merge(actual_df, on="valid_time", how="inner").dropna(subset=["power"])
        metrics = {"n": len(joined)}
        if len(joined):
            for col, label in [("prediction", "ml"), ("baseline", "baseline")]:
                error = joined[col] - joined.power
                metrics[f"{label}_mae"] = float(error.abs().mean())
                metrics[f"{label}_rmse"] = float(np.sqrt((error ** 2).mean()))
                nonzero = joined.power >= 0.05
                metrics[f"{label}_mape_pct"] = float((error[nonzero].abs() / joined.power[nonzero]).mean() * 100) if nonzero.any() else None
            metrics["mape_eligible_n"] = int((joined.power >= 0.05).sum())
            if "wind_ms" in joined and "observed_wind_ms" in joined:
                wind_error = joined.wind_ms - joined.observed_wind_ms
                metrics["wind_mae_ms"] = float(wind_error.abs().mean())
                metrics["wind_bias_ms"] = float(wind_error.mean())
            metrics["high_forecast_near_zero_actual_hours"] = int(((joined.prediction >= 0.4) & (joined.power < 0.05)).sum())
        self.log(site, as_of, "compute_metrics", {"pred_rows": len(pred_df), "actual_rows": len(actual_df)},
                 "compare_only_matching_known_hours", metrics)
        return metrics

    def decide_recompute(self, site: str, as_of: datetime, metrics: dict | None,
                         new_data_flag: bool, weather_delta_ms: float = 0.0) -> dict:
        if self.mode == "validation":
            required = bool(metrics and metrics.get("n", 0) >= 6 and metrics.get("ml_mae", 0) > 0.12 and new_data_flag)
            reason = f"last_known_mae={metrics.get('ml_mae'):.4f}>0.12" if required else "validation_error_below_threshold_or_no_new_actual"
        else:
            required = bool(new_data_flag and weather_delta_ms >= 0.5)
            reason = f"overlap_wind_change={weather_delta_ms:.3f}m/s>=0.5" if required else "weather_change_below_threshold_or_no_overlap"
        decision = {"recompute": required, "reason": reason, "weather_delta_ms": weather_delta_ms}
        self.log(site, as_of, "decide_recompute", {"metrics": metrics, "new_data_flag": new_data_flag,
                 "weather_delta_ms": weather_delta_ms}, reason, decision)
        return decision

    def get_agent_log(self, date: str, site: str | None = None) -> list[dict]:
        if not self.log_path.exists():
            return []
        rows = [json.loads(line) for line in self.log_path.read_text().splitlines() if line]
        return [r for r in rows if r["as_of"].startswith(date) and (site is None or r["site"] == site)]

    def get_day_report(self, date: str, site: str) -> dict:
        path = ARTIFACTS / f"selected_{self.mode}.csv"
        if not path.exists():
            return {"error": "Run backtest first"}
        pred = pd.read_csv(path, parse_dates=["valid_time", "run_time", "as_of"])
        pred["valid_time"] = pd.to_datetime(pred.valid_time, utc=True)
        day = pd.Timestamp(date, tz="UTC")
        daily = pred[(pred.site == site) & (pred.valid_time >= day) &
                     (pred.valid_time < day + pd.Timedelta(days=1))]
        if daily.empty:
            return {"date": date, "site": site, "hours": 0, "error": "No forecast for this date"}
        actual = self.get_actual_generation(site, date, str((day + pd.Timedelta(days=1)).date()))
        metrics = self.compute_metrics(site, daily, actual, datetime.now(timezone.utc)) if len(actual) else None
        return {"date": date, "site": site, "hours": len(daily),
                "forecast_mean": float(daily.operational_prediction.mean() if "operational_prediction" in daily else daily.prediction.mean()),
                "chosen_model": str(daily.chosen_model.iloc[0]) if "chosen_model" in daily else "ml",
                "weather_run": str(daily.run_time.max()), "metrics": metrics,
                "actual_available": bool(len(actual))}

    def get_overall_metrics(self) -> dict:
        if self.mode == "production":
            return {"mode": "production", "actual_available": False,
                    "message": "February actual generation was not supplied; February accuracy is unknown."}
        path = ARTIFACTS / "metrics_validation.json"
        return json.loads(path.read_text()) if path.exists() else {"error": "Run validation first"}


def load_tools(mode: str = "production") -> AgentTools:
    import joblib
    tag = "validation" if mode == "validation" else "production"
    bundles = {site: joblib.load(ARTIFACTS / f"model_{site}_{tag}.joblib") for site in SITES}
    data = {site: load_scada(site, write_report=False)[0] for site in SITES}
    selection = ARTIFACTS / "model_selection.json"
    champions = json.loads(selection.read_text()) if mode == "production" and selection.exists() else None
    return AgentTools(mode, bundles, data, champions)
