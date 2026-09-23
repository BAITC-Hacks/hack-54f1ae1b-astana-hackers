"""Explicit daily finite-state machine, including a real recompute loop."""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
from .config import SITES, ARTIFACTS
from .tools import AgentTools
from .analysis import summarize


class ForecastFSM:
    def __init__(self, tools: AgentTools):
        self.tools = tools

    def run(self, start_as_of: str, end_as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        predictions, revisions = [], []
        previous = {site: None for site in SITES}
        previous_weather = {site: None for site in SITES}
        self.tools.log_path.unlink(missing_ok=True)
        for day in pd.date_range(start_as_of, end_as_of, freq="D", tz="UTC"):
            as_of = day.to_pydatetime()
            for site in SITES:
                weather = self.tools.get_weather_forecast(site, as_of, 48)
                old = previous[site]
                old_weather = previous_weather[site]
                metrics = None
                if old is not None and self.tools.mode == "validation":
                    actual = self.tools.get_actual_generation(
                        site, str((day - pd.Timedelta(days=1)).date()), str(day.date()), as_of)
                    metrics = self.tools.compute_metrics(site, old, actual, as_of)
                delta = 0.0
                if old_weather is not None:
                    overlap = weather[["valid_time", "wind_speed_100m"]].merge(
                        old_weather[["valid_time", "wind_speed_100m"]], on="valid_time", suffixes=("_new", "_old"))
                    if len(overlap):
                        delta = float((overlap.wind_speed_100m_new - overlap.wind_speed_100m_old).abs().mean())
                decision = self.tools.decide_recompute(site, as_of, metrics, old is not None, delta)
                summary = summarize(site, as_of.isoformat(), metrics, decision)
                self.tools.log(site, as_of, "analysis_agent", {"metrics": metrics},
                               decision["reason"], {"summary": summary})
                if decision["recompute"] and old is not None:
                    self.tools.log(site, as_of, "recompute_trigger", {"prior_as_of": str(old.as_of.iloc[0])},
                                   "reenter_weather_feature_prediction_states", {"reason": decision["reason"]})
                    # Re-enter the tool cycle with the newly published weather run.
                    newer = self.tools.get_weather_forecast(site, as_of, 48)
                    revised_features = self.tools.prepare_features(site, newer, as_of)
                    revised = self.tools.run_prediction(site, revised_features, as_of)
                    shared = revised[revised.valid_time.isin(old.valid_time)].copy()
                    shared["revision_of"] = old.as_of.iloc[0]
                    shared["reason"] = decision["reason"]
                    revisions.append(shared)
                features = self.tools.prepare_features(site, weather, as_of)
                pred = self.tools.run_prediction(site, features, as_of)
                pred["analysis_summary"] = summary
                predictions.append(pred)
                previous[site], previous_weather[site] = pred, weather
        result = pd.concat(predictions, ignore_index=True)
        revised = pd.concat(revisions, ignore_index=True) if revisions else pd.DataFrame()
        result.to_csv(ARTIFACTS / f"predictions_{self.tools.mode}.csv", index=False)
        revised.to_csv(ARTIFACTS / f"revisions_{self.tools.mode}.csv", index=False)
        return result, revised
