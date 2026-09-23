"""Train, validate on January, then issue February without using hidden actuals."""
from __future__ import annotations
import argparse
import json
import pandas as pd
from .config import ARTIFACTS, SITES, VALIDATION_START, VALIDATION_END, PRODUCTION_START, PRODUCTION_END
from .data import load_all
from .model import train_model
from .tools import AgentTools
from .orchestrator import ForecastFSM
from .weather_agent import fetch_previous_runs_forecast, fetch_single_run_forecast, assert_no_leakage


def selected_issuance(predictions: pd.DataFrame, target_start: str, target_end: str) -> pd.DataFrame:
    start = pd.Timestamp(target_start, tz="UTC")
    end = pd.Timestamp(target_end, tz="UTC") + pd.Timedelta(days=1)
    rows = predictions[(predictions.valid_time >= start) & (predictions.valid_time < end)].copy()
    rows = rows.sort_values("as_of").drop_duplicates(["site", "valid_time"], keep="last")
    return rows.sort_values(["site", "valid_time"])


def make_weather_training(data: dict, train_cutoff: str) -> dict:
    """Reconstruct available daily runs, never use a later weather revision."""
    result = {}
    last_day = pd.Timestamp(train_cutoff, tz="UTC") - pd.Timedelta(days=2)
    for site, location in SITES.items():
        frames = []
        for day in pd.date_range(pd.Timestamp("2025-12-01", tz="UTC"), last_day, freq="D"):
            as_of = day.to_pydatetime()
            weather = fetch_single_run_forecast(location, as_of, 48)
            assert_no_leakage(weather, as_of)
            weather = weather[weather.valid_time <= day + pd.Timedelta(hours=24)].copy()
            weather["as_of"] = day
            frames.append(weather)
        weather = pd.concat(frames, ignore_index=True)
        weather = weather.rename(columns={"wind_speed_100m":"wind_ms", "temperature_2m":"temperature_c"})
        actual = data[site][["valid_time", "power", "is_outage"]]
        aligned = weather.merge(actual, on="valid_time", how="inner")
        aligned = aligned[~aligned.is_outage].dropna(subset=["power"])
        aligned.to_csv(ARTIFACTS / f"training_alignment_{site}_{train_cutoff}.csv", index=False)
        result[site] = aligned
    return result


def run(mode: str, data: dict, train_cutoff: str, start: str, end: str, target_start: str, target_end: str):
    weather_training = make_weather_training(data, train_cutoff)
    bundles = {site: train_model(site, data[site], train_cutoff, mode, weather_training[site]) for site in SITES}
    selection = ARTIFACTS / "model_selection.json"
    champions = json.loads(selection.read_text()) if mode == "production" and selection.exists() else None
    tools = AgentTools(mode, bundles, data, champions)
    predictions, revisions = ForecastFSM(tools).run(start, end)
    chosen = selected_issuance(predictions, target_start, target_end)
    chosen.to_csv(ARTIFACTS / f"selected_{mode}.csv", index=False)
    metrics = {}
    if mode == "validation":
        for site in SITES:
            actual = tools.get_actual_generation(site, target_start, str((pd.Timestamp(target_end) + pd.Timedelta(days=1)).date()))
            metrics[site] = tools.compute_metrics(site, chosen[chosen.site == site], actual,
                                                  pd.Timestamp("2026-02-01", tz="UTC").to_pydatetime())
        (ARTIFACTS / "metrics_validation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    return {"mode": mode, "all_forecast_rows": len(predictions), "selected_rows": len(chosen),
            "revision_rows": len(revisions), "metrics": metrics}


def previous_runs_check(data: dict):
    result = {}
    for site, location in SITES.items():
        try:
            df = fetch_previous_runs_forecast(location, "2026-01-14", "2026-01-31")
            actual = data[site][["valid_time", "wind_ms"]].rename(columns={"wind_ms": "scada_wind_ms"})
            comparison = df.merge(actual, on="valid_time", how="left")
            single = fetch_single_run_forecast(location, pd.Timestamp("2026-01-19", tz="UTC").to_pydatetime(), 72)
            comparison = comparison.merge(single[["valid_time", "wind_speed_100m"]], on="valid_time", how="left")
            lead_proxy = {}
            for days in (1, 2):
                col = f"wind_speed_100m_previous_day{days}"
                lead_proxy[f"day{days}"] = {
                    "scada_comparable_hours": int(comparison[[col, "scada_wind_ms"]].dropna().shape[0]),
                    "wind_mae_vs_scada_ms": float((comparison[col] - comparison.scada_wind_ms).abs().mean()),
                    "single_run_comparable_hours": int(comparison[[col, "wind_speed_100m"]].dropna().shape[0]),
                    "mean_abs_difference_from_single_run_ms": float((comparison[col] - comparison.wind_speed_100m).abs().mean()),
                }
            result[site] = {"rows": len(df), "first_valid_time": str(df.valid_time.iloc[0]),
                            "historical_range_accepted": bool(len(df) == 432),
                            "lead_proxy": lead_proxy,
                            "comparison_note": "SCADA wind is not a 100m weather observation; Single Runs and Previous Runs have different issue times."}
        except Exception as exc:
            result[site] = {"error": repr(exc), "historical_range_accepted": False}
    (ARTIFACTS / "previous_runs_check.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "production", "all"], default="all")
    parser.add_argument("--previous-runs-check", action="store_true")
    args = parser.parse_args()
    data, reports = load_all()
    results = {"missing_reports": reports}
    if args.previous_runs_check:
        results["previous_runs_check"] = previous_runs_check(data)
    if args.mode in ("validation", "all"):
        results["validation"] = run("validation", data, "2026-01-13", "2026-01-13", "2026-01-30",
                                     VALIDATION_START, VALIDATION_END)
        # At the first February issuance (2026-01-31 00 UTC), Jan 31 actuals
        # are still in the future. Select the champion only from known hours.
        selected = pd.read_csv(ARTIFACTS / "selected_validation.csv", parse_dates=["valid_time", "as_of", "run_time"])
        selected.valid_time = pd.to_datetime(selected.valid_time, utc=True)
        known_cutoff = pd.Timestamp("2026-01-31", tz="UTC")
        selection_metrics = {}
        for site in SITES:
            pred = selected[(selected.site == site) & (selected.valid_time < known_cutoff)]
            actual = data[site][["valid_time", "power"]]
            joined = pred.merge(actual, on="valid_time", how="inner").dropna(subset=["power"])
            selection_metrics[site] = {"n": len(joined),
                "ml_mae": float((joined.prediction - joined.power).abs().mean()),
                "baseline_mae": float((joined.baseline - joined.power).abs().mean())}
        (ARTIFACTS / "model_selection_basis.json").write_text(json.dumps(selection_metrics, indent=2))
        champions = {site: "ml" if score["ml_mae"] < score["baseline_mae"] else "baseline"
                     for site, score in selection_metrics.items()}
        (ARTIFACTS / "model_selection.json").write_text(json.dumps(champions, indent=2))
        results["model_selection"] = champions
        results["model_selection_basis"] = selection_metrics
    if args.mode in ("production", "all"):
        results["production"] = run("production", data, "2026-01-31", "2026-01-31", "2026-02-27",
                                     PRODUCTION_START, PRODUCTION_END)
    (ARTIFACTS / "run_summary.json").write_text(json.dumps(results, default=str, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in results.items() if k != "missing_reports"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
