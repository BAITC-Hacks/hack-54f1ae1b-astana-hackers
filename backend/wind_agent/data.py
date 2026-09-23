"""SCADA preparation. The missing timestamp report precedes any filling."""
from __future__ import annotations
import json
from datetime import timezone, timedelta
import numpy as np
import pandas as pd
from .config import CSV_FILES, CSV_UTC_OFFSET_HOURS, ARTIFACTS

TIME = "Статистическое время"
WIND = "Средняя скорость ветра(m/s)"
POWER = "Нормализованная активная мощность"
TEMP = "Средняя температура окружающей среды(°C)"
LOCAL_TZ = timezone(timedelta(hours=CSV_UTC_OFFSET_HOURS))


def _missing_spans(index: pd.DatetimeIndex, observed: pd.DatetimeIndex) -> pd.DataFrame:
    missing = index.difference(observed)
    if len(missing) == 0:
        return pd.DataFrame(columns=["start_local", "end_local", "samples", "hours"])
    series = pd.Series(missing)
    groups = series.diff().ne(pd.Timedelta(minutes=10)).cumsum()
    result = series.groupby(groups).agg(["min", "max", "count"])
    result.columns = ["start_local", "end_local", "samples"]
    result["hours"] = result["samples"] / 6
    return result.reset_index(drop=True)


def load_scada(site: str, write_report: bool = True) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(CSV_FILES[site])
    raw[TIME] = pd.to_datetime(raw[TIME], errors="raise")
    if raw[TIME].duplicated().any():
        raise ValueError(f"Duplicate SCADA timestamps: {site}")
    raw = raw.set_index(TIME).sort_index()
    full = pd.date_range(raw.index.min(), raw.index.max(), freq="10min", name=TIME)
    spans = _missing_spans(full, raw.index)
    report = {
        "site": site,
        "rows": len(raw),
        "expected_rows": len(full),
        "missing_samples": int(len(full) - len(raw)),
        "missing_hours": round((len(full) - len(raw)) / 6, 2),
        "short_gaps_1h_or_less": int((spans.samples <= 6).sum()),
        "medium_gaps_1h_to_24h": int(((spans.samples > 6) & (spans.samples < 144)).sum()),
        "long_gaps_24h_or_more": int((spans.samples >= 144).sum()),
        "largest_gaps": [
            {"start_local": str(r.start_local), "end_local": str(r.end_local),
             "samples": int(r.samples), "hours": round(float(r.hours), 2)}
            for r in spans.sort_values("samples", ascending=False).head(12).itertuples()
        ],
    }
    if write_report:
        (ARTIFACTS / f"missing_{site}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    frame = raw[[WIND, POWER, TEMP]].reindex(full)
    # Causal fill: use past telemetry only, and only for complete gaps <= 1h.
    absent = frame[POWER].isna()
    gap_id = absent.ne(absent.shift()).cumsum()
    gap_size = absent.groupby(gap_id).transform("sum")
    short_gap = absent & (gap_size <= 6)
    forward = frame.ffill()
    frame.loc[short_gap, [WIND, POWER, TEMP]] = forward.loc[short_gap, [WIND, POWER, TEMP]]
    frame["original"] = ~absent
    frame["is_outage"] = absent & (gap_size >= 144)
    frame.index = frame.index.tz_localize(LOCAL_TZ).tz_convert("UTC")
    hourly = frame.resample("1h").agg({WIND:"mean", POWER:"mean", TEMP:"mean", "original":"sum", "is_outage":"max"})
    hourly = hourly.rename(columns={WIND:"wind_ms", POWER:"power", TEMP:"temperature_c", "original":"original_samples"})
    hourly.loc[hourly.original_samples < 4, ["wind_ms", "power", "temperature_c"]] = np.nan
    hourly["is_outage"] = hourly["is_outage"].astype(bool)
    hourly["valid_time"] = hourly.index
    return hourly.reset_index(drop=True), report


def load_all() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    data, reports = {}, {}
    for site in CSV_FILES:
        data[site], reports[site] = load_scada(site)
    return data, reports
