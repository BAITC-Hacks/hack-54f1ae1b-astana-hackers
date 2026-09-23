from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest
from wind_agent.data import load_scada
from wind_agent.weather_agent import select_available_run, assert_no_leakage


def test_run_selection_respects_publication_delay():
    as_of = datetime(2026, 1, 31, tzinfo=timezone.utc)
    assert select_available_run(as_of) == datetime(2026, 1, 30, 18, tzinfo=timezone.utc)


def test_leakage_assertion_rejects_future_run_and_past_target():
    as_of = datetime(2026, 1, 31, tzinfo=timezone.utc)
    valid = pd.DataFrame({"valid_time":[pd.Timestamp(as_of) + pd.Timedelta(hours=1)],
                          "run_time":[pd.Timestamp(as_of) - pd.Timedelta(hours=6)],
                          "lead_hours":[7]})
    assert_no_leakage(valid, as_of)
    with pytest.raises(AssertionError):
        assert_no_leakage(valid.assign(run_time=pd.Timestamp(as_of)), as_of)
    with pytest.raises(AssertionError):
        assert_no_leakage(valid.assign(valid_time=pd.Timestamp(as_of)), as_of)


def test_long_scada_gap_stays_unfilled():
    hourly, report = load_scada("turbine_1", write_report=False)
    assert report["missing_samples"] == 9992
    assert report["largest_gaps"][0]["samples"] == 5993
    # 2024-06-01 local 12:00 is inside the observed missing interval.
    utc = pd.Timestamp("2024-06-01 07:00", tz="UTC")
    row = hourly.loc[hourly.valid_time == utc].iloc[0]
    assert pd.isna(row.power)
    assert bool(row.is_outage)
