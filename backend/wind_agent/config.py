from pathlib import Path
from .weather_agent import TurbineSite

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

SITES = {
    "turbine_1": TurbineSite("turbine_1", 43.645150, 78.535604, 100),
    "turbine_2": TurbineSite("turbine_2", 43.643198, 78.538828, 100),
}
CSV_FILES = {name: RAW / f"{name}.csv" for name in SITES}
CSV_UTC_OFFSET_HOURS = 5  # confirmed by the data owner for the entire export
VALIDATION_START = "2026-01-14"
VALIDATION_END = "2026-01-31"
PRODUCTION_START = "2026-02-01"
PRODUCTION_END = "2026-02-28"
