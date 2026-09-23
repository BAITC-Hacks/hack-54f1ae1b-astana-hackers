"""
weather_agent.py
=================

Weather Agent для агентного пайплайна прогнозирования выработки ВЭС.

КЛЮЧЕВАЯ ИДЕЯ
-------------
Для walk-forward бэктеста нужно на каждый день D получить погодный прогноз
для D+1..D+2 РОВНО ТАКИМ, каким он был известен на момент D — а не то, что
Open-Meteo "задним числом" знает сейчас о погоде на эти даты.

Наивный вызов Historical Forecast API (`historical-forecast-api.open-meteo.com`)
для этого НЕ подходит: этот эндпоинт склеивает в один непрерывный ряд самые
СВЕЖИЕ (короткие) лид-таймы каждого запуска модели — для часа H он отдаёт
данные из последнего запуска модели, который его покрывает. Это ближе к
nowcast-качеству и может быть заметно точнее того, что реально было известно
за 24-48ч вперёд. Это скрытая утечка данных.

Правильные источники — те, что хранят прогнозы, привязанные к конкретному
историческому моменту выпуска:

1. Single Runs API (https://single-runs-api.open-meteo.com/v1/forecast)
   Параметр &run=<ISO дата-время инициализации модели>. Возвращает полный
   горизонт прогноза именно ЭТОГО запуска. Для ECMWF IFS HRES 9km архив
   начинается с 2024-03-14 — февраль 2026 полностью покрыт.
   Это САМЫЙ строгий по анти-утечке источник: физически невозможно получить
   данные "из будущего" относительно run.

2. Previous Runs API (https://previous-runs-api.open-meteo.com/v1/forecast)
   Переменные вида `<var>_previous_day1`, `_previous_day2` и т.д. — значение,
   предсказанное за 24ч / 48ч до фактического времени. Хорошо для метрик
   деградации точности по лид-тайму и как второй источник для сверки.

В этом модуле реализован (1) как основной путь и обёртка над (2) для
дополнительной валидации.

ВАЖНЫЙ НЮАНС РЕАЛИЗМА
----------------------
Запуск, инициализированный в 00 UTC, физически появляется в архиве/операционно
не раньше чем через ~4-6 часов (для глобальных моделей). Если агент "встаёт"
в 00:00 UTC дня D и наивно запрашивает run=D 00:00, он получает несуществующие
на тот момент данные (Single Runs API либо кинет ошибку на будущий/недоступный
run, либо это будет "подглядыванием", если запрос идёт сейчас, а не в момент
D в реальном времени). Поэтому select_available_run() выбирает ПОСЛЕДНИЙ
запуск, который гарантированно был бы уже опубликован к моменту cutoff.
"""

from __future__ import annotations

import time
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests
import pandas as pd

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# ECMWF IFS HRES: 4 запуска в сутки, публикация начинается примерно через
# эти часы после времени инициализации (консервативная оценка, см.
# https://open-meteo.com/en/docs/model-updates для точных цифр по вашей модели)
ECMWF_RUN_HOURS = (0, 6, 12, 18)
ECMWF_PUBLISH_DELAY_HOURS = 6  # берём с запасом

CACHE_DIR = Path(os.getenv("WEATHER_CACHE_DIR", Path(__file__).resolve().parents[2] / "artifacts" / "weather_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TurbineSite:
    name: str
    latitude: float
    longitude: float
    hub_height_m: int = 100  # явное допущение: паспортная высота отсутствует


def select_available_run(as_of: datetime, run_hours: Iterable[int] = ECMWF_RUN_HOURS,
                          publish_delay_hours: int = ECMWF_PUBLISH_DELAY_HOURS) -> datetime:
    """
    Возвращает время инициализации последнего запуска модели, который
    гарантированно уже опубликован к моменту as_of (UTC, naive или aware).

    Это и есть защита от утечки на уровне выбора run: агент, "живущий" в
    моменте as_of, физически не мог видеть более поздних запусков.
    """
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    else:
        as_of = as_of.astimezone(timezone.utc)

    candidates = []
    for day_offset in (0, -1):
        day = (as_of + timedelta(days=day_offset)).date()
        for h in run_hours:
            run_dt = datetime(day.year, day.month, day.day, h, tzinfo=timezone.utc)
            published_at = run_dt + timedelta(hours=publish_delay_hours)
            if published_at <= as_of:
                candidates.append(run_dt)

    if not candidates:
        raise ValueError(f"Нет доступного запуска модели к моменту {as_of.isoformat()}")

    return max(candidates)


def _cache_key(**kwargs) -> str:
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def fetch_single_run_forecast(
    site: TurbineSite,
    as_of: datetime,
    horizon_hours: int = 48,
    variables: tuple[str, ...] = (
        "wind_speed_100m", "wind_direction_100m", "temperature_2m",
    ),
    model: str = "ecmwf_ifs",
    max_retries: int = 3,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Получает погодный прогноз на horizon_hours часов вперёд от as_of,
    используя ровно тот модельный запуск, который существовал бы на
    момент as_of в реальном времени (не позже него).

    Возвращает DataFrame с колонками: valid_time, <variables...>, run_time,
    lead_hours (сколько часов вперёд от run_time сделан прогноз на valid_time).
    lead_hours сохраняется явно — пригодится для проверки на утечку и для
    фичей ("насколько 'свежий' был прогноз").
    """
    run_time = select_available_run(as_of)
    as_of_ts = pd.Timestamp(as_of)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    hours_since_run = int((as_of_ts - pd.Timestamp(run_time)).total_seconds() / 3600)

    cache_key = _cache_key(
        site=site.name, lat=site.latitude, lon=site.longitude,
        run=run_time.isoformat(), horizon=horizon_hours,
        variables=variables, model=model,
    )
    cache_path = CACHE_DIR / f"single_run_{cache_key}.parquet"
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        assert_no_leakage(cached, as_of)
        return cached

    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "hourly": ",".join(variables),
        "models": model,
        # forecast_hours отсчитывается от run, а не от as_of.
        "forecast_hours": horizon_hours + hours_since_run + 1,
        "wind_speed_unit": "ms",
        "timeformat": "iso8601",
        "timezone": "UTC",
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(SINGLE_RUNS_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Single Runs API недоступен после {max_retries} попыток: {last_err}")

    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise RuntimeError(f"Пустой ответ Single Runs API для run={run_time}: {payload}")

    df = pd.DataFrame(hourly)
    df["valid_time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"])
    df["run_time"] = run_time
    df["lead_hours"] = (df["valid_time"] - run_time).dt.total_seconds() / 3600.0

    # оставляем только строго будущее относительно as_of, в пределах горизонта
    df = df[(df["valid_time"] > as_of_ts) &
            (df["valid_time"] <= as_of_ts + pd.Timedelta(hours=horizon_hours))].reset_index(drop=True)
    assert_no_leakage(df, as_of)

    if use_cache:
        df.to_parquet(cache_path)

    return df


def assert_no_leakage(df: pd.DataFrame, as_of: datetime) -> None:
    """
    Самопроверка перед тем, как фичи уйдут в модель: все valid_time должны
    быть строго позже as_of, а run_time — строго не позже as_of минус
    задержку публикации. Бросает AssertionError при малейшем подозрении
    на утечку. Вызывать в конце каждого шага Weather Agent в оркестраторе.
    """
    if df.empty:
        raise AssertionError("Пустой прогноз не допускается")
    as_of_ts = pd.Timestamp(as_of)
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    assert (df["valid_time"] > as_of_ts).all(), \
        "Найдены valid_time <= as_of — прогноз включает 'прошлое' относительно момента симуляции"
    run_time = df["run_time"].iloc[0]
    run_ts = pd.Timestamp(run_time, tz="UTC") if run_time.tzinfo is None else pd.Timestamp(run_time)
    assert run_ts + timedelta(hours=ECMWF_PUBLISH_DELAY_HOURS) <= as_of_ts, \
        "Запуск модели ещё не мог быть опубликован к моменту as_of"
    assert df["run_time"].nunique() == 1, "Смешаны разные запуски модели"
    assert (df["lead_hours"] >= 0).all(), "Отрицательный lead_hours"
    actual_lead = (df["valid_time"] - pd.Timestamp(run_ts)).dt.total_seconds() / 3600
    assert (actual_lead.sub(df["lead_hours"]).abs() < 1e-9).all(), "Некорректный lead_hours"


def fetch_previous_runs_forecast(
    site: TurbineSite,
    valid_date_start: str,
    valid_date_end: str,
    lead_days: tuple[int, ...] = (1, 2),
    variables: tuple[str, ...] = ("wind_speed_100m", "temperature_2m"),
) -> pd.DataFrame:
    """
    Альтернативный / валидационный путь через Previous Runs API:
    для диапазона фактических дат возвращает, что было спрогнозировано
    заранее на lead_days дней вперёд (temperature_2m_previous_day1 и т.п.).

    Полезно для (а) сверки с результатом Single Runs API, (б) быстрого
    расчёта агрегированных метрик деградации точности по лид-тайму, не
    перебирая по одному запуску на день.

    ВНИМАНИЕ: перед боевым использованием проверьте одним тестовым запросом,
    что параметры start_date/end_date действительно возвращают архивный
    (а не относительный "от сегодня") диапазон для previous-runs-api —
    задокументированный способ навигации по прошлому у этого эндпоинта
    в основном описан через past_days относительно текущей даты запроса,
    и это стоит явно перепроверить перед тем, как полагаться на него в
    бэктесте.
    """
    hourly_vars = [
        f"{v}_previous_day{d}" for v in variables for d in lead_days
    ]
    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": valid_date_start,
        "end_date": valid_date_end,
        "hourly": ",".join(hourly_vars),
        "timeformat": "iso8601",
        "timezone": "UTC",
        "models": "ecmwf_ifs",
        "wind_speed_unit": "ms",
    }
    resp = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=30)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    df = pd.DataFrame(hourly)
    df["valid_time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop(columns=["time"])


def walk_forward_weather(
    site: TurbineSite,
    day_start: str,
    day_end: str,
    horizon_hours: int = 48,
) -> pd.DataFrame:
    """
    Прогоняет Weather Agent по каждому дню бэктеста D в [day_start, day_end]
    (включительно), для каждого D запрашивая прогноз на D+1..D+2 ровно так,
    как он выглядел бы на момент 00:00 UTC дня D. Возвращает единую таблицу
    со всеми прогнозами, помеченными meta-колонкой as_of_date — это и есть
    "метка, на основании чего сделан прогноз" из README (п.4 пайплайна).
    """
    frames = []
    for d in pd.date_range(day_start, day_end, freq="D", tz="UTC"):
        as_of = d.to_pydatetime()
        df = fetch_single_run_forecast(site, as_of, horizon_hours=horizon_hours)
        assert_no_leakage(df, as_of)
        df["as_of_date"] = as_of.date()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # Пример использования на одной турбине и одном дне бэктеста
    site = TurbineSite(name="turbine_1", latitude=43.645150, longitude=78.535604, hub_height_m=100)
    as_of = datetime(2026, 2, 1, tzinfo=timezone.utc)

    df = fetch_single_run_forecast(site, as_of, horizon_hours=48)
    assert_no_leakage(df, as_of)
    print(df[["valid_time", "run_time", "lead_hours", "wind_speed_100m"]].head(10))
