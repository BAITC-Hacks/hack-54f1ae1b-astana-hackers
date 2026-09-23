"""Claude function calling over the exact AgentTools used by ForecastFSM."""
from __future__ import annotations
import os
import re
import anthropic
import pandas as pd
from .tools import load_tools

TOOL_SCHEMAS = [
    {"name":"get_day_report", "description":"Retrieve a dated forecast and available actual error for one turbine.",
     "input_schema":{"type":"object","properties":{"date":{"type":"string","description":"YYYY-MM-DD"},
                                               "site":{"type":"string","enum":["turbine_1","turbine_2"]}},
                     "required":["date","site"]}},
    {"name":"get_overall_metrics", "description":"Retrieve measured validation metrics, or state that February actuals are unavailable.",
     "input_schema":{"type":"object","properties":{}}},
    {"name":"get_agent_log", "description":"Retrieve actual FSM decisions and analysis logs for a date and turbine.",
     "input_schema":{"type":"object","properties":{"date":{"type":"string", "description":"Issue date; for an error on target day D use D+1, when actuals became known."},
                                               "site":{"type":"string","enum":["turbine_1","turbine_2"]}},
                     "required":["date"]}},
]


def _dispatch(tools, name: str, args: dict):
    if name == "get_day_report":
        return tools.get_day_report(args["date"], args["site"])
    if name == "get_overall_metrics":
        return tools.get_overall_metrics()
    if name == "get_agent_log":
        return tools.get_agent_log(args["date"], args.get("site"))
    raise ValueError(name)


def _fallback(question: str, tools) -> dict:
    date = re.search(r"\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4}", question)
    site = "turbine_2" if re.search(r"(?:турбин[а-я]*|turbine)[ _-]*2", question, re.I) else "turbine_1"
    if date:
        day = date.group()
        if "." in day:
            day = "-".join(reversed(day.split(".")))
        mode = "production" if day >= "2026-02-01" else "validation"
        if mode != tools.mode:
            tools = load_tools(mode)
        report = tools.get_day_report(day, site)
        log_day = str((pd.Timestamp(day) + pd.Timedelta(days=1)).date()) if re.search(r"почему|ошиб", question, re.I) else day
        logs = tools.get_agent_log(log_day, site)
        summary = [r["output"].get("summary") for r in logs if r["tool"] == "analysis_agent"]
        if report.get("error"):
            spoken = f"{site}, {day}: {report['error']}."
        else:
            spoken = (f"{site}, {day}: средний прогноз {report['forecast_mean']:.3f} "
                      f"(нормализованная мощность), выпуск погоды {report['weather_run']}, "
                      f"рабочая модель {report['chosen_model']}.")
            metric = report.get("metrics")
            spoken += (f" По {metric['n']} известным часам MAE={metric['ml_mae']:.4f}, "
                       f"RMSE={metric['ml_rmse']:.4f}." if metric else " Фактическая мощность отсутствует.")
        spoken += f" Анализ агента: {summary[-1] if summary else 'лог за день отсутствует'}."
        return {"answer": spoken,
                "tool_calls": ["get_day_report", "get_agent_log"]}
    metric = tools.get_overall_metrics()
    if not metric.get("actual_available", True):
        spoken = "Фактическая выработка за февраль не предоставлена, поэтому точность за февраль неизвестна."
    elif "turbine_1" in metric:
        spoken = "Январская проверка: " + "; ".join(
            f"{site}: MAE LightGBM {m['ml_mae']:.4f}, кривая {m['baseline_mae']:.4f}"
            for site, m in metric.items()) + "."
    else:
        spoken = str(metric)
    return {"answer": spoken, "tool_calls": ["get_overall_metrics"]}


def answer(question: str, mode: str = "production") -> dict:
    if re.search(r"феврал|2026-02", question, re.I):
        mode = "production"
    elif re.search(r"январ|2026-01", question, re.I):
        mode = "validation"
    tools = load_tools(mode)
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _fallback(question, tools)
    client = anthropic.Anthropic()
    messages = [{"role":"user", "content":question}]
    called = []
    system = ("Ты помощник оператора ВЭС. Для каждого факта о прогнозе, ошибке или причине "
              "обязательно вызови инструмент. Не выдумывай февральский факт: он не дан. "
              "Для вопроса об ошибке целевого дня D смотри анализ в логе следующего дня D+1, когда стал известен факт. "
              "Если дата в январе, используй режим валидации; если в феврале — production. "
              "Отвечай по-русски кратко. Нормализованная мощность от 0 до 1.")
    for _ in range(5):
        response = client.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
                                          max_tokens=500, system=system, tools=TOOL_SCHEMAS, messages=messages)
        uses = [block for block in response.content if block.type == "tool_use"]
        if not uses:
            return {"answer":" ".join(b.text for b in response.content if b.type == "text"), "tool_calls":called}
        messages.append({"role":"assistant", "content":response.content})
        results = []
        for use in uses:
            args = use.input
            selected_tools = load_tools("validation" if args.get("date", "2026-02-01") < "2026-02-01" else mode)
            value = _dispatch(selected_tools, use.name, args)
            called.append(use.name)
            results.append({"type":"tool_result", "tool_use_id":use.id, "content":str(value)})
        messages.append({"role":"user", "content":results})
    return {"answer":"Достигнут предел вызовов инструментов.", "tool_calls":called}
