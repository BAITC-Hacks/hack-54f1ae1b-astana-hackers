"""Grounded numeric summaries; optional LLM wording never changes metrics."""
from __future__ import annotations
import os
import anthropic
from openai import OpenAI
from .llm import provider, openai_model


def summarize(site: str, as_of: str, metrics: dict | None, decision: dict) -> str:
    if not metrics or not metrics.get("n"):
        return f"{site}: на {as_of} проверяемого факта пока нет; {decision['reason']}."
    facts = (f"{site}, {as_of}: n={metrics['n']}, ML MAE={metrics['ml_mae']:.4f}, "
             f"RMSE={metrics['ml_rmse']:.4f}, MAPE={metrics['ml_mape_pct']}, "
             f"baseline MAE={metrics['baseline_mae']:.4f}, "
             f"wind MAE={metrics.get('wind_mae_ms')}, wind bias={metrics.get('wind_bias_ms')}, "
             f"hours with forecast>=0.4 and actual<0.05={metrics.get('high_forecast_near_zero_actual_hours')}; "
             f"решение={decision['reason']}.")
    if metrics.get("wind_mae_ms", 0) >= 2:
        facts += " Прогноз ветра заметно расходился с измеренным ветром; это возможный вклад в ошибку мощности."
    elif metrics.get("high_forecast_near_zero_actual_hours", 0) > 0:
        facts += " Есть часы почти нулевой мощности при высоком прогнозе; причина остановки или ограничения не подтверждена."
    else:
        facts += " По доступным измерениям точную причину ошибки установить нельзя."
    selected = provider()
    if selected == "none":
        return facts
    try:
        instructions = ("Ты аналитик ВЭС. Опиши только переданные числовые факты, "
                        "без новых чисел или причин, которых нет в данных. Один-два предложения по-русски.")
        if selected == "openai":
            response = OpenAI().responses.create(model=openai_model(), instructions=instructions,
                                                 input=facts, max_output_tokens=180, store=False)
            return response.output_text or facts
        response = anthropic.Anthropic().messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"), max_tokens=180,
            system=instructions, messages=[{"role":"user", "content":facts}])
        return " ".join(block.text for block in response.content if block.type == "text")
    except Exception as exc:
        return facts + f" [LLM unavailable: {type(exc).__name__}]"
