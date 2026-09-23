"""Read-only dashboard API plus a tool-grounded chat endpoint."""
from __future__ import annotations
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .config import ARTIFACTS
from .tools import load_tools
from .chat import answer

app = FastAPI(title="Wind Agent")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/api/summary")
def summary():
    path = ARTIFACTS / "run_summary.json"
    return json.loads(path.read_text()) if path.exists() else {"status":"Run the backtest first"}


@app.get("/api/forecast")
def forecast(mode: str = "production", site: str = "turbine_1"):
    if mode not in ("production", "validation") or site not in ("turbine_1", "turbine_2"):
        raise HTTPException(400, "Unknown mode or site")
    path = ARTIFACTS / f"selected_{mode}.csv"
    if not path.exists():
        return []
    pred = pd.read_csv(path, parse_dates=["valid_time", "run_time", "as_of"])
    pred.valid_time = pd.to_datetime(pred.valid_time, utc=True)
    pred = pred[pred.site == site]
    if mode == "validation":
        actual = load_tools(mode).data[site][["valid_time", "power"]]
        pred = pred.merge(actual, on="valid_time", how="left")
    else:
        pred["power"] = None
    return json.loads(pred.to_json(orient="records", date_format="iso"))


@app.get("/api/logs")
def logs(date: str, mode: str = "production", site: str | None = None):
    if mode not in ("production", "validation"):
        raise HTTPException(400, "Unknown mode")
    path = ARTIFACTS / f"agent_log_{mode}.jsonl"
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r for r in records if r["as_of"].startswith(date) and (site is None or r["site"] == site)]


class ChatRequest(BaseModel):
    question: str
    mode: str = "production"


@app.post("/api/chat")
def chat(request: ChatRequest):
    if request.mode not in ("production", "validation"):
        raise HTTPException(400, "Unknown mode")
    return answer(request.question, request.mode)
