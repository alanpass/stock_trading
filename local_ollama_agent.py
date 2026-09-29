# -*- coding: utf-8 -*-
"""Local Ollama Agent for the Taiwan-stock dashboard.

No OpenAI API is used.  Ollama runs locally and Qwen3:8b can call a small set of
read-only tools that expose the dashboard's already-fetched market/model data.
The LLM is an orchestration/explanation layer only; it never overwrites numeric
model predictions or the deterministic final decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from business_master import profile_for_symbol, build_business_research_context
except Exception:
    profile_for_symbol = None

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, pd.DataFrame):
        return v.tail(5).to_dict("records")
    return v


def _safe(v, default=None):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_fugle_quote",
                "description": "取得目前股票的 Fugle 報價摘要。只能讀取，不下單。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_technical_snapshot",
                "description": "分析 5 分鐘 K、MA5/20/60、VWAP、ATR、趨勢與近期價格結構。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_entry_model_result",
                "description": "取得現有逢低進場 Random Forest 模型結果，不重新訓練。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_next_day_prediction",
                "description": "取得已保存的隔日股價預測結果，不重新訓練。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "research_model",
                "description": "查詢目前模型的樣本量、5-fold OOF 指標、快取與模型狀態。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "backtest_model",
                "description": "讀取最近保存的回測/驗證摘要；若沒有資料則明確回報。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_model_tuning",
                "description": "根據現有驗證指標提出調參候選與研究方向；不直接修改生產模型。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_business_profile",
                "description": "查詢公司主要業務與官方產業價值鏈子分類。優先讀取本機快取，必要時查詢TPEx個體公司產業鏈頁面。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_business_research_plan",
                "description": "依官方產業鏈/公司主要業務建立本檔股票的研究主題、搜尋關鍵字與查證重點。",
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string", "description": "股票代號"}},
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_latest_research_report",
                "description": "讀取最新盤後模型研究 Agent 報告，包含模型健康、特徵漂移、產業新聞查證與重訓建議。只讀，不修改模型。",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]


class LocalOllamaAgent:
    def __init__(self, context: dict[str, Any], log_dir: Path, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST):
        self.ctx = context
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.host = host
        self._ollama = None

    def _client(self):
        if self._ollama is None:
            try:
                import ollama
            except ImportError as exc:
                raise RuntimeError("尚未安裝 ollama Python 套件，請執行：python -m pip install ollama") from exc
            self._ollama = ollama.Client(host=self.host)
        return self._ollama

    def _quote(self, symbol: str):
        q = self.ctx.get("quote") or {}
        return {
            "symbol": symbol,
            "price": _safe(q.get("lastPrice", q.get("closePrice"))),
            "change": _safe(q.get("change")),
            "change_percent": _safe(q.get("changePercent")),
            "previous_close": _safe(q.get("previousClose")),
            "date": q.get("date"),
            "market_open": self.ctx.get("market_open"),
        }

    def _technical(self, symbol: str):
        d = self.ctx.get("intraday")
        if not isinstance(d, pd.DataFrame) or d.empty:
            return {"symbol": symbol, "error": "目前沒有可用的 5 分鐘 K"}
        x = d.copy().sort_values("date")
        close = pd.to_numeric(x["close"], errors="coerce")
        high = pd.to_numeric(x["high"], errors="coerce")
        low = pd.to_numeric(x["low"], errors="coerce")
        vol = pd.to_numeric(x.get("volume"), errors="coerce").fillna(0)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        typical = pd.to_numeric(x.get("average", close), errors="coerce").fillna(close)
        denom = float(vol.sum())
        vwap = float((typical * vol).sum() / denom) if denom > 0 else float(close.iloc[-1])
        prev = close.shift(1)
        tr = pd.concat([(high-low), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
        atr12 = tr.rolling(12).mean().iloc[-1]
        recent_high = close.tail(12).max()
        recent_low = close.tail(12).min()
        return {
            "symbol": symbol,
            "last_close": _safe(close.iloc[-1]),
            "ma5": _safe(ma5), "ma20": _safe(ma20), "ma60": _safe(ma60),
            "vwap": _safe(vwap), "atr12": _safe(atr12),
            "return_6bars": _safe(close.pct_change(6).iloc[-1]),
            "recent_high_12bars": _safe(recent_high), "recent_low_12bars": _safe(recent_low),
            "bars": int(len(x)), "latest_bar": str(x.iloc[-1]["date"]),
        }

    def _entry(self, symbol: str):
        result = self.ctx.get("entry_result")
        if not isinstance(result, dict):
            return {"symbol": symbol, "error": "目前尚無進場模型結果"}
        keys = [
            "action", "decision_reason", "reference_price", "best_entry_price",
            "entry_score", "confidence", "trend_score", "pattern_name", "trend_state",
            "opportunity_probability", "vwap", "overextended", "entry_lower", "entry_upper",
        ]
        return {"symbol": symbol, **{k: _jsonable(result.get(k)) for k in keys}}

    def _next_day(self, symbol: str):
        result = self.ctx.get("ai_result")
        if not isinstance(result, dict):
            return {"symbol": symbol, "error": "目前尚無已保存的隔日預測結果"}
        keys = ["current_price", "predicted_price", "predicted_return", "direction", "confidence", "probability_up", "probability_down", "prediction_base_date", "prediction_date", "train_samples", "train_stocks"]
        return {"symbol": symbol, **{k: _jsonable(result.get(k)) for k in keys}}

    def _business(self, symbol: str):
        if profile_for_symbol is None:
            return {"symbol": symbol, "error": "business_master.py 不可用"}
        try:
            path = Path(self.ctx.get("base_dir", ".")) / "data" / "business_master.csv"
            p = profile_for_symbol(symbol, path, force=False)
            return {
                "symbol": symbol,
                "industry": p.get("industry"),
                "industry_name": p.get("industry_name"),
                "business_group": p.get("business_group"),
                "primary_chain": p.get("primary_chain"),
                "all_chain_paths": p.get("all_chain_paths"),
                "source_type": p.get("source_type"),
                "source_url": p.get("source_url"),
            }
        except Exception as exc:
            return {"symbol": symbol, "error": str(exc)}

    def _business_research_plan(self, symbol: str):
        if profile_for_symbol is None:
            return {"symbol": symbol, "error": "business_master.py 不可用"}
        try:
            path = Path(self.ctx.get("base_dir", ".")) / "data" / "business_master.csv"
            p = profile_for_symbol(symbol, path, force=False)
            ctx = build_business_research_context(p)
            return {
                "symbol": symbol,
                "business_scope": ctx.get("business_scope_text"),
                "industry": ctx.get("industry_name"),
                "business_group": ctx.get("business_group"),
                "primary_chain": ctx.get("primary_chain"),
                "research_keywords": ctx.get("research_keywords"),
                "all_chain_paths": ctx.get("all_chain_paths"),
                "source_url": ctx.get("source_url"),
            }
        except Exception as exc:
            return {"symbol": symbol, "error": str(exc)}

    def _research(self, symbol: str):
        entry = self.ctx.get("entry_result") or {}
        ai = self.ctx.get("ai_result") or {}
        return {
            "symbol": symbol,
            "entry_model": {
                "train_samples": entry.get("train_samples"),
                "train_days": entry.get("train_days"),
                "oof_mae": entry.get("oof_mae"),
                "oof_direction_accuracy": entry.get("oof_direction_accuracy"),
                "oof_opportunity_accuracy": entry.get("oof_opportunity_accuracy"),
            },
            "next_day_model": {
                "train_samples": ai.get("train_samples"),
                "train_stocks": ai.get("train_stocks"),
                "metrics": ai.get("metrics"),
                "confidence": ai.get("confidence"),
            },
        }

    def _backtest(self, symbol: str):
        out = self.ctx.get("output_dir")
        if not out:
            return {"symbol": symbol, "error": "沒有提供 output 路徑"}
        root = Path(out)
        candidates = list(root.glob("**/*backtest*.csv")) + list(root.glob("**/*validation*.csv"))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return {"symbol": symbol, "error": "找不到已保存的回測/驗證 CSV"}
        p = candidates[0]
        try:
            df = pd.read_csv(p)
            return {"symbol": symbol, "file": str(p), "rows": int(len(df)), "columns": list(df.columns), "tail": df.tail(5).to_dict("records")}
        except Exception as exc:
            return {"symbol": symbol, "file": str(p), "error": str(exc)}

    def _tuning(self, symbol: str):
        research = self._research(symbol)
        ideas = []
        e = research.get("entry_model", {})
        acc = _safe(e.get("oof_opportunity_accuracy"))
        if acc is not None and acc < 0.70:
            ideas.append("優先檢驗進場標籤門檻與時間窗（目前 30 分鐘目標）")
        dir_acc = _safe(e.get("oof_direction_accuracy"))
        if dir_acc is not None and dir_acc < 0.80:
            ideas.append("增加 MA60/VWAP/市場狀態分層，並檢查趨勢型態在不同波動 regime 的穩定性")
        if not ideas:
            ideas.append("目前 OOF 指標沒有明顯需要立即調參的訊號，建議先做 walk-forward drift 分析")
        return {"symbol": symbol, "do_not_auto_apply": True, "candidate_changes": ideas}

    def _latest_research(self, symbol: str):
        root = Path(self.ctx.get("output_dir") or "output") / "research_reports"
        files = sorted(root.glob("research_*.json")) if root.exists() else []
        if not files:
            return {"symbol": symbol, "error": "尚無盤後模型研究報告"}
        try:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
            # 避免將過量新聞全文塞給 LLM；保留關鍵治理與該股票相關證據。
            related = []
            sym = str(symbol)
            for item in data.get("news", []):
                text = f"{item.get('title','')} {item.get('summary','')}"
                if sym in text:
                    related.append(item)
            return {
                "report_date": data.get("report_date"),
                "research_conclusion": data.get("research_conclusion"),
                "recommend_retrain": data.get("recommend_retrain"),
                "model_health_count": len(data.get("model_health", [])),
                "related_news": related[:10],
                "ollama_summary": (data.get("ollama_analysis") or {}).get("content"),
                "json_path": str(files[-1]),
            }
        except Exception as exc:
            return {"symbol": symbol, "error": str(exc)}

    def dispatch(self, name: str, args: dict[str, Any]):
        symbol = str(args.get("symbol") or self.ctx.get("symbol"))
        mapping: dict[str, Callable[[str], Any]] = {
            "get_fugle_quote": self._quote,
            "get_technical_snapshot": self._technical,
            "get_entry_model_result": self._entry,
            "get_next_day_prediction": self._next_day,
            "research_model": self._research,
            "backtest_model": self._backtest,
            "suggest_model_tuning": self._tuning,
            "get_business_profile": self._business,
            "get_business_research_plan": self._business_research_plan,
            "get_latest_research_report": self._latest_research,
        }
        fn = mapping.get(name)
        if fn is None:
            return {"error": f"unknown tool: {name}"}
        return fn(symbol)

    def run(self, user_prompt: str, max_rounds: int = 5) -> dict[str, Any]:
        client = self._client()
        symbol = str(self.ctx.get("symbol"))
        system = (
            "你是本機執行的台股研究型 AI Agent。完全禁止使用雲端 API；所有工具資料都來自本機 Dashboard 已取得的資料、快取或模型輸出。"
            "你可以呼叫工具蒐集證據，但不能改寫量化模型的數值結果。不要自行捏造價格。"
            "最終回答要區分：已驗證數值、工具觀察、研究建議。不可自行保證投資獲利。"
            "若需要重新訓練或調參，只能提出研究建議，不可自動覆蓋 production model。"
            "分析個股新聞前，優先呼叫 get_business_profile 與 get_business_research_plan；按照公司實際產品/服務、產業鏈位置與研究關鍵字查找訊息。不要把『產業利多』直接當成『公司利多』。"
            "若新聞與公司主要業務無直接關聯，降低其證據權重；若只有單一媒體描述而沒有官方公告/財報/營收/法人或同業佐證，標示待查證。"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"目前股票：{symbol}\n任務：{user_prompt}"},
        ]
        trace: list[dict[str, Any]] = []
        for _ in range(max_rounds):
            response = client.chat(model=self.model, messages=messages, tools=_tool_defs(), options={"temperature": 0.1})
            message = response.message
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                content = getattr(message, "content", "") or ""
                result = {"symbol": symbol, "content": content, "trace": trace, "model": self.model}
                self._log(result)
                return result
            messages.append(message)
            for call in tool_calls:
                fn = call.function.name
                args = call.function.arguments or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                output = self.dispatch(fn, args)
                trace.append({"tool": fn, "arguments": args, "output": output})
                messages.append({"role": "tool", "tool_name": fn, "content": json.dumps(output, ensure_ascii=False, default=_jsonable)})
        result = {"symbol": symbol, "content": "Agent 達到最大工具輪次，請查看工具追蹤。", "trace": trace, "model": self.model}
        self._log(result)
        return result

    def _log(self, result):
        try:
            p = self.log_dir / "ollama_agent_runs.jsonl"
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=_jsonable) + "\n")
        except Exception:
            pass


def ollama_available(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    try:
        import ollama
    except ImportError:
        return False, "尚未安裝 Python 套件 ollama"
    try:
        client = ollama.Client(host=host)
        models = client.list().models
        names = []
        for m in models:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(str(name))
        if not any(n == model or n.startswith(model + ":") for n in names):
            return False, f"Ollama 已連線，但找不到模型 {model}；請執行 ollama pull {model}"
        return True, "ok"
    except Exception as exc:
        return False, f"無法連線 Ollama：{exc}"
