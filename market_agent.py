# -*- coding: utf-8 -*-
"""AI Agent orchestration layer for the stock dashboard.

The agent layer is intentionally deterministic for the numeric decision. It orchestrates
specialist stages (data, technical, flow, risk, decision) and optionally uses the
OpenAI Agents SDK only for narrative explanation when OPENAI_API_KEY is configured.
The LLM is never allowed to overwrite the numeric prediction/score.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

TW_TZ = "Asia/Taipei"


def _safe_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _last_close(df):
    if df is None or df.empty or "close" not in df.columns:
        return np.nan
    return _safe_float(df.sort_values("date").iloc[-1]["close"])


def _atr(df, n=12):
    if df is None or df.empty:
        return np.nan
    x = df.copy().sort_values("date")
    prev = pd.to_numeric(x["close"], errors="coerce").shift(1)
    tr = pd.concat([
        pd.to_numeric(x["high"], errors="coerce") - pd.to_numeric(x["low"], errors="coerce"),
        (pd.to_numeric(x["high"], errors="coerce") - prev).abs(),
        (pd.to_numeric(x["low"], errors="coerce") - prev).abs(),
    ], axis=1).max(axis=1)
    return _safe_float(tr.rolling(n).mean().iloc[-1])


def _vwap(df):
    if df is None or df.empty:
        return np.nan
    x = df.copy().sort_values("date")
    vol = pd.to_numeric(x.get("volume"), errors="coerce").fillna(0)
    avg = pd.to_numeric(x.get("average", x["close"]), errors="coerce")
    close = pd.to_numeric(x["close"], errors="coerce")
    typical = avg.where(avg.notna(), close)
    den = float(vol.sum())
    if den <= 0:
        return _safe_float(close.iloc[-1])
    return _safe_float((typical * vol).sum() / den)


@dataclass
class AgentContext:
    symbol: str
    as_of: str
    market_open: bool
    quote: dict[str, Any]
    intraday: pd.DataFrame | None = None
    daily: pd.DataFrame | None = None
    entry_result: dict[str, Any] | None = None
    ai_result: dict[str, Any] | None = None
    institutions: pd.DataFrame | None = None
    tdcc: pd.DataFrame | None = None


@dataclass
class AgentFinding:
    agent: str
    status: str
    score: float
    confidence: float
    reasons: list[str]
    evidence: dict[str, Any]


class BaseAgent:
    name = "base"

    def run(self, ctx: AgentContext) -> AgentFinding:
        raise NotImplementedError


class DataQualityAgent(BaseAgent):
    name = "data_quality"

    def run(self, ctx):
        reasons = []
        evidence = {}
        score = 100.0
        if not ctx.quote:
            score -= 35
            reasons.append("即時報價資料不足")
        if ctx.intraday is None or ctx.intraday.empty:
            score -= 35
            reasons.append("盤中K線資料不足")
        else:
            last = pd.to_datetime(ctx.intraday["date"], errors="coerce").max()
            evidence["latest_intraday"] = str(last)
            if pd.isna(last):
                score -= 15
                reasons.append("K線時間無法確認")
        if ctx.daily is None or ctx.daily.empty:
            score -= 20
            reasons.append("日K資料不足")
        evidence["market_open"] = ctx.market_open
        status = "ok" if score >= 80 else "warning" if score >= 60 else "blocked"
        return AgentFinding(self.name, status, max(0, score), max(0, score) / 100.0, reasons or ["資料完整度正常"], evidence)


class TechnicalAgent(BaseAgent):
    name = "technical"

    def run(self, ctx):
        d = ctx.intraday
        if d is None or d.empty:
            return AgentFinding(self.name, "blocked", 0, 0, ["沒有可用的5分K"], {})
        x = d.copy().sort_values("date")
        close = pd.to_numeric(x["close"], errors="coerce")
        ema3 = close.ewm(span=3, adjust=False).mean().iloc[-1]
        ema6 = close.ewm(span=6, adjust=False).mean().iloc[-1]
        ema12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        last = _safe_float(close.iloc[-1])
        vwap = _vwap(x)
        atr = _atr(x, 12)
        ret6 = _safe_float(close.pct_change(6).iloc[-1])
        hh = _safe_float(close.tail(12).max())
        ll = _safe_float(close.tail(12).min())
        trend = 0.0
        trend += 30 if ema3 > ema6 else 0
        trend += 25 if ema6 > ema12 else 0
        trend += 20 if last > vwap else 0
        trend += 15 if ret6 > 0 else 0
        trend += 10 if last > close.tail(12).mean() else 0
        reasons = []
        if trend >= 70:
            reasons.append("短線均線結構偏多")
        elif trend >= 50:
            reasons.append("短線趨勢中性偏多")
        else:
            reasons.append("短線趨勢偏弱或震盪")
        if np.isfinite(vwap) and last < vwap:
            reasons.append("價格位於VWAP下方")
        if np.isfinite(atr) and atr > 0 and np.isfinite(last) and np.isfinite(hh) and last > hh - 0.25 * atr:
            reasons.append("價格接近日內高檔，不宜追價")
        evidence = {
            "ema3": _safe_float(ema3), "ema6": _safe_float(ema6), "ema12": _safe_float(ema12),
            "vwap": vwap, "atr12": atr, "return_6bars": ret6, "recent_high": hh, "recent_low": ll,
        }
        return AgentFinding(self.name, "ok", float(np.clip(trend, 0, 100)), float(np.clip(trend / 100, 0, 1)), reasons, evidence)


class FlowAgent(BaseAgent):
    name = "flow"

    def run(self, ctx):
        score = 50.0
        reasons = ["法人/籌碼資料作為輔助訊號"]
        evidence = {}
        if ctx.institutions is not None and not ctx.institutions.empty:
            last = ctx.institutions.iloc[-1]
            total = _safe_float(last.get("total_net"))
            foreign = _safe_float(last.get("foreign_net"))
            evidence["institution_total_net"] = total
            evidence["foreign_net"] = foreign
            if np.isfinite(total):
                score += 15 if total > 0 else -15 if total < 0 else 0
                reasons.append("最近法人合計偏買" if total > 0 else "最近法人合計偏賣" if total < 0 else "最近法人合計接近中性")
        if ctx.tdcc is not None and not ctx.tdcc.empty:
            row = ctx.tdcc.iloc[0]
            big = _safe_float(row.get("big_pct"))
            evidence["big_pct"] = big
            if np.isfinite(big) and big >= 50:
                score += 8
                reasons.append("大戶持股結構相對穩定")
        return AgentFinding(self.name, "ok", float(np.clip(score, 0, 100)), float(np.clip(score / 100, 0, 1)), reasons, evidence)


class RiskGuardAgent(BaseAgent):
    name = "risk_guard"

    def run(self, ctx):
        score = 100.0
        reasons = []
        evidence = {}
        if ctx.entry_result:
            er = ctx.entry_result
            if er.get("overextended"):
                score -= 35
                reasons.append("偵測到高檔延伸，禁止追高")
            if _safe_float(er.get("trend_score"), 0) < 55:
                score -= 25
                reasons.append("趨勢強度不足")
            if _safe_float(er.get("entry_score"), 0) < 55:
                score -= 20
                reasons.append("進場分數未達確認門檻")
        if ctx.ai_result:
            conf = _safe_float(ctx.ai_result.get("confidence"), 0.5)
            evidence["next_day_confidence"] = conf
            if conf < 0.55:
                score -= 10
                reasons.append("隔日模型可信度偏低")
        return AgentFinding(self.name, "ok" if score >= 60 else "warning", float(np.clip(score, 0, 100)), float(np.clip(score / 100, 0, 1)), reasons or ["未偵測到主要風險觸發器"], evidence)


class DecisionAgent(BaseAgent):
    name = "decision"

    def run(self, ctx):
        findings = self._findings
        tech = findings.get("technical")
        risk = findings.get("risk_guard")
        data = findings.get("data_quality")
        entry = ctx.entry_result or {}
        score = _safe_float(entry.get("entry_score"), 0)
        confidence = _safe_float(entry.get("confidence"), np.nan)
        if not np.isfinite(confidence):
            confidence = 0.5
        if tech:
            score = 0.55 * score + 0.45 * tech.score
        if risk:
            score = 0.85 * score + 0.15 * risk.score
        if data and data.status == "blocked":
            score = min(score, 30)
        if tech and tech.score >= 68 and score >= 68 and not entry.get("overextended", False):
            action = "適合回檔進場"
        elif tech and tech.score >= 55 and entry.get("overextended", False):
            action = "上漲趨勢，但不追高"
        elif tech and tech.score >= 55 and score >= 52:
            action = "等待回檔確認"
        else:
            action = "不建議進場"
        reasons = []
        if tech: reasons.extend(tech.reasons[:2])
        if risk: reasons.extend(risk.reasons[:2])
        evidence = {"composite_score": float(np.clip(score, 0, 100)), "action": action}
        return AgentFinding(self.name, "ok", float(np.clip(score, 0, 100)), confidence, reasons or ["依現有量化結果綜合判斷"], evidence)

    def attach_findings(self, findings):
        self._findings = findings


class MarketSupervisorAgent:
    """Deterministic supervisor + optional LLM explanation layer."""

    def __init__(self, log_dir: Path | None = None):
        self.agents = [DataQualityAgent(), TechnicalAgent(), FlowAgent(), RiskGuardAgent()]
        self.decision = DecisionAgent()
        self.log_dir = log_dir
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        findings = {}
        for agent in self.agents:
            findings[agent.name] = agent.run(ctx)
        self.decision.attach_findings(findings)
        findings[self.decision.name] = self.decision.run(ctx)
        final = findings["decision"]
        result = {
            "symbol": ctx.symbol,
            "as_of": ctx.as_of,
            "decision": final.evidence.get("action", "不建議進場"),
            "score": final.score,
            "confidence": final.confidence,
            "reasons": final.reasons,
            "findings": {k: asdict(v) for k, v in findings.items()},
        }
        self._write_log(result)
        return result

    def _write_log(self, result: dict[str, Any]):
        if not self.log_dir:
            return
        path = self.log_dir / "agent_decisions.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass


def llm_explain(result: dict[str, Any]) -> str | None:
    """Backward-compatible local explanation using Ollama/Qwen3; never uses OpenAI."""
    try:
        from local_ollama_agent import LocalOllamaAgent
        context = {
            "symbol": result.get("symbol"),
            "market_open": False,
            "quote": {},
            "intraday": pd.DataFrame(),
            "daily": pd.DataFrame(),
            "entry_result": result.get("entry_result"),
            "ai_result": result.get("ai_result"),
            "output_dir": str(Path(result.get("output_dir", "output"))),
        }
        agent = LocalOllamaAgent(context, Path(result.get("log_dir", "output/agent_logs")))
        out = agent.run("請用繁體中文摘要目前 Agent 的量化證據與風險，最多 5 句，不改寫任何數值。")
        return out.get("content") if isinstance(out, dict) else None
    except Exception:
        return None
