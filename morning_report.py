# -*- coding: utf-8 -*-
"""早報資料組裝：最近 5 天法說會、近 2 天財經新聞摘要、AI 看好的產業與個股。

設計成「只讀已經存在的東西」，不重新抓 TWSE/TPEx/Fugle：

- 法說會、題材、財報這些都是盤後 15:00 那條 pipeline 已經算好、寫成
  output/research_reports/research_YYYY-MM-DD.json 的東西（EmailAgent.load_latest_report()
  讀的也是同一批檔案）。早報只是把最近幾天的檔案重新讀回來、合併、換一種角度呈現。
- 財經新聞摘要交給 cnyes_news_agent.CnyesNewsAgent.build_digest()（讀它自己的快取，
  不重新爬網頁；爬網頁是 23:00 --refresh 排程的事）。
- 「看好的產業與個股」是本機 Qwen3 8B 讀「最近幾天已經算好的題材強度／營收 YoY／法說會
  正面因素」這些結構化數字做的統整，不是憑空生成；Ollama 不可用時改用純資料排序的版本
  （不寫因為-所以的敘述句，只列數字），兩種情況都清楚標示，不會假裝是同一等級的判斷。

⚠️ 我沒有這個專案 research_agent.py 的原始碼可以核對「report_date/檔名格式/欄位名稱」
是否跟這裡假設的完全一致，這裡沿用 email_agent.py 已經在用的假設（檔名含
YYYY-MM-DD、欄位名稱與 EmailAgent 讀的相同）。接上之前請先跑
`python test_morning_report.py` 對照你實際存的 JSON 檔案。
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from cnyes_news_agent import CnyesNewsAgent

TAIPEI_TZ = timezone(timedelta(hours=8))


class MorningReportAgent:
    def __init__(
        self,
        base_dir: str | Path = ".",
        earnings_lookback_days: int = 5,
        outlook_lookback_days: int = 3,
        news_days: int = 2,
        ollama_model: str = "qwen3:8b",
        ollama_host: str = "http://127.0.0.1:11434",
    ):
        self.base = Path(base_dir).resolve()
        self.output_dir = self.base / "output" / "research_reports"
        self.earnings_lookback_days = max(1, earnings_lookback_days)
        self.outlook_lookback_days = max(1, outlook_lookback_days)
        self.news_days = max(1, news_days)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip("/")
        self.news_agent = CnyesNewsAgent(base_dir=base_dir, ollama_model=ollama_model, ollama_host=ollama_host)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            x = float(value)
        except Exception:
            return default
        return x if math.isfinite(x) else default

    # --------------------- 讀最近幾天已經存好的盤後研究 JSON ---------------------
    def _recent_reports(self, days: int) -> list[dict[str, Any]]:
        """依檔名裡的日期挑出最近 N 天的 research_*.json，按日期由舊到新排序。
        單一檔案壞掉（JSON 損毀等）只跳過那一份，不影響其他天。"""
        if not self.output_dir.exists():
            return []
        cutoff = (datetime.now(TAIPEI_TZ) - timedelta(days=days)).date()
        dated: list[tuple[Any, dict[str, Any]]] = []
        for p in sorted(self.output_dir.glob("research_*.json")):
            m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
            if not m:
                continue
            try:
                d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < cutoff:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                dated.append((d, data))
        dated.sort(key=lambda pair: pair[0])
        return [d for _, d in dated]

    def _recent_earnings_calls(self, reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for r in reports:
            for c in r.get("earnings_calls", []) or []:
                if isinstance(c, dict) and c.get("symbol"):
                    merged[(str(c.get("symbol", "")), str(c.get("event_date", "")))] = c
        return list(merged.values())

    # ------------------------------ 看好產業與個股 ------------------------------
    def _outlook_evidence(self, reports: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
        themes: dict[str, dict[str, Any]] = {}
        fin: dict[str, dict[str, Any]] = {}
        calls: dict[tuple[str, str], dict[str, Any]] = {}
        name_map: dict[str, str] = {}
        for r in reports:
            for t in r.get("bullish_themes", []) or []:
                if isinstance(t, dict) and t.get("theme"):
                    prev = themes.get(t["theme"])
                    if not prev or self._safe_float(t.get("strength")) >= self._safe_float(prev.get("strength")):
                        themes[t["theme"]] = t
                    for c in t.get("affected_companies") or []:
                        if isinstance(c, dict) and c.get("symbol") and c.get("name"):
                            name_map[str(c["symbol"])] = str(c["name"])
            for f in r.get("financial_snapshots", []) or []:
                if isinstance(f, dict) and f.get("symbol"):
                    fin[str(f["symbol"])] = f
            for c in r.get("earnings_calls", []) or []:
                if isinstance(c, dict) and c.get("symbol"):
                    calls[(str(c["symbol"]), str(c.get("event_date", "")))] = c
                    if c.get("name"):
                        name_map[str(c["symbol"])] = str(c["name"])

        for f in fin.values():
            sym = str(f.get("symbol", ""))
            if sym in name_map:
                f.setdefault("name", name_map[sym])
        for c in calls.values():
            sym = str(c.get("symbol", ""))
            if sym in name_map:
                c.setdefault("name", name_map[sym])

        top_themes = sorted(themes.values(), key=lambda t: self._safe_float(t.get("strength")), reverse=True)[:6]
        top_fin = sorted(fin.values(), key=lambda f: self._safe_float(f.get("revenue_yoy")), reverse=True)[:10]
        bullish_calls = [c for c in calls.values() if "利多" in str(c.get("impact", "")) and c.get("memo_opened")]
        bullish_calls.sort(key=lambda c: self._safe_float(c.get("confidence")), reverse=True)
        return top_themes, top_fin, bullish_calls[:8]

    def _fallback_outlook(self, themes: list[dict], fin: list[dict], calls: list[dict]) -> dict[str, Any]:
        """Ollama 不可用或回傳格式不對時的退路：只依現有數字排序，不生成因果敘述句，
        避免在沒有模型檢查的情況下編出聽起來很像分析、其實只是排序的句子。"""
        industries = [
            {"name": t.get("theme", ""), "reason": f"題材強度 {self._safe_float(t.get('strength')):.1f}/100｜狀態：{t.get('status','')}"}
            for t in themes[:5]
        ]
        stocks = [
            {
                "symbol": str(f.get("symbol", "")),
                "name": str(f.get("name", "")) or str(f.get("symbol", "")),
                "reason": f"最新月營收 YoY {self._safe_float(f.get('revenue_yoy')):+.1f}%｜EPS {f.get('eps', '—')}",
            }
            for f in fin[:5]
        ]
        return {
            "available": True, "model": None, "fallback": True,
            "industries": industries, "stocks": stocks,
            "overall_take": "本機 Qwen3 8B 暫時無法使用，以下改用資料直接排序（題材強度／營收 YoY），未經 AI 綜合判斷。",
        }

    def _outlook_prompt(self, themes: list[dict], fin: list[dict], calls: list[dict]) -> str:
        lines: list[str] = []
        for i, t in enumerate(themes):
            lines.append(
                f"[題材{i}] {t.get('theme','')}｜強度 {self._safe_float(t.get('strength')):.1f}/100｜狀態 {t.get('status','')}"
                f"｜官方證據 {t.get('official_evidence_count',0)}｜接單/出貨證據 {t.get('order_evidence_count',0)}"
                f"｜營收支撐 {t.get('revenue_support_count',0)}"
            )
        for i, f in enumerate(fin):
            lines.append(
                f"[財報{i}] {f.get('symbol','')} {f.get('name','')}｜最新月營收 YoY "
                f"{self._safe_float(f.get('revenue_yoy')):+.1f}%｜EPS {f.get('eps','—')}"
            )
        for i, c in enumerate(calls):
            factors = "、".join((c.get("positive_factors") or [])[:3])
            lines.append(
                f"[法說{i}] {c.get('symbol','')} {c.get('name','')}｜信心 {self._safe_float(c.get('confidence')):.0f}%"
                f"｜利多因素：{factors or '—'}"
            )
        joined = "\n".join(lines)
        return f"""你是台股研究團隊的產業分析師。以下是最近幾天盤後研究流程已經算好的結構化資料
（題材強度、月營收年增率、法說會正面因素），不是新聞，也還沒有經過你之外的任何人判斷。

請只根據「以下資料」統整：
1. 3~5 個目前值得關注的產業／題材類別，每個附一句話原因，原因必須引用資料裡的具體數字
   （強度分數、YoY、信心度…），不要自己編數字或編資料以外的事件。
2. 3~6 檔目前值得關注的個股，附代號、名稱（資料裡有的話）與一句話原因，同樣只能引用資料。
3. 這只是研究線索整理、不是投資建議，用詞要保守（「值得關注」而非「建議買進」「一定會漲」）。

只輸出下面這個 JSON，不要其他文字、不要用 ```：
{{"industries": [{{"name": "string", "reason": "string"}}],
"stocks": [{{"symbol": "string", "name": "string", "reason": "string"}}],
"overall_take": "string"}}

資料：
{joined}
"""

    def _run_outlook(self, themes: list[dict], fin: list[dict], calls: list[dict]) -> dict[str, Any]:
        if not (themes or fin or calls):
            return {
                "available": False, "model": self.ollama_model,
                "error": f"最近 {self.outlook_lookback_days} 天沒有足夠的題材／財報／法說會資料可供統整。",
                "industries": [], "stocks": [], "overall_take": "",
            }
        prompt = self._outlook_prompt(themes, fin, calls)
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False,
                      "format": "json", "options": {"temperature": 0.2}},
                timeout=90,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        except Exception as exc:
            fb = self._fallback_outlook(themes, fin, calls)
            fb["error"] = f"呼叫本機 Ollama 失敗，改用資料排序版本：{exc}"
            return fb

        parsed = CnyesNewsAgent._safe_json(raw)
        if not isinstance(parsed, dict):
            fb = self._fallback_outlook(themes, fin, calls)
            fb["validation"] = "failed"
            fb["error"] = "Qwen 回傳內容不是合法 JSON，改用資料排序版本。"
            return fb

        industries = [
            {"name": str(x.get("name", ""))[:40], "reason": str(x.get("reason", ""))[:150]}
            for x in (parsed.get("industries") or []) if isinstance(x, dict) and x.get("name")
        ][:6]
        stocks = [
            {"symbol": str(x.get("symbol", "")), "name": str(x.get("name", "")), "reason": str(x.get("reason", ""))[:150]}
            for x in (parsed.get("stocks") or []) if isinstance(x, dict) and x.get("symbol")
        ][:8]
        if not industries and not stocks:
            fb = self._fallback_outlook(themes, fin, calls)
            fb["validation"] = "empty"
            fb["error"] = "Qwen 回傳的 JSON 沒有有效項目，改用資料排序版本。"
            return fb
        return {
            "available": True, "model": self.ollama_model,
            "industries": industries, "stocks": stocks,
            "overall_take": str(parsed.get("overall_take", ""))[:200],
        }

    # ---------------------------------- 組報告 ----------------------------------
    def build(self) -> dict[str, Any]:
        now = datetime.now(TAIPEI_TZ)
        earnings_reports = self._recent_reports(self.earnings_lookback_days)
        outlook_reports = self._recent_reports(self.outlook_lookback_days)
        themes, fin, calls = self._outlook_evidence(outlook_reports)

        latest = earnings_reports[-1] if earnings_reports else {}
        return {
            "report_date": now.strftime("%Y-%m-%d"),
            "generated_at": now.isoformat(),
            "earnings_lookback_days": self.earnings_lookback_days,
            "earnings_calls": self._recent_earnings_calls(earnings_reports),
            "outlook_lookback_days": self.outlook_lookback_days,
            "outlook": self._run_outlook(themes, fin, calls),
            "cnyes_news": self.news_agent.build_digest(days=self.news_days),
            "research_scope": {"watchlist_symbols": (latest.get("research_scope", {}) or {}).get("watchlist_symbols", [])},
            "reports_found": len(earnings_reports),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="組出早報資料（不寄信，純輸出 JSON）")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--earnings-days", type=int, default=5)
    parser.add_argument("--outlook-days", type=int, default=3)
    parser.add_argument("--news-days", type=int, default=2)
    args = parser.parse_args()

    agent = MorningReportAgent(
        base_dir=args.base_dir, earnings_lookback_days=args.earnings_days,
        outlook_lookback_days=args.outlook_days, news_days=args.news_days,
    )
    print(json.dumps(agent.build(), ensure_ascii=False, indent=2, default=str))
