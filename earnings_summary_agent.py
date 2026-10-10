# -*- coding: utf-8 -*-
"""
法說會摘要 Agent（範圍：最近 N 天，預設 5 天）。

做法與 finance_agent.FinanceNewsAgent 的「財經新聞摘要」一致，並且直接沿用它的方法，
所以不需要、也沒有修改 finance_agent.py：

1. 摘要      每場法說整理成 2~4 個重點（extract_points：數字、財測、展望優先）
2. 畫重點    挑出要用紅字／螢光標示的關鍵詞（extract_highlights）
3. 分析      利多／利空／中性／混合、產業、為什麼值得注意、後續追蹤
4. 總覽      「法說會 AI 重點」：最重要的幾場、利多／利空、產業熱度、待追蹤、風險

兩層設計：
- 規則層（永遠可用）：從 Fugle 法說會 Agent 已產生的結構化欄位＋備忘錄正文抽重點。
- LLM 層（Ollama Qwen3 可用時）：重要度前 N 場重寫摘要；結果存在 earnings_summary_cache.json，
  下次更新不重做。

用法（run_finance_info_update.py 已改成這樣呼叫）：

    agent = EarningsSummaryAgent(BASE)
    earnings, digest = agent.analyze_earnings(items, days=5)
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime
from typing import Any

from finance_agent import (
    IMPORTANT_WORDS,
    NUMBER_PATTERN,
    SECTOR_KEYWORDS,
    TAIPEI,
    FinanceNewsAgent,
    _clean,
    _dedupe_keep_order,
    _parse_dt,
)

LIST_FIELDS = (
    "financial_highlights",
    "operating_highlights",
    "guidance",
    "positive_factors",
    "negative_factors",
    "key_risks",
    "qa_highlights",
)
SENTIMENTS = ("利多", "利空", "中性", "混合")


def _flat(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "。".join(str(x).strip().rstrip("。") for x in value if str(x).strip())
    return str(value or "").strip()


def _code(value: Any) -> str:
    m = re.search(r"[1-9]\d{3}[A-Z]?", str(value or ""))
    return m.group(0) if m else ""


def _row_key(row: dict[str, Any]) -> str:
    return str(
        row.get("url")
        or row.get("source_url")
        or f"{row.get('symbol', '')}|{row.get('published_date', '')}|{row.get('title', '')}"
    )


class EarningsSummaryAgent(FinanceNewsAgent):
    """沿用 FinanceNewsAgent 的摘要／畫重點／LLM 連線，只是把輸入換成法說會。"""

    EARNINGS_SYSTEM = (
        "你是台股法說會研究助理。只能根據提供的法說會內容整理，不可編造任何數字、公司或事件，也不給買賣建議。"
        "輸出 JSON，欄位：points（2~4 條繁體中文重點，每條 70 字內，優先寫營收／獲利／毛利率等財務數字、"
        "訂單與產能、展望與財測、風險；保留關鍵數字）、"
        "sentiment（利多／利空／中性／混合，指法說內容對該公司基本面的訊號）、"
        "sectors（最多 3 個，只能從這份清單挑：" + "、".join(SECTOR_KEYWORDS) + "）、"
        "why（一句話說明這場法說最值得注意的地方，40 字內）、"
        "highlights（3~8 個關鍵詞，必須逐字出現在 points 裡，例如數字、公司、事件）、"
        "watch（0~2 條後續要追蹤的事項，每條 30 字內）。"
    )
    DIGEST_SYSTEM = (
        "你是台股法說會研究主編。只能根據提供的條列整理，不可編造，也不給買賣建議。輸出 JSON："
        "headline（60 字內總結這幾天法說會的共同焦點）、key_points（4~6 條最重要的事件重點，每條 60 字內）、"
        "watch_items（3~5 條接下來要追蹤的事項）、risks（最多 3 條風險）。"
    )

    def __init__(self, base_dir: str = ".", **kwargs: Any):
        super().__init__(base_dir, **kwargs)
        # 與新聞摘要分開存，避免互相覆蓋
        self.cache_path = self.out_dir / "earnings_summary_cache.json"
        self.llm_max_articles = int(os.getenv("EARNINGS_LLM_MAX", "30"))
        self.llm_time_budget = int(os.getenv("EARNINGS_LLM_TIME_BUDGET_SEC", "480"))

    # ---------------- 讀取欄位 ----------------
    @staticmethod
    def _texts(row: dict[str, Any]) -> tuple[str, str]:
        """回傳 (結構化重點文字, 備忘錄正文)。結構化欄位是法說會 Agent 已整理過的，訊號最乾淨。"""
        parts = [_flat(row.get("one_line_summary") or row.get("summary"))]
        parts += [_flat(row.get(k)) for k in LIST_FIELDS]
        structured = "。".join(p for p in parts if p)
        structured = re.sub(r"。{2,}", "。", structured)
        memo = _clean(row.get("memo_text") or row.get("content") or "")
        return structured, memo

    @staticmethod
    def _company(row: dict[str, Any]) -> str:
        return str(row.get("name") or row.get("company") or row.get("company_name") or row.get("symbol") or "").strip()

    # ---------------- 規則層 ----------------
    def _importance(self, row: dict[str, Any], text: str, now: datetime) -> float:
        score = 1.0
        if row.get("in_watchlist"):
            score += 2.0
        score += min(2.0, len(NUMBER_PATTERN.findall(text)) * 0.3)
        score += min(2.0, sum(0.4 for w in IMPORTANT_WORDS if w in text))
        if row.get("guidance"):
            score += 1.0
        dt = _parse_dt(row.get("published_date") or row.get("event_date") or row.get("date"))
        if dt:
            age_days = max(0.0, (now - dt).total_seconds() / 86400)
            score += 1.5 if age_days <= 1 else 1.0 if age_days <= 2 else 0.5 if age_days <= 3 else 0.0
        if str(row.get("impact") or "") in ("利多", "利空"):
            score += 0.5
        try:
            score += min(1.0, float(row.get("confidence") or 0) / 100)
        except (TypeError, ValueError):
            pass
        return round(min(10.0, score), 2)

    def _rule_row(self, row: dict[str, Any], now: datetime) -> dict[str, Any]:
        structured, memo = self._texts(row)
        title = str(row.get("title") or "")
        body = structured if len(structured) >= 60 else f"{structured}。{memo[:3000]}"
        fake = {"title": title, "content": body, "summary": structured}
        points = self.extract_points(fake, max_points=4, max_chars=100)
        text = f"{title} {self._company(row)} {body}"
        sentiment = str(row.get("impact") or row.get("judgement") or "").strip()
        if sentiment not in SENTIMENTS:
            sentiment, _ = self.detect_sentiment(text)
        sectors = self.detect_sectors(f"{row.get('industry_name', '')} {text}")
        stocks = _dedupe_keep_order([c for c in (_code(row.get("symbol") or row.get("code")),) if c])
        guidance = _flat(row.get("guidance"))
        return {
            "points": points,
            "sentiment": sentiment,
            "sectors": sectors,
            "stocks": stocks,
            "why": guidance[:60] + ("…" if len(guidance) > 60 else ""),
            "watch": [],
            "highlights": self.extract_highlights(points, title, stocks, sectors),
            "importance": self._importance(row, f"{structured} {memo[:1500]}", now),
            "ai_source": "rule",
        }

    # ---------------- LLM 層 ----------------
    def _llm_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        structured, memo = self._texts(row)
        body = (structured + "\n" + memo)[:2600]
        user = (
            f"公司：{self._company(row)}（{row.get('symbol', '')}）\n標題：{row.get('title', '')}\n"
            f"日期：{row.get('published_date') or row.get('event_date') or ''}\n內容：{body}"
        )
        obj = self._chat_json(self.EARNINGS_SYSTEM, user)
        if not obj:
            return None
        points = [str(x).strip() for x in (obj.get("points") or []) if str(x).strip()][:4]
        if not points:
            return None
        sentiment = str(obj.get("sentiment") or "").strip()
        blob = " ".join(points)
        return {
            "points": points,
            "sentiment": sentiment if sentiment in SENTIMENTS else "",
            "sectors": self._canon_sectors(obj.get("sectors") or [], f"{row.get('title', '')} {body}"),
            "why": str(obj.get("why") or "").strip()[:80],
            "highlights": [str(h).strip() for h in (obj.get("highlights") or []) if str(h).strip() and str(h).strip() in blob],
            "watch": [str(x).strip()[:40] for x in (obj.get("watch") or []) if str(x).strip()][:2],
        }

    @staticmethod
    def _fingerprint_row(row: dict[str, Any]) -> str:
        raw = f"{row.get('title', '')}|{row.get('content_hash', '')}|{len(str(row.get('memo_text') or ''))}|{len(_flat(row.get('one_line_summary')))}"
        return hashlib.md5(raw.encode("utf-8", "ignore")).hexdigest()[:12]

    # ---------------- 主流程 ----------------
    def analyze_earnings(  # type: ignore[override]
        self,
        items: list[dict[str, Any]],
        days: int = 5,
        now: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        now = now or datetime.now(TAIPEI)
        t0 = time.time()
        rows = [dict(x) for x in items if isinstance(x, dict)]
        for row in rows:
            row["_rule"] = self._rule_row(row, now)

        cache = self._load_cache()
        llm_on = self._check_llm()
        llm_done = llm_cached = llm_fail = 0
        budget_hit = False
        ranked = sorted(rows, key=lambda r: -r["_rule"]["importance"])
        for idx, row in enumerate(ranked):
            key, fp = _row_key(row), self._fingerprint_row(row)
            hit = cache.get(key)
            if hit and hit.get("fp") == fp and hit.get("ai"):
                row["_llm"] = hit["ai"]
                llm_cached += 1
                continue
            if not llm_on or idx >= self.llm_max_articles:
                continue
            if time.time() - t0 > self.llm_time_budget:
                budget_hit = True
                continue
            try:
                ai = self._llm_row(row)
            except Exception as exc:  # noqa: BLE001
                ai = None
                if llm_fail == 0:
                    print(f"[earnings-agent] Qwen3 呼叫失敗（之後同類錯誤不再重複顯示）：{type(exc).__name__}: {exc}", flush=True)
            if ai:
                row["_llm"] = ai
                cache[key] = {"fp": fp, "ai": ai, "ts": now.isoformat()}
                llm_done += 1
            else:
                llm_fail += 1

        # 合併：LLM 優先、規則補洞；法說會 Agent 已給的 impact 視為權威，不被覆蓋
        for row in rows:
            r = row.pop("_rule")
            l = row.pop("_llm", None)
            use = dict(r)
            if l:
                use.update({k: v for k, v in l.items() if v not in (None, "", [])})
                use["ai_source"] = "qwen3"
                if not l.get("highlights"):
                    use["highlights"] = self.extract_highlights(use["points"], row.get("title", ""), use.get("stocks", []), use.get("sectors", []))
                if str(row.get("impact") or "") in SENTIMENTS:
                    use["sentiment"] = str(row["impact"])
                elif not l.get("sentiment"):
                    use["sentiment"] = r["sentiment"]
            row["ai_points"] = use["points"]
            row["ai_summary"] = "；".join(use["points"])
            row["sentiment"] = use["sentiment"]
            row["sectors"] = use["sectors"]
            row["stocks"] = use["stocks"]
            row["why"] = use.get("why", "")
            row["watch"] = use.get("watch", [])
            row["highlights"] = use["highlights"]
            row["importance"] = r["importance"]
            row["ai_source"] = use["ai_source"]

        # 快取只留目前視窗內的資料（取代舊資料）
        live = {_row_key(r) for r in rows}
        try:
            self._save_cache({k: v for k, v in cache.items() if k in live})
        except Exception:  # noqa: BLE001
            pass

        rows.sort(key=lambda r: -r["importance"])
        digest = self._earnings_digest(rows, days, now)
        digest["llm"] = {
            "enabled": llm_on,
            "note": self.llm_note,
            "model": self.model if llm_on else "",
            "summarized_now": llm_done,
            "from_cache": llm_cached,
            "failed": llm_fail,
            "time_budget_hit": budget_hit,
        }
        self.stats["earnings"] = {"total": len(rows), "llm_summarized": llm_done, "llm_cached": llm_cached,
                                  "llm_failed": llm_fail, "seconds": round(time.time() - t0, 1)}
        return rows, digest

    # ---------------- 總覽 ----------------
    def _earnings_digest(self, rows: list[dict[str, Any]], days: int, now: datetime) -> dict[str, Any]:
        total = len(rows)
        counts = {s: 0 for s in SENTIMENTS}
        sector_stat: dict[str, dict[str, int]] = {}
        for r in rows:
            counts[r.get("sentiment", "中性")] = counts.get(r.get("sentiment", "中性"), 0) + 1
            for sec in r.get("sectors") or []:
                st = sector_stat.setdefault(sec, {"count": 0, "利多": 0, "利空": 0})
                st["count"] += 1
                if r.get("sentiment") in ("利多", "利空"):
                    st[r["sentiment"]] += 1
        sector_heat = sorted(
            (
                {"sector": k, **v, "tone": "偏多" if v["利多"] > v["利空"] else "偏空" if v["利空"] > v["利多"] else "中性"}
                for k, v in sector_stat.items()
            ),
            key=lambda x: -x["count"],
        )[:10]

        def label(r: dict[str, Any]) -> str:
            name = self._company(r)
            sym = str(r.get("symbol") or "")
            return f"{name}（{sym}）" if sym and sym not in name else name or str(r.get("title", ""))

        key_points = []
        for r in rows[:6]:
            p = (r.get("ai_points") or [""])[0]
            if p:
                key_points.append({
                    "title": label(r), "point": p, "sentiment": r.get("sentiment", "中性"),
                    "url": r.get("source_url") or r.get("url", ""), "highlights": (r.get("highlights") or [])[:6],
                })

        def theme(sent: str) -> list[dict[str, Any]]:
            out = []
            for r in rows:
                if r.get("sentiment") == sent and len(out) < 4:
                    out.append({"theme": label(r), "reason": r.get("why") or (r.get("ai_points") or [""])[0],
                                "sectors": r.get("sectors", []), "url": r.get("source_url") or r.get("url", "")})
            return out

        watch_items = _dedupe_keep_order([w for r in rows for w in (r.get("watch") or [])])[:5]
        risks = _dedupe_keep_order([_flat(r.get("key_risks"))[:60] for r in rows if _flat(r.get("key_risks"))])[:3]
        top_sec = "、".join(x["sector"] for x in sector_heat[:3]) or "—"
        headline = (
            f"最近 {days} 天共 {total} 場法說會備忘錄：利多 {counts['利多']}、利空 {counts['利空']}、"
            f"中性 {counts['中性']}、混合 {counts['混合']}；集中產業：{top_sec}。"
            if total else f"最近 {days} 天沒有可用的法說會備忘錄。"
        )
        digest: dict[str, Any] = {
            "generated_at": now.isoformat(timespec="seconds"),
            "days": days,
            "total": total,
            **counts,
            "stats": {"total": total, **counts},
            "headline": headline,
            "key_points": key_points,
            "bullish": theme("利多"),
            "bearish": theme("利空"),
            "sector_heat": sector_heat,
            "watch_items": watch_items,
            "risks": risks,
            "generated_by": "rule",
        }
        if self.llm_ready and rows:
            try:
                lines = [
                    f"- [{r.get('sentiment', '中性')}]{label(r)} {str(r.get('published_date') or '')[:10]}｜{'；'.join((r.get('ai_points') or [])[:2])}"
                    for r in rows[:20]
                ]
                obj = self._chat_json(self.DIGEST_SYSTEM, "\n".join(lines))
                if obj and obj.get("key_points"):
                    digest["agent_headline"] = str(obj.get("headline") or "").strip()
                    digest["agent_key_points"] = [str(x).strip() for x in obj["key_points"] if str(x).strip()][:6]
                    wi = [str(x).strip() for x in obj.get("watch_items", []) if str(x).strip()][:5]
                    rk = [str(x).strip() for x in obj.get("risks", []) if str(x).strip()][:3]
                    digest["watch_items"] = wi or watch_items
                    digest["risks"] = rk or risks
                    digest["generated_by"] = "qwen3"
            except Exception:  # noqa: BLE001
                pass
        return digest
