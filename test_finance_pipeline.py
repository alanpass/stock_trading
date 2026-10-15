# -*- coding: utf-8 -*-
"""離線驗證：不連網路。

用假的鉅亨 API 驗證
  多分類滾動爬取 → 去重 → 規則摘要／畫重點 → finance_info_latest.json 取代舊資料
  → Claude 寫回 AI 今日重點與逐篇重點（finance_digest.py apply）→ 下一次規則更新仍保留 Claude 的內容
執行：python test_finance_pipeline.py
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TAIPEI = ZoneInfo("Asia/Taipei")
os.environ["CNYES_DETAIL_ENRICH_MAX"] = "0"
os.environ["FINANCE_MIN_NEWS"] = "0"

import cnyes_news_crawler as cc  # noqa: E402


# ---------------- 假的鉅亨 API ----------------
def _fake_rows(prefix: str, n: int, hours_back_start: int, step_min: int):
    now = datetime.now(TAIPEI)
    rows = []
    for i in range(n):
        t = now - timedelta(hours=hours_back_start, minutes=step_min * i)
        rows.append({
            "newsId": f"{prefix}{i}",
            "title": f"{prefix}新聞{i}：台積電（2330）營收年增{10 + i}%，AI 伺服器需求強勁" if i % 3 == 0 else f"{prefix}新聞{i}：美股下跌 聯準會升息疑慮",
            "summary": "摘要" * 5,
            "content": ("台積電今日公布營收，年增 20%，優於預期。AI 伺服器與先進封裝需求強勁。" * 6) if i % 3 == 0 else ("美股收盤大跌，費半重挫 3%，市場擔憂升息。" * 8),
            "publishAt": int(t.timestamp()),
            "categoryName": prefix,
            "stock": ["2330"] if i % 3 == 0 else [],
        })
    return rows


DATA = {
    "headline": _fake_rows("頭條", 35, 1, 50),    # 35 篇，橫跨約 1.2 天
    "tw_stock": _fake_rows("台股", 40, 2, 40),
    "wd_stock": _fake_rows("國際", 20, 3, 90),
    "future": _fake_rows("期貨", 5, 4, 120),
}


class FakeResp:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p


def fake_get(self, url, params=None, accept_json=False):
    cat = "headline" if url.endswith("/headline") else url.rsplit("/", 1)[-1]
    if cat not in DATA:
        raise RuntimeError("404 unknown category")
    rows = [r for r in DATA[cat]]
    page, limit = int(params["page"]), int(params["limit"])
    start, end = int(params["startAt"]), int(params["endAt"])
    rows = [r for r in rows if start <= r["publishAt"] <= end + 120]
    chunk = rows[(page - 1) * limit: page * limit]
    last = max(1, -(-len(rows) // limit))
    return FakeResp({"items": {"data": chunk, "total": len(rows), "last_page": last}})


cc.CnyesNewsCrawler._get = fake_get


def main() -> int:
    import run_finance_info_update as run
    import finance_digest as fd

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        reports = base / "output/research_reports"
        r1 = run.update_finance_info(base, watchlist=[])
        latest = json.loads((reports / "finance_info_latest.json").read_text(encoding="utf-8"))
        print("第 1 次更新：新聞", latest["news_count"], "各分類", latest["crawl"].get("per_category_count"))
        dg = latest["news_digest"]
        assert latest["news_count"] >= 60, "應該遠多於 2 篇"
        assert "forex" in latest["crawl"]["per_category_count"] and latest["crawl"]["per_category_count"]["forex"] == 0, "不存在的分類應被略過而不是中斷"
        first = latest["news"][0]
        assert first["ai_points"] and first["sentiment"] in {"利多", "利空", "中性", "混合"}
        assert all(h in " ".join(first["ai_points"]) for h in first["highlights"]), "highlights 必須出現在摘要裡"
        assert dg["generated_by"] == "rule" and first["ai_source"] == "rule"

        files = sorted(p.name for p in reports.iterdir())
        assert "cnyes_news_latest.json" in files and "finance_info_latest.json" in files
        assert not any(n.startswith("finance_info_2") for n in files), "不應再產生一堆時間戳快照"

        # Claude 寫回 AI 今日重點與第一則新聞的逐篇重點
        fd.BASE, fd.LATEST, fd.JOB_DIR = base, reports / "finance_info_latest.json", base / "output/claude_jobs/finance"
        assert fd.export() == 0 and (fd.JOB_DIR / "brief.md").exists()
        notes = {
            "news": {"headline": "AI 供應鏈營收強勁，但美股回檔壓抑情緒", "key_points": ["台積電營收年增優於預期", "費半重挫 3%"],
                     "watch_items": ["Fed 談話"], "risks": ["升息疑慮"]},
            "articles": [{"id": first["article_id"], "points": ["台積電營收年增 20% 優於預期", "AI 伺服器與先進封裝需求強勁"],
                          "sentiment": "利多", "sectors": ["半導體"], "stocks": ["2330"], "why": "AI 需求帶動營收",
                          "highlights": ["年增 20%", "優於預期", "不在文中的詞"]}],
        }
        notes_path = fd.JOB_DIR / "digest_notes.json"
        notes_path.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
        assert fd.apply(notes_path, publish=False) == 0
        applied = json.loads((reports / "finance_info_latest.json").read_text(encoding="utf-8"))
        a1 = next(x for x in applied["news"] if x["article_id"] == first["article_id"])
        assert applied["news_digest"]["generated_by"] == "claude"
        assert a1["ai_source"] == "claude" and "不在文中的詞" not in a1["highlights"], "highlights 必須出現在重點裡"

        # 第 2 次：規則層重算，Claude 的今日重點與逐篇重點要保留
        r2 = run.update_finance_info(base, watchlist=[])
        latest2 = json.loads((reports / "finance_info_latest.json").read_text(encoding="utf-8"))
        a2 = next(x for x in latest2["news"] if x["article_id"] == first["article_id"])
        print("第 2 次：Claude 重點", latest2["news_digest"].get("generated_by"), "逐篇沿用", latest2["agent_stats"]["ai_cached"])
        assert latest2["news_digest"]["generated_by"] == "claude"
        assert a2["ai_source"] == "claude" and latest2["agent_stats"]["ai_cached"] >= 1
        assert latest2["updated_at"] >= latest["updated_at"]

        # 第 3 次：爬蟲整個壞掉，要保留上一版而不是變空
        def boom(self, *a, **k): raise RuntimeError("network down")
        cc.CnyesNewsCrawler._get = boom
        for f in reports.glob("cnyes_news_*.json"):
            f.unlink()
        r3 = run.update_finance_info(base, watchlist=[])
        latest3 = json.loads((reports / "finance_info_latest.json").read_text(encoding="utf-8"))
        print("第 3 次（斷網）：新聞", latest3["news_count"], "stale =", latest3["stale"])
        assert latest3["news_count"] >= 60 and latest3["stale"] is True
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
