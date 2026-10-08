# -*- coding: utf-8 -*-
"""離線驗證：不連網路、不需要 Ollama。

用假的鉅亨 API 與假的 Ollama 伺服器，驗證
  多分類滾動爬取 → 去重 → Agent 摘要／畫重點 → finance_info_latest.json 取代舊資料
執行：python test_finance_pipeline.py
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
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


# ---------------- 假的 Ollama ----------------
class Ollama(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen3:8b"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        system = req["messages"][0]["content"]
        user = req["messages"][1]["content"]
        if "主編" in system:
            out = {"headline": "AI 供應鏈營收強勁，但美股回檔壓抑情緒", "key_points": ["台積電營收年增優於預期", "費半重挫 3%"], "watch_items": ["Fed 談話"], "risks": ["升息疑慮"]}
        else:
            out = {"points": ["台積電營收年增 20% 優於預期", "AI 伺服器與先進封裝需求強勁"], "sentiment": "利多", "sectors": ["半導體"], "stocks": ["2330"], "why": "AI 需求帶動營收", "highlights": ["年增 20%", "優於預期", "不在文中的詞"]}
        body = json.dumps({"message": {"content": json.dumps(out, ensure_ascii=False)}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["OLLAMA_HOST"] = f"http://127.0.0.1:{srv.server_port}"

    import importlib
    import finance_agent
    importlib.reload(finance_agent)
    import run_finance_info_update as run
    importlib.reload(run)

    ok = True
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        # 假的法說會模組：避免真的連 Fugle
        r1 = run.update_finance_info(base, watchlist=[])
        latest = json.loads((base / "output/research_reports/finance_info_latest.json").read_text(encoding="utf-8"))
        print("第 1 次更新：新聞", latest["news_count"], "各分類", latest["crawl"].get("per_category_count"))
        print("Agent：", latest["agent_stats"])
        dg = latest["news_digest"]
        print("今日重點：", dg.get("agent_headline") or dg.get("headline"))
        assert latest["news_count"] >= 60, "應該遠多於 2 篇"
        assert "forex" in latest["crawl"]["per_category_count"] and latest["crawl"]["per_category_count"]["forex"] == 0, "不存在的分類應被略過而不是中斷"
        first = latest["news"][0]
        assert first["ai_points"] and first["sentiment"] in {"利多", "利空", "中性", "混合"}
        assert all(h in " ".join(first["ai_points"]) for h in first["highlights"]), "highlights 必須出現在摘要裡"
        assert dg["generated_by"] == "qwen3"

        files = sorted(p.name for p in (base / "output/research_reports").iterdir())
        print("輸出檔：", files)
        assert "cnyes_news_latest.json" in files and "finance_info_latest.json" in files
        assert not any(n.startswith("finance_info_2") for n in files), "不應再產生一堆時間戳快照"

        # 第 2 次：資料要被取代，且 LLM 摘要走快取
        r2 = run.update_finance_info(base, watchlist=[])
        latest2 = json.loads((base / "output/research_reports/finance_info_latest.json").read_text(encoding="utf-8"))
        print("第 2 次：LLM 新摘要", latest2["agent_stats"]["llm_summarized"], "快取", latest2["agent_stats"]["llm_cached"])
        assert latest2["agent_stats"]["llm_cached"] > 0
        assert latest2["updated_at"] >= latest["updated_at"]

        # 第 3 次：爬蟲整個壞掉，要保留上一版而不是變空
        def boom(self, *a, **k): raise RuntimeError("network down")
        cc.CnyesNewsCrawler._get = boom
        for f in (base / "output/research_reports").glob("cnyes_news_*.json"):
            f.unlink()
        r3 = run.update_finance_info(base, watchlist=[])
        latest3 = json.loads((base / "output/research_reports/finance_info_latest.json").read_text(encoding="utf-8"))
        print("第 3 次（斷網）：新聞", latest3["news_count"], "stale =", latest3["stale"])
        assert latest3["news_count"] >= 60 and latest3["stale"] is True
    print("ALL OK" if ok else "FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
