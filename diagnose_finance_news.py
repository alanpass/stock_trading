# -*- coding: utf-8 -*-
"""財經新聞診斷：一眼看出「為什麼只有幾篇」。  python diagnose_finance_news.py"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "output" / "research_reports"
TZ = ZoneInfo("Asia/Taipei")


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    print("=" * 70, "\n財經新聞診斷", datetime.now(TZ).isoformat(timespec="seconds"), "\n" + "=" * 70)

    src = (BASE / "cnyes_news_crawler.py").read_text(encoding="utf-8")
    uses_os = "os.getenv" in src or "os.environ" in src
    has_import = re.search(r"^import os\b", src, re.M) is not None
    print(f"[1] 爬蟲程式：用到 os={uses_os}，有 import os={has_import}", "" if (has_import or not uses_os) else "  ← ❌ 爬蟲會在寫入快取前就中斷！")

    print("\n[2] 鉅亨快取檔（每個檔案幾篇）")
    files = sorted(REPORTS.glob("cnyes_news_*.json"))
    if not files:
        print("  找不到任何 cnyes_news_*.json → 爬蟲從來沒成功寫入過")
    for f in files:
        d = load(f)
        arts = d.get("articles", []) if isinstance(d, dict) else []
        print(f"  {f.name:32s} {len(arts):4d} 篇  抓取時間={str(d.get('crawl_started_at',''))[:19]}  分類={d.get('per_category_count','-')}")
        for e in (d.get("errors") or [])[:3]:
            print("      ⚠", str(e)[:120])

    print("\n[3] 財經資訊總表 finance_info_latest.json")
    fin = load(REPORTS / "finance_info_latest.json")
    if not fin or "_error" in fin:
        print("  讀不到：", fin.get("_error") if fin else "檔案不存在")
    else:
        news = fin.get("news", [])
        days = Counter(str(n.get("published_at") or n.get("published") or "")[:10] for n in news)
        print(f"  更新時間={fin.get('updated_at')}  新聞={len(news)} 篇  法說會={fin.get('earnings_count')}  stale={fin.get('stale')}")
        print("  各日期篇數：", dict(sorted(days.items())))
        print("  各分類抓取數：", fin.get("crawl", {}).get("per_category_count"))
        print("  Agent：", fin.get("agent_stats"))
        for e in (fin.get("errors") or [])[:8]:
            print("   ⚠", str(e)[:140])

    print("\n[4] 最近執行紀錄（每個時段抓了幾篇）")
    for r in load(REPORTS / "finance_run_history.json").get("runs", [])[-10:]:
        print(f"  {r.get('at','')[:19]}  新聞={r.get('news'):4}  法說會={r.get('earnings'):3}  LLM新摘要={r.get('llm')}  stale={r.get('stale')}  錯誤={r.get('errors')}")

    print("\n[5] Ollama / Qwen3")
    try:
        from finance_agent import FinanceNewsAgent
        a = FinanceNewsAgent(BASE)
        print("  可用" if a._check_llm() else f"  不可用：{a.llm_note}（仍會用規則層摘要，只是少了 Qwen3 的重寫）")
    except Exception as exc:
        print("  檢查失敗：", exc)


if __name__ == "__main__":
    main()
