# -*- coding: utf-8 -*-
"""接進排程前，先用這支腳本確認 CnyesNewsAgent 真的連得上、抓得到東西。

    python test_cnyes_news_agent.py --dry-run     # 只測連線＋列表頁掃描＋單篇解析，不寫快取
    python test_cnyes_news_agent.py --refresh      # 實際跑 23:00 排程會做的事（爬取＋寫快取）
    python test_cnyes_news_agent.py --digest       # 實際跑早報會做的事（讀快取＋本機 Qwen3 8B 摘要）

建議第一次先用 --dry-run：如果「候選文章 id」是 0，代表列表頁掃描失敗（很可能被 User-Agent
或頻率限制擋下、或鉅亨網版面已改版），這時候再去接 23:00 排程也不會有用，要先解決這一步。
"""
from __future__ import annotations

import argparse
import json

from cnyes_news_agent import CnyesNewsAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="只測連線與解析，不寫入快取")
    parser.add_argument("--refresh", action="store_true", help="實際爬取並寫入快取（23:00 排程用）")
    parser.add_argument("--digest", action="store_true", help="只讀快取並跑摘要（早報用）")
    args = parser.parse_args()

    agent = CnyesNewsAgent(base_dir=args.base_dir)

    if args.digest:
        result = agent.build_digest(days=args.days)
        print(f"快取裡近 {result['window_days']} 天共 {result['total_in_window']} 篇")
        digest = result["digest"]
        if not digest.get("available"):
            print(f"摘要未完成：{digest.get('error')}")
            return
        print(f"\n整體重點：{digest.get('overall_take')}")
        for i, item in enumerate(digest.get("items", []), 1):
            print(f"\n[{i}] {item['headline']}（{item['impact']}｜{item['relevance']}）")
            print(f"    {item['summary']}")
        return

    if args.refresh:
        print(json.dumps(agent.refresh_cache(), ensure_ascii=False, indent=2))
        return

    # --dry-run（預設）：只測連線與解析邏輯，不寫檔。
    ids = agent._candidate_ids()
    print(f"列表頁掃到 {len(ids)} 個候選文章 id：{ids[:10]}{'...' if len(ids) > 10 else ''}")
    if not ids:
        print("\n沒抓到任何候選 id —— 最可能是被擋下，或 /news/cat/headline 版面已改版。")
        print("可以先用瀏覽器開同一個網址，確認頁面本身看不看得到新聞清單，再檢查 _candidate_ids()。")
        return

    print("\n嘗試解析第一篇文章...")
    art = agent._parse_article(ids[0])
    if not art:
        print(f"解析失敗：{ids[0]} 這篇抓不到內容（_get 回傳 None，或 _parse_article 內部出錯）。")
        return
    print(json.dumps(art.to_dict(), ensure_ascii=False, indent=2))
    if not art.published_at:
        print("\n⚠️ 抓到文章但沒有 published_at——代表 article:published_time 這個 meta 標籤抓不到，"
              "請檢查鉅亨網文章頁的 meta 標籤是否有改版。沒有 published_at 的文章不會被收進快取。")


if __name__ == "__main__":
    main()
