# -*- coding: utf-8 -*-
"""快速驗證：目前 Fugle 法說會備忘錄是否能被 Agent 逐一讀取。"""
from pathlib import Path
from earnings_call_agent import FugleEarningsCallAgent

CURRENT = {
    "6582": "https://blog.fugle.tw/post/earnings-call-6582-2026-09-24",
    "6142": "https://blog.fugle.tw/post/earnings-call-6142-2026-09-24",
    "2029": "https://blog.fugle.tw/post/earnings-call-2029-2026-09-24",
    "1473": "https://blog.fugle.tw/post/earnings-call-1473-2026-09-24",
    "1416": "https://blog.fugle.tw/post/earnings-call-1416-2026-09-24",
}

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("symbols", nargs="*", default=list(CURRENT)); args=ap.parse_args()
    agent=FugleEarningsCallAgent(Path(__file__).resolve().parent)
    for sym in args.symbols:
        url=CURRENT.get(sym, sym)
        print(f"\n=== {sym} ===\nURL={url}")
        try:
            article=agent.crawler.extract_article(url)
            print(f"MEMO_OPENED=True chars={len(article['memo_text'])} sections={list(article['sections'])}")
            result=agent.analyze_one(url)
            print(f"AGENT_TOOL_USED={result.get('agent_tool_used')} impact={result.get('impact')} confidence={result.get('confidence')}")
            print("SUMMARY:", result.get("one_line_summary") or result.get("summary"))
            print("FIN:", "；".join(result.get("financial_highlights",[])[:3]))
            print("GUIDE:", "；".join(result.get("guidance",[])[:3]))
        except Exception as exc:
            print("ERROR:", exc)

if __name__ == "__main__": main()
