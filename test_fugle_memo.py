# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, json
from earnings_call_agent import EarningsCallAgent

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("symbols",nargs="*",default=["6582"]); ap.add_argument("--crawl-only",action="store_true"); args=ap.parse_args()
    agent=EarningsCallAgent(Path(__file__).resolve().parent)
    try:
        for s in args.symbols:
            print(f"\n=== {s} ===")
            events=agent._fugle_memo_events(s,max_events=3)
            if not events: print("NO_EVENT"); continue
            for e in events:
                print(e.get("url"))
                try:
                    art=agent.crawler.extract_article(e["url"])
                    print(json.dumps({"title":art.get("title"),"reader_method":art.get("reader_method"),"article_reader":art.get("article_reader"),"memo_read_success":art.get("detail_read_verified"),"memo_chars":art.get("memo_chars"),"sections":list(art.get("sections",{}).keys()),"preview":art.get("memo_text","")[:1200]},ensure_ascii=False,indent=2))
                    if not args.crawl_only:
                        a=agent.analyze_one(e["url"]); print(json.dumps(a,ensure_ascii=False,indent=2)); print("AGENT_TOOL_USED:",a.get("agent_tool_used")); print("MEMO_READ_SUCCESS:",a.get("memo_read_success"))
                except Exception as exc: print("ERROR:",exc)
    finally: agent.crawler.close()
if __name__=="__main__": main()
