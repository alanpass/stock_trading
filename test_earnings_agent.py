# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, json
from earnings_call_agent import FugleEarningsCallAgent

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("symbols", nargs="*", default=["6582","6142","2029","1473","1416"]); ap.add_argument("--limit", type=int, default=20); ap.add_argument("--crawl-only", action="store_true"); args=ap.parse_args()
    agent=FugleEarningsCallAgent(Path(__file__).resolve().parent)
    discovered=agent.crawler.discover()
    print(f"DISCOVERED={len(discovered)}")
    selected=[x for x in discovered if not args.symbols or str(x.get("symbol")) in set(args.symbols)]
    if not selected: selected=discovered[:args.limit]
    for x in selected[:args.limit]:
        print("\n===",x.get("symbol"),x.get("url"),"===")
        try:
            if args.crawl_only:
                article=agent.crawler.extract_article(x["url"])
                print(json.dumps({
                    "title":article.get("title"),
                    "reader_method":article.get("reader_method"),
                    "detail_read_verified":article.get("detail_read_verified"),
                    "memo_chars":len(article.get("memo_text", "")),
                    "sections":list(article.get("sections",{}).keys()),
                    "preview":article.get("memo_text","")[:1200],
                    "source_url":article.get("url"),
                },ensure_ascii=False,indent=2))
            else:
                a=agent.analyze_one(x["url"]); print(json.dumps(a,ensure_ascii=False,indent=2))
                print("AGENT_TOOL_USED:",a.get("agent_tool_used"))
        except Exception as exc: print("ERROR:",exc)

if __name__=="__main__": main()
