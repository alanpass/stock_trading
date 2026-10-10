# -*- coding: utf-8 -*-
"""診斷法說會爬取卡在哪一步（只探索＋讀 1 篇，不跑 Qwen3 分析）。
執行：python diagnose_earnings.py
"""
import sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from earnings_call_agent import EarningsCallAgent

base = Path(__file__).resolve().parent
a = EarningsCallAgent(base)
t = time.time()
print("1) 探索主題頁 ...", flush=True)
found = a.crawler.discover()
print(f"   找到 {len(found)} 個，{time.time() - t:.0f} 秒")
for x in found[:8]:
    print("  ", x.get("published_date"), x.get("url"))
print("   debug:", str(getattr(a.crawler, "_last_discovery_debug", {}))[:1500])
if found:
    t = time.time()
    print("2) 讀取第 1 篇正文 ...", flush=True)
    try:
        art = a.crawler.extract_article(found[0]["url"])
        print(f"   OK 正文 {art.get('memo_chars')} 字，方式 {art.get('reader_method')}，{time.time() - t:.0f} 秒")
    except Exception as exc:
        print(f"   失敗（{time.time() - t:.0f} 秒）：{type(exc).__name__}: {exc}")
    t = time.time()
    print("3) Qwen3 分析 1 篇 ...", flush=True)
    try:
        r = a.analyze_one(found[0]["url"])
        print(f"   OK {time.time() - t:.0f} 秒；impact={r.get('impact')}")
    except Exception as exc:
        print(f"   失敗（{time.time() - t:.0f} 秒）：{type(exc).__name__}: {exc}")
a.crawler.close()
