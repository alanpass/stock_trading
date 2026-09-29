# -*- coding: utf-8 -*-
"""單獨執行 Fugle 法說會備忘錄 AI Agent。"""
from pathlib import Path
import argparse, json
from earnings_call_agent import FugleEarningsCallAgent
BASE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--days',type=int,default=14)
    ap.add_argument('--limit',type=int,default=80)
    ap.add_argument('--force',action='store_true')
    a=ap.parse_args()
    agent=FugleEarningsCallAgent(BASE,ollama_model='qwen3:8b')
    out=agent.daily_run(days=a.days,limit=a.limit,force=a.force)
    print(json.dumps({
        'topic_url': out.get('topic_url'),
        'discovered_count': out.get('discovered_count'),
        'processed_count': out.get('processed_count'),
        'errors': len(out.get('errors',[])),
        'agent_flow': out.get('agent_flow')
    }, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
