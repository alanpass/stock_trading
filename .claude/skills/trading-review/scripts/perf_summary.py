"""紙上交易績效彙整（當沖＋波段）。數字全部從 output/ 底下的檔案計算，不估算。

python perf_summary.py            # 印出 Markdown 表格
python perf_summary.py --xlsx     # 另外寫 output/trading_performance.xlsx（當沖逐日、當沖逐筆、波段淨值、波段成交）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DT = ROOT / "output" / "daytrade"
SW = ROOT / "output" / "swing"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def daytrade() -> tuple[pd.DataFrame, pd.DataFrame]:
    days, trades = [], []
    for f in sorted(DT.glob("*/summary.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        cap = s.get("capital") or (1_000_000 if s["date"] < "2026-10-22" else 300_000)
        days.append({"日期": s["date"], "本金": cap, "筆數": s.get("trades", 0), "勝": s.get("wins", 0),
                     "損益": s.get("pnl", 0), "報酬%": round(s.get("pnl", 0) / cap * 100, 3)})
        for p in s.get("positions", []):
            if p.get("exit") is None:
                continue
            trades.append({"日期": s["date"], "策略": p.get("strategy", "未分類"), "股票": p["symbol"], "股數": p["shares"],
                           "進場時間": p["opened_at"], "進場": p["entry"], "出場時間": p["closed_at"], "出場": p["exit"],
                           "原因": p["exit_reason"], "損益": p["pnl"], "Claude標記": bool(p.get("claude_flag"))})
    return pd.DataFrame(days), pd.DataFrame(trades)


def swing() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nav = pd.read_csv(SW / "nav.csv", encoding="utf-8-sig") if (SW / "nav.csv").exists() else pd.DataFrame()
    tr = pd.read_csv(SW / "trades.csv", encoding="utf-8-sig") if (SW / "trades.csv").exists() else pd.DataFrame()
    st = json.loads((SW / "state.json").read_text(encoding="utf-8")) if (SW / "state.json").exists() else {}
    return nav, tr, st


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "（無資料）"
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(f"{v:,}" if isinstance(v, (int,)) and not isinstance(v, bool) else str(v) for v in r) + " |"
             for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main() -> int:
    d, t = daytrade()
    nav, tr, st = swing()
    print("## 當沖（逐日）\n" + md(d))
    if not d.empty:
        n, w = int(d["筆數"].sum()), int(d["勝"].sum())
        print(f"\n累計 {n} 筆、勝 {w}（勝率 {w / n * 100:.0f}%）、損益 {int(d['損益'].sum()):,} 元；"
              f"最大單日虧損 {int(d['損益'].min()):,} 元" if n else "\n累計 0 筆")
    if not t.empty:
        g = t.groupby("策略").agg(筆數=("損益", "size"), 勝=("損益", lambda x: int((x > 0).sum())), 損益=("損益", "sum")).reset_index()
        print("\n### 當沖分策略\n" + md(g))
    print("\n## 波段\n" + md(nav))
    if st:
        cap = st.get("capital", 300_000)
        print(f"\n淨值 {st.get('nav', cap):,}（{(st.get('nav', cap) / cap - 1) * 100:+.2f}%），現金 {st.get('cash', 0):,.0f}；"
              f"上次換股 {st.get('last_rebalance')}，已過 {st.get('days_since_rebalance')} 個交易日")
        if not nav.empty:
            eq = nav["nav"]
            print(f"最大回落 {((eq / eq.cummax()) - 1).min() * 100:.2f}%")
        rows = [{"股票": f"{s}{p.get('name', '')}", "股數": p["shares"], "成本": p.get("avg"), "最新": p.get("last"),
                 "損益": round(p["last"] * p["shares"] - p["cost_total"])} for s, p in st.get("positions", {}).items()]
        print("\n### 波段持股\n" + md(pd.DataFrame(rows)))
    if "--xlsx" in sys.argv:
        out = ROOT / "output" / "trading_performance.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as x:
            d.to_excel(x, sheet_name="當沖逐日", index=False)
            t.to_excel(x, sheet_name="當沖逐筆", index=False)
            nav.to_excel(x, sheet_name="波段淨值", index=False)
            tr.to_excel(x, sheet_name="波段成交", index=False)
        print(f"\n已寫入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
