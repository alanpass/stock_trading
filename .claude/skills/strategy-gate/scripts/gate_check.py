"""策略驗證關卡：讀逐筆回測結果，判斷能不能上線。

python gate_check.py trades.csv --name 跌深反彈 [--ratio] [--universe "每日成交值前 40 名（用當天以前資料）"] [--hindsight]

CSV 需要 day（YYYY-MM-DD）與 net（扣成本單筆報酬，預設單位 %；--ratio 表示 0.01＝1%）。
輸出 PASS / WATCH / FAIL，並寫入 output/strategy_gate/<name>.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
MIN_TRADES, MIN_DAYS, MIN_PF_OOS, MAX_DD = 60, 20, 1.2, -25.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check(t: pd.DataFrame, hindsight: bool) -> dict:
    t = t.sort_values("day").reset_index(drop=True)
    days = sorted(t["day"].unique())
    res = {"trades": int(len(t)), "days": len(days), "checks": []}

    def add(name, ok, detail):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    add("樣本數", len(t) >= MIN_TRADES and len(days) >= MIN_DAYS, f"{len(t)} 筆／{len(days)} 天（需 ≥{MIN_TRADES} 筆、≥{MIN_DAYS} 天）")
    if len(days) >= 3:
        cuts = [days[len(days) // 3], days[2 * len(days) // 3]]
        segs = [t[t.day < cuts[0]], t[(t.day >= cuts[0]) & (t.day < cuts[1])], t[t.day >= cuts[1]]]
        means = [round(float(s.net.mean()), 3) if len(s) else float("nan") for s in segs]
        add("三段都賺", all(m > 0 for m in means), f"三段平均 {means}%")
        oos = segs[2]
        gain, loss = oos.net[oos.net > 0].sum(), -oos.net[oos.net < 0].sum()
        pf = gain / loss if loss > 0 else float("inf")
        add("樣本外獲利因子", pf >= MIN_PF_OOS, f"最後一段 PF {pf:.2f}（需 ≥ {MIN_PF_OOS}）")
    # 每個訊號日的平均報酬依序累加（不複利），持有多天的重疊部位才不會被重複放大
    daily = t.groupby("day").net.mean()
    eq = daily.cumsum()
    dd = float((eq - eq.cummax()).min())
    add("最大回落", dd >= MAX_DD, f"{dd:.1f} 個百分點（訊號日平均報酬累加；需 ≥ {MAX_DD}）")
    res["avg_net_pct"] = round(float(t.net.mean()), 3)
    res["win_rate"] = round(float((t.net > 0).mean() * 100), 1)
    passed = all(c["ok"] for c in res["checks"])
    res["verdict"] = "FAIL" if not passed else ("WATCH" if hindsight else "PASS")
    if hindsight and passed:
        res["note"] = "候選股是事後挑選，回測結果高估；最多只能小額紙上前推觀察"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ratio", action="store_true", help="net 欄是比例（0.01＝1%）")
    ap.add_argument("--universe", default="", help="候選股怎麼選（必填說明）")
    ap.add_argument("--hindsight", action="store_true", help="候選股是用今天的名單回測過去")
    a = ap.parse_args()
    t = pd.read_csv(a.csv)
    if not {"day", "net"} <= set(t.columns):
        print("CSV 需要 day 與 net 欄位"); return 2
    t["day"] = pd.to_datetime(t["day"]).dt.date.astype(str)
    if a.ratio:
        t["net"] = t["net"] * 100
    if not a.universe:
        print("請用 --universe 說明候選股怎麼選（避免事後選股）"); return 2
    res = check(t, a.hindsight)
    res.update({"name": a.name, "universe": a.universe, "hindsight": a.hindsight,
                "checked_at": datetime.now().isoformat(timespec="seconds"), "source": str(a.csv)})
    out = ROOT / "output" / "strategy_gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{a.name}：{res['verdict']}（{res['trades']} 筆，平均 {res['avg_net_pct']:+.2f}%，勝率 {res['win_rate']}%）")
    for c in res["checks"]:
        print(f"  {'✓' if c['ok'] else '✗'} {c['name']}：{c['detail']}")
    if res.get("note"):
        print("  注意：" + res["note"])
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
