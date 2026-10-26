# -*- coding: utf-8 -*-
"""未來分析頁「最近系統推薦股票」：波段／長期名單與看多原因，下方追蹤過去推薦的漲跌。

資料：system_picks_public.json（GitHub finance-data，失敗退回本機），由 system_picks.py 每天盤後產生。
"""
from __future__ import annotations

import json
import time
import urllib.request
from html import escape
from pathlib import Path

import streamlit as st

PUBLIC_URL = ("https://raw.githubusercontent.com/alanpass/stock_trading/finance-data/"
              "output/research_reports/system_picks_public.json")
HORIZON_TABS = {"波段推薦": "swing", "長期推薦": "long"}

_CSS = """<style>
.sp-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:6px 0 4px}
.sp-card{background:#fff;border:1px solid #eadfdf;border-radius:14px;padding:16px 18px;box-shadow:0 6px 16px -14px rgba(110,30,36,.5)}
.sp-top{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.sp-rank{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;background:#8e2b2f;color:#fff;font-weight:800;font-size:13px;flex:none;align-self:center}
.sp-name{font-size:20px;font-weight:800;color:#2f2a2a}.sp-code{font-size:13px;color:#8d8585}
.sp-px{margin-left:auto;text-align:right;font-variant-numeric:tabular-nums}
.sp-px b{font-size:18px;color:#2f2a2a}.sp-px small{display:block;font-size:12px}
.sp-themes{margin:6px 0 10px;display:flex;flex-wrap:wrap;gap:5px}
.sp-themes span{font-size:12px;color:#8a5a12;background:#f6efe2;border-radius:999px;padding:1px 9px}
.sp-why-title{font-size:13px;font-weight:800;color:#8e2b2f;margin:2px 0 6px}
.sp-why{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.sp-why li{display:flex;gap:8px;align-items:flex-start;font-size:14px;line-height:1.55;color:#3b3434}
.sp-cat{flex:none;font-size:11.5px;font-weight:800;color:#fff;background:#c94848;border-radius:5px;padding:1px 7px;margin-top:2px;min-width:52px;text-align:center}
.sp-cat.c-籌碼{background:#b5523a}.sp-cat.c-基本面{background:#7a4fa0}.sp-cat.c-研究筆記{background:#34508f}.sp-cat.c-題材{background:#a86a12}
.sp-cat.c-技術,.sp-cat.c-型態{background:#8e2b2f}.sp-cat.c-評價,.sp-cat.c-股利{background:#2f7a68}
.sp-sum{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:8px 0 12px}
.sp-tile{background:#fff;border:1px solid #eadfdf;border-radius:12px;padding:12px 16px}
.sp-tile b{display:block;font-size:14px;color:#8e2b2f;margin-bottom:4px}
.sp-tile .v{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}
.sp-tile small{display:block;font-size:12.5px;color:#7d7474;margin-top:2px}
.sp-up,.sp-table td.sp-up{color:#d23c3c}.sp-down,.sp-table td.sp-down{color:#14874f}.sp-flat,.sp-table td.sp-flat{color:#6b6464}
.sp-wrap{overflow-x:auto;background:#fff;border:1px solid #eadfdf;border-radius:12px;max-height:460px;overflow-y:auto}
.sp-table{width:100%;border-collapse:collapse;font-size:14px;min-width:620px}
.sp-table th{position:sticky;top:0;background:#fbf1f1;color:#6f2024;font-weight:700;text-align:left;padding:9px 12px;border-bottom:1px solid #eadada;white-space:nowrap}
.sp-table td{padding:8px 12px;border-bottom:1px solid #f2eaea;color:#3b3434}
.sp-table td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.sp-table td.sp-wait{color:#9a9090;font-size:12.5px}
.sp-h{display:inline-block;font-size:12px;font-weight:700;border-radius:999px;padding:1px 9px;white-space:nowrap}
.sp-h.swing{background:#fde8e8;color:#8e2b2f}.sp-h.long{background:#e6f2ee;color:#2f7a68}
.sp-sub{font-size:16px;font-weight:800;color:#6f2024;margin:18px 0 6px}
.sp-note{font-size:12.5px;color:#7d7474;line-height:1.8;margin-top:10px}
@media(max-width:800px){.sp-grid,.sp-sum{grid-template-columns:1fr}}
</style>"""


@st.cache_data(ttl=600, show_spinner=False)
def load_system_picks(base_dir: str) -> dict:
    try:
        req = urllib.request.Request(f"{PUBLIC_URL}?v={int(time.time() // 600)}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
        if isinstance(data, dict) and data.get("report_type") == "system_picks_public" and data.get("latest"):
            return data
    except Exception:
        pass
    try:
        data = json.loads((Path(base_dir) / "output" / "research_reports" / "system_picks_public.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("latest") else {}
    except (OSError, ValueError):
        return {}


def _cls(v) -> str:
    return "sp-flat" if v is None else "sp-up" if v > 0 else "sp-down" if v < 0 else "sp-flat"


def _pct(v) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def _price(v) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _card(i: int, p: dict) -> str:
    themes = "".join(f"<span>{escape(t)}</span>" for t in p.get("themes") or [])
    why = "".join(f'<li><span class="sp-cat c-{escape(r["cat"])}">{escape(r["cat"])}</span><span>{escape(r["text"])}</span></li>'
                  for r in p.get("reasons") or [])
    r20 = p.get("r20")
    sub = "" if r20 is None else f'<small class="{_cls(r20)}">近 20 日 {r20:+.1f}%</small>'
    return (f'<div class="sp-card"><div class="sp-top"><span class="sp-rank">{i}</span><span class="sp-name">{escape(p["name"])}</span>'
            f'<span class="sp-code">{escape(p["symbol"])}</span><span class="sp-px"><b>{_price(p.get("close"))}</b>{sub}</span></div>'
            f'<div class="sp-themes">{themes}</div><div class="sp-why-title">看多的原因（{len(p.get("reasons") or [])} 項）</div>'
            f'<ul class="sp-why">{why}</ul></div>')


def render_system_picks(base_dir: str | Path) -> None:
    data = load_system_picks(str(base_dir))
    latest = data.get("latest") or {}
    if not latest.get("picks"):
        st.info("目前還沒有系統推薦名單；盤後研究完成後會自動更新。")
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.caption(f"依 {latest.get('date')} 收盤資料篩選；名單更新：{str(data.get('generated_at', ''))[:16].replace('T', ' ')}。"
               "每檔都列出符合的看多條件，數字皆來自公開資料。")

    tab = st.segmented_control("期間", list(HORIZON_TABS), default="波段推薦", key="sp_horizon") or "波段推薦"
    horizon = HORIZON_TABS[tab]
    picks = [p for p in latest["picks"] if p["horizon"] == horizon]
    if picks:
        st.markdown('<div class="sp-grid">' + "".join(_card(i, p) for i, p in enumerate(picks, 1)) + "</div>", unsafe_allow_html=True)
    else:
        st.info("今天沒有股票符合這一組條件。")

    # 過去推薦的追蹤
    st.markdown('<div class="sp-sub">過去推薦追蹤</div>', unsafe_allow_html=True)
    tiles = []
    for key, s in (data.get("summary") or {}).items():
        win = "—" if s.get("win_rate") is None else f'{s["win_rate"]:.0f}%'
        tiles.append(f'<div class="sp-tile"><b>{escape(s["label"])}</b><span class="v {_cls(s.get("avg_ret"))}">{_pct(s.get("avg_ret"))}</span>'
                     f'<small>推薦後平均漲跌・{s["count"]} 筆有後續行情・上漲比例 {win}</small></div>')
    st.markdown(f'<div class="sp-sum">{"".join(tiles)}</div>', unsafe_allow_html=True)
    tracked = sorted(data.get("tracked") or [], key=lambda r: (r["date"], r["ret_pct"] if r["ret_pct"] is not None else -999), reverse=True)
    body = "".join(
        f'<tr><td>{escape(r["date"])}</td><td><b>{escape(r["name"])}</b> <span style="color:#8d8585;font-size:12px">{escape(r["symbol"])}</span></td>'
        f'<td><span class="sp-h {r["horizon"]}">{"波段" if r["horizon"] == "swing" else "長期"}</span></td>'
        f'<td class="num">{_price(r.get("ref_price"))}</td><td class="num">{_price(r.get("last_close"))}</td>'
        + (f'<td class="num {_cls(r["ret_pct"])}"><b>{_pct(r["ret_pct"])}</b></td></tr>' if r.get("ret_pct") is not None
           else '<td class="num sp-wait">尚無後續行情</td></tr>')
        for r in tracked)
    st.markdown('<div class="sp-wrap"><table class="sp-table"><thead><tr><th>推薦日</th><th>股票</th><th>類型</th>'
                '<th style="text-align:right">推薦日收盤</th><th style="text-align:right">最新收盤</th><th style="text-align:right">推薦後漲跌</th>'
                f'</tr></thead><tbody>{body}</tbody></table></div>', unsafe_allow_html=True)
    if not any(r.get("ret_pct") is not None for r in tracked):
        st.caption("名單從今天開始記錄，下一個交易日收盤後才會出現推薦後漲跌。")

    rules = "".join(f"<br><b>{escape(k)}</b>：{escape(v)}" for k, v in data.get("rules") or [])
    st.markdown(f'<div class="sp-note">篩選規則{rules}<br>{escape(data.get("disclaimer") or "")}</div>', unsafe_allow_html=True)
