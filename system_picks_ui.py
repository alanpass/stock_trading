# -*- coding: utf-8 -*-
"""未來分析頁「最近系統推薦股票」：讀取 system_picks_public.json（GitHub finance-data，失敗退回本機）。"""
from __future__ import annotations

import json
import time
import urllib.request
from html import escape
from pathlib import Path

import streamlit as st

PUBLIC_URL = ("https://raw.githubusercontent.com/alanpass/stock_trading/finance-data/"
              "output/research_reports/system_picks_public.json")

_CSS = """<style>
.sp-sum{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:4px 0 14px}
.sp-tile{background:#fff;border:1px solid #eadfdf;border-radius:12px;padding:12px 16px}
.sp-tile b{display:block;font-size:14px;color:#8e2b2f;margin-bottom:6px}
.sp-tile .v{font-size:24px;font-weight:800;font-variant-numeric:tabular-nums}
.sp-tile small{display:block;font-size:12.5px;color:#7d7474;margin-top:2px}
.sp-up,.sp-table td.sp-up{color:#d23c3c}.sp-down,.sp-table td.sp-down{color:#14874f}.sp-flat,.sp-table td.sp-flat{color:#6b6464}
.sp-wrap{overflow-x:auto;background:#fff;border:1px solid #eadfdf;border-radius:12px}
.sp-table{width:100%;border-collapse:collapse;font-size:14px;min-width:720px}
.sp-table th{position:sticky;top:0;background:#fbf1f1;color:#6f2024;font-weight:700;text-align:left;padding:9px 12px;border-bottom:1px solid #eadada;white-space:nowrap}
.sp-table td{padding:9px 12px;border-bottom:1px solid #f2eaea;vertical-align:top;color:#3b3434}
.sp-table td.sp-wait{color:#9a9090;font-size:12.5px}
.sp-table tr:last-child td{border-bottom:0}
.sp-table td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.sp-stock b{font-size:15px;color:#2f2a2a}.sp-stock span{color:#8d8585;font-size:12px;margin-left:6px}
.sp-src{display:inline-block;font-size:12px;font-weight:700;border-radius:999px;padding:1px 9px;white-space:nowrap}
.sp-src.swing{background:#fde8e8;color:#8e2b2f}.sp-src.daytrade{background:#eef2fb;color:#34508f}.sp-src.theme{background:#f6efe2;color:#8a5a12}
.sp-reason{color:#5b5252;font-size:13px}
.sp-note{font-size:12.5px;color:#7d7474;line-height:1.8;margin-top:10px}
@media(max-width:700px){.sp-sum{grid-template-columns:1fr}}
</style>"""


@st.cache_data(ttl=600, show_spinner=False)
def load_system_picks(base_dir: str) -> dict:
    try:
        req = urllib.request.Request(f"{PUBLIC_URL}?v={int(time.time() // 600)}", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
        if isinstance(data, dict) and data.get("report_type") == "system_picks_public":
            return data
    except Exception:
        pass
    try:
        data = json.loads((Path(base_dir) / "output" / "research_reports" / "system_picks_public.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _cls(v) -> str:
    return "sp-flat" if v is None else "sp-up" if v > 0 else "sp-down" if v < 0 else "sp-flat"


def _price(v) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _pct(v) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def render_system_picks(base_dir: str | Path) -> None:
    data = load_system_picks(str(base_dir))
    picks = data.get("picks") or []
    if not picks:
        st.info("目前還沒有系統推薦紀錄；盤後研究完成後會自動更新。")
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    days = data.get("pick_days") or sorted({p["date"] for p in picks}, reverse=True)
    st.caption(f"最近 {len(days)} 個有推薦的日子（{days[-1]} ～ {days[0]}）；推薦後漲跌以 {data.get('data_asof') or '最新'} 收盤計算。"
               f"名單更新：{str(data.get('generated_at', ''))[:16].replace('T', ' ')}")

    tiles = []
    for key, s in (data.get("summary") or {}).items():
        win = "—" if s.get("win_rate") is None else f'{s["win_rate"]:.0f}%'
        tiles.append(f'<div class="sp-tile"><b>{escape(s["label"])}</b><span class="v {_cls(s.get("avg_ret"))}">{_pct(s.get("avg_ret"))}</span>'
                     f'<small>推薦後平均漲跌・{s["count"]} 筆・上漲比例 {win}</small></div>')
    st.markdown(f'<div class="sp-sum">{"".join(tiles)}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 2])
    # 預設選最近一個已經有後續行情的日子（假日前產生的名單還算不出漲跌）
    scored = [d for d in days if any(p["date"] == d and p.get("ret_pct") is not None for p in picks)]
    day = c1.selectbox("推薦日期", days, index=days.index(scored[0]) if scored else 0, key="sp_day",
                       format_func=lambda d: d + ("" if d in scored else "（尚無後續行情）"))
    labels = {"全部": None, **{s["label"]: k for k, s in (data.get("summary") or {}).items()}}
    src = c2.segmented_control("來源", list(labels), default="全部", key="sp_src") or "全部"
    rows = [p for p in picks if p["date"] == day and (labels[src] is None or p["source"] == labels[src])]
    rows.sort(key=lambda p: (p["source"], -(p["ret_pct"] if p["ret_pct"] is not None else -999)))

    body = "".join(
        f'<tr><td class="sp-stock"><b>{escape(p["name"])}</b><span>{escape(p["symbol"])}</span></td>'
        f'<td><span class="sp-src {p["source"]}">{escape(p["source_label"])}</span></td>'
        f'<td class="sp-reason">{escape(p.get("reason") or "")}</td>'
        f'<td class="num">{_price(p.get("ref_price"))}</td><td class="num">{_price(p.get("last_close"))}</td>'
        + (f'<td class="num {_cls(p["ret_pct"])}"><b>{_pct(p["ret_pct"])}</b></td></tr>' if p.get("ret_pct") is not None
           else '<td class="num sp-wait">尚無後續行情</td></tr>')
        for p in rows)
    st.markdown('<div class="sp-wrap"><table class="sp-table"><thead><tr><th>股票</th><th>來源</th><th>推薦理由</th>'
                '<th style="text-align:right">推薦日收盤</th><th style="text-align:right">最新收盤</th><th style="text-align:right">推薦後漲跌</th>'
                f'</tr></thead><tbody>{body or "<tr><td colspan=6>這一天沒有符合的推薦。</td></tr>"}</tbody></table></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sp-note">來源說明：<b>題材波段選股</b>＝前 3 強題材裡 20 日動能最強的個股；'
        '<b>當沖候選</b>＝當沖引擎當天判定可做多的股票；<b>盤後利多題材</b>＝盤後研究強度前 2 名題材的自選成分股。'
        f'<br>{escape(data.get("disclaimer") or "")}</div>', unsafe_allow_html=True)
