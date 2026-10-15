# -*- coding: utf-8 -*-
"""Claude 盤後個股研究筆記：網站顯示層。

筆記由 Claude 排程任務在每個交易日盤後預先產生（run_after_close_research.py --finalize），
發布到 GitHub finance-data 分支；網站只讀取並標示產生時間，不在網頁上即時呼叫任何 LLM。
"""
from __future__ import annotations

import html
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st

PUBLIC_URL = (
    "https://raw.githubusercontent.com/"
    "alanpass/stock_trading/finance-data/"
    "output/research_reports/stock_notes_public.json"
)
TAIPEI = ZoneInfo("Asia/Taipei")

_CSS = """
<style>
.sn-meta{font-size:12px;color:var(--site-muted,#787878);margin:0 0 12px;line-height:1.7}
.sn-card{background:#fff;border:1px solid var(--site-line,#d8cfcf);border-left:4px solid var(--site-red,#8e2b2f);border-radius:4px;padding:16px 18px;margin:8px 0 14px}
.sn-card .sn-title{font-family:Georgia,"Noto Serif TC","Microsoft JhengHei",serif;font-size:20px;font-weight:700;color:var(--site-ink,#252525);margin:0 0 4px}
.sn-head{font-size:14px;color:var(--site-red,#8e2b2f);font-weight:700;margin:0 0 10px}
.sn-sum{font-size:14px;color:#3d3838;line-height:1.85;margin:0 0 12px}
.sn-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.sn-col{background:var(--site-blush,#fbf1f1);border-radius:4px;padding:10px 12px}
.sn-col b{display:block;font-size:12px;color:var(--site-red-dark,#6f2024);margin-bottom:4px}
.sn-col ul{margin:0;padding-left:18px;font-size:13px;color:#4a4545;line-height:1.7}
.sn-foot{font-size:12px;color:var(--site-muted,#787878);margin-top:10px;line-height:1.7;overflow-wrap:anywhere}
.sn-foot a{color:var(--site-red,#8e2b2f)}
@media(max-width:760px){.sn-grid{grid-template-columns:1fr}}
</style>
"""


@st.cache_data(ttl=300, show_spinner=False)
def load_stock_notes(base_dir: str) -> dict:
    """優先讀 GitHub 公開 JSON；失敗時退回本機 stock_notes_latest.json。"""
    try:
        req = urllib.request.Request(f"{PUBLIC_URL}?v={int(time.time())}", headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode("utf-8"))
        if isinstance(data, dict) and data.get("report_type") == "stock_notes_public" and isinstance(data.get("notes"), list):
            return data
    except Exception:
        pass
    path = Path(base_dir) / "output" / "research_reports" / "stock_notes_latest.json"
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("notes"), list):
                return data
    except Exception:
        pass
    return {}


def _fmt_time(value: str) -> str:
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TAIPEI)
        return dt.astimezone(TAIPEI).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value or "—")


def _ul(items: list) -> str:
    rows = [f"<li>{html.escape(str(x))}</li>" for x in items or [] if str(x).strip()]
    return "<ul>" + "".join(rows) + "</ul>" if rows else "<ul><li>—</li></ul>"


def note_card_html(note: dict) -> str:
    links = [str(u) for u in note.get("source_links", []) or [] if str(u).startswith("http")]
    link_html = "、".join(f'<a href="{html.escape(u)}" target="_blank" rel="noopener">來源 {i}</a>' for i, u in enumerate(links, 1))
    evidence = "；".join(html.escape(str(x)) for x in note.get("evidence", []) or [])
    conf = float(note.get("confidence", 0) or 0)
    return (
        f'<div class="sn-card"><div class="sn-title">{html.escape(str(note.get("name") or ""))}（{html.escape(str(note.get("symbol", "")))}）</div>'
        + (f'<div class="sn-head">{html.escape(str(note.get("headline", "")))}</div>' if note.get("headline") else "")
        + f'<p class="sn-sum">{html.escape(str(note.get("summary", "")))}</p>'
        + '<div class="sn-grid">'
        + f'<div class="sn-col"><b>正面因素</b>{_ul(note.get("positives"))}</div>'
        + f'<div class="sn-col"><b>風險</b>{_ul(note.get("risks"))}</div>'
        + f'<div class="sn-col"><b>後續觀察</b>{_ul(note.get("watch"))}</div>'
        + "</div>"
        + f'<div class="sn-foot">證據：{evidence or "—"}<br>研究信心：{conf * 100:.0f}%'
        + (f"｜深度研究更新 {_fmt_time(note.get('updated_at'))}" if note.get("updated_at") else "")
        + (f"｜{link_html}" if link_html else "")
        + "</div></div>"
    )


def render_stock_note_for_symbol(base_dir: str | Path, symbol: str) -> None:
    """個股頁使用：只顯示目前選取股票的最新筆記。"""
    data = load_stock_notes(str(base_dir))
    note = next((n for n in data.get("notes", []) if str(n.get("symbol")) == str(symbol)), None)
    if not note:
        st.caption("此股票尚無 Claude 盤後研究筆記；自選股與當日熱門股會在每個交易日盤後自動產生。")
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="sn-meta">Claude 盤後研究筆記｜產生時間 {_fmt_time(data.get("generated_at", ""))}（台灣時間）</div>', unsafe_allow_html=True)
    st.markdown(note_card_html(note), unsafe_allow_html=True)


def render_stock_notes_workspace(base_dir: str | Path, selected_symbol: str = "") -> None:
    data = load_stock_notes(str(base_dir))
    notes = [n for n in data.get("notes", []) if isinstance(n, dict)]
    if not notes:
        st.info("目前尚無個股研究筆記。系統會在每個交易日盤後，由 Claude 為自選股與當日熱門股預先撰寫。")
        return
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="sn-meta">產生時間：{_fmt_time(data.get("generated_at", ""))}（台灣時間）'
        f'｜研究日期 {html.escape(str(data.get("report_date", "")))}｜{html.escape(str(data.get("generated_by", "Claude 研究 Agent")))}'
        f'｜共 {len(notes)} 檔<br>{html.escape(str(data.get("disclaimer", "研究筆記僅整理公開資訊與證據，不構成投資建議。")))}</div>',
        unsafe_allow_html=True,
    )
    groups = {
        "自選股": [n for n in notes if n.get("category") == "watchlist"],
        "當日熱門股": [n for n in notes if n.get("category") == "hot"],
        "深度研究": [n for n in notes if n.get("category") == "deep"],
    }
    group = st.radio("筆記範圍", [k for k, v in groups.items() if v] or ["自選股"], horizontal=True, key="stock_notes_group")
    rows = groups.get(group, [])
    if not rows:
        return
    labels = [f'{n.get("name") or ""}（{n.get("symbol")}）' for n in rows]
    default = next((i for i, n in enumerate(rows) if str(n.get("symbol")) == str(selected_symbol)), 0)
    pick = st.selectbox("選擇股票", labels, index=default, key=f"stock_notes_pick_{group}")
    st.markdown(note_card_html(rows[labels.index(pick)]), unsafe_allow_html=True)
