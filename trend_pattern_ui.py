# -*- coding: utf-8 -*-
"""Streamlit 趨勢型態頁面。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_api import FugleClient, clean_symbol
from trend_pattern_engine import CATEGORY_ORDER, load_published_trend_patterns

PATTERN_GUIDES: dict[str, dict[str, Any]] = {
    "頭肩頂": {"values": [0.18, 0.42, 0.24, 0.85, 0.25, 0.47, 0.16], "note": "三個波峰，中間頭部較高；跌破頸線才確認轉弱。"},
    "頭肩頂跌破": {"values": [0.18, 0.42, 0.24, 0.85, 0.25, 0.47, 0.06], "note": "頭肩頂已向下跌破頸線。"},
    "倒頭肩": {"values": [0.80, 0.52, 0.87, 0.20, 0.84, 0.54, 0.79], "note": "三個波谷，中間頭部較低；等待向上突破頸線。"},
    "倒頭肩突破": {"values": [0.80, 0.52, 0.87, 0.20, 0.84, 0.54, 0.98], "note": "倒頭肩已向上突破頸線。"},
    "雙頂": {"values": [0.15, 0.82, 0.35, 0.78, 0.20], "note": "兩次相近高點，支撐跌破前仍屬潛在型態。"},
    "雙頂跌破": {"values": [0.15, 0.82, 0.35, 0.78, 0.06], "note": "兩次高點後跌破中間支撐。"},
    "雙底": {"values": [0.82, 0.30, 0.60, 0.34, 0.68], "note": "兩次相近低點，等待突破中間反彈高點。"},
    "雙底突破": {"values": [0.82, 0.30, 0.60, 0.34, 0.96], "note": "雙底已向上突破中間壓力。"},
    "圓底": {"values": [0.78, 0.61, 0.39, 0.24, 0.32, 0.50, 0.73, 0.91], "note": "價格呈弧形築底，右側均價開始回升。"},
    "杯柄": {"values": [0.87, 0.69, 0.42, 0.28, 0.43, 0.70, 0.83, 0.72, 0.77], "note": "杯底回升後形成淺回檔柄部，杯沿突破才確認。"},
    "上升楔形": {"values": [0.18, 0.28, 0.40, 0.35, 0.55, 0.52, 0.72, 0.67, 0.74], "note": "高低點都上移、下方支撐更陡；留意動能轉弱。"},
    "下降楔形": {"values": [0.91, 0.78, 0.82, 0.60, 0.66, 0.44, 0.49, 0.28, 0.56], "note": "高低點都下移、壓力收斂；留意向上突破。"},
    "多頭旗形": {"values": [0.12, 0.28, 0.43, 0.72, 0.91, 0.77, 0.82, 0.72, 0.79, 0.96], "note": "快速上漲後短暫整理，向上突破才確認延續。"},
    "空頭旗形": {"values": [0.92, 0.72, 0.53, 0.25, 0.12, 0.24, 0.18, 0.30, 0.20, 0.06], "note": "快速下跌後有限反彈，跌破整理區才確認延續。"},
    "上升三角形": {"values": [0.24, 0.72, 0.42, 0.73, 0.52, 0.75, 0.62, 0.95], "note": "高點大致水平、低點墊高，收盤突破壓力才確認。"},
    "下降三角形": {"values": [0.88, 0.35, 0.73, 0.34, 0.58, 0.33, 0.45, 0.08], "note": "低點大致水平、高點下移，跌破支撐偏弱。"},
    "對稱三角形": {"values": [0.88, 0.36, 0.75, 0.46, 0.63, 0.51, 0.56, 0.95], "note": "高點下移、低點上移，方向需等待有效突破。"},
    "箱型整理": {"values": [0.45, 0.75, 0.38, 0.69, 0.42, 0.72, 0.40, 0.64], "note": "價格在水平支撐與壓力間震盪；留意箱頂／箱底突破。"},
    "持續創高": {"values": [0.12, 0.21, 0.34, 0.30, 0.49, 0.46, 0.65, 0.80, 0.77, 0.98], "note": "近期接近區間新高，並由中短期均線確認偏多。"},
    "上升趨勢": {"values": [0.10, 0.22, 0.18, 0.41, 0.36, 0.61, 0.53, 0.82, 0.96], "note": "高點與低點逐步墊高，趨勢仍需留意量價變化。"},
    "下降趨勢": {"values": [0.95, 0.79, 0.83, 0.62, 0.68, 0.42, 0.46, 0.22, 0.08], "note": "反彈高點與低點下移，暫未出現明確止跌訊號。"},
    "盤整待突破": {"values": [0.48, 0.72, 0.34, 0.68, 0.37, 0.70, 0.46, 0.62], "note": "暫未符合單一典型形態，觀察支撐、壓力與突破方向。"},
}


def _guide_key(pattern_name: str) -> str:
    if pattern_name in PATTERN_GUIDES:
        return pattern_name
    for known in PATTERN_GUIDES:
        if pattern_name.startswith(known):
            return known
    return "盤整待突破"


def _schematic_figure(pattern_name: str, height: int = 155) -> go.Figure:
    key = _guide_key(pattern_name)
    values = PATTERN_GUIDES[key]["values"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(values) + 1)), y=values, mode="lines+markers",
        line={"width": 3}, marker={"size": 5}, name=key,
        hovertemplate="示意點 %{x}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": f"{key}｜示意圖（非實際股價）", "font": {"size": 13}},
        height=height, margin={"l": 8, "r": 8, "t": 35, "b": 8},
        showlegend=False, xaxis={"visible": False}, yaxis={"visible": False},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def _load_live_three_month_chart(symbol: str, end_day: str) -> pd.DataFrame:
    end = pd.Timestamp(end_day).normalize()
    start = (end - pd.Timedelta(days=130)).date()
    client = FugleClient()
    frame = client.historical_candles(clean_symbol(symbol), start, end.date(), "D", adjusted=False)
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for field in ("close", "volume"):
        if field not in frame:
            frame[field] = 0.0
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").tail(75)
    frame["MA20"] = frame["close"].rolling(20).mean()
    frame["MA50"] = frame["close"].rolling(50).mean()
    return frame.reset_index(drop=True)


def _render_live_chart(symbol: str) -> None:
    try:
        frame = _load_live_three_month_chart(symbol, date.today().isoformat())
    except Exception as exc:
        st.warning(f"無法取得 {symbol} 最新日 K：{type(exc).__name__}: {exc}")
        return
    if frame.empty:
        st.info("行情來源沒有回傳可用的三個月日 K 資料。")
        return
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["close"], mode="lines", name="收盤價", line={"width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["MA20"], mode="lines", name="MA20", line={"width": 1.5}), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["MA50"], mode="lines", name="MA50", line={"width": 1.2, "dash": "dot"}), row=1, col=1)
    fig.add_trace(go.Bar(x=frame["date"], y=frame["volume"], name="成交量", opacity=0.45), row=2, col=1)
    fig.update_layout(
        title=f"{symbol}｜近三個月簡易技術圖（資料至 {frame['date'].iloc[-1].date()}）",
        height=420, margin={"l": 10, "r": 10, "t": 48, "b": 15},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key=f"trend_patterns_chart_{symbol}")


def render_trend_patterns_workspace(base_dir: str | Path) -> None:
    """呈現按型態分類的概念股；個股詳細 K 線採使用者要求後再載入。"""
    base = Path(base_dir)
    payload = load_published_trend_patterns(base)
    stocks = payload.get("stocks", []) if isinstance(payload, dict) else []
    st.markdown(
        '<div class="content-heading"><div class="eyebrow">TREND PATTERN ANALYSIS</div>'
        '<h1>趨勢型態</h1><p>以產業分析的全部概念股為範圍，觀察近三個月圖表形態與近六個月所處位置；'
        '只在箱頂突破、三日轉折或主要型態改變時更新公開判讀。</p></div>',
        unsafe_allow_html=True,
    )
    guide_url = str(payload.get("pattern_guide_url") or "https://www.ig.com/cn/trading-strategies/10-chart-patterns-every-trader-needs-to-know-220907")
    st.markdown(f"型態判讀參考：[IG｜10 種常見圖表形態]({guide_url})。型態辨識以程式規則估算，所有型態均可能失效，不保證未來走勢。")
    if not stocks:
        st.info("目前尚無公開型態快照。先在本機執行 python run_trend_pattern_update.py --force 建立初始快照，確認結果後即可發布到網站。")
        return

    frame = pd.DataFrame([row for row in stocks if isinstance(row, dict)])
    if frame.empty or "symbol" not in frame:
        st.info("型態快照目前沒有可顯示的概念股。")
        return
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(4)
    if "themes" not in frame:
        frame["themes"] = [[] for _ in range(len(frame))]
    for col in ("classification", "pattern_name", "last_major_change_at"):
        if col not in frame:
            frame[col] = ""

    updated_at = str(payload.get("updated_at") or "尚無更新紀錄")
    data_asof = str(payload.get("data_asof") or "未知")
    st.caption(
        f"資料 K 棒日期：{data_asof}｜最後一次重大型態更新：{updated_at}｜"
        f"涵蓋概念股 {int(payload.get('tracked_symbols', len(frame)) or len(frame))} 檔｜"
        f"本次重大變化 {int(payload.get('changed_count', 0) or 0)} 檔"
    )
    counts = frame["classification"].value_counts()
    metric_cols = st.columns(5)
    for col, label in zip(metric_cols, CATEGORY_ORDER[:5]):
        col.metric(label, int(counts.get(label, 0)))
    st.divider()

    all_themes = sorted({theme for items in frame["themes"].tolist() if isinstance(items, list) for theme in items})
    selected_themes = st.multiselect("篩選產業主題（預設全部）", all_themes, default=all_themes, key="trend_pattern_theme_filter_v1")
    if selected_themes:
        frame = frame[frame["themes"].apply(lambda items: bool(set(items if isinstance(items, list) else []) & set(selected_themes)))]
    else:
        frame = frame.iloc[0:0]
    selected_classifications = st.multiselect(
        "篩選趨勢狀態", CATEGORY_ORDER, default=CATEGORY_ORDER, key="trend_pattern_classification_filter_v1"
    )
    frame = frame[frame["classification"].isin(selected_classifications)]
    if frame.empty:
        st.info("目前篩選條件沒有股票。")
        return

    frame = frame.sort_values(["last_major_change_at", "symbol"], ascending=[False, True], na_position="last")
    for category in CATEGORY_ORDER:
        section = frame[frame["classification"] == category]
        if section.empty:
            continue
        st.markdown(f'<div class="feature-section-title">{category}｜{len(section)} 檔</div>', unsafe_allow_html=True)
        explanations = {
            "底部反轉候選": "常見於跌深後轉穩、雙底／倒頭肩／圓底等結構；仍須觀察壓力突破與成交量。",
            "多頭延續／持續創高": "價格趨勢偏多或突破短期壓力；需留意追高風險及突破失敗。",
            "整理待突破": "三角形、旗形、杯柄或箱型整理；方向未確認前不預設一定向上。",
            "高檔轉弱／反轉警示": "雙頂、頭肩頂、上升楔形或上漲後三日轉弱，需留意支撐。",
            "下降趨勢／弱勢整理": "均線與報酬偏弱，或跌破重要支撐；不代表一定會持續下跌。",
            "中性觀察": "目前訊號不足以歸入更明確的方向。",
        }
        st.caption(explanations.get(category, ""))

        for pattern_name, sub in section.groupby("pattern_name", dropna=False, sort=True):
            pattern_name = str(pattern_name or "盤整待突破")
            cols = st.columns([1.1, 2.4])
            with cols[0]:
                st.plotly_chart(
                    _schematic_figure(pattern_name), use_container_width=True,
                    key=f"trend_pattern_schematic_{category}_{pattern_name}",
                )
            with cols[1]:
                st.markdown(f"**{pattern_name}｜{len(sub)} 檔**")
                st.caption(PATTERN_GUIDES.get(_guide_key(pattern_name), PATTERN_GUIDES["盤整待突破"])["note"])
                st.caption("此圖為形態概念示意，不是個股實際股價；個股日 K 請展開後載入。")

            for _, row in sub.sort_values("symbol").iterrows():
                symbol = str(row.get("symbol") or "").zfill(4)
                name = str(row.get("name") or symbol)
                themes = row.get("themes") if isinstance(row.get("themes"), list) else []
                with st.expander(f"{name}（{symbol}）｜{pattern_name}", expanded=False):
                    left, right = st.columns(2)
                    with left:
                        st.write(f"**型態分類：** {category}")
                        st.write(f"**近 3 個月趨勢：** {row.get('trend_3m', '未提供')}")
                        st.write(f"**近 6 個月背景：** {row.get('market_direction_6m', '未提供')}")
                        st.write(f"**型態信心參考：** {float(row.get('confidence') or 0):.0%}")
                    with right:
                        st.write(f"**最近判讀收盤價：** {row.get('close_at_assessment', '—')}")
                        st.write(f"**近 3 日報酬：** {row.get('return_3d_pct', '—')}%")
                        st.write(f"**近 3 個月報酬：** {row.get('return_3m_pct', '—')}%")
                        st.write(f"**支撐／壓力估計：** {row.get('support', '—')}／{row.get('resistance', '—')}")
                    st.write(f"**六個月方向判讀：** {row.get('market_direction_6m', '未提供')}")
                    st.write(f"**判讀說明：** {row.get('reason', '目前沒有額外說明。')}")
                    st.write(f"**所屬產業主題：** {'、'.join(themes) if themes else '未分類'}")
                    st.caption(
                        f"最近重大型態變化：{row.get('major_change', '初始型態建檔')}｜"
                        f"日期：{row.get('last_major_change_at', '—')}｜"
                        f"最新用於判讀的 K 棒：{row.get('last_bar_date', '—')}"
                    )
                    if st.button("載入／更新此股近三個月技術圖", key=f"trend_pattern_load_chart_{symbol}"):
                        _render_live_chart(symbol)
                    st.caption("近三個月圖由 Fugle 歷史日 K 即時查詢，顯示收盤價、MA20、MA50 與成交量；按下按鈕才查詢，避免一次呼叫所有股票。")

    with st.expander("分析方法與更新規則", expanded=False):
        st.markdown(
            """
            **資料範圍：** 產業分析 ROTATION_THEMES 的全部概念股；同一檔股票屬於多個主題時會保留全部主題。

            **時間尺度：** 近 3 個月日 K 辨識圖表形態，近 6 個月位置辨識股價相對高低位與背景方向。支撐、壓力與型態信心均為啟發式估算。

            **重大變化：** 包含箱型／三角形／杯柄等壓力突破、連續下跌後三日轉穩或上漲，以及多頭趨勢三日轉弱、主要型態改變。沒有重大變化時不改寫公開快照。

            **型態限制：** 自動規則是近似判斷，無法完全取代人眼辨識；形態可能失敗、假突破也可能發生，請勿單憑型態交易。
            """
        )
