# -*- coding: utf-8 -*-
"""Streamlit 趨勢型態頁面。"""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stock_api import FugleClient, clean_symbol
from trend_pattern_engine import CATEGORY_ORDER, load_published_trend_patterns, load_trend_backtest

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
    "價格基準待確認": {"values": [0.85, 0.87, 0.86, 0.34, 0.36, 0.35, 0.38], "note": "疑似價格基準斷層，先核對除權息／股票股利／分割與還原股價資料，不直接判斷趨勢。"},
}


def _guide_key(pattern_name: str) -> str:
    if pattern_name in PATTERN_GUIDES:
        return pattern_name
    for known in PATTERN_GUIDES:
        if pattern_name.startswith(known):
            return known
    return "盤整待突破"


def _schematic_svg(pattern_name: str, height: int = 120) -> str:
    """形態示意圖用靜態 SVG：一頁十幾張互動 Plotly 圖會讓瀏覽器卡十幾秒，示意線不需要互動。"""
    key = _guide_key(pattern_name)
    values = PATTERN_GUIDES[key]["values"]
    width, pad = 300, 10
    step = (width - 2 * pad) / max(1, len(values) - 1)
    points = [(pad + i * step, pad + (1 - v) * (height - 2 * pad)) for i, v in enumerate(values)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#6c8ba3"/>' for x, y in points)
    return (
        f'<div style="font-size:13px;color:#8e2b2f;margin-bottom:4px">{escape(key)}｜示意圖（非實際股價）</div>'
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="display:block;max-width:340px;height:auto" role="img" aria-label="{escape(key)} 示意圖">'
        f'<polyline points="{path}" fill="none" stroke="#6c8ba3" stroke-width="3" stroke-linejoin="round"/>{dots}</svg>'
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_live_six_month_chart(symbol: str, end_day: str) -> pd.DataFrame:
    """載入六個月還原日 K；額外取得歷史暖身資料計算可切換的長週期均線。"""
    end = pd.Timestamp(end_day).normalize()
    # 六個月圖表之外要保留 MA120/MA240 的計算暖身；分段下載避免單次歷史 API 區間過長。
    start = end - pd.Timedelta(days=560)
    client = FugleClient()
    parts: list[pd.DataFrame] = []
    cursor = start
    failures: list[str] = []
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=299), end)
        try:
            part = client.historical_candles(
                clean_symbol(symbol), cursor.date(), chunk_end.date(), "D", adjusted=True
            )
            if part is not None and not part.empty:
                parts.append(part)
        except Exception as exc:
            failures.append(f"{cursor.date()}~{chunk_end.date()}: {type(exc).__name__}")
        cursor = chunk_end + pd.Timedelta(days=1)

    if not parts:
        detail = "；".join(failures[-2:])
        raise RuntimeError(f"Fugle 沒有回傳可用的還原日 K。{detail}")

    frame = pd.concat(parts, ignore_index=True).copy()
    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Taipei").dt.tz_localize(None).dt.normalize()
    for field in ("open", "high", "low", "close", "volume"):
        if field not in frame:
            frame[field] = np.nan if field != "volume" else 0.0
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    # 僅保留 OHLC 齊全、價格有效的真實交易日；不補週末或休市日。
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame[
        (frame["open"] > 0) & (frame["high"] > 0)
        & (frame["low"] > 0) & (frame["close"] > 0)
    ]
    frame = (
        frame.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if frame.empty:
        return pd.DataFrame()

    frame["volume"] = frame["volume"].fillna(0).clip(lower=0)
    # 先用完整暖身歷史計算均線，再裁切成六個日曆月，長週期均線才不會從圖表左端重新起算。
    for window in (5, 10, 20, 60, 120, 240):
        frame[f"MA{window}"] = frame["close"].rolling(window=window, min_periods=window).mean()

    chart_start = end - pd.DateOffset(months=6)
    visible = frame[(frame["date"] >= chart_start) & (frame["date"] <= end)].copy()
    visible["date_label"] = visible["date"].dt.strftime("%Y-%m-%d")
    return visible.reset_index(drop=True)


def _render_live_chart(symbol: str) -> None:
    """使用與操作策略相同的 K 線互動方式；範圍固定為最近六個日曆月。"""
    try:
        frame = _load_live_six_month_chart(
            symbol, datetime.now(ZoneInfo("Asia/Taipei")).date().isoformat()
        )
    except Exception as exc:
        st.warning(f"無法取得 {symbol} 最新六個月日 K：{type(exc).__name__}: {exc}")
        return
    if frame.empty:
        st.info("行情來源沒有回傳可用的六個月 OHLC K 棒資料。")
        return

    chart_x = frame["date_label"].tolist()
    display_labels = frame["date"].dt.strftime("%m/%d").tolist()
    n = len(chart_x)
    if n == 0:
        st.info("最近六個月沒有可繪製的交易日資料。")
        return

    # 還原後若最近可見區間仍有異常價格跳空，明確提醒但不隱藏 K 棒。
    daily_return = frame["close"].pct_change()
    suspicious = daily_return[daily_return.abs() >= 0.35]
    if not suspicious.empty:
        gap_idx = int(suspicious.index[-1])
        gap_date = pd.Timestamp(frame.loc[gap_idx, "date"]).strftime("%Y-%m-%d")
        gap_pct = float(suspicious.loc[gap_idx] * 100)
        st.warning(
            f"{gap_date} 還原收盤價仍出現 {gap_pct:+.1f}% 單日跳空；"
            "請先核對股票分割、股票股利或除權息資料，勿只依圖表判斷趨勢。"
        )

    reset_state_key = f"trend_technical_chart_reset_version_{symbol}"
    control_col, reset_col = st.columns([5, 1])
    with control_col:
        st.caption(f"{symbol}｜近六個月日 K，最新交易日：{chart_x[-1]}")
    with reset_col:
        if st.button(
            "回到最新",
            key=f"trend_technical_reset_latest_{symbol}",
            help="重設圖表範圍，回到這六個月的完整 K 線",
        ):
            st.session_state[reset_state_key] = int(
                st.session_state.get(reset_state_key, 0)
            ) + 1
    reset_version = int(st.session_state.get(reset_state_key, 0))

    ma_specs = [
        ("MA5", 5, True), ("MA10", 10, False), ("MA20", 20, True),
        ("MA60", 60, True), ("MA120", 120, False), ("MA240", 240, False),
    ]
    ma_cols = st.columns(6)
    enabled_ma: dict[int, bool] = {}
    for col, (label, window, default_on) in zip(ma_cols, ma_specs):
        with col:
            enabled_ma[window] = st.checkbox(
                label,
                value=default_on,
                key=f"trend_technical_ma_{symbol}_{label}",
            )

    # 與操作策略一致：紅漲綠跌、價格與成交量共用交易日類別軸。
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.76, 0.24],
    )
    fig.add_trace(
        go.Candlestick(
            x=chart_x,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="K 線",
            increasing_line_color="#ef4444",
            decreasing_line_color="#10b981",
            increasing_fillcolor="#ef4444",
            decreasing_fillcolor="#10b981",
        ),
        row=1, col=1,
    )

    ma_colors = {
        5: "#2563eb", 10: "#f59e0b", 20: "#8b5cf6",
        60: "#0891b2", 120: "#64748b", 240: "#a16207",
    }
    for label, window, _default_on in ma_specs:
        if enabled_ma.get(window):
            fig.add_trace(
                go.Scatter(
                    x=chart_x,
                    y=frame[f"MA{window}"],
                    mode="lines",
                    name=label,
                    line=dict(width=1.4, color=ma_colors[window]),
                    connectgaps=False,
                ),
                row=1, col=1,
            )

    volume_colors = [
        "#ef4444" if close >= open_ else "#10b981"
        for open_, close in zip(frame["open"], frame["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=chart_x,
            y=frame["volume"],
            name="成交量",
            marker_color=volume_colors,
            hovertemplate="%{x}<br>成交量 %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        width=1400,
        height=760,
        autosize=False,
        margin=dict(l=38, r=45, t=28, b=36),
        hovermode="x unified",
        dragmode="zoom",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        uirevision=f"{symbol}-{reset_version}",
        bargap=0.18,
    )

    tick_step = max(1, int(np.ceil((n - 1) / 8)))
    tick_indexes = list(range(0, n, tick_step))
    if tick_indexes[-1] != n - 1:
        tick_indexes.append(n - 1)
    tickvals = [chart_x[i] for i in tick_indexes]
    ticktext = [display_labels[i] for i in tick_indexes]
    for row_no in (1, 2):
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=chart_x,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=[-0.5, n - 0.5],
            minallowed=-0.5,
            maxallowed=n - 0.5,
            showgrid=True,
            gridcolor="#e6eaee",
            rangeslider_visible=False,
            row=row_no, col=1,
        )

    price_low = float(frame["low"].min())
    price_high = float(frame["high"].max())
    price_pad = max(
        (price_high - price_low) * 0.04, abs(price_high) * 0.005, 0.01
    )
    price_range = [price_low - price_pad, price_high + price_pad]
    volume_top = max(float(frame["volume"].max()), 1.0) * 1.08
    volume_range = [0.0, volume_top]
    fig.update_yaxes(
        range=price_range, minallowed=price_range[0], maxallowed=price_range[1],
        showgrid=True, gridcolor="#e6eaee", row=1, col=1,
    )
    fig.update_yaxes(
        range=volume_range, minallowed=volume_range[0], maxallowed=volume_range[1],
        showgrid=True, gridcolor="#e6eaee", row=2, col=1,
    )

    st.caption(
        "操作方式與「操作策略」一致：滑鼠滾輪放大／縮小、雙擊重設回完整六個月；"
        "放大後可切換工具列平移工具移動視窗。原始視圖範圍會鎖定，不能縮小到原圖之外。"
        "日期軸只包含真實交易日，週末與休市日不留空格。"
    )
    st.plotly_chart(
        fig,
        use_container_width=False,
        key=f"trend_patterns_chart_{symbol}_{reset_version}",
        config={
            "scrollZoom": True,
            "doubleClick": "reset",
            "displaylogo": False,
            "responsive": False,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["zoomOut2d", "autoScale2d", "resetScale2d"],
        },
    )


def _render_backtest(base: Path) -> None:
    """40 個交易日前的型態，之後 40 個交易日的平均報酬（排程每天產生的 trend_pattern_backtest.json）。"""
    bt = load_trend_backtest(base)
    st.markdown('<div class="feature-section-title">型態回測｜40 日前的型態，之後 40 日誰最賺錢</div>', unsafe_allow_html=True)
    if not bt.get("by_classification"):
        st.info("尚未產生回測結果；下一次趨勢型態排程執行後會出現。")
        return
    horizon = int(bt.get("horizon_trading_days", 40))
    best_c, best_p = bt.get("best_classification"), bt.get("best_pattern")
    rows_c = {r["classification"]: r for r in bt["by_classification"]}
    rows_p = {r["pattern_name"]: r for r in bt["by_pattern"]}
    st.caption(
        f"判讀日 {bt.get('assessed_on')}（用當時最近 40 個交易日判讀型態）→ 之後 {horizon} 個交易日到 {bt.get('ended_on')}｜"
        f"{bt.get('symbols')} 檔概念股｜檔數少於 {bt.get('min_samples', 5)} 的分組不參與排名"
    )
    cols = st.columns(4)
    cols[0].metric("全部概念股平均", f"{bt['all_avg_return_pct']:+.2f}%", f"中位數 {bt['all_median_return_pct']:+.2f}%", delta_color="off")
    if best_c:
        r = rows_c[best_c]
        cols[1].metric("最賺錢的趨勢狀態", best_c, f"平均 {r['avg_return_pct']:+.2f}%（{r['count']} 檔）", delta_color="off")
    if best_p:
        r = rows_p[best_p]
        cols[2].metric("最賺錢的型態", best_p, f"平均 {r['avg_return_pct']:+.2f}%（{r['count']} 檔）", delta_color="off")
    cols[3].metric("全部上漲比例", f"{bt.get('all_win_rate_pct', 0):.0f}%")

    def table(items: list[dict], key: str, label: str) -> pd.DataFrame:
        return pd.DataFrame([{
            label: r[key], "檔數": r["count"], "平均報酬%": r["avg_return_pct"], "中位數報酬%": r["median_return_pct"],
            "上漲比例%": r["win_rate_pct"], "相對全體%": r["excess_vs_all_pct"], "期間最大回檔%": r["avg_max_drawdown_pct"],
            "樣本": "足夠" if r["enough_samples"] else "不足", "表現最好": "、".join(r["best"]), "表現最差": "、".join(r["worst"]),
        } for r in items])

    fmt = {c: "{:+.2f}" for c in ("平均報酬%", "中位數報酬%", "相對全體%", "期間最大回檔%")} | {"上漲比例%": "{:.0f}"}
    st.dataframe(table(bt["by_classification"], "classification", "40 日前趨勢狀態").style.format(fmt),
                 use_container_width=True, hide_index=True)
    with st.expander("依型態細分（頭肩、雙底、楔形…）", expanded=False):
        st.dataframe(table(bt["by_pattern"], "pattern_name", "40 日前型態").style.format(fmt),
                     use_container_width=True, hide_index=True)
    with st.expander(f"個股明細（{len(bt.get('stocks', []))} 檔）", expanded=False):
        stocks = pd.DataFrame(bt.get("stocks", []))
        if not stocks.empty:
            stocks = stocks.rename(columns={"symbol": "代號", "name": "名稱", "classification": "40 日前趨勢狀態",
                                            "pattern_name": "40 日前型態", "forward_return_pct": "之後報酬%",
                                            "max_drawdown_pct": "期間最大回檔%"})
            st.dataframe(stocks[["代號", "名稱", "40 日前趨勢狀態", "40 日前型態", "之後報酬%", "期間最大回檔%"]],
                         use_container_width=True, hide_index=True)
    st.caption(
        "讀法：平均報酬容易被少數大漲股拉高，請同時看中位數與上漲比例。這只是單一期間的歷史統計，"
        "市場整體漲跌會影響所有分組，不代表未來會重複，也不構成投資建議。"
    )


def render_trend_patterns_workspace(base_dir: str | Path) -> None:
    """呈現按型態分類的概念股；個股詳細 K 線採使用者要求後再載入。"""
    base = Path(base_dir)
    payload = load_published_trend_patterns(base)
    stocks = payload.get("stocks", []) if isinstance(payload, dict) else []
    st.markdown(
        '<div class="content-heading"><div class="eyebrow">TREND PATTERN ANALYSIS</div>'
        '<h1>趨勢型態</h1><p>以產業分析的全部概念股為範圍，使用最近 40 個交易日辨識型態、近六個月判斷背景趨勢；'
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
    for start_index in range(0, len(CATEGORY_ORDER), 4):
        metric_cols = st.columns(min(4, len(CATEGORY_ORDER) - start_index))
        for col, label in zip(metric_cols, CATEGORY_ORDER[start_index:start_index + 4]):
            col.metric(label, int(counts.get(label, 0)))
    _render_backtest(base)
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
            "資料調整檢查": "還原股價仍有疑似價格斷層，先核對股票分割、股票股利或除權息，不進行方向判斷。",
        "高檔轉弱／反轉警示": "雙頂、頭肩頂、上升楔形或上漲後三日轉弱，需留意支撐。",
            "下降趨勢／弱勢整理": "均線與報酬偏弱，或跌破重要支撐；不代表一定會持續下跌。",
            "中性觀察": "目前訊號不足以歸入更明確的方向。",
        }
        st.caption(explanations.get(category, ""))

        for pattern_name, sub in section.groupby("pattern_name", dropna=False, sort=True):
            pattern_name = str(pattern_name or "盤整待突破")
            cols = st.columns([1.1, 2.4])
            with cols[0]:
                st.markdown(_schematic_svg(pattern_name), unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"**{pattern_name}｜{len(sub)} 檔**")
                st.caption(PATTERN_GUIDES.get(_guide_key(pattern_name), PATTERN_GUIDES["盤整待突破"])["note"])
                st.caption("此圖為形態概念示意，不是個股實際股價；個股日 K 請展開後載入。")

            for _, row in sub.sort_values("symbol").iterrows():
                symbol = str(row.get("symbol") or "").zfill(4)
                name = str(row.get("name") or symbol)
                themes = row.get("themes") if isinstance(row.get("themes"), list) else []
                with st.expander(f"{name}（{symbol}）｜{pattern_name}", expanded=False):
                    # 一檔股票的說明合成一個 markdown 元件：整頁有一百多檔，逐行 st.write 會產生兩千多個元件拖慢瀏覽器。
                    facts = [
                        ("型態分類", category),
                        ("最近判讀收盤價", row.get("close_at_assessment", "—")),
                        ("近 40 個交易日趨勢", row.get("trend_40d", "未提供")),
                        ("近 3 日報酬", f"{row.get('return_3d_pct', '—')}%"),
                        ("近 6 個月背景", row.get("market_direction_6m", "未提供")),
                        ("近 40 個交易日報酬", f"{row.get('return_40d_pct', '—')}%"),
                        ("型態信心參考", f"{float(row.get('confidence') or 0):.0%}"),
                        ("支撐／壓力估計", f"{row.get('support', '—')}／{row.get('resistance', '—')}"),
                    ]
                    grid = "".join(f"<div><b>{escape(k)}：</b>{escape(str(v))}</div>" for k, v in facts)
                    st.markdown(
                        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 18px">{grid}</div>\n\n'
                        f"**價格調整方式：** {row.get('adjustment_method', '舊快照／未標記還原方式')}  \n"
                        f"**判讀說明：** {row.get('reason', '目前沒有額外說明。')}  \n"
                        f"**所屬產業主題：** {'、'.join(themes) if themes else '未分類'}\n\n"
                        f"<small>最近重大型態變化：{escape(str(row.get('major_change', '初始型態建檔')))}｜"
                        f"日期：{escape(str(row.get('last_major_change_at', '—')))}｜"
                        f"最新用於判讀的 K 棒：{escape(str(row.get('last_bar_date', '—')))}<br>"
                        "按下按鈕才查詢六個月還原日 K；均線可自行勾選，並具備與「操作策略」相同的縮放、雙擊還原、固定視窗範圍及回到最新操作。</small>",
                        unsafe_allow_html=True,
                    )
                    if bool(row.get("corporate_action_review", False)):
                        st.warning(
                            f"此股於 {row.get('suspected_gap_date', '未知日期')} 出現 "
                            f"{row.get('suspected_gap_pct', '未知')}% 價格跳空，已暫停趨勢型態判斷。"
                        )
                    chart_visible_key = f"trend_pattern_chart_visible_{symbol}"
                    if st.button("載入／更新此股近六個月 K 棒圖", key=f"trend_pattern_load_chart_{symbol}"):
                        st.session_state[chart_visible_key] = True
                    if st.session_state.get(chart_visible_key, False):
                        _render_live_chart(symbol)

    with st.expander("分析方法與更新規則", expanded=False):
        st.markdown(
            """
            **資料範圍：** 產業分析 ROTATION_THEMES 的全部概念股；同一檔股票屬於多個主題時會保留全部主題。

            **價格基準：** 趨勢分析與個股 K 棒均使用 Fugle `adjusted=True` 還原股價，避免股票分割、股票股利與除權息造成的價格斷層。還原後若仍出現 35% 以上單日跳空，會放入「資料調整檢查」，暫停方向判斷。

            **時間尺度：** 最近 40 個交易日日 K 辨識目前圖表型態與 3／10／20／40 日報酬；近 6 個月資料判斷高低位置與背景方向。展開個股後顯示近六個月 OHLC K 棒、成交量，並可切換 MA5、MA10、MA20、MA60、MA120、MA240。均線使用圖表顯示期間以外的歷史暖身資料計算。

            **空值處理：** 圖表只使用 OHLC 齊全的真實交易日，日期軸採交易日類別序列，因此週末、國定假日與缺失行情不會形成空白日期。

            **重大變化：** 包含箱型／三角形／杯柄等壓力突破、連續下跌後三日轉穩或上漲，以及多頭趨勢三日轉弱、主要型態改變。沒有重大變化時不改寫公開快照。

            **型態限制：** 自動規則是近似判斷，無法完全取代人眼辨識；形態可能失敗，假突破也可能發生，請勿單憑型態交易。
            """
        )
