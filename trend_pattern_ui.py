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

from section_ui import section as _sec
from plotly.subplots import make_subplots

from stock_api import FugleClient, clean_symbol
from industry_rotation import ROTATION_DISPLAY_NAMES, load_published_rotation
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
    """第 80～41 個交易日前那 40 根日 K：各趨勢分類的股票在期間內平均漲跌（排程每天產生 trend_pattern_backtest.json）。"""
    bt = load_trend_backtest(base)
    if bt.get("report_type") != "trend_pattern_window_stats" or not bt.get("by_classification"):
        st.info("尚未產生期間統計；下一次趨勢型態排程執行後會出現。")
        return
    offset = int(bt.get("offset_trading_days", 40))
    st.markdown(
        f"**統計期間：** {bt.get('window_start')} ～ {bt.get('window_end')}"
        f"（第 {offset + 40}～{offset + 1} 個交易日前，共 40 根日 K）  \n"
        f"用這 40 根日 K 判讀趨勢型態，漲跌＝期間前一日收盤到 {bt.get('window_end')} 收盤（還原股價）。"
        f"{bt.get('symbols')} 檔概念股。"
    )
    pct = lambda v: "—" if v is None else f"{v:+.2f}%"
    a = bt["all"]
    cols = st.columns(4)
    cols[0].metric("全部平均漲跌", pct(a["avg_return_pct"]), f"中位數 {pct(a['median_return_pct'])}", delta_color="off")
    cols[1].metric("上漲股平均漲幅", pct(a["up_avg_pct"]), f"{a['up_count']} 檔上漲", delta_color="off")
    cols[2].metric("下跌股平均跌幅", pct(a["down_avg_pct"]), f"{a['down_count']} 檔下跌", delta_color="off")
    best = bt.get("best_classification")
    if best:
        r = next(x for x in bt["by_classification"] if x["classification"] == best)
        cols[3].metric(f"平均最賺錢（≥{bt.get('min_samples', 5)} 檔）", best, f"平均 {pct(r['avg_return_pct'])}（{r['count']} 檔）", delta_color="off")

    def table(items: list[dict], key: str, label: str) -> pd.DataFrame:
        return pd.DataFrame([{
            label: r[key], "檔數": r["count"], "平均漲跌%": r["avg_return_pct"], "中位數%": r["median_return_pct"],
            "上漲檔數": r["up_count"], "上漲股平均漲幅%": r["up_avg_pct"],
            "下跌檔數": r["down_count"], "下跌股平均跌幅%": r["down_avg_pct"],
            "期間最大回檔%": r["avg_max_drawdown_pct"],
            "漲最多": "、".join(r["best"]), "跌最多": "、".join(r["worst"]),
        } for r in items])

    fmt = {c: "{:+.2f}" for c in ("平均漲跌%", "中位數%", "上漲股平均漲幅%", "下跌股平均跌幅%", "期間最大回檔%")}
    st.dataframe(table(bt["by_classification"], "classification", "趨勢狀態").style.format(fmt, na_rep="—"),
                 use_container_width=True, hide_index=True)
    with st.expander("依型態細分（頭肩、雙底、楔形…）", expanded=False):
        st.dataframe(table(bt["by_pattern"], "pattern_name", "型態").style.format(fmt, na_rep="—"),
                     use_container_width=True, hide_index=True)
    with st.expander(f"個股明細（{len(bt.get('stocks', []))} 檔）", expanded=False):
        stocks = pd.DataFrame(bt.get("stocks", []))
        if not stocks.empty:
            stocks = stocks.rename(columns={"symbol": "代號", "name": "名稱", "classification": "趨勢狀態",
                                            "pattern_name": "型態", "window_return_pct": "期間漲跌%",
                                            "max_drawdown_pct": "期間最大回檔%"})
            st.dataframe(stocks[["代號", "名稱", "趨勢狀態", "型態", "期間漲跌%", "期間最大回檔%"]],
                         use_container_width=True, hide_index=True)
    st.caption("型態是用同一段 K 棒判讀的，這裡描述的是「各型態在形成期間的漲跌」，不是之後的報酬；"
               "檔數少的分組容易被單一個股左右。歷史統計，不構成投資建議。")


# 由多到空的顯示順序與顏色（台股：紅多綠空）
_BIAS = [
    ("多頭延續／持續創高", "偏多", "#c62828"),
    ("底部反轉候選", "轉強候選", "#e58a7f"),
    ("整理待突破", "整理", "#c9b48a"),
    ("中性觀察", "中性", "#b9c2cb"),
    ("高檔轉弱／反轉警示", "轉弱警示", "#7cc0a0"),
    ("下降趨勢／弱勢整理", "偏弱", "#16803c"),
]
_BULL = {"多頭延續／持續創高", "底部反轉候選"}
_BEAR = {"高檔轉弱／反轉警示", "下降趨勢／弱勢整理"}
_BULL_EVENTS = {"箱頂突破", "近20日壓力突破", "連跌後轉穩（3日）", "雙底突破確認", "倒頭肩突破確認"}
_FEED_DAYS = 5
_FEED_NAMES = 12
_RADAR_DISTANCE_PCT = 3.0
STALE_DATA_DAYS = 5
_RADAR_LIMIT = 8

_TREND_CSS = """<style>
.tp-brief{background:#fff;border:1px solid #eadada;border-left:4px solid #8e2b2f;border-radius:10px;padding:14px 18px;font-size:15.5px;line-height:1.85;color:#3f2f2f;margin-bottom:14px}
.tp-brief b{color:#8e2b2f}
.tp-stack{display:flex;height:30px;border-radius:7px;overflow:hidden;background:#eef0f2}
.tp-stack span{display:block;height:100%}
.tp-legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:9px 0 4px;font-size:13.5px;color:#594b4b}
.tp-legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.tp-legend b{font-size:15px;color:#2f2626;margin-left:4px}
.tp-chart-label{font-size:13px;color:#7a6a6a;margin:12px 0 2px}
.tp-chart-label em{font-style:normal;font-weight:700}
.tp-chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.tp-chip{display:inline-block;border-radius:999px;padding:2px 10px;font-size:12.5px;background:#f4eeee;color:#594b4b;white-space:nowrap}
.tp-chip.up{background:#fbe9e7;color:#b3261e}.tp-chip.down{background:#e6f4ec;color:#14703a}
.tp-feed{border:1px solid #eadada;border-radius:10px;background:#fff;overflow:hidden}
.tp-feed-row{display:grid;grid-template-columns:52px 190px minmax(0,1fr) 78px;gap:10px;align-items:center;padding:8px 14px;border-top:1px solid #f3ecec;font-size:13.5px;color:#3f2f2f}
.tp-feed-row:first-child{border-top:0}
.tp-feed-head{font-size:12px;color:#8a7676;background:#fbf7f7}
.tp-num{text-align:right;font-variant-numeric:tabular-nums}
.tp-muted{color:#8a7676;font-size:12.5px}
.tp-up{color:#c62828}.tp-down{color:#16803c}
.tp-map{border:1px solid #eadada;border-radius:10px;background:#fff;overflow:hidden}
.tp-map-row{display:grid;grid-template-columns:minmax(120px,1.1fr) 62px 46px minmax(120px,3fr) 58px;gap:12px;align-items:center;padding:7px 14px;border-top:1px solid #f3ecec;font-size:13.5px;color:#3f2f2f}
.tp-map-row:first-child{border-top:0}
.tp-map-row .tp-stack{height:14px;border-radius:4px}
.tp-radar-title{font-weight:700;font-size:14.5px;margin:0 0 8px;color:#3f2f2f}
.tp-radar-row{background:#fff;border:1px solid #eadada;border-radius:9px;padding:8px 12px;margin-bottom:7px;font-size:13.5px;color:#3f2f2f}
.tp-radar-line{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.tp-range{position:relative;height:5px;border-radius:3px;background:#eee7e7;margin-top:7px}
.tp-range i{position:absolute;top:-3px;width:3px;height:11px;border-radius:2px;background:#8e2b2f}
.tp-table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #eadada;font-size:13.5px}
.tp-table th{font-weight:500;font-size:12.5px;color:#7a6a6a;background:#fbf7f7;padding:7px 10px;text-align:right}
.tp-table td{padding:7px 10px;border-top:1px solid #f3ecec;text-align:right;font-variant-numeric:tabular-nums;color:#3f2f2f}
.tp-table th:first-child,.tp-table td:first-child{text-align:left}
.tp-table tr.base td{background:#fbf7f7;font-weight:600}
.tp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(232px,1fr));gap:10px}
.tp-card{background:#fff;border:1px solid #eadada;border-radius:10px;padding:10px 12px;font-size:13px;color:#3f2f2f;min-width:0}
.tp-card-top{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.tp-card-name{font-weight:700;font-size:14.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tp-card svg{display:block;width:100%;height:38px;margin:6px 0}
.tp-card-foot{display:flex;justify-content:space-between;gap:8px;align-items:center}
.tp-card-foot .tp-chip{overflow:hidden;text-overflow:ellipsis}
.tp-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:4px 18px;font-size:14px;color:#3f2f2f;margin-bottom:8px}
@media(max-width:700px){
.tp-feed-row{grid-template-columns:44px 1fr 66px;gap:4px 8px}
.tp-feed-row>:nth-child(3){grid-column:1/4;grid-row:2}
.tp-feed-row>:nth-child(4){grid-column:3;grid-row:1}
.tp-feed-head>:nth-child(3){display:none}
.tp-map-row{grid-template-columns:minmax(90px,1fr) 52px minmax(70px,1.5fr) 48px}
.tp-map-row>:nth-child(3){display:none}
}
</style>"""


def _pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:+.{digits}f}%" if np.isfinite(number) else "—"


def _tone(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "tp-up" if number > 0 else "tp-down" if number < 0 else ""


def _stack_html(counts: dict[str, int]) -> str:
    total = sum(int(counts.get(c, 0)) for c, _, _ in _BIAS) or 1
    return '<div class="tp-stack">' + "".join(
        f'<span style="width:{int(counts.get(c, 0)) / total * 100:.2f}%;background:{color}" title="{escape(label)} {int(counts.get(c, 0))} 檔"></span>'
        for c, label, color in _BIAS if counts.get(c)) + "</div>"


def _share(row: dict[str, Any], names: set[str]) -> float:
    total = sum(int(row.get(c, 0) or 0) for c, _, _ in _BIAS) or 1
    return sum(int(row.get(c, 0) or 0) for c in names) / total * 100


def _breadth_svg(breadth: list[dict[str, Any]]) -> str:
    """偏多＋轉強、轉弱＋偏弱占比的走勢；靜態 SVG，不需要互動。"""
    width, height, pad = 1000, 150, 26
    n = len(breadth)
    if n < 2:
        return ""
    x = lambda i: pad + i * (width - 2 * pad) / (n - 1)
    y = lambda v: pad / 2 + (100 - v) * (height - pad) / 100
    line = lambda names: " ".join(f"{x(i):.1f},{y(_share(row, names)):.1f}" for i, row in enumerate(breadth))
    grid = "".join(
        f'<line x1="{pad}" x2="{width - pad}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eee7e7"/>'
        f'<text x="{pad - 5}" y="{y(v) + 4:.1f}" font-size="12" fill="#9a8a8a" text-anchor="end">{v}</text>'
        for v in (0, 50, 100))
    ticks = "".join(
        f'<text x="{x(i):.1f}" y="{height - 1}" font-size="12" fill="#9a8a8a" text-anchor="middle">{escape(str(breadth[i]["date"])[5:])}</text>'
        for i in sorted({0, n // 2, n - 1}))
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" style="display:block;height:auto" role="img" aria-label="多空占比走勢">{grid}{ticks}'
            f'<polyline points="{line(_BULL)}" fill="none" stroke="#c62828" stroke-width="2.4" stroke-linejoin="round"/>'
            f'<polyline points="{line(_BEAR)}" fill="none" stroke="#16803c" stroke-width="2.4" stroke-linejoin="round"/></svg>')


def _theme_stats(stocks: list[dict[str, Any]], rotation: dict[str, Any]) -> list[dict[str, Any]]:
    quadrants = {str(g.get("name")): str(g.get("quadrant") or "") for g in rotation.get("groups", []) if isinstance(g, dict)}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in stocks:
        for theme in row.get("themes") or []:
            groups.setdefault(str(theme), []).append(row)
    out = []
    for theme, rows in groups.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[str(row.get("classification"))] = counts.get(str(row.get("classification")), 0) + 1
        bull = sum(counts.get(c, 0) for c in _BULL)
        bear = sum(counts.get(c, 0) for c in _BEAR)
        out.append({"theme": theme, "label": ROTATION_DISPLAY_NAMES.get(theme, theme), "quadrant": quadrants.get(theme, ""),
                    "n": len(rows), "counts": counts, "net": (bull - bear) / len(rows) * 100})
    return sorted(out, key=lambda r: (-r["net"], r["label"]))


def _render_brief(stocks: list[dict[str, Any]], intel: dict[str, Any], themes: list[dict[str, Any]]) -> None:
    _sec("01｜今日型態情報")
    counts: dict[str, int] = {}
    for row in stocks:
        counts[str(row.get("classification"))] = counts.get(str(row.get("classification")), 0) + 1
    total = sum(counts.get(c, 0) for c, _, _ in _BIAS) or 1
    bull = sum(counts.get(c, 0) for c in _BULL)
    bear = sum(counts.get(c, 0) for c in _BEAR)
    sentences = [f"追蹤 {total} 檔概念股，<b>轉弱與偏弱 {bear} 檔（{bear / total * 100:.0f}%）</b>、<b>偏多與轉強候選 {bull} 檔（{bull / total * 100:.0f}%）</b>。"]
    breadth = intel.get("breadth") or []
    if len(breadth) > 5:
        diff = _share(breadth[-1], _BULL) - _share(breadth[-6], _BULL)
        sentences.append(f"偏多占比較 5 個交易日前{'增加' if diff >= 0 else '減少'} {abs(diff):.0f} 個百分點。")
    events = intel.get("events") or []
    days = sorted({str(e.get("date")) for e in events})[-_FEED_DAYS:]
    recent = [e for e in events if str(e.get("date")) in days]
    if recent:
        up = sum(1 for e in recent if e.get("event") in _BULL_EVENTS)
        sentences.append(f"近 {len(days)} 個交易日出現 {len(recent)} 次異動，偏多 {up} 次、偏空 {len(recent) - up} 次。")
    if len(themes) >= 4:
        sentences.append(f"型態最強的族群是 <b>{escape(themes[0]['label'])}</b>、{escape(themes[1]['label'])}；最弱的是 <b>{escape(themes[-1]['label'])}</b>、{escape(themes[-2]['label'])}。")
    legend = "".join(f'<span><i style="background:{color}"></i>{escape(label)}<b>{counts.get(c, 0)}</b></span>' for c, label, color in _BIAS if counts.get(c))
    chart = ""
    if len(breadth) >= 2:
        chart = (f'<div class="tp-chart-label">近 {len(breadth)} 個交易日占比（%）：<em class="tp-up">偏多＋轉強候選</em>、<em class="tp-down">轉弱警示＋偏弱</em></div>'
                 + _breadth_svg(breadth))
    st.markdown(f'<div class="tp-brief">{"".join(sentences)}</div>{_stack_html(counts)}<div class="tp-legend">{legend}</div>{chart}', unsafe_allow_html=True)


def _render_feed(intel: dict[str, Any]) -> None:
    _sec("02｜型態異動快訊")
    events = intel.get("events") or []
    if not events:
        st.info("下一次型態更新後會出現異動資料。")
        return
    days = sorted({str(e.get("date")) for e in events})[-_FEED_DAYS:]
    recent = [e for e in events if str(e.get("date")) in days]

    @st.fragment
    def feed() -> None:
        side = st.radio("異動方向", ["全部", "偏多", "偏空"], horizontal=True, key="trend_feed_side", label_visibility="collapsed")
        rows = [e for e in recent if side == "全部" or (e.get("event") in _BULL_EVENTS) == (side == "偏多")]
        summary: dict[str, int] = {}
        for e in rows:
            summary[str(e.get("event"))] = summary.get(str(e.get("event")), 0) + 1
        chips = "".join(f'<span class="tp-chip {"up" if name in _BULL_EVENTS else "down"}">{escape(name)} {count}</span>'
                        for name, count in sorted(summary.items(), key=lambda item: -item[1]))
        # 同一天同一種異動合併成一列：大跌日一次上百檔，逐檔列出只會看到同一個標籤
        watch = {clean_symbol(c) for c in st.session_state.get("watchlist", [])}
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for e in rows:
            groups.setdefault((str(e.get("date")), str(e.get("event"))), []).append(e)
        lines = ""
        for (day, name), items in sorted(groups.items(), key=lambda item: (item[0][0], len(item[1])), reverse=True):
            items = sorted(items, key=lambda e: clean_symbol(str(e.get("symbol"))) not in watch)
            names = "、".join(
                f"<b>{escape(str(e.get('name')))}</b>" if clean_symbol(str(e.get("symbol"))) in watch else escape(str(e.get("name")))
                for e in items[:_FEED_NAMES])
            more = f'<span class="tp-muted"> 等 {len(items)} 檔</span>' if len(items) > _FEED_NAMES else ""
            since = float(np.mean([float(e.get("since_pct") or 0) for e in items]))
            lines += (f'<div class="tp-feed-row"><span class="tp-muted">{escape(day[5:])}</span>'
                      f'<span><span class="tp-chip {"up" if name in _BULL_EVENTS else "down"}">{escape(name)} {len(items)}</span></span>'
                      f'<span>{names}{more}</span>'
                      f'<span class="tp-num {_tone(since)}">{_pct(since)}</span></div>')
        header = '<div class="tp-feed-row tp-feed-head"><span>日期</span><span>異動</span><span>股票（粗體是目前自選股）</span><span class="tp-num">至今平均</span></div>'
        st.markdown(f'<div class="tp-chips">{chips}</div><div class="tp-feed">{header}{lines}</div>', unsafe_allow_html=True)
        st.caption(f"近 {len(days)} 個交易日。「至今平均」是異動當天收盤到最新收盤的平均漲跌（還原股價）；最新一天的異動還沒有後續資料，所以是 0。")

    feed()


def _render_theme_map(themes: list[dict[str, Any]], has_rotation: bool) -> None:
    _sec("03｜族群型態地圖")
    st.caption("每個族群裡的股票各屬於哪種型態，依「（偏多＋轉強候選）−（轉弱警示＋偏弱）」的占比排序。"
               + ("象限來自「產業分析」的輪動圖：型態和資金輪動方向一致時，訊號比較值得留意。" if has_rotation else ""))
    quadrant_tone = {"領先": "up", "改善": "up", "轉弱": "down", "落後": "down"}
    rows = "".join(
        f'<div class="tp-map-row"><span><b>{escape(r["label"])}</b></span>'
        f'<span>{f"""<span class="tp-chip {quadrant_tone.get(r["quadrant"], "")}">{escape(r["quadrant"])}</span>""" if r["quadrant"] else ""}</span>'
        f'<span class="tp-muted tp-num">{r["n"]} 檔</span>{_stack_html(r["counts"])}'
        f'<span class="tp-num {_tone(r["net"])}">{r["net"]:+.0f}</span></div>'
        for r in themes)
    legend = "".join(f'<span><i style="background:{color}"></i>{escape(label)}</span>' for c, label, color in _BIAS)
    st.markdown(f'<div class="tp-map">{rows}</div><div class="tp-legend">{legend}</div>', unsafe_allow_html=True)


def _radar_html(rows: list[dict[str, Any]], key: str) -> str:
    html = ""
    for row in rows:
        high, low, close = float(row["prior_high_20d"]), float(row["prior_low_20d"]), float(row["close_at_assessment"])
        position = max(0.0, min(100.0, (close - low) / max(high - low, 1e-9) * 100))
        level = high if key == "high" else low
        html += (f'<div class="tp-radar-row"><div class="tp-radar-line"><span><b>{escape(str(row.get("name")))}</b> <span class="tp-muted">{escape(str(row.get("symbol")))}</span></span>'
                 f'<span class="tp-num {"tp-up" if key == "high" else "tp-down"}">距 {abs(row["_distance"]):.1f}%</span></div>'
                 f'<div class="tp-radar-line tp-muted"><span>收盤 {close:g}｜{"前高" if key == "high" else "前低"} {level:g}</span><span>{escape(str(row.get("pattern_name")))}</span></div>'
                 f'<div class="tp-range"><i style="left:{position:.1f}%"></i></div></div>')
    return html or '<div class="tp-muted">目前沒有股票在這個範圍內。</div>'


def _render_radar(stocks: list[dict[str, Any]]) -> None:
    _sec("04｜關鍵價雷達")
    st.caption(f"收盤距離前 20 個交易日高點或低點 {_RADAR_DISTANCE_PCT:g}% 以內的股票，方便先知道哪些股票接近關鍵價。下方小條是收盤在前 20 日區間的位置。接近不代表一定會突破或跌破。")
    near_high, near_low = [], []
    for row in stocks:
        try:
            close, high, low = float(row["close_at_assessment"]), float(row["prior_high_20d"]), float(row["prior_low_20d"])
        except (KeyError, TypeError, ValueError):
            continue
        to_high, to_low = (high / close - 1) * 100, (close / low - 1) * 100
        if 0 <= to_high <= _RADAR_DISTANCE_PCT:
            near_high.append({**row, "_distance": to_high})
        if 0 <= to_low <= _RADAR_DISTANCE_PCT:
            near_low.append({**row, "_distance": to_low})
    near_high.sort(key=lambda r: r["_distance"])
    near_low.sort(key=lambda r: r["_distance"])
    left, right = st.columns(2, gap="medium")
    left.markdown(f'<div class="tp-radar-title">接近前高｜{len(near_high)} 檔</div>{_radar_html(near_high[:_RADAR_LIMIT], "high")}', unsafe_allow_html=True)
    right.markdown(f'<div class="tp-radar-title">接近前低｜{len(near_low)} 檔</div>{_radar_html(near_low[:_RADAR_LIMIT], "low")}', unsafe_allow_html=True)


def _forward_table(rows: list[dict[str, Any]], key: str, baseline: dict[str, Any]) -> str:
    cell = lambda r, name: f'<td class="{_tone(r.get(name))}">{_pct(r.get(name), 2)}</td>'
    win = lambda r, name: f'<td>{float(r[name]):.0f}%</td>' if r.get(name) is not None else "<td>—</td>"
    line = lambda r, label, css="": (f'<tr class="{css}"><td>{escape(label)}</td><td>{int(r.get("samples", 0))}</td>'
                                     f'{cell(r, "avg_10d")}{win(r, "win_10d")}{cell(r, "excess_10d")}{cell(r, "avg_20d")}{win(r, "win_20d")}{cell(r, "excess_20d")}</tr>')
    head = ('<tr><th>分類</th><th>樣本</th><th>10 日平均</th><th>10 日上漲比例</th><th>10 日超額</th>'
            '<th>20 日平均</th><th>20 日上漲比例</th><th>20 日超額</th></tr>')
    body = "".join(line(r, str(r.get(key))) for r in rows)
    return f'<table class="tp-table">{head}{body}{line(baseline, "全部樣本", "base") if baseline else ""}</table>'


def _render_forward(intel: dict[str, Any], base: Path) -> None:
    _sec("05｜型態出現後的表現")
    forward = intel.get("forward") or {}
    rows = forward.get("by_classification") or []
    if not rows:
        st.info("下一次型態更新後會出現統計。")
    else:
        order = {c: i for i, (c, _, _) in enumerate(_BIAS)}
        rows = sorted(rows, key=lambda r: order.get(str(r.get("classification")), 99))
        excess = [float(r["excess_20d"]) for r in rows if r.get("excess_20d") is not None]
        if excess:
            spread = max(excess) - min(excess)
            verdict = "各分類差距很小，型態適合用來閱讀盤面，不適合單獨當成進出依據。" if spread <= 3 else "分類之間有一些差距，但樣本期間短，仍不宜單獨當成進出依據。"
            st.markdown(f'<div class="tp-brief">這些分類過去準不準？訊號出現後 20 個交易日，各分類相對全部樣本的超額報酬介於 <b>{min(excess):+.1f}%～{max(excess):+.1f}%</b>。{verdict}</div>', unsafe_allow_html=True)
        st.markdown(_forward_table(rows, "classification", forward.get("baseline") or {}), unsafe_allow_html=True)
        st.caption(f"{forward.get('start')}～{forward.get('end')}，{forward.get('symbols')} 檔，每 {forward.get('sample_step_days')} 個交易日取一次樣本。"
                   "超額＝減去同一天全部樣本的平均，扣掉大盤漲跌的影響。"
                   "追蹤名單是今天選出的概念股，有存活者偏差，期間也只涵蓋單一段行情；同一檔股票的樣本彼此相關。歷史統計，不構成投資建議。")
        with st.expander("依型態細分"):
            detail = sorted((r for r in forward.get("by_pattern") or [] if int(r.get("samples", 0)) >= 30), key=lambda r: -int(r.get("samples", 0)))
            st.markdown(_forward_table(detail, "pattern_name", {}), unsafe_allow_html=True)
            st.caption("只列樣本 30 筆以上的型態。")
    with st.expander("型態形成期間的漲跌（第 80～41 個交易日前）"):
        _render_backtest(base)


def _spark_svg(values: Any) -> str:
    if not isinstance(values, list) or len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    width, height = 200, 38
    points = " ".join(f"{i * width / (len(values) - 1):.1f},{2 + (high - v) / span * (height - 4):.1f}" for i, v in enumerate(values))
    color = "#c62828" if values[-1] >= values[0] else "#16803c"
    return f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>'


def _stock_card(row: dict[str, Any]) -> str:
    colors = {c: (label, color) for c, label, color in _BIAS}
    label, color = colors.get(str(row.get("classification")), ("待確認", "#b9c2cb"))
    return (f'<div class="tp-card"><div class="tp-card-top"><span class="tp-card-name">{escape(str(row.get("name")))}</span><span class="tp-muted">{escape(str(row.get("symbol")))}</span></div>'
            f'{_spark_svg(row.get("spark"))}'
            f'<div class="tp-card-foot"><span class="tp-chip" style="border-left:3px solid {color}">{escape(str(row.get("pattern_name")))}</span>'
            f'<span class="tp-muted">{escape(label)}</span></div>'
            f'<div class="tp-card-foot" style="margin-top:6px"><span>3 日 <b class="{_tone(row.get("return_3d_pct"))}">{_pct(row.get("return_3d_pct"))}</b></span>'
            f'<span>20 日 <b class="{_tone(row.get("return_20d_pct"))}">{_pct(row.get("return_20d_pct"))}</b></span></div></div>')


def _render_stock_detail(row: dict[str, Any]) -> None:
    symbol = str(row.get("symbol"))
    pattern_name = str(row.get("pattern_name") or "盤整待突破")
    themes = row.get("themes") if isinstance(row.get("themes"), list) else []
    facts = [
        ("型態", pattern_name), ("分類", row.get("classification", "—")),
        ("收盤", row.get("close_at_assessment", "—")), ("前 20 日高／低", f"{row.get('prior_high_20d', '—')}／{row.get('prior_low_20d', '—')}"),
        ("近 3 日", _pct(row.get("return_3d_pct"))), ("近 40 日", _pct(row.get("return_40d_pct"))),
        ("近 40 日趨勢", row.get("trend_40d", "—")), ("近 6 個月背景", row.get("market_direction_6m", "—")),
        ("規則信心", f"{float(row.get('confidence') or 0):.0%}"), ("型態支撐／壓力", f"{row.get('support', '—')}／{row.get('resistance', '—')}"),
    ]
    grid = "".join(f"<div><b>{escape(k)}：</b>{escape(str(v))}</div>" for k, v in facts)
    left, right = st.columns([1.1, 2.4])
    left.markdown(_schematic_svg(pattern_name), unsafe_allow_html=True)
    right.markdown(
        f'<div class="tp-facts">{grid}</div>\n\n'
        f"**判讀：** {row.get('reason', '目前沒有額外說明。')}  \n"
        f"**族群：** {'、'.join(themes) if themes else '未分類'}  \n"
        f"<small>最近重大變化：{escape(str(row.get('major_change', '—')))}（{escape(str(row.get('last_major_change_at', '—'))[:10])}）｜"
        f"判讀 K 棒：{escape(str(row.get('last_bar_date', '—')))}｜{escape(str(row.get('adjustment_method', '')))}</small>",
        unsafe_allow_html=True,
    )
    if bool(row.get("corporate_action_review", False)):
        st.warning(f"此股於 {row.get('suspected_gap_date', '未知日期')} 出現 {row.get('suspected_gap_pct', '未知')}% 價格跳空，已暫停趨勢型態判斷。")
    _render_live_chart(symbol)


def _render_explorer(stocks: list[dict[str, Any]]) -> None:
    _sec("06｜個股型態總覽")

    @st.fragment
    def explorer() -> None:
        theme_labels = {ROTATION_DISPLAY_NAMES.get(t, t): t for t in sorted({t for r in stocks for t in (r.get("themes") or [])})}
        class_labels = {label: c for c, label, _ in _BIAS if any(r.get("classification") == c for r in stocks)}
        if any(r.get("classification") == "資料調整檢查" for r in stocks):
            class_labels["待確認"] = "資料調整檢查"
        first, second, third, fourth = st.columns([1.5, 1.5, 1, 0.8], vertical_alignment="bottom")
        chosen_themes = first.multiselect("產業主題", list(theme_labels), placeholder="全部主題", key="trend_explorer_themes")
        chosen_classes = second.multiselect("型態分類", list(class_labels), placeholder="全部分類", key="trend_explorer_classes")
        sort_name = third.selectbox("排序", ["近 3 日漲幅", "近 3 日跌幅", "近 20 日漲幅", "近 20 日跌幅", "代號"], key="trend_explorer_sort")
        only_watch = fourth.toggle("只看自選股", key="trend_explorer_watch")

        rows = stocks
        if chosen_themes:
            wanted = {theme_labels[t] for t in chosen_themes}
            rows = [r for r in rows if wanted & set(r.get("themes") or [])]
        if chosen_classes:
            wanted = {class_labels[c] for c in chosen_classes}
            rows = [r for r in rows if r.get("classification") in wanted]
        if only_watch:
            watch = {clean_symbol(c) for c in st.session_state.get("watchlist", [])}
            rows = [r for r in rows if clean_symbol(str(r.get("symbol"))) in watch]
        field, reverse = {"近 3 日漲幅": ("return_3d_pct", True), "近 3 日跌幅": ("return_3d_pct", False),
                          "近 20 日漲幅": ("return_20d_pct", True), "近 20 日跌幅": ("return_20d_pct", False), "代號": ("symbol", False)}[sort_name]
        if field == "symbol":
            rows = sorted(rows, key=lambda r: str(r.get("symbol")))
        else:
            rows = sorted(rows, key=lambda r: float(r.get(field) or 0), reverse=reverse)
        if not rows:
            st.info("目前篩選條件沒有股票。")
            return

        by_symbol = {str(r.get("symbol")): r for r in rows}
        picked = st.selectbox(f"查看個股型態與近六個月 K 線（{len(rows)} 檔）", [""] + list(by_symbol), key="trend_explorer_pick",
                              format_func=lambda code: "選擇股票…" if not code else f"{by_symbol[code].get('name')}（{code}）")
        if picked:
            _render_stock_detail(by_symbol[picked])
        st.markdown('<div class="tp-grid">' + "".join(_stock_card(r) for r in rows) + "</div>", unsafe_allow_html=True)

    explorer()


def render_trend_patterns_workspace(base_dir: str | Path) -> None:
    """市場情報頁：先看整體多空與異動，再看族群、關鍵價，最後才到個股。"""
    base = Path(base_dir)
    payload = load_published_trend_patterns(base)
    raw_stocks = payload.get("stocks", []) if isinstance(payload, dict) else []
    st.markdown(
        '<div class="content-heading"><div class="eyebrow">TREND PATTERN ANALYSIS</div>'
        '<h1>趨勢型態</h1><p>用 K 線型態讀盤面：今天市場偏多還是偏空、哪些股票出現異動、哪些族群最強或最弱、誰正接近關鍵價。</p></div>',
        unsafe_allow_html=True,
    )
    stocks = [row for row in raw_stocks if isinstance(row, dict) and row.get("symbol")]
    if not stocks:
        st.info("目前尚無公開型態快照。先在本機執行 python run_trend_pattern_update.py --force 建立初始快照，確認結果後即可發布到網站。")
        return
    for row in stocks:
        row["symbol"] = clean_symbol(str(row["symbol"]))
    intel = payload.get("market_intel") if isinstance(payload.get("market_intel"), dict) else {}
    rotation = load_published_rotation(base) or {}
    themes = _theme_stats(stocks, rotation)

    st.markdown(_TREND_CSS, unsafe_allow_html=True)
    try:
        data_age = (pd.Timestamp(datetime.now(ZoneInfo("Asia/Taipei")).date()) - pd.Timestamp(payload.get("data_asof"))).days
    except (TypeError, ValueError):
        data_age = 0
    if data_age >= STALE_DATA_DAYS:
        st.warning(f"型態資料停在 {payload.get('data_asof')}，已經 {data_age} 天沒有更新，下方內容不代表目前盤面。")
    st.caption(f"資料日期 {payload.get('data_asof') or '未知'}｜涵蓋概念股 {int(payload.get('tracked_symbols', len(stocks)) or len(stocks))} 檔｜"
               "型態由程式規則估算，可能失效，不構成投資建議。")
    _render_brief(stocks, intel, themes)
    _render_feed(intel)
    _render_theme_map(themes, bool(rotation.get("groups")))
    _render_radar(stocks)
    _render_forward(intel, base)
    _render_explorer(stocks)

    with st.expander("型態圖鑑"):
        guide_url = str(payload.get("pattern_guide_url") or "https://www.ig.com/cn/trading-strategies/10-chart-patterns-every-trader-needs-to-know-220907")
        st.markdown(f"型態判讀參考：[IG｜10 種常見圖表形態]({guide_url})。下列為概念示意，不是實際股價。")
        present: dict[str, int] = {}
        for row in stocks:
            present[_guide_key(str(row.get("pattern_name") or ""))] = present.get(_guide_key(str(row.get("pattern_name") or "")), 0) + 1
        columns = st.columns(3)
        for index, key in enumerate(PATTERN_GUIDES):
            with columns[index % 3]:
                st.markdown(_schematic_svg(key), unsafe_allow_html=True)
                st.caption(f"{PATTERN_GUIDES[key]['note']}｜目前 {present.get(key, 0)} 檔")

    with st.expander("分析方法與更新規則"):
        st.markdown(
            """
            **資料範圍：** 產業分析 ROTATION_THEMES 的全部概念股；同一檔股票屬於多個主題時會保留全部主題。

            **價格基準：** 趨勢分析與個股 K 棒均使用 Fugle `adjusted=True` 還原股價，避免股票分割、股票股利與除權息造成的價格斷層。還原後若仍出現 35% 以上單日跳空，會放入「資料調整檢查」，暫停方向判斷。

            **時間尺度：** 最近 40 個交易日日 K 辨識目前圖表型態與 3／10／20／40 日報酬；近 6 個月資料判斷高低位置與背景方向。查看個股時顯示近六個月 OHLC K 棒、成交量，並可切換 MA5、MA10、MA20、MA60、MA120、MA240。

            **雙底／雙頂：** 要有前段 8% 以上的跌勢（雙頂是漲勢）、第二個低點（高點）在最近 15 根內，且之後沒有破底（過頂）。兩者都成立時取較新的。

            **異動快訊：** 箱型／三角形等壓力突破、跌破前 20 日低點、連跌後三日轉穩、上漲後三日轉弱，以及雙底、倒頭肩、雙頂、頭肩頂的確認。每天用同一套規則逐日重算。

            **型態出現後的表現：** 只用訊號當天之後的收盤計算，不會用到未來資料判斷型態。

            **空值處理：** 圖表只使用 OHLC 齊全的真實交易日，日期軸採交易日類別序列，因此週末、國定假日與缺失行情不會形成空白日期。

            **型態限制：** 自動規則是近似判斷，無法完全取代人眼辨識；形態可能失敗，假突破也可能發生，請勿單憑型態交易。
            """
        )
