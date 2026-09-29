# -*- coding: utf-8 -*-
"""Interactive market intelligence panel for the Taiwan stock dashboard.

This module reads the most recent local research-agent report and presents
interactive sector/news intelligence without polling external news sources
on every dashboard refresh.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from research_agent import ResearchAgent
from business_master import business_map_for_symbols, build_business_research_context
from stock_api import INDUSTRIES, INDUSTRY_OVERRIDES, clean_symbol
from post_market_analysis import PostMarketAnalyzer


def _latest_report(output_dir: Path) -> dict[str, Any] | None:
    report_dir = output_dir / "research_reports"
    if not report_dir.exists():
        return None
    files = sorted(report_dir.glob("research_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def _industry_for_symbol(symbol: str, base_dir: Path) -> str:
    symbol = clean_symbol(symbol)
    override = INDUSTRY_OVERRIDES.get(symbol)
    if override:
        return str(override).zfill(2)
    path = base_dir / "data" / "industry_master.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, dtype={"symbol": str, "industry": str})
            df["symbol"] = df["symbol"].map(clean_symbol)
            row = df[df["symbol"] == symbol]
            if not row.empty:
                return str(row.iloc[0]["industry"]).zfill(2)
        except Exception:
            pass
    return "00"


def _label(score: float) -> str:
    if pd.isna(score):
        return "中性 / 待判定"
    score = float(score)
    if score >= 0.20:
        return "利多"
    if score <= -0.20:
        return "利空"
    return "中性"


def _verification_class(text: str) -> str:
    text = str(text or "")
    if text.startswith("高可信"):
        return "高可信"
    if text.startswith("較可信"):
        return "較可信"
    if text.startswith("中度可信"):
        return "中度可信"
    return "待查證"


def _prepare_news(report: dict[str, Any]) -> pd.DataFrame:
    rows = report.get("news", []) or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    df["impact"] = df["sentiment_score"].map(_label)
    if "verification" not in df.columns:
        df["verification"] = "待查證"
    df["verification_level"] = df["verification"].map(_verification_class)
    if "published" in df.columns:
        df["published_dt"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    else:
        df["published_dt"] = pd.NaT
    return df


def _selected_news(df: pd.DataFrame, symbol: str, industry_code: str) -> pd.DataFrame:
    if df.empty:
        return df
    symbol = clean_symbol(symbol)
    industry_code = str(industry_code).zfill(2)
    sector_name = INDUSTRIES.get(industry_code, "")
    code_text = symbol
    sector_terms = sector_name.replace("業", "").split()
    text = (df.get("title", "").astype(str) + " " + df.get("summary", "").astype(str)).str.lower()
    mask_symbol = text.str.contains(code_text.lower(), regex=False, na=False)
    mask_sector = pd.Series(False, index=df.index)
    for term in sector_terms:
        term = str(term).strip()
        if len(term) >= 2:
            mask_sector = mask_sector | text.str.contains(term.lower(), regex=False, na=False)
    out = df[mask_symbol | mask_sector].copy()
    return out


def render_market_intelligence(base_dir: Path, output_dir: Path, selected_symbol: str | None = None, selected_industry: str = "00", blocked: bool = False) -> None:
    """Render interactive sector/news intelligence using saved research data."""
    st.markdown("## 市場情報與利多 / 利空分析")
    st.caption(
        "盤後研究 Agent 蒐集新聞、官方重大訊息、營收與公開供應鏈資訊；"
        "以下圖表只使用已保存的研究報告，不會因每 2 秒行情刷新而重複抓新聞。"
    )

    if blocked:
        st.info("目前正在執行優先 AI 預測任務；市場研究區暫停更新，預測完成後恢復。")
        return

    c1, c2, c3 = st.columns([1.1, 1.1, 3.8])
    with c1:
        run_now = st.button("更新市場情報", type="secondary", use_container_width=True, key="refresh_market_intelligence")
    with c2:
        load_latest = st.button("載入最新報告", use_container_width=True, key="load_market_intelligence")
    with c3:
        latest_hint = _latest_report(output_dir)
        if latest_hint:
            st.caption(f"最新研究日期：{latest_hint.get('report_date', '未知')}｜來源：本機研究 Agent")
        else:
            st.caption("尚無已保存研究報告；首次執行可建立市場情報資料。")

    if run_now:
        try:
            symbol = clean_symbol(selected_symbol or "")
            industry_code = str(selected_industry or _industry_for_symbol(symbol, base_dir)).zfill(2)
            with st.spinner("本機 Qwen3 / Research Agent 正在更新市場情報…"):
                agent = ResearchAgent(base_dir, ollama_model="qwen3:8b")
                report = agent.run_daily_research(symbols=[symbol] if symbol else [], sector_codes=[industry_code] if industry_code != "00" else None)
            st.session_state["market_intelligence_report"] = report
        except Exception as exc:
            st.error(f"市場情報更新失敗：{exc}")

    if load_latest or st.session_state.get("market_intelligence_report") is None:
        latest = _latest_report(output_dir)
        if latest:
            st.session_state["market_intelligence_report"] = latest

    selected_symbol = clean_symbol(selected_symbol or st.session_state.get("selected") or "")
    st.session_state["selected_symbol"] = selected_symbol

    report = st.session_state.get("market_intelligence_report")
    if not isinstance(report, dict):
        st.info("尚無市場情報報告。請在盤後研究時段執行「更新市場情報」。")
        return

    news_df = _prepare_news(report)
    sector_df = pd.DataFrame(report.get("sector_news", []) or [])

    relevant = _selected_news(news_df, selected_symbol or "", str(selected_industry or "00")) if selected_symbol else news_df
    if relevant.empty:
        relevant = news_df

    positive = int((relevant["sentiment_score"] >= 0.20).sum()) if not relevant.empty else 0
    negative = int((relevant["sentiment_score"] <= -0.20).sum()) if not relevant.empty else 0
    neutral = max(0, len(relevant) - positive - negative)
    official = int(relevant["official_match"].fillna(False).astype(bool).sum()) if (not relevant.empty and "official_match" in relevant.columns) else 0
    avg_sent = float(relevant["sentiment_score"].mean()) if not relevant.empty else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("相關新聞", f"{len(relevant)}")
    m2.metric("利多 / 利空", f"{positive} / {negative}")
    m3.metric("官方佐證", f"{official}")
    m4.metric("平均消息情緒", f"{avg_sent:+.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(["產業利多 / 利空", "新聞查證", "目前股票相關消息", "今日漲跌原因"])

    with tab1:
        if sector_df.empty:
            st.info("目前沒有產業情緒資料。")
        else:
            plot = sector_df.copy()
            plot["weighted_sentiment"] = pd.to_numeric(plot["weighted_sentiment"], errors="coerce").fillna(0.0)
            plot = plot.sort_values("weighted_sentiment")
            fig = px.bar(
                plot,
                x="weighted_sentiment",
                y="name",
                orientation="h",
                hover_data=["code", "news_count", "confidence"],
                title="各產業消息情緒",
            )
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(height=540, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="加權消息情緒", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            sector_rows = plot.copy()
            sector_rows["訊息方向"] = sector_rows["weighted_sentiment"].map(_label)
            sector_rows["證據信心"] = sector_rows["confidence"].map(lambda x: f"{float(x):.0%}")
            st.dataframe(
                sector_rows[["code", "name", "訊息方向", "weighted_sentiment", "news_count", "證據信心"]],
                use_container_width=True,
                hide_index=True,
            )

    with tab2:
        if relevant.empty:
            st.info("目前沒有新聞資料。")
        else:
            chart = relevant.dropna(subset=["published_dt"]).copy()
            if not chart.empty:
                fig = px.scatter(
                    chart,
                    x="published_dt",
                    y="sentiment_score",
                    color="impact",
                    size="corroboration" if "corroboration" in chart.columns else None,
                    hover_name="title",
                    hover_data=["source", "verification_level"],
                    title="新聞時間 × 利多 / 利空情緒",
                )
                fig.add_hline(y=0, line_dash="dash")
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="消息情緒", xaxis_title="發布時間")
                st.plotly_chart(fig, use_container_width=True)

            cols = [c for c in ["title", "source", "published", "impact", "sentiment_score", "corroboration", "official_match", "verification", "link"] if c in relevant.columns]
            st.dataframe(relevant[cols].head(80), use_container_width=True, hide_index=True)

    with tab3:
        if not selected_symbol:
            st.info("目前沒有選取股票。")
        else:
            sym = clean_symbol(selected_symbol)
            st.markdown(f"**{sym}**｜產業：{INDUSTRIES.get(str(selected_industry).zfill(2), '未知產業')} ({str(selected_industry).zfill(2)})")
            # 顯示「這家公司到底做什麼」以及 Agent 實際使用的研究關鍵字，讓使用者知道新聞為何被選進來。
            try:
                profiles = business_map_for_symbols([sym], path=base_dir / "data" / "business_master.csv", force=False)
                profile = profiles.get(sym, {})
                ctx = build_business_research_context(profile)
            except Exception:
                ctx = {}
            if ctx:
                b1, b2 = st.columns([1.1, 2.9])
                with b1:
                    st.markdown("**主要業務**")
                    st.write(ctx.get("business_group") or "待建立")
                with b2:
                    st.markdown("**產業鏈位置**")
                    st.write(ctx.get("primary_chain") or "待建立")
                kws = ctx.get("research_keywords") or []
                if kws:
                    st.caption("Agent 研究關鍵字：" + "、".join(kws[:12]))
            company_rows = []
            for item in report.get("company_research", []) or []:
                if clean_symbol(item.get("symbol", "")) == sym:
                    company_rows.append(item)
            if company_rows:
                cr = company_rows[0]
                with st.expander("Agent 為本公司建立的研究問題", expanded=True):
                    for q in cr.get("query_plan", [])[:5]:
                        st.write("• " + str(q))
                    if cr.get("positive_evidence"):
                        st.markdown("**主要利多證據**")
                        for n in cr.get("positive_evidence", [])[:5]:
                            st.write(f"• {n.get('title','')}｜{n.get('verification','待查證')}")
                    if cr.get("negative_evidence"):
                        st.markdown("**主要利空證據**")
                        for n in cr.get("negative_evidence", [])[:5]:
                            st.write(f"• {n.get('title','')}｜{n.get('verification','待查證')}")
            stock_news = _selected_news(news_df, selected_symbol, selected_industry)
            if stock_news.empty:
                st.info("最新研究報告尚未找到與此股票直接相關的新聞；可手動更新市場情報。")
            else:
                for _, row in stock_news.head(12).iterrows():
                    title = str(row.get("title", ""))
                    impact = str(row.get("impact", "中性"))
                    verification = str(row.get("verification", "待查證"))
                    relevance = float(pd.to_numeric(row.get("business_relevance", 0), errors="coerce") or 0.0)
                    score = float(pd.to_numeric(row.get("sentiment_score", 0), errors="coerce") or 0.0)
                    st.markdown(f"**{impact}｜情緒 {score:+.2f}｜業務相關度 {relevance:.0%}｜{verification}**")
                    st.write(title)
                    link = str(row.get("link", "")).strip()
                    if link:
                        st.markdown(f"[查看原始來源]({link})")
                    st.divider()

    with tab4:
        if not selected_symbol:
            st.warning("目前沒有選取股票；請先在自選股或搜尋區選取股票。")
        else:
            st.markdown(f"### {selected_symbol}｜盤後今日漲跌原因研究")
            st.caption("系統先描述可量化的價格、TAIEX、同細分業務群與法人資料，再依公司主要業務查核財報、營收、公告、主管/董事公開談話、接單、供應鏈與國際事件；每則消息都標示證據狀態，不把相關性直接當因果。")
            c1, c2, c3 = st.columns([1.1, 1.1, 4.0])
            with c1:
                run_post = st.button("分析今日漲跌原因", type="primary", use_container_width=True, key=f"run_post_{selected_symbol}")
            with c2:
                load_post = st.button("載入已保存", use_container_width=True, key=f"load_post_{selected_symbol}")
            with c3:
                st.caption("盤後使用；盤中可手動查看最近一次研究，不會每 2 秒抓新聞。")

            analyzer = PostMarketAnalyzer(base_dir, ollama_model=st.session_state.get("ollama_model", "qwen3:8b"))
            if run_post:
                try:
                    watch = st.session_state.get("watchlist", [selected_symbol])
                    with st.spinner("本機 Qwen3 正在查核今日漲跌原因：價格／大盤／同業／法人／財報／公告／供應鏈／國際事件…"):
                        post = analyzer.analyze(selected_symbol, industry_symbols=watch)
                    st.session_state[f"post_market_{selected_symbol}"] = post
                except Exception as exc:
                    st.error(f"盤後漲跌原因分析失敗：{exc}")
            if load_post or st.session_state.get(f"post_market_{selected_symbol}") is None:
                cached_post = analyzer.load(selected_symbol)
                if cached_post:
                    st.session_state[f"post_market_{selected_symbol}"] = cached_post

            post = st.session_state.get(f"post_market_{selected_symbol}")
            if not isinstance(post, dict):
                st.info("目前尚無此股票的盤後漲跌原因研究。按「分析今日漲跌原因」建立第一份報告。")
            else:
                move = post.get("price_context", {}) or {}
                inst = post.get("institution_context", {}) or {}
                vals = st.columns(5)
                price = move.get("current_price")
                pct = move.get("today_change_percent")
                taiex_idx = (move.get("taiex") or {}).get("index")
                peer = move.get("peer_median_change_percent")
                foreign = (inst.get("totals_10d") or {}).get("foreign_net")
                vals[0].metric("今日價格", "--" if price is None else f"{float(price):,.2f}")
                vals[1].metric("今日漲跌幅", "--" if pct is None else f"{float(pct):+.2f}%")
                vals[2].metric("TAIEX", "--" if taiex_idx is None else f"{float(taiex_idx):,.2f}")
                vals[3].metric("同業中位數", "--" if peer is None else f"{float(peer):+.2f}%")
                vals[4].metric("外資10日買賣超", "--" if foreign is None else f"{float(foreign):,.0f}")
                st.markdown(f"**主要業務：** {post.get('business_group') or '待建立'}　｜　**產業鏈位置：** {post.get('primary_chain') or '待建立'}")

                reasons = post.get("reason_candidates", []) or []
                if reasons:
                    rows = []
                    for r in reasons[:10]:
                        score = float(r.get("score", 0) or 0)
                        rows.append({"候選驅動因素": r.get("factor"), "方向": r.get("direction"), "證據強度": f"{score*100:.0f}%", "證據/說明": r.get("evidence"), "來源": r.get("source", "")})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                with st.expander("消息逐一查證", expanded=True):
                    news_rows = post.get("verified_news", []) or []
                    if not news_rows:
                        st.info("沒有足夠的公司相關消息；這不代表沒有原因，還要以公告、財報與法人資料判斷。")
                    else:
                        verify_rows = []
                        for n in news_rows[:40]:
                            verify_rows.append({"消息": n.get("title"), "來源": n.get("source"), "業務相關度": f"{float(n.get('business_relevance',0))*100:.0f}%", "跨來源數": int(n.get("corroboration",0) or 0), "官方匹配": "是" if n.get("official_match") else "否", "查證結果": n.get("verification"), "連結": n.get("link")})
                        st.dataframe(pd.DataFrame(verify_rows), use_container_width=True, hide_index=True)

                with st.expander("Qwen3 盤後歸因摘要", expanded=True):
                    llm = post.get("ollama_analysis", {}) or {}
                    if llm.get("available"):
                        st.markdown(llm.get("content") or "未產生摘要。")
                    else:
                        st.warning(llm.get("error", "Qwen3 尚不可用；仍可查看規則式證據。"))

                st.caption("注意：此區是事件歸因研究，不是確定因果判定。單一新聞、缺乏原始公告、與營收/法人/價格反應不一致時，會標示待查證／疑似誇大／市場反應與基本面不一致。")

    st.caption(
        "查證邏輯：官方公告／原始資料優先；多來源一致可提高信心；單一來源或與營收、法人、價格反應不一致時標示待查證。"
        "『疑似假利多／假利空』只作研究標籤，不視為已證實的真偽結論。"
    )
