# -*- coding: utf-8 -*-
"""功能區塊卡片：讓每頁的功能／小標題之間有明顯區隔（沿用「操作策略」頁的淺色交錯卡片）。

兩種用法：
  with section_box("fin-global"):   # 包住一段內容
      ...
  section("02｜產業輪動圖")           # 串流式：關掉上一張卡片、開新卡片並放標題，不必改縮排
  section_end()                      # 頁面結束時關掉最後一張（_render_page 會呼叫）

section() 只能在頁面最外層呼叫（不在 with 區塊或 fragment 裡）；偵測到不在最外層時退回成只畫標題。
"""
from __future__ import annotations

import threading
from html import escape

import streamlit as st
from streamlit.delta_generator_singletons import context_dg_stack

PALETTE = 6
_local = threading.local()   # Streamlit 每個 session 的腳本在自己的執行緒跑


def _state() -> dict:
    if not hasattr(_local, "s"):
        _local.s = {"n": 0, "open": None, "depth": None}
    return _local.s


def reset() -> None:
    """每頁開始時呼叫：記下頁面容器的層數，section() 只在這一層開卡片。"""
    _local.s = {"n": 0, "open": None, "depth": len(context_dg_stack.get())}


def section_box(name: str, tone: int | None = None):
    """tone 固定時 key 也固定（給 fragment 裡的區塊用，自動刷新時卡片不重建）。"""
    s = _state()
    s["n"] += 1
    if tone is not None:
        return st.container(key=f"secbox-p{tone % PALETTE}-{name}")
    return st.container(key=f"secbox-p{(s['n'] - 1) % PALETTE}-{name}-{s['n']}")


def section(title: str, *, html: bool = False) -> None:
    s = _state()
    section_end()
    if s["depth"] is not None and len(context_dg_stack.get()) == s["depth"]:
        box = section_box("s")
        box.__enter__()
        s["open"] = box
    st.markdown(f'<div class="feature-section-title">{title if html else escape(title)}</div>', unsafe_allow_html=True)


def section_end() -> None:
    s = _state()
    box = s.get("open")
    if box is not None:
        stack = context_dg_stack.get()
        if stack and stack[-1] is box:
            box.__exit__(None, None, None)
        s["open"] = None


CSS = """<style>
/* 功能區塊卡片：淺色交錯底、圓角、細框，與操作策略頁一致 */
[class*="st-key-secbox-"]{border-radius:14px!important;padding:22px 24px 18px!important;margin:0 0 14px!important;
  border:1px solid rgba(142,43,47,.10)!important;box-shadow:none!important}
[class*="st-key-secbox-p0-"]{background:#fbf1f1!important}
[class*="st-key-secbox-p1-"]{background:#f8f4f2!important}
[class*="st-key-secbox-p2-"]{background:#f9f5ef!important}
[class*="st-key-secbox-p3-"]{background:#f4f7f6!important}
[class*="st-key-secbox-p4-"]{background:#fbf3f3!important}
[class*="st-key-secbox-p5-"]{background:#f5f3f4!important}
[class*="st-key-secbox-"] .feature-section-title{margin-top:0!important}
[class*="st-key-secbox-"] .finance-card-title{margin-top:0!important}
@media(max-width:600px){[class*="st-key-secbox-"]{padding:16px 12px 12px!important}}
</style>"""
