# -*- coding: utf-8 -*-
"""Streamlit Community Cloud entry point.

Cloud Secrets is loaded before importing stock_dashboard so stock_api.py
can continue to read FUGLE_API_KEY from the environment.
"""
from __future__ import annotations

import os
import streamlit as st

try:
    fugle_key = st.secrets.get("FUGLE_API_KEY")
    if fugle_key:
        os.environ["FUGLE_API_KEY"] = str(fugle_key)
except Exception:
    pass

import stock_dashboard  # noqa: E402,F401
