@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run stock_dashboard.py
