# -*- coding: utf-8 -*-
"""
AI 台股隔日價格預測系統
============================================================
檔案：stock_month_prediction.py

本版本針對先前專案問題重新整理，重點：

1. 3044 預設股票
2. 可輸入單一股票或多股票
3. 多股票共同訓練
4. 模型會把不同股票的歷史特徵放進同一訓練資料集
5. 每次執行重新取得/更新市場資料並重新訓練
6. 預測隔日股價
7. 目標改為 Future 1-Day Return，再換算成價格
8. Random Forest Regression + Classification
9. 5-Fold Time Series Cross Validation
10. OOF (Out-of-Fold) 預測
11. MAE / RMSE / MAPE
12. Accuracy / Precision / Recall / F1
13. OOF 殘差 Q10~Q90 價格區間
14. 歷史滾動回測
15. 可用「前面資料」預測「隔日股價」
16. 技術指標與量價特徵
17. 0050 Benchmark 為可選特徵；0050 下載失敗不會讓主模型停止
18. Fugle Historical Candles 使用正確的 /{symbol} 路徑
19. 330 天分段下載
20. 429 Rate Limit 自動退避重試
21. CSV / PKL / PNG / JSON / TXT
22. 不再使用 month_period
23. 避免歷史回測 MAE/RMSE/MAPE 全部 NaN

重要：
「持續學習」在本專案定義為：
每次重新執行時，把最新取得的所有股票歷史資料重新加入共同訓練集，
再重新訓練模型。這不是線上 incremental learning；但對目前
Random Forest 架構而言，能確實讓新增市場資料影響下一次模型結果。

Fugle v1.0 Historical Candles：
GET /marketdata/v1.0/stock/historical/candles/{symbol}
使用 X-API-KEY 驗證。
"""

import os
import sys
import json
import time
import math
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import joblib

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit


# ============================================================
# 1. 專案設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = OUTPUT_DIR / "error_log.txt"

FUGLE_API_KEY = os.getenv("FUGLE_API_KEY", "").strip()

DEFAULT_STOCKS = ["3044"]
DEFAULT_PREDICT_STOCK = "3044"
BENCHMARK = "0050"

HORIZON = 1  # 隔日：1 個交易日後
DOWNLOAD_CHUNK_DAYS = 330

# 最少有效樣本
MIN_STOCK_SAMPLES = 80
MIN_GLOBAL_SAMPLES = 150
MIN_OOF_SAMPLES = 20

RANDOM_STATE = 42

# Fugle v1.0 正確 endpoint：
# /marketdata/v1.0/stock/historical/candles/{symbol}
FUGLE_HISTORICAL_URL = (
    "https://api.fugle.tw/marketdata/v1.0/"
    "stock/historical/candles"
)


# ============================================================
# 2. 基本工具
# ============================================================

def log_error(message):
    """把錯誤寫入 output/error_log.txt。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{timestamp}] {message}\n"

    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def clean_symbol(symbol):
    """將 3044.TW / 3044 等輸入統一成 Fugle symbol。"""
    s = str(symbol).strip().upper()

    if "." in s:
        s = s.split(".")[0]

    return s


def parse_stock_input(text):
    """解析 3044,2330,2454,0050。"""
    if not text.strip():
        return DEFAULT_STOCKS.copy()

    text = text.replace("，", ",")
    text = text.replace("、", ",")

    result = []

    for item in text.split(","):
        symbol = clean_symbol(item)

        if symbol and symbol not in result:
            result.append(symbol)

    return result or DEFAULT_STOCKS.copy()


def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def calculate_rmse(y_true, y_pred):
    return float(
        math.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def calculate_mape(y_true, y_pred):
    """
    MAPE 排除實際值為 0 的資料。
    """
    a = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(a)
        & np.isfinite(p)
        & (np.abs(a) > 1e-12)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(
                (a[mask] - p[mask])
                / a[mask]
            )
        )
        * 100
    )


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def finite_series(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# 3. Fugle API
# ============================================================

def request_fugle_history(
    symbol,
    start_date,
    end_date,
    retries=5,
):
    """
    呼叫 Fugle v1.0 Historical Candles。

    正確格式：
    GET
    https://api.fugle.tw/marketdata/v1.0/
    stock/historical/candles/{symbol}

    不是：
    /historical/candles?symbol=0050

    因此可修正先前 404：
    Cannot GET /v1.0/stock/historical/candles?symbol=0050
    """

    if not FUGLE_API_KEY:
        raise RuntimeError(
            "FUGLE_API_KEY 未設定。"
        )

    symbol = clean_symbol(symbol)

    url = (
        f"{FUGLE_HISTORICAL_URL}/"
        f"{symbol}"
    )

    params = {
        "from": pd.Timestamp(start_date).strftime(
            "%Y-%m-%d"
        ),
        "to": pd.Timestamp(end_date).strftime(
            "%Y-%m-%d"
        ),
        "timeframe": "D",
        "fields": (
            "open,high,low,close,volume,change"
        ),
        "sort": "asc",
    }

    headers = {
        "X-API-KEY": FUGLE_API_KEY,
        "Accept": "application/json",
    }

    last_error = None

    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=40,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:
                wait_seconds = min(
                    60,
                    2 ** attempt + 2
                )

                message = (
                    f"Fugle HTTP 429：{symbol} "
                    f"{params['from']} ~ {params['to']}，"
                    f"{wait_seconds} 秒後重試"
                )

                print("    " + message)
                log_error(message)

                time.sleep(wait_seconds)
                last_error = response.text
                continue

            if response.status_code == 404:
                # 404 通常不是 retry 能解決的問題。
                last_error = (
                    "HTTP 404\n"
                    f"URL：{response.url}\n"
                    f"內容：{response.text}"
                )
                break

            if response.status_code >= 500:
                wait_seconds = min(
                    30,
                    2 ** attempt
                )

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text}"
                )

                time.sleep(wait_seconds)
                continue

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )
            break

        except requests.RequestException as exc:
            last_error = repr(exc)
            time.sleep(
                min(30, 2 ** attempt)
            )

    raise RuntimeError(
        "Fugle API 錯誤\n"
        f"股票：{symbol}\n"
        f"期間：{params['from']} ~ {params['to']}\n"
        f"{last_error}"
    )


def extract_fugle_data(payload):
    """
    Fugle Historical Candles response：
    {
        "symbol": "...",
        "data": [...]
    }
    """

    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        data = payload.get("data")

        if isinstance(data, list):
            return data

        # 相容其他可能包裝
        for key in [
            "candles",
            "results",
        ]:
            data = payload.get(key)
            if isinstance(data, list):
                return data

    return []


def candles_to_dataframe(
    payload,
    symbol,
):
    candles = extract_fugle_data(payload)

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles)

    wanted = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in wanted:
        if col not in df.columns:
            df[col] = np.nan

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    for col in wanted[1:]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["stock_code"] = clean_symbol(symbol)

    df = df[
        [
            "date",
            "stock_code",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    df = df.dropna(
        subset=["date", "close"]
    )

    df = df.sort_values("date")
    df = df.drop_duplicates(
        subset=["date", "stock_code"]
    )

    return df


# ============================================================
# 4. Cache + 330 天下載
# ============================================================

def read_history_cache(symbol):
    path = DATA_DIR / f"{symbol}_history.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            path,
            parse_dates=["date"],
        )

        if df.empty:
            return pd.DataFrame()

        required = [
            "date",
            "stock_code",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        return df

    except Exception as exc:
        log_error(
            f"{symbol} cache 讀取失敗：{exc}"
        )
        return pd.DataFrame()


def save_history_cache(
    symbol,
    df,
):
    path = DATA_DIR / f"{symbol}_history.csv"

    try:
        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["date", "close"]
        )

        df = df.sort_values("date")
        df = df.drop_duplicates(
            subset=["date", "stock_code"]
        )

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

    except Exception as exc:
        log_error(
            f"{symbol} cache 儲存失敗：{exc}"
        )


def download_history(
    symbol,
    start_date,
    end_date,
    sleep_seconds=0.5,
):
    """
    由於 Historical Candles 單次限制為 1 年內，
    這裡使用 330 天區間。

    同時會保留 data/ 既有 cache，
    讓每次執行都能更新並累積資料。
    """

    symbol = clean_symbol(symbol)

    start_date = pd.Timestamp(
        start_date
    ).normalize()

    end_date = pd.Timestamp(
        end_date
    ).normalize()

    cache = read_history_cache(symbol)

    pieces = []

    if not cache.empty:
        cache_part = cache[
            (cache["date"] >= start_date)
            & (cache["date"] <= end_date)
        ].copy()

        if not cache_part.empty:
            pieces.append(cache_part)

    # 仍然重新下載目前時間範圍，
    # 讓每次執行可以取得新市場資料。
    print(
        f"\n[資料下載] 股票：{symbol}"
    )

    current = start_date

    while current <= end_date:
        chunk_end = min(
            current
            + pd.Timedelta(
                days=DOWNLOAD_CHUNK_DAYS - 1
            ),
            end_date,
        )

        print(
            f"  {current.date()} ~ "
            f"{chunk_end.date()}"
        )

        try:
            payload = request_fugle_history(
                symbol,
                current,
                chunk_end,
            )

            chunk_df = candles_to_dataframe(
                payload,
                symbol,
            )

            if chunk_df.empty:
                print(
                    "    此區間沒有有效資料"
                )
            else:
                print(
                    f"    取得 {len(chunk_df)} 筆"
                )
                pieces.append(chunk_df)

        except Exception as exc:
            print(
                f"    下載失敗：{exc}"
            )

            log_error(
                f"{symbol} "
                f"{current.date()} ~ "
                f"{chunk_end.date()} "
                f"下載失敗：{exc}"
            )

        current = (
            chunk_end
            + pd.Timedelta(days=1)
        )

        time.sleep(sleep_seconds)

    if not pieces:
        raise RuntimeError(
            f"{symbol} 無法取得任何歷史資料"
        )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["date", "close"]
    )

    result = result.sort_values("date")
    result = result.drop_duplicates(
        subset=["date", "stock_code"]
    )

    # 保存完整 cache。
    # 若舊 cache 有更早資料，也一併保留。
    if not cache.empty:
        result = pd.concat(
            [cache, result],
            ignore_index=True,
        )

        result = result.sort_values("date")
        result = result.drop_duplicates(
            subset=["date", "stock_code"],
            keep="last",
        )

    save_history_cache(
        symbol,
        result,
    )

    return result[
        (result["date"] >= start_date)
        & (result["date"] <= end_date)
    ].copy()


# ============================================================
# 5. 技術指標
# ============================================================

def calc_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    return 100 - (
        100 / (1 + rs)
    )


def add_features(
    market_df,
    benchmark_df=None,
):
    """
    所有特徵都以當日以前資訊計算。

    target_return：
    close[t+21] / close[t] - 1

    因此模型學習的是未來 1 個交易日報酬率。
    """

    df = market_df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # Fugle / CSV 讀回資料偶爾會把 OHLCV 當成 object/string。
    # 在任何 rolling、isfinite、sklearn 操作前先強制數值化。
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=["date", "close"]
    )

    df = df.sort_values(
        ["stock_code", "date"]
    ).reset_index(drop=True)

    group = df.groupby(
        "stock_code",
        group_keys=False,
    )

    # --------------------------------------------------------
    # 基礎報酬率
    # --------------------------------------------------------

    for n in [
        1,
        2,
        3,
        5,
        10,
        20,
        21,
        60,
    ]:
        df[f"ret_{n}"] = group[
            "close"
        ].pct_change(n)

    # --------------------------------------------------------
    # OHLC 結構
    # --------------------------------------------------------

    df["hl_range"] = (
        (df["high"] - df["low"])
        / df["close"].replace(
            0,
            np.nan,
        )
    )

    df["oc_change"] = (
        (df["close"] - df["open"])
        / df["open"].replace(
            0,
            np.nan,
        )
    )

    max_oc = df[
        ["open", "close"]
    ].max(axis=1)

    min_oc = df[
        ["open", "close"]
    ].min(axis=1)

    df["upper_shadow"] = (
        (df["high"] - max_oc)
        / df["close"].replace(
            0,
            np.nan,
        )
    )

    df["lower_shadow"] = (
        (min_oc - df["low"])
        / df["close"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # MA
    # --------------------------------------------------------

    for n in [
        5,
        10,
        20,
        60,
    ]:
        ma = group[
            "close"
        ].transform(
            lambda s, n=n:
            s.rolling(n).mean()
        )

        df[f"ma_{n}_ratio"] = (
            df["close"]
            / ma.replace(
                0,
                np.nan,
            )
            - 1
        )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for n in [
        12,
        26,
    ]:
        ema = group[
            "close"
        ].transform(
            lambda s, n=n:
            s.ewm(
                span=n,
                adjust=False,
            ).mean()
        )

        df[f"ema_{n}_ratio"] = (
            df["close"]
            / ema.replace(
                0,
                np.nan,
            )
            - 1
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    def macd_func(s):
        ema12 = s.ewm(
            span=12,
            adjust=False,
        ).mean()

        ema26 = s.ewm(
            span=26,
            adjust=False,
        ).mean()

        return ema12 - ema26

    df["macd"] = group[
        "close"
    ].transform(macd_func)

    df["macd_signal"] = (
        df.groupby(
            "stock_code"
        )["macd"]
        .transform(
            lambda s:
            s.ewm(
                span=9,
                adjust=False,
            ).mean()
        )
    )

    df["macd_hist"] = (
        df["macd"]
        - df["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi_14"] = group[
        "close"
    ].transform(
        calc_rsi
    )

    # --------------------------------------------------------
    # 波動率
    # --------------------------------------------------------

    for n in [
        5,
        10,
        20,
        60,
    ]:
        df[f"volatility_{n}"] = (
            group["ret_1"]
            .transform(
                lambda s, n=n:
                s.rolling(n).std()
            )
        )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    df["previous_close"] = (
        group["close"].shift(1)
    )

    tr1 = (
        df["high"]
        - df["low"]
    )

    tr2 = (
        df["high"]
        - df["previous_close"]
    ).abs()

    tr3 = (
        df["low"]
        - df["previous_close"]
    ).abs()

    df["true_range"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = (
        df.groupby(
            "stock_code"
        )["true_range"]
        .transform(
            lambda s:
            s.rolling(14).mean()
        )
        / df["close"].replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_change"] = (
        group["volume"].pct_change()
    )

    for n in [
        5,
        20,
        60,
    ]:
        volume_ma = group[
            "volume"
        ].transform(
            lambda s, n=n:
            s.rolling(n).mean()
        )

        df[f"volume_ratio_{n}"] = (
            df["volume"]
            / volume_ma.replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # 20 / 60 日高低位置
    # --------------------------------------------------------

    for n in [
        20,
        60,
    ]:
        rolling_high = group[
            "high"
        ].transform(
            lambda s, n=n:
            s.rolling(n).max()
        )

        rolling_low = group[
            "low"
        ].transform(
            lambda s, n=n:
            s.rolling(n).min()
        )

        df[f"position_{n}"] = (
            (
                df["close"]
                - rolling_low
            )
            / (
                rolling_high
                - rolling_low
            ).replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    for n in [
        20,
        60,
    ]:
        rolling_mean = group[
            "close"
        ].transform(
            lambda s, n=n:
            s.rolling(n).mean()
        )

        rolling_std = group[
            "close"
        ].transform(
            lambda s, n=n:
            s.rolling(n).std()
        )

        df[f"zscore_{n}"] = (
            (
                df["close"]
                - rolling_mean
            )
            / rolling_std.replace(
                0,
                np.nan,
            )
        )

    # --------------------------------------------------------
    # 日期
    # --------------------------------------------------------

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["quarter"] = (
        df["date"].dt.quarter
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    if (
        benchmark_df is not None
        and not benchmark_df.empty
    ):
        b = benchmark_df[
            [
                "date",
                "close",
            ]
        ].copy()

        b["date"] = pd.to_datetime(
            b["date"],
            errors="coerce",
        )

        b = b.dropna(
            subset=["date", "close"]
        )

        b = b.sort_values("date")

        for n in [
            1,
            5,
            20,
            60,
        ]:
            b[
                f"benchmark_ret_{n}"
            ] = b[
                "close"
            ].pct_change(n)

        b = b.rename(
            columns={
                "close":
                "benchmark_close"
            }
        )

        # 保留 stock_code 的原排序，
        # 只依日期做 asof merge。
        original_order = df.index

        df = pd.merge_asof(
            df.sort_values("date"),
            b.sort_values("date"),
            on="date",
            direction="backward",
        )

        # merge 後補回可能遺失欄位
        if "stock_code" not in df.columns:
            df["stock_code"] = ""

        for n in [
            1,
            5,
            20,
            60,
        ]:
            col = (
                f"benchmark_ret_{n}"
            )

            if col not in df.columns:
                df[col] = np.nan

        df["relative_ret_20"] = (
            df["ret_20"]
            - df["benchmark_ret_20"]
        )

        df["relative_ret_60"] = (
            df["ret_60"]
            - df["benchmark_ret_60"]
        )

    else:
        for n in [
            1,
            5,
            20,
            60,
        ]:
            df[
                f"benchmark_ret_{n}"
            ] = 0.0

        df["relative_ret_20"] = (
            df["ret_20"]
        )

        df["relative_ret_60"] = (
            df["ret_60"]
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    future_close = (
        df.groupby(
            "stock_code"
        )["close"]
        .shift(-HORIZON)
    )

    df["target_price"] = (
        future_close
    )

    df["target_return"] = (
        future_close
        / df["close"]
        - 1
    )

    df["target_direction"] = (
        df["target_return"] > 0
    ).astype(int)

    # --------------------------------------------------------
    # 清理
    # --------------------------------------------------------

    # 只對數值欄位處理 inf，避免 object/datetime 欄位觸發
    # numpy ufunc isfinite 型別錯誤。
    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_cols) > 0:
        df[numeric_cols] = (
            df[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
        )

    return df


# ============================================================
# 6. Feature list
# ============================================================

def get_feature_columns():
    """
    所有模型使用的特徵。

    不直接使用股價本身作為主要模型輸入，
    讓不同價格尺度股票可以共同訓練。

    stock_code 不直接 one-hot，
    因為 Random Forest 對類別代碼沒有自然的距離概念。

    反而利用：
    - 個股自身報酬
    - MA ratio
    - Z-score
    - 波動率
    - 量比
    - 相對大盤報酬
    等尺度無關特徵。
    """

    return [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_21",
        "ret_60",

        "hl_range",
        "oc_change",
        "upper_shadow",
        "lower_shadow",

        "ma_5_ratio",
        "ma_10_ratio",
        "ma_20_ratio",
        "ma_60_ratio",

        "ema_12_ratio",
        "ema_26_ratio",

        "macd",
        "macd_signal",
        "macd_hist",

        "rsi_14",

        "volatility_5",
        "volatility_10",
        "volatility_20",
        "volatility_60",

        "atr_14",

        "volume_change",
        "volume_ratio_5",
        "volume_ratio_20",
        "volume_ratio_60",

        "position_20",
        "position_60",

        "zscore_20",
        "zscore_60",

        "day_of_week",
        "month",
        "quarter",

        "benchmark_ret_1",
        "benchmark_ret_5",
        "benchmark_ret_20",
        "benchmark_ret_60",

        "relative_ret_20",
        "relative_ret_60",
    ]


def prepare_model_dataframe(
    feature_df,
):
    """
    對模型資料做：
    1. target 完整性確認
    2. feature 補值
    3. 無限值處理
    """

    df = feature_df.copy()

    features = get_feature_columns()

    # 先把 target / price 轉成純 numeric，
    # 再使用 numpy.isfinite。
    for col in ["target_return", "target_price", "close"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    numeric_target = (
        df["target_return"].to_numpy(dtype=float, na_value=np.nan)
    )
    numeric_price = (
        df["target_price"].to_numpy(dtype=float, na_value=np.nan)
    )
    numeric_close = (
        df["close"].to_numpy(dtype=float, na_value=np.nan)
    )

    valid_mask = (
        np.isfinite(numeric_target)
        & np.isfinite(numeric_price)
        & np.isfinite(numeric_close)
    )

    df = df.loc[valid_mask].copy()

    for col in features:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        median = df[col].median()

        if not np.isfinite(median):
            median = 0.0

        df[col] = df[col].fillna(
            median
        )

    for col in features:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df[features] = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )

    df = df.sort_values(
        ["date", "stock_code"]
    ).reset_index(
        drop=True
    )

    return df, features


# ============================================================
# 7. Model
# ============================================================

def create_regressor():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=3,
        max_features=0.75,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_classifier():
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_leaf=3,
        max_features=0.75,
        bootstrap=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_final_models(
    train_df,
    features,
):
    X = train_df[features]

    y_reg = train_df[
        "target_return"
    ].astype(float)

    y_cls = train_df[
        "target_direction"
    ].astype(int)

    reg = create_regressor()
    clf = create_classifier()

    reg.fit(
        X,
        y_reg,
    )

    clf.fit(
        X,
        y_cls,
    )

    return reg, clf


# ============================================================
# 8. Time Series OOF
# ============================================================

def time_series_oof(
    model_df,
    features,
    n_splits=5,
):
    """
    全部股票共同做時間切分。

    注意：
    這裡不是普通 random KFold，
    避免未來資料進入訓練集。
    """

    df = model_df.sort_values(
        ["date", "stock_code"]
    ).reset_index(drop=True)

    n = len(df)

    if n < MIN_GLOBAL_SAMPLES:
        raise RuntimeError(
            f"OOF 資料不足：{n} 筆，"
            f"至少需要 {MIN_GLOBAL_SAMPLES} 筆"
        )

    # 避免某些資料量較少的情況
    actual_splits = min(
        n_splits,
        max(2, n // 50),
    )

    splitter = TimeSeriesSplit(
        n_splits=actual_splits
    )

    X = df[features]
    y_reg = df[
        "target_return"
    ].astype(float)

    y_cls = df[
        "target_direction"
    ].astype(int)

    oof_return = np.full(
        n,
        np.nan,
        dtype=float,
    )

    oof_direction = np.full(
        n,
        np.nan,
        dtype=float,
    )

    fold_records = []

    for fold_no, (
        train_idx,
        valid_idx,
    ) in enumerate(
        splitter.split(X),
        start=1,
    ):
        if len(train_idx) < 50:
            continue

        X_train = X.iloc[
            train_idx
        ]

        X_valid = X.iloc[
            valid_idx
        ]

        y_reg_train = y_reg.iloc[
            train_idx
        ]

        y_reg_valid = y_reg.iloc[
            valid_idx
        ]

        y_cls_train = y_cls.iloc[
            train_idx
        ]

        y_cls_valid = y_cls.iloc[
            valid_idx
        ]

        # Classification 如果某一折只有單一 class，
        # RandomForestClassifier 無法建立二元模型。
        if y_cls_train.nunique() < 2:
            print(
                f"Fold {fold_no}："
                "訓練集只有單一方向，"
                "Classification 跳過"
            )

            reg = create_regressor()

            reg.fit(
                X_train,
                y_reg_train,
            )

            pred_return = reg.predict(
                X_valid
            )

            pred_direction = (
                pred_return > 0
            ).astype(int)

        else:
            reg = create_regressor()
            clf = create_classifier()

            reg.fit(
                X_train,
                y_reg_train,
            )

            clf.fit(
                X_train,
                y_cls_train,
            )

            pred_return = reg.predict(
                X_valid
            )

            pred_direction = clf.predict(
                X_valid
            )

        oof_return[
            valid_idx
        ] = pred_return

        oof_direction[
            valid_idx
        ] = pred_direction

        fold_mae = mean_absolute_error(
            y_reg_valid,
            pred_return,
        )

        fold_rmse = calculate_rmse(
            y_reg_valid,
            pred_return,
        )

        fold_mape = calculate_mape(
            y_reg_valid,
            pred_return,
        )

        fold_acc = accuracy_score(
            y_cls_valid,
            pred_direction,
        )

        fold_precision = precision_score(
            y_cls_valid,
            pred_direction,
            zero_division=0,
        )

        fold_recall = recall_score(
            y_cls_valid,
            pred_direction,
            zero_division=0,
        )

        fold_f1 = f1_score(
            y_cls_valid,
            pred_direction,
            zero_division=0,
        )

        fold_records.append({
            "fold": fold_no,
            "train_samples": len(
                train_idx
            ),
            "validation_samples": len(
                valid_idx
            ),
            "MAE": fold_mae,
            "RMSE": fold_rmse,
            "MAPE_percent": fold_mape,
            "Accuracy": fold_acc,
            "Precision": fold_precision,
            "Recall": fold_recall,
            "F1": fold_f1,
        })

    df["oof_pred_return"] = (
        oof_return
    )

    df["oof_pred_direction"] = (
        oof_direction
    )

    valid = (
        np.isfinite(oof_return)
        & np.isfinite(oof_direction)
    )

    if valid.sum() < MIN_OOF_SAMPLES:
        raise RuntimeError(
            f"有效 OOF 樣本只有 "
            f"{valid.sum()} 筆"
        )

    y_r = y_reg.iloc[
        np.where(valid)[0]
    ]

    p_r = pd.Series(
        oof_return[valid]
    )

    y_c = y_cls.iloc[
        np.where(valid)[0]
    ]

    p_c = pd.Series(
        oof_direction[valid]
    ).astype(int)

    metrics = {
        "samples": int(
            valid.sum()
        ),
        "MAE": float(
            mean_absolute_error(
                y_r,
                p_r,
            )
        ),
        "RMSE": float(
            calculate_rmse(
                y_r,
                p_r,
            )
        ),
        "MAPE_percent": calculate_mape(
            y_r,
            p_r,
        ),
        "Accuracy": float(
            accuracy_score(
                y_c,
                p_c,
            )
        ),
        "Precision": float(
            precision_score(
                y_c,
                p_c,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_c,
                p_c,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_c,
                p_c,
                zero_division=0,
            )
        ),
    }

    fold_df = pd.DataFrame(
        fold_records
    )

    return (
        df,
        metrics,
        fold_df,
    )


# ============================================================
# 9. 歷史回測
# ============================================================

def historical_backtest(
    feature_df,
    stock_code,
):
    """
    隔日歷史滾動回測：

    假設今天是測試點 t：
      - 只使用 t 以前可取得的資料訓練
      - 使用 t 當天特徵
      - 預測 t+1 交易日（隔日）
      - 與實際 t+1 價格比較

    因此可以回答：
    「如果當時不知道後來的價格，
     模型能不能預測？」

    為避免 500 棵樹 * 每一個交易日都重訓造成
    執行時間過長，預設每 10 個交易日測一次。
    """

    df = feature_df[
        feature_df["stock_code"]
        == stock_code
    ].copy()

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    df, features = (
        prepare_model_dataframe(df)
    )

    if len(df) < 150:
        raise RuntimeError(
            f"{stock_code} 回測有效資料不足："
            f"{len(df)} 筆"
        )

    results = []

    # 至少 120 筆歷史資料後才開始
    start_index = 120

    # 改為每天做一次隔日回測
    test_indices = range(
        start_index,
        len(df),
        1,
    )

    for test_index in test_indices:
        train_df = df.iloc[
            :test_index
        ].copy()

        test_df = df.iloc[
            [test_index]
        ].copy()

        # 需要實際 t+1 target（隔日）
        if not np.isfinite(
            test_df[
                "target_price"
            ].iloc[0]
        ):
            continue

        if len(train_df) < 100:
            continue

        try:
            reg = create_regressor()
            clf = create_classifier()

            reg.fit(
                train_df[features],
                train_df[
                    "target_return"
                ],
            )

            # 若早期訓練集剛好只有一類，
            # 直接用 regression 方向代替。
            if (
                train_df[
                    "target_direction"
                ].nunique()
                >= 2
            ):
                clf.fit(
                    train_df[features],
                    train_df[
                        "target_direction"
                    ],
                )

                pred_direction = int(
                    clf.predict(
                        test_df[features]
                    )[0]
                )
            else:
                pred_direction = int(
                    reg.predict(
                        test_df[features]
                    )[0] > 0
                )

            predicted_return = float(
                reg.predict(
                    test_df[features]
                )[0]
            )

            current_price = float(
                test_df[
                    "close"
                ].iloc[0]
            )

            actual_price = float(
                test_df[
                    "target_price"
                ].iloc[0]
            )

            actual_return = float(
                test_df[
                    "target_return"
                ].iloc[0]
            )

            predicted_price = (
                current_price
                * (
                    1
                    + predicted_return
                )
            )

            actual_direction = int(
                actual_return > 0
            )

            results.append({
                "stock_code": stock_code,
                "prediction_date": (
                    test_df[
                        "date"
                    ].iloc[0]
                ),
                "current_price": (
                    current_price
                ),
                "predicted_return": (
                    predicted_return
                ),
                "predicted_price": (
                    predicted_price
                ),
                "actual_return": (
                    actual_return
                ),
                "actual_price": (
                    actual_price
                ),
                "predicted_direction": (
                    pred_direction
                ),
                "actual_direction": (
                    actual_direction
                ),
            })

        except Exception as exc:
            log_error(
                "Historical Backtest error: "
                f"{stock_code}, "
                f"index={test_index}, "
                f"{exc}\n"
                f"{traceback.format_exc()}"
            )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        raise RuntimeError(
            "沒有產生任何有效回測樣本"
        )

    valid = result_df[
        np.isfinite(
            result_df[
                "actual_price"
            ]
        )
        & np.isfinite(
            result_df[
                "predicted_price"
            ]
        )
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "回測結果沒有有效價格"
        )

    metrics = {
        "samples": int(
            len(valid)
        ),
        "MAE": float(
            mean_absolute_error(
                valid["actual_price"],
                valid["predicted_price"],
            )
        ),
        "RMSE": float(
            calculate_rmse(
                valid["actual_price"],
                valid["predicted_price"],
            )
        ),
        "MAPE_percent": calculate_mape(
            valid["actual_price"],
            valid["predicted_price"],
        ),
        "direction_accuracy": float(
            accuracy_score(
                valid[
                    "actual_direction"
                ],
                valid[
                    "predicted_direction"
                ],
            )
        ),
    }

    return (
        result_df,
        metrics,
    )


# ============================================================
# 10. 價格區間
# ============================================================

def calculate_price_interval(
    oof_df,
    stock_code,
    current_price,
    predicted_return,
):
    """
    不使用任意 ±20%。

    先計算：
      residual =
      actual_future_return - OOF_predicted_return

    再取得 residual Q10 / Q90。

    最後：
      lower_return =
      predicted_return + Q10

      upper_return =
      predicted_return + Q90

      lower_price =
      current_price * (1 + lower_return)

      upper_price =
      current_price * (1 + upper_return)
    """

    subset = oof_df[
        oof_df["stock_code"]
        == stock_code
    ].copy()

    subset = subset[
        np.isfinite(
            subset[
                "oof_pred_return"
            ]
        )
        & np.isfinite(
            subset[
                "target_return"
            ]
        )
    ].copy()

    if len(subset) >= 10:
        residual = (
            subset[
                "target_return"
            ]
            - subset[
                "oof_pred_return"
            ]
        )

        q10 = float(
            residual.quantile(0.10)
        )

        q90 = float(
            residual.quantile(0.90)
        )

    else:
        # 個股 OOF 不足時，
        # 使用所有股票共同 OOF 殘差。
        all_valid = oof_df[
            np.isfinite(
                oof_df[
                    "oof_pred_return"
                ]
            )
            & np.isfinite(
                oof_df[
                    "target_return"
                ]
            )
        ].copy()

        if len(all_valid) >= 10:
            residual = (
                all_valid[
                    "target_return"
                ]
                - all_valid[
                    "oof_pred_return"
                ]
            )

            q10 = float(
                residual.quantile(0.10)
            )

            q90 = float(
                residual.quantile(0.90)
            )
        else:
            q10 = -0.10
            q90 = 0.10

    lower_return = (
        predicted_return
        + q10
    )

    upper_return = (
        predicted_return
        + q90
    )

    lower_price = max(
        0.01,
        current_price
        * (1 + lower_return),
    )

    upper_price = max(
        lower_price,
        current_price
        * (1 + upper_return),
    )

    return (
        lower_price,
        upper_price,
        q10,
        q90,
    )


# ============================================================
# 11. Feature Importance
# ============================================================

def save_feature_importance(
    model,
    features,
    stock_code,
):
    importance = pd.DataFrame({
        "feature": features,
        "importance": (
            model.feature_importances_
        ),
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    csv_path = (
        OUTPUT_DIR
        / f"{stock_code}_feature_importance.csv"
    )

    png_path = (
        OUTPUT_DIR
        / f"{stock_code}_feature_importance.png"
    )

    importance.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    top = (
        importance
        .head(20)
        .sort_values("importance")
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top["feature"],
        top["importance"],
    )

    plt.xlabel(
        "Random Forest Importance"
    )

    plt.title(
        f"{stock_code} Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        png_path,
        dpi=160,
    )

    plt.close()

    return (
        importance,
        csv_path,
        png_path,
    )


# ============================================================
# 12. Prediction plot
# ============================================================

def save_prediction_plot(
    history,
    stock_code,
    current_price,
    predicted_price,
    lower_price,
    upper_price,
    prediction_date,
):
    path = (
        OUTPUT_DIR
        / f"{stock_code}_prediction.png"
    )

    hist = history.sort_values(
        "date"
    ).tail(180)

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        hist["date"],
        hist["close"],
        label="Historical Close",
    )

    plt.scatter(
        [prediction_date],
        [predicted_price],
        s=80,
        label="Predicted Price",
    )

    plt.axhline(
        current_price,
        linestyle="--",
        label="Current Price",
    )

    plt.axhline(
        lower_price,
        linestyle=":",
        label="Q10 Lower",
    )

    plt.axhline(
        upper_price,
        linestyle=":",
        label="Q90 Upper",
    )

    plt.title(
        f"{stock_code} Next-Day Prediction"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Price"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=160,
    )

    plt.close()

    return path


# ============================================================
# 13. 預測單一股票
# ============================================================

def predict_stock(
    feature_df,
    model_df,
    reg_model,
    clf_model,
    features,
    oof_df,
    stock_code,
):
    raw = feature_df[
        feature_df["stock_code"]
        == stock_code
    ].copy()

    if raw.empty:
        raise RuntimeError(
            f"{stock_code} 沒有任何歷史資料"
        )

    raw = raw.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    latest = raw.iloc[
        [-1]
    ].copy()

    # 最新一筆不需要 target。
    # 建立 prediction features。
    for col in features:
        if col not in latest.columns:
            latest[col] = 0.0

        latest[col] = pd.to_numeric(
            latest[col],
            errors="coerce",
        )

        value = safe_float(
            latest[col].iloc[0]
        )

        if not np.isfinite(value):
            median = model_df[
                col
            ].median()

            if not np.isfinite(median):
                median = 0.0

            latest.loc[
                :,
                col
            ] = median

    latest[features] = (
        latest[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    current_price = float(
        latest[
            "close"
        ].iloc[0]
    )

    # sklearn 必須收到 2-D：shape=(1, feature_count)
    prediction_X = latest[
        features
    ].to_numpy(dtype=float).reshape(
        1,
        len(features),
    )

    predicted_return = float(
        reg_model.predict(
            prediction_X
        )[0]
    )

    predicted_price = (
        current_price
        * (
            1
            + predicted_return
        )
    )

    # Classification
    if hasattr(
        clf_model,
        "classes_"
    ):
        pred_direction = int(
            clf_model.predict(
                prediction_X
            )[0]
        )
    else:
        pred_direction = int(
            predicted_return > 0
        )

    (
        lower_price,
        upper_price,
        q10,
        q90,
    ) = calculate_price_interval(
        oof_df,
        stock_code,
        current_price,
        predicted_return,
    )

    latest_date = pd.Timestamp(
        latest[
            "date"
        ].iloc[0]
    )

    prediction_date = (
        latest_date
        + pd.offsets.BDay(
            HORIZON
        )
    )

    direction = (
        "上漲"
        if pred_direction == 1
        else "下跌"
    )

    return {
        "stock_code": stock_code,
        "latest_date": latest_date,
        "prediction_date": prediction_date,
        "current_price": current_price,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "direction": direction,
        "residual_Q10": q10,
        "residual_Q90": q90,
    }


# ============================================================
# 14. 主程式
# ============================================================

def main():
    print_header(
        "AI 台股隔日價格預測系統"
    )

    print(
        "模型：Random Forest Regression + Classification"
    )

    print(
        f"預測：隔日（{HORIZON} 個交易日後）"
    )

    print(
        "目標：Future 1-Day Return"
    )

    print(
        "訓練：多股票共同訓練"
    )

    print(
        "驗證：5-Fold Time Series + OOF"
    )

    print(
        "\n重要：每次重新執行會重新訓練，"
        "新增的歷史資料可以影響下一次模型。"
    )

    # --------------------------------------------------------
    # 輸入股票
    # --------------------------------------------------------

    stock_text = input(
        "\n請輸入共同訓練股票代號 "
        "(例如 3044,2330,2454,0050，"
        "直接 Enter 使用 3044)："
    )

    stocks = parse_stock_input(
        stock_text
    )

    predict_text = input(
        "請輸入要預測的股票 "
        "(直接 Enter 使用 3044)："
    ).strip()

    predict_symbol = (
        clean_symbol(
            predict_text
        )
        if predict_text
        else DEFAULT_PREDICT_STOCK
    )

    if predict_symbol not in stocks:
        stocks.append(
            predict_symbol
        )

    print(
        "\n共同訓練股票："
    )

    print(
        ", ".join(stocks)
    )

    print(
        f"預測股票：{predict_symbol}"
    )

    # --------------------------------------------------------
    # API Key
    # --------------------------------------------------------

    if not FUGLE_API_KEY:
        raise RuntimeError(
            "FUGLE_API_KEY 未設定。\n"
            "PowerShell：\n"
            '$env:FUGLE_API_KEY="你的 API Key"'
        )

    # --------------------------------------------------------
    # 日期
    # --------------------------------------------------------

    end_date = pd.Timestamp.today().normalize()

    # 歷史資料。
    # 每次 API request 仍切成 330 天。
    start_date = (
        end_date
        - pd.Timedelta(days=1095)
    )

    # --------------------------------------------------------
    # 下載股票
    # --------------------------------------------------------

    print_header(
        "下載共同訓練股票"
    )

    stock_frames = []

    failed_stocks = []

    for symbol in stocks:
        try:
            df = download_history(
                symbol,
                start_date,
                end_date,
            )

            print(
                f"{symbol} 最終有效下載資料："
                f"{len(df)} 筆"
            )

            if len(df) >= 100:
                stock_frames.append(
                    df
                )
            else:
                print(
                    f"{symbol} 資料不足，"
                    "不加入共同模型"
                )

                failed_stocks.append(
                    symbol
                )

        except Exception as exc:
            failed_stocks.append(
                symbol
            )

            log_error(
                f"股票 {symbol} 下載失敗："
                f"{exc}\n"
                f"{traceback.format_exc()}"
            )

            print(
                f"{symbol} 下載失敗："
                f"{exc}"
            )

    if not stock_frames:
        raise RuntimeError(
            "沒有任何股票可以建立訓練資料"
        )

    market_df = pd.concat(
        stock_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print_header(
        "下載 0050 Benchmark（可選）"
    )

    benchmark_df = pd.DataFrame()

    try:
        benchmark_df = (
            download_history(
                BENCHMARK,
                start_date,
                end_date,
            )
        )

        if len(benchmark_df) < 100:
            print(
                "0050 Benchmark 資料不足，"
                "改用無 Benchmark 模式"
            )

            benchmark_df = pd.DataFrame()

        else:
            print(
                f"0050 Benchmark："
                f"{len(benchmark_df)} 筆"
            )

    except Exception as exc:
        print(
            "0050 Benchmark 下載失敗。"
        )

        print(
            "主模型不會因此停止，"
            "改用無 Benchmark 模式。"
        )

        log_error(
            "0050 Benchmark 失敗："
            f"{exc}\n"
            f"{traceback.format_exc()}"
        )

        benchmark_df = pd.DataFrame()

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print_header(
        "建立技術與量價特徵"
    )

    feature_df = add_features(
        market_df,
        benchmark_df,
    )

    model_df, features = (
        prepare_model_dataframe(
            feature_df
        )
    )

    print(
        f"原始市場資料："
        f"{len(market_df):,} 筆"
    )

    print(
        f"模型有效資料："
        f"{len(model_df):,} 筆"
    )

    print(
        f"模型特徵數："
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # 股票樣本統計
    # --------------------------------------------------------

    print(
        "\n各股票有效訓練樣本："
    )

    stock_counts = (
        model_df
        .groupby("stock_code")
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        stock_counts.to_string()
    )

    usable_stocks = []

    for symbol in stocks:
        count = int(
            stock_counts.get(
                symbol,
                0
            )
        )

        if count >= MIN_STOCK_SAMPLES:
            usable_stocks.append(
                symbol
            )

    if predict_symbol not in (
        usable_stocks
    ):
        raise RuntimeError(
            f"預測股票 {predict_symbol} "
            "有效訓練資料不足。"
        )

    train_df = model_df[
        model_df[
            "stock_code"
        ].isin(
            usable_stocks
        )
    ].copy()

    if len(train_df) < MIN_GLOBAL_SAMPLES:
        raise RuntimeError(
            f"共同訓練資料不足："
            f"{len(train_df)}"
        )

    print(
        "\n實際共同訓練股票："
    )

    print(
        ", ".join(
            usable_stocks
        )
    )

    # --------------------------------------------------------
    # 5 Fold OOF
    # --------------------------------------------------------

    print_header(
        "5-Fold 時序交叉驗證"
    )

    (
        oof_df,
        oof_metrics,
        fold_df,
    ) = time_series_oof(
        train_df,
        features,
        n_splits=5,
    )

    validation_path = (
        OUTPUT_DIR
        / "5fold_validation.csv"
    )

    fold_df.to_csv(
        validation_path,
        index=False,
        encoding="utf-8-sig",
    )

    oof_path = (
        OUTPUT_DIR
        / "oof_predictions.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nRegression"
    )

    print(
        f"MAE  : "
        f"{oof_metrics['MAE']:.6f}"
    )

    print(
        f"RMSE : "
        f"{oof_metrics['RMSE']:.6f}"
    )

    if np.isfinite(
        oof_metrics[
            "MAPE_percent"
        ]
    ):
        print(
            f"MAPE : "
            f"{oof_metrics['MAPE_percent']:.4f}%"
        )
    else:
        print(
            "MAPE : 無法計算"
        )

    print(
        "\nClassification"
    )

    print(
        f"Accuracy : "
        f"{oof_metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{oof_metrics['Precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{oof_metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{oof_metrics['F1']:.4f}"
    )

    # --------------------------------------------------------
    # 歷史回測
    # --------------------------------------------------------

    print_header(
        f"隔日歷史回測：{predict_symbol}"
    )

    backtest_path = None

    try:
        (
            backtest_df,
            backtest_metrics,
        ) = historical_backtest(
            feature_df,
            predict_symbol,
        )

        backtest_path = (
            OUTPUT_DIR
            / f"{predict_symbol}_historical_backtest.csv"
        )

        backtest_df.to_csv(
            backtest_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"樣本數："
            f"{backtest_metrics['samples']}"
        )

        print(
            f"MAE："
            f"{backtest_metrics['MAE']:.4f}"
        )

        print(
            f"RMSE："
            f"{backtest_metrics['RMSE']:.4f}"
        )

        if np.isfinite(
            backtest_metrics[
                "MAPE_percent"
            ]
        ):
            print(
                f"MAPE："
                f"{backtest_metrics['MAPE_percent']:.4f}%"
            )
        else:
            print(
                "MAPE：無法計算"
            )

        print(
            "方向準確率："
            f"{backtest_metrics['direction_accuracy']:.2%}"
        )

        print(
            f"回測結果："
            f"{backtest_path}"
        )

    except Exception as exc:
        backtest_metrics = {
            "samples": 0,
            "MAE": None,
            "RMSE": None,
            "MAPE_percent": None,
            "direction_accuracy": None,
        }

        print(
            "歷史回測失敗："
            f"{exc}"
        )

        log_error(
            "Historical backtest failed："
            f"{exc}\n"
            f"{traceback.format_exc()}"
        )

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    print_header(
        "訓練最終共同模型"
    )

    reg_model, clf_model = (
        train_final_models(
            train_df,
            features,
        )
    )

    # --------------------------------------------------------
    # 模型儲存
    # --------------------------------------------------------

    model_package = {
        "model_type":
            "Random Forest Regression + Classification",

        "regressor":
            reg_model,

        "classifier":
            clf_model,

        "features":
            features,

        "training_stocks":
            usable_stocks,

        "prediction_stock":
            predict_symbol,

        "horizon":
            HORIZON,

        "horizon_description":
            "Next Trading Day",

        "target":
            "1-trading-day-future-return",

        "created_at":
            datetime.now().isoformat(),

        "oof_metrics":
            oof_metrics,
    }

    model_path = (
        OUTPUT_DIR
        / "stock_month_prediction_model.pkl"
    )

    joblib.dump(
        model_package,
        model_path,
    )

    print(
        f"模型已儲存："
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    (
        importance_df,
        importance_csv,
        importance_png,
    ) = save_feature_importance(
        reg_model,
        features,
        predict_symbol,
    )

    print(
        f"Feature Importance："
        f"{importance_png}"
    )

    # --------------------------------------------------------
    # 最新預測
    # --------------------------------------------------------

    print_header(
        "產生最新預測"
    )

    prediction = predict_stock(
        feature_df,
        model_df,
        reg_model,
        clf_model,
        features,
        oof_df,
        predict_symbol,
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    prediction_csv = (
        OUTPUT_DIR
        / f"{predict_symbol}_prediction_result.csv"
    )

    prediction_row = {
        "stock_code":
            prediction["stock_code"],

        "latest_date":
            prediction["latest_date"],

        "prediction_date":
            prediction["prediction_date"],

        "current_price":
            prediction["current_price"],

        "predicted_return_percent":
            prediction[
                "predicted_return"
            ] * 100,

        "predicted_price":
            prediction[
                "predicted_price"
            ],

        "lower_price_Q10":
            prediction[
                "lower_price"
            ],

        "upper_price_Q90":
            prediction[
                "upper_price"
            ],

        "direction":
            prediction[
                "direction"
            ],

        "residual_Q10":
            prediction[
                "residual_Q10"
            ],

        "residual_Q90":
            prediction[
                "residual_Q90"
            ],

        "OOF_Accuracy":
            oof_metrics[
                "Accuracy"
            ],

        "OOF_Precision":
            oof_metrics[
                "Precision"
            ],

        "OOF_Recall":
            oof_metrics[
                "Recall"
            ],

        "OOF_F1":
            oof_metrics[
                "F1"
            ],
    }

    prediction_df = pd.DataFrame(
        [prediction_row]
    )

    prediction_df.to_csv(
        prediction_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # PNG
    # --------------------------------------------------------

    history_for_plot = market_df[
        market_df[
            "stock_code"
        ] == predict_symbol
    ].copy()

    prediction_png = (
        save_prediction_plot(
            history_for_plot,
            predict_symbol,
            prediction[
                "current_price"
            ],
            prediction[
                "predicted_price"
            ],
            prediction[
                "lower_price"
            ],
            prediction[
                "upper_price"
            ],
            prediction[
                "prediction_date"
            ],
        )
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    execution_log = {
        "execution_time":
            datetime.now().isoformat(),

        "model_type":
            "Random Forest Regression + Classification",

        "stocks_input":
            stocks,

        "usable_training_stocks":
            usable_stocks,

        "failed_stocks":
            failed_stocks,

        "prediction_stock":
            predict_symbol,

        "horizon_trading_days":
            HORIZON,

        "download_start":
            str(start_date.date()),

        "download_end":
            str(end_date.date()),

        "benchmark":
            BENCHMARK,

        "benchmark_available":
            not benchmark_df.empty,

        "training_samples":
            int(len(train_df)),

        "feature_count":
            len(features),

        "features":
            features,

        "OOF_metrics":
            oof_metrics,

        "historical_backtest":
            backtest_metrics,

        "prediction":
            {
                "latest_date":
                    str(
                        prediction[
                            "latest_date"
                        ]
                    ),

                "prediction_date":
                    str(
                        prediction[
                            "prediction_date"
                        ]
                    ),

                "current_price":
                    prediction[
                        "current_price"
                    ],

                "predicted_return_percent":
                    prediction[
                        "predicted_return"
                    ] * 100,

                "predicted_price":
                    prediction[
                        "predicted_price"
                    ],

                "lower_price":
                    prediction[
                        "lower_price"
                    ],

                "upper_price":
                    prediction[
                        "upper_price"
                    ],

                "direction":
                    prediction[
                        "direction"
                    ],

                "residual_Q10":
                    prediction[
                        "residual_Q10"
                    ],

                "residual_Q90":
                    prediction[
                        "residual_Q90"
                    ],
            },

        "files":
            {
                "model":
                    str(model_path),

                "prediction_csv":
                    str(prediction_csv),

                "oof_csv":
                    str(oof_path),

                "validation_csv":
                    str(validation_path),

                "feature_importance_csv":
                    str(importance_csv),

                "feature_importance_png":
                    str(importance_png),

                "prediction_png":
                    str(prediction_png),

                "historical_backtest_csv":
                    (
                        str(backtest_path)
                        if backtest_path
                        else None
                    ),

                "error_log":
                    str(ERROR_LOG),
            },
    }

    json_path = (
        OUTPUT_DIR
        / "execution_log.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            execution_log,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # 最終結果
    # --------------------------------------------------------

    print_header(
        "預測結果"
    )

    print(
        f"股票："
        f"{prediction['stock_code']}"
    )

    print(
        f"日期："
        f"{prediction['latest_date'].date()}"
    )

    print(
        f"隔日預測日期："
        f"{prediction['prediction_date'].date()}"
    )

    print(
        f"目前價格："
        f"{prediction['current_price']:.2f}"
    )

    print(
        "預測隔日報酬率："
        f"{prediction['predicted_return'] * 100:.2f}%"
    )

    print(
        f"預測價格："
        f"{prediction['predicted_price']:.2f}"
    )

    print(
        "價格區間："
        f"{prediction['lower_price']:.2f}"
        " ~ "
        f"{prediction['upper_price']:.2f}"
    )

    print(
        f"方向："
        f"{prediction['direction']}"
    )

    print_header(
        "可信度驗證（OOF 5-Fold）"
    )

    print(
        "Accuracy : "
        f"{oof_metrics['Accuracy']:.4f}"
    )

    print(
        "Precision: "
        f"{oof_metrics['Precision']:.4f}"
    )

    print(
        "Recall   : "
        f"{oof_metrics['Recall']:.4f}"
    )

    print(
        "F1 Score : "
        f"{oof_metrics['F1']:.4f}"
    )

    print_header(
        "檔案輸出"
    )

    print(
        f"模型：{model_path}"
    )

    print(
        f"預測 CSV：{prediction_csv}"
    )

    print(
        f"OOF CSV：{oof_path}"
    )

    print(
        f"5-Fold CSV：{validation_path}"
    )

    print(
        f"Feature Importance："
        f"{importance_csv}"
    )

    print(
        f"Feature Importance PNG："
        f"{importance_png}"
    )

    print(
        f"預測 PNG：{prediction_png}"
    )

    if backtest_path:
        print(
            f"隔日歷史回測："
            f"{backtest_path}"
        )

    print(
        f"JSON：{json_path}"
    )

    print(
        f"錯誤紀錄：{ERROR_LOG}"
    )

    print_header(
        "程式執行完成"
    )


# ============================================================
# 15. Entry Point
# ============================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n使用者中止程式。"
        )

    except Exception as exc:
        print(
            "\n"
            + "=" * 60
        )

        print(
            "程式執行發生錯誤"
        )

        print(
            "=" * 60
        )

        print(
            str(exc)
        )

        detail = (
            str(exc)
            + "\n"
            + traceback.format_exc()
        )

        log_error(
            "MAIN ERROR\n"
            + detail
        )

        print(
            "\n詳細錯誤："
            f"{ERROR_LOG}"
        )

        print(
            "\n請確認："
        )

        print(
            "1. FUGLE_API_KEY 是否正確"
        )

        print(
            "2. 股票代號是否正確"
        )

        print(
            "3. 網路是否正常"
        )

        print(
            "4. requirements.txt 套件是否安裝"
        )

        print(
            "5. Fugle API 是否暫時 Rate Limit"
        )

        print(
            "6. output/error_log.txt"
        )

        print(
            "7. 不要使用 3044.TW，"
            "Fugle 請使用 3044"
        )

        sys.exit(1)
