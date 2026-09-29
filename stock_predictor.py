# -*- coding: utf-8 -*-
"""AI 台股產業共同訓練隔日預測模組。
核心設計沿用使用者提供的 stock_month_prediction_next_day_v2：
Random Forest Regression + Classification、Future 1-Day Return、44 個尺度無關特徵、
5-fold Time Series OOF，並將預測報酬率轉回價格。
"""
from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
API_BASE = "https://api.fugle.tw/marketdata/v1.0/stock"
HORIZON = 1
RANDOM_STATE = 42

FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20","ret_21","ret_60",
    "hl_range","oc_change","upper_shadow","lower_shadow",
    "ma_5_ratio","ma_10_ratio","ma_20_ratio","ma_60_ratio",
    "ema_12_ratio","ema_26_ratio","macd","macd_signal","macd_hist","rsi_14",
    "volatility_5","volatility_10","volatility_20","volatility_60","atr_14",
    "volume_change","volume_ratio_5","volume_ratio_20","volume_ratio_60",
    "position_20","position_60","zscore_20","zscore_60",
    "day_of_week","month","quarter",
    "benchmark_ret_1","benchmark_ret_5","benchmark_ret_20","benchmark_ret_60",
    "relative_ret_20","relative_ret_60"
]

class FugleClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FUGLE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("找不到 FUGLE_API_KEY。請先設定環境變數。")
        self.session = requests.Session()
        self.session.headers.update({"X-API-KEY": self.api_key, "Accept": "application/json"})
        self.last_request = 0.0

    def get(self, path: str, params: dict, retries: int = 4):
        wait = max(0, 0.25 - (time.time() - self.last_request))
        if wait: time.sleep(wait)
        url = f"{API_BASE}/{path.lstrip('/')}"
        for attempt in range(retries):
            try:
                r = self.session.get(url, params=params, timeout=30)
                self.last_request = time.time()
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and data.get("statusCode", 200) not in (200, None):
                    raise RuntimeError(str(data))
                return data
            except Exception:
                if attempt == retries - 1: raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("API request failed")

    def candles(self, symbol: str, start: str, end: str, timeframe: str = "D") -> pd.DataFrame:
        # Fugle requires the queried historical interval to be less than one year.
        data = self.get("historical/candles", {"symbol": symbol, "from": start, "to": end, "timeframe": timeframe})
        rows = data.get("data", data if isinstance(data, list) else [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        rename = {"date":"date","time":"date","open":"open","high":"high","low":"low","close":"close","volume":"volume","turnover":"turnover"}
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        if "date" not in df: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in ["open","high","low","close","volume"]:
            if c not in df: df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["date","open","high","low","close","volume"]].dropna(subset=["date","close"]).sort_values("date").drop_duplicates("date")

    def historical(self, symbol: str, days: int = 900) -> pd.DataFrame:
        end = datetime.now().date()
        start = end - timedelta(days=days)
        pieces=[]
        cur=start
        while cur < end:
            chunk_end=min(cur+timedelta(days=329), end)
            try: p=self.candles(symbol, cur.isoformat(), chunk_end.isoformat())
            except Exception: p=pd.DataFrame()
            if not p.empty: pieces.append(p)
            cur=chunk_end+timedelta(days=1)
        if not pieces: return pd.DataFrame()
        return pd.concat(pieces, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)

def clean_symbol(s: str) -> str:
    s=s.strip().upper().replace(".TW","")
    return s

def add_features(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    x=df.copy().sort_values("date").reset_index(drop=True)
    c=x["close"].astype(float); o=x["open"].astype(float); h=x["high"].astype(float); l=x["low"].astype(float); v=x["volume"].astype(float)
    for n in [1,2,3,5,10,20,21,60]: x[f"ret_{n}"]=c.pct_change(n)
    x["hl_range"]=(h-l)/c.replace(0,np.nan); x["oc_change"]=(c-o)/o.replace(0,np.nan)
    x["upper_shadow"]=(h-np.maximum(o,c))/c.replace(0,np.nan); x["lower_shadow"]=(np.minimum(o,c)-l)/c.replace(0,np.nan)
    for n in [5,10,20,60]: x[f"ma_{n}_ratio"]=c/c.rolling(n).mean()-1
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean()
    x["ema_12_ratio"]=c/e12-1; x["ema_26_ratio"]=c/e26-1; x["macd"]=macd; x["macd_signal"]=sig; x["macd_hist"]=macd-sig
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean(); rs=gain/loss.replace(0,np.nan); x["rsi_14"]=100-100/(1+rs)
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); x["atr_14"]=tr.rolling(14).mean()/c
    for n in [5,10,20,60]: x[f"volatility_{n}"]=x["ret_1"].rolling(n).std()
    x["volume_change"]=v.pct_change(); x["volume_ratio_5"]=v/v.rolling(5).mean(); x["volume_ratio_20"]=v/v.rolling(20).mean(); x["volume_ratio_60"]=v/v.rolling(60).mean()
    for n in [20,60]:
        lo=c.rolling(n).min(); hi=c.rolling(n).max(); x[f"position_{n}"]=(c-lo)/(hi-lo).replace(0,np.nan); x[f"zscore_{n}"]=(c-c.rolling(n).mean())/c.rolling(n).std()
    x["day_of_week"]=x.date.dt.dayofweek; x["month"]=x.date.dt.month; x["quarter"]=x.date.dt.quarter
    if benchmark is not None and not benchmark.empty:
        b=benchmark[["date","close"]].rename(columns={"close":"benchmark_close"}).sort_values("date")
        x=pd.merge_asof(x.sort_values("date"), b, on="date", direction="backward")
        for n in [1,5,20,60]: x[f"benchmark_ret_{n}"]=x["benchmark_close"].pct_change(n)
    else:
        for n in [1,5,20,60]: x[f"benchmark_ret_{n}"]=0.0
    x["relative_ret_20"]=x["ret_20"]-x["benchmark_ret_20"]; x["relative_ret_60"]=x["ret_60"]-x["benchmark_ret_60"]
    future=x.groupby(lambda _: True)["close"].shift(-HORIZON)
    x["target_price"]=future; x["target_return"]=future/x["close"]-1; x["target_direction"]=(x["target_return"]>0).astype(int)
    x=x.replace([np.inf,-np.inf],np.nan)
    return x

def prepare_training(frames: dict[str,pd.DataFrame], benchmark: pd.DataFrame|None=None):
    parts=[]
    for code,df in frames.items():
        if df.empty: continue
        z=add_features(df, benchmark); z["stock_code"]=code; parts.append(z)
    if not parts: raise RuntimeError("沒有可用的產業歷史資料")
    all_df=pd.concat(parts,ignore_index=True).sort_values(["date","stock_code"]).reset_index(drop=True)
    all_df=all_df.dropna(subset=["target_return","target_price","close"])
    for c in FEATURES:
        all_df[c]=pd.to_numeric(all_df[c],errors="coerce")
        med=all_df[c].median(); all_df[c]=all_df[c].replace([np.inf,-np.inf],np.nan).fillna(0 if not np.isfinite(med) else med)
    return all_df

def models():
    return (RandomForestRegressor(n_estimators=500,max_depth=14,min_samples_leaf=3,max_features=0.75,random_state=RANDOM_STATE,n_jobs=-1), RandomForestClassifier(n_estimators=500,max_depth=14,min_samples_leaf=3,max_features=0.75,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1))

def oof(df:pd.DataFrame):
    n=len(df); splits=min(5,max(2,n//50)); tscv=TimeSeriesSplit(n_splits=splits); preds=[]; yt=[]; yp=[]; yc=[]; ycp=[]
    X=df[FEATURES]; yr=df.target_return.astype(float); yc_true=df.target_direction.astype(int)
    for tr,te in tscv.split(X):
        reg,clf=models(); reg.fit(X.iloc[tr],yr.iloc[tr]); clf.fit(X.iloc[tr],yc_true.iloc[tr]); pr=reg.predict(X.iloc[te]); pc=clf.predict(X.iloc[te])
        preds.extend(te.tolist()); yt.extend(yr.iloc[te]); yp.extend(pr); yc.extend(yc_true.iloc[te]); ycp.extend(pc)
    return {"Accuracy":accuracy_score(yc,ycp),"Precision":precision_score(yc,ycp,zero_division=0),"Recall":recall_score(yc,ycp,zero_division=0),"F1":f1_score(yc,ycp,zero_division=0),"MAE_return":mean_absolute_error(yt,yp),"RMSE_return":float(np.sqrt(mean_squared_error(yt,yp)))}

def cache_key(sector,codes): return hashlib.sha1((sector+"|"+"|".join(sorted(codes))).encode()).hexdigest()[:16]

def train_sector(client:FugleClient, sector:str, codes:list[str], benchmark_code="0050", days=900, progress=None):
    codes=[clean_symbol(c) for c in dict.fromkeys(codes)]
    key=cache_key(sector,codes); model_path=MODEL_DIR/f"{sector}_{key}.pkl"
    if model_path.exists(): return joblib.load(model_path)
    benchmark=pd.DataFrame()
    if benchmark_code not in codes:
        try: benchmark=client.historical(benchmark_code,days)
        except Exception: benchmark=pd.DataFrame()
    frames={}
    for i,code in enumerate(codes):
        try: frames[code]=client.historical(code,days)
        except Exception: frames[code]=pd.DataFrame()
        if progress: progress(i+1,len(codes),code)
    usable=[k for k,v in frames.items() if len(v)>=100]
    frames={k:frames[k] for k in usable}
    if len(frames)<2: raise RuntimeError(f"產業「{sector}」有效股票不足，無法共同訓練。有效數：{len(frames)}")
    df=prepare_training(frames,benchmark)
    metrics=oof(df)
    reg,clf=models(); reg.fit(df[FEATURES],df.target_return.astype(float)); clf.fit(df[FEATURES],df.target_direction.astype(int))
    package={"sector":sector,"training_stocks":usable,"features":FEATURES,"regressor":reg,"classifier":clf,"metrics":metrics,"created_at":datetime.now().isoformat(),"horizon":1}
    joblib.dump(package,model_path); return package

def predict_latest(client:FugleClient, package, symbol:str, benchmark_code="0050"):
    hist=client.historical(symbol,900)
    if hist.empty: raise RuntimeError(f"{symbol} 沒有歷史資料")
    benchmark=pd.DataFrame()
    try: benchmark=client.historical(benchmark_code,900)
    except Exception: pass
    feat=add_features(hist,benchmark)
    latest=feat.iloc[[-1]].copy()
    for c in FEATURES: latest[c]=pd.to_numeric(latest[c],errors="coerce").fillna(0)
    X=latest[FEATURES]
    ret=float(package["regressor"].predict(X)[0]); price=float(latest.close.iloc[0]); pred=price*(1+ret)
    direction="上漲" if int(package["classifier"].predict(X)[0])==1 else "下跌"
    # 用 OOF return RMSE 做保守價格區間，若沒有就以 ±1 std。
    spread=float(package["metrics"].get("RMSE_return",0.02)); low=max(0,pred*(1-spread)); high=pred*(1+spread)
    return {"stock_code":symbol,"latest_date":str(pd.Timestamp(latest.date.iloc[0]).date()),"prediction_date":str((pd.Timestamp(latest.date.iloc[0])+pd.offsets.BDay(1)).date()),"current_price":price,"predicted_return":ret,"predicted_price":pred,"lower_price":low,"upper_price":high,"direction":direction,"metrics":package["metrics"],"training_stocks":package["training_stocks"],"sector":package["sector"]}
