# -*- coding: utf-8 -*-
"""Compact multi-stock next-day AI engine compatible with the dashboard."""
from __future__ import annotations
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

FEATURES = [
    "ret_1","ret_2","ret_3","ret_5","ret_10","ret_20","ret_21","ret_60",
    "hl_range","oc_change","upper_shadow","lower_shadow","ma_5_ratio","ma_10_ratio",
    "ma_20_ratio","ma_60_ratio","ema_12_ratio","ema_26_ratio","macd","macd_signal",
    "macd_hist","rsi_14","volatility_5","volatility_10","volatility_20","volatility_60",
    "atr_14","volume_change","volume_ratio_5","volume_ratio_20","volume_ratio_60",
    "position_20","position_60","zscore_20","zscore_60","day_of_week","month","quarter",
    "benchmark_ret_1","benchmark_ret_5","benchmark_ret_20","benchmark_ret_60","relative_ret_20","relative_ret_60"
]


def _next_trading_date(date_value):
    """取得預測基準日之後的下一個交易日（排除週末）。"""
    d = pd.Timestamp(date_value).normalize()
    nd = d + pd.Timedelta(days=1)
    while nd.weekday() >= 5:
        nd += pd.Timedelta(days=1)
    return nd


def _normalise_metrics(metrics):
    """相容舊版 joblib 模型，補齊 Dashboard 需要的 OOF 指標欄位。"""
    src = dict(metrics or {})
    aliases = {
        "MAE": ["mae", "MAE"],
        "RMSE": ["rmse", "RMSE"],
        "MAPE": ["mape", "MAPE"],
        "Accuracy": ["accuracy", "Accuracy"],
        "Precision": ["precision", "Precision"],
        "Recall": ["recall", "Recall"],
        "F1": ["f1", "F1"],
        "RegressionDirectionAccuracy": ["regression_direction_accuracy", "RegressionDirectionAccuracy"],
        "ClassificationDirectionAccuracy": ["classification_direction_accuracy", "ClassificationDirectionAccuracy"],
    }
    out = {}
    for key, names in aliases.items():
        value = np.nan
        for name in names:
            if name in src:
                value = src[name]
                break
        try:
            value = float(value)
        except Exception:
            value = np.nan
        out[key] = value
    return out


def _rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ag=up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al=dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-100/(1+rs)


def _atr(df,n=14):
    prev=df["close"].shift(1)
    tr=pd.concat([(df.high-df.low),(df.high-prev).abs(),(df.low-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_features(df, benchmark=None):
    x=df.copy(); x["date"]=pd.to_datetime(x["date"],errors="coerce")
    if "stock_code" not in x.columns: x["stock_code"]="UNKNOWN"
    x=x.dropna(subset=["date","close"]).sort_values(["stock_code","date"]).reset_index(drop=True)
    parts=[]
    for code,g in x.groupby("stock_code",sort=False):
        g=g.copy(); c=pd.to_numeric(g.close,errors="coerce"); o=pd.to_numeric(g.open,errors="coerce"); h=pd.to_numeric(g.high,errors="coerce"); l=pd.to_numeric(g.low,errors="coerce"); v=pd.to_numeric(g.volume,errors="coerce").fillna(0)
        g["ret_1"]=c.pct_change(1); g["ret_2"]=c.pct_change(2); g["ret_3"]=c.pct_change(3); g["ret_5"]=c.pct_change(5); g["ret_10"]=c.pct_change(10); g["ret_20"]=c.pct_change(20); g["ret_21"]=c.pct_change(21); g["ret_60"]=c.pct_change(60)
        rng=(h-l).replace(0,np.nan); body=(c-o); g["hl_range"]=rng/c; g["oc_change"]=body/o.replace(0,np.nan); g["upper_shadow"]=(h-np.maximum(o,c))/c; g["lower_shadow"]=(np.minimum(o,c)-l)/c
        for n in [5,10,20,60]: g[f"ma_{n}_ratio"]=c/c.rolling(n).mean()-1
        e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); macd=e12-e26; sig=macd.ewm(span=9,adjust=False).mean(); g["ema_12_ratio"]=c/e12-1; g["ema_26_ratio"]=c/e26-1; g["macd"]=macd; g["macd_signal"]=sig; g["macd_hist"]=macd-sig
        g["rsi_14"]=_rsi(c); g["volatility_5"]=c.pct_change().rolling(5).std(); g["volatility_10"]=c.pct_change().rolling(10).std(); g["volatility_20"]=c.pct_change().rolling(20).std(); g["volatility_60"]=c.pct_change().rolling(60).std(); g["atr_14"]=_atr(g)/c
        g["volume_change"]=v.pct_change(); g["volume_ratio_5"]=v/v.rolling(5).mean(); g["volume_ratio_20"]=v/v.rolling(20).mean(); g["volume_ratio_60"]=v/v.rolling(60).mean()
        r20=h.rolling(20).max()-l.rolling(20).min(); r60=h.rolling(60).max()-l.rolling(60).min(); g["position_20"]=(c-l.rolling(20).min())/r20.replace(0,np.nan); g["position_60"]=(c-l.rolling(60).min())/r60.replace(0,np.nan)
        g["zscore_20"]=(c-c.rolling(20).mean())/c.rolling(20).std(); g["zscore_60"]=(c-c.rolling(60).mean())/c.rolling(60).std(); g["day_of_week"]=g.date.dt.dayofweek; g["month"]=g.date.dt.month; g["quarter"]=g.date.dt.quarter
        future=c.shift(-1); g["target_return"]=future/c-1; g["target_price"]=future; g["target_direction"]=(g["target_return"]>0).astype(int)
        parts.append(g)
    x=pd.concat(parts,ignore_index=True) if parts else x
    if benchmark is not None and not benchmark.empty:
        b=benchmark.copy(); b["date"]=pd.to_datetime(b.date,errors="coerce"); b=b.sort_values("date"); bc=pd.to_numeric(b.close,errors="coerce"); bp=bc.pct_change()
        bm=pd.DataFrame({"date":b.date,"benchmark_ret_1":bp,"benchmark_ret_5":bc.pct_change(5),"benchmark_ret_20":bc.pct_change(20),"benchmark_ret_60":bc.pct_change(60)})
        x=pd.merge_asof(x.sort_values("date"),bm.sort_values("date"),on="date",direction="backward")
        x["relative_ret_20"]=x["ret_20"]-x["benchmark_ret_20"]; x["relative_ret_60"]=x["ret_60"]-x["benchmark_ret_60"]
    else:
        for c in ["benchmark_ret_1","benchmark_ret_5","benchmark_ret_20","benchmark_ret_60","relative_ret_20","relative_ret_60"]: x[c]=0.0
    for f in FEATURES:
        x[f]=pd.to_numeric(x.get(f,0),errors="coerce")
    return x


def _reg(): return RandomForestRegressor(n_estimators=500,max_depth=14,min_samples_leaf=3,max_features=0.75,random_state=42,n_jobs=-1)
def _clf(): return RandomForestClassifier(n_estimators=500,max_depth=14,min_samples_leaf=3,max_features=0.75,class_weight="balanced",random_state=42,n_jobs=-1)


def train(df, selected=None):
    x=df.copy(); x=x[np.isfinite(x[FEATURES].replace([np.inf,-np.inf],np.nan)).all(axis=1)].copy(); x=x.dropna(subset=["target_return","target_price"])
    med=x[FEATURES].median(numeric_only=True); x[FEATURES]=x[FEATURES].fillna(med).fillna(0.0)
    x=x.sort_values(["date","stock_code"]).reset_index(drop=True); X=x[FEATURES]; y=x.target_return.astype(float); yc=x.target_direction.astype(int)
    reg=_reg(); clf=_clf(); tscv=TimeSeriesSplit(n_splits=min(5,max(2,len(x)//50)))
    oof_reg=np.full(len(x),np.nan); oof_cls=np.full(len(x),np.nan)
    for tr,va in tscv.split(X):
        rr=_reg(); rr.fit(X.iloc[tr],y.iloc[tr]); oof_reg[va]=rr.predict(X.iloc[va])
        if yc.iloc[tr].nunique()>1:
            cc=_clf(); cc.fit(X.iloc[tr],yc.iloc[tr]); oof_cls[va]=cc.predict(X.iloc[va])
        else:
            oof_cls[va]=(oof_reg[va]>0).astype(int)
    valid=np.isfinite(oof_reg)
    yv=y.iloc[np.where(valid)[0]]; prv=oof_reg[valid]; cv=np.isfinite(oof_cls); acc=accuracy_score(yc.iloc[np.where(cv)[0]],oof_cls[cv]) if cv.any() else np.nan
    reg_dir=accuracy_score((yv>0).astype(int),(prv>0).astype(int)) if len(yv) else 0.5
    prec=precision_score(yc.iloc[np.where(cv)[0]],oof_cls[cv],zero_division=0) if cv.any() else np.nan; rec=recall_score(yc.iloc[np.where(cv)[0]],oof_cls[cv],zero_division=0) if cv.any() else np.nan; f1=f1_score(yc.iloc[np.where(cv)[0]],oof_cls[cv],zero_division=0) if cv.any() else np.nan
    reg.fit(X,y); clf.fit(X,yc)
    feature_profile = {}
    for feat in FEATURES:
        vals = pd.to_numeric(X[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            feature_profile[feat] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "q05": float(vals.quantile(0.05)),
                "q50": float(vals.quantile(0.50)),
                "q95": float(vals.quantile(0.95)),
            }
    # OOF 可能因前幾折沒有預測而只涵蓋部分樣本；
    # 必須與 prv 使用相同的 valid mask，不能拿完整 y 直接相減。
    residual=(yv.to_numpy(dtype=float)-prv)
    q10=float(np.nanquantile(residual,0.10)) if np.isfinite(residual).any() else -0.01; q90=float(np.nanquantile(residual,0.90)) if np.isfinite(residual).any() else 0.01
    rw=max(0.1,reg_dir); cw=max(0.1,float(acc) if np.isfinite(acc) else 0.5); rw=rw/(rw+cw); cw=1-rw
    result=_normalise_metrics({
        "MAE": float(mean_absolute_error(yv, prv)) if len(yv) else np.nan,
        "RMSE": math.sqrt(mean_squared_error(yv, prv)) if len(yv) else np.nan,
        "MAPE": float(np.mean(np.abs((yv.to_numpy(dtype=float) - prv) / yv.to_numpy(dtype=float))) * 100)
            if len(yv) and np.all(np.abs(yv.to_numpy(dtype=float)) > 1e-12) else np.nan,
        "Accuracy": float(acc) if np.isfinite(acc) else np.nan,
        "Precision": float(prec) if np.isfinite(prec) else np.nan,
        "Recall": float(rec) if np.isfinite(rec) else np.nan,
        "F1": float(f1) if np.isfinite(f1) else np.nan,
        "RegressionDirectionAccuracy": float(reg_dir),
        "ClassificationDirectionAccuracy": float(acc) if np.isfinite(acc) else 0.5,
    })
    return {"reg":reg,"clf":clf,"features":FEATURES,"result":result,"residuals":[q10,q90],"regression_weight":rw,"classification_weight":cw,"training_samples":int(len(x)),"training_stock_count":int(x.stock_code.nunique()),"feature_profile":feature_profile}


def _bundle_path(models_dir, industry_code): return Path(models_dir)/f"ai_sector_{industry_code}.joblib"
def save_model_bundle(models_dir, industry_code, out, sector, training_date):
    p=_bundle_path(models_dir,industry_code); p.parent.mkdir(parents=True,exist_ok=True)
    bundle=dict(out)
    bundle.update({
        "sector_codes": sector,
        "training_data_date": training_date,
        "training_samples": int(out.get("training_samples", 0)),
        "training_stock_count": int(out.get("training_stock_count", len(sector))),
        "trained_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
    })
    joblib.dump(bundle, p)
    return p

def load_model_bundle(models_dir,industry_code):
    p=_bundle_path(models_dir,industry_code)
    return joblib.load(p) if p.exists() else None

def bundle_is_usable(bundle, sector, training_date):
    if not bundle:
        return False
    required = ["reg", "clf", "result", "residuals", "regression_weight", "classification_weight"]
    return (
        sorted(bundle.get("sector_codes", [])) == sorted(sector)
        and str(bundle.get("training_data_date")) == str(training_date)
        and all(k in bundle for k in required)
    )

def predict_from_bundle(bundle, feat, selected):
    d=feat[feat.stock_code.astype(str)==str(selected)].copy().sort_values("date")
    if d.empty: raise RuntimeError(f"找不到 {selected} 的最新特徵")
    row=d.iloc[-1]; X=row[FEATURES].to_frame().T.fillna(0)
    pr=float(bundle["reg"].predict(X)[0]); proba=float(bundle["clf"].predict_proba(X)[0,1]) if hasattr(bundle["clf"],"predict_proba") else float(pr>0)
    rw=float(bundle.get("regression_weight",0.5)); cw=float(bundle.get("classification_weight",0.5)); reg_prob=1/(1+np.exp(-pr/0.01)); p_up=rw*reg_prob+cw*proba; p_down=1-p_up; conf=max(p_up,p_down); direction="上漲" if p_up>=0.55 else "下跌" if p_up<=0.45 else "持平"
    price=float(row.close); lower=price*(1+float(bundle.get("residuals",[-0.01,0.01])[0])); upper=price*(1+float(bundle.get("residuals",[-0.01,0.01])[1]))
    base_date = pd.Timestamp(row.date)
    return {"current_price":price,"predicted_return":pr,"predicted_price":price*(1+pr),"lower_price":lower,"upper_price":upper,"direction":direction,"confidence":float(conf),"probability_up":float(p_up),"probability_down":float(p_down),"prediction_base_date":base_date.strftime("%Y-%m-%d"),"prediction_date":_next_trading_date(base_date).strftime("%Y-%m-%d"),"metrics":_normalise_metrics(bundle.get("result", bundle.get("result_meta",{}))),"regression_weight":rw,"classification_weight":cw,"train_samples":int(bundle.get("training_samples", len(feat))),"train_stocks":int(bundle.get("training_stock_count", len(bundle.get("sector_codes", [])) or feat.stock_code.nunique()))}
