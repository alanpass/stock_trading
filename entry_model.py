# -*- coding: utf-8 -*-
"""Lightweight intraday entry model used by the dashboard."""
from __future__ import annotations
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class EntryModel:
    """Joblib-safe container for the intraday entry model."""

    def __init__(
        self,
        reg,
        clf,
        features,
        horizon_bars,
        train_samples,
        train_days,
        oof_mae,
        oof_direction_accuracy,
        oof_opportunity_accuracy,
    ):
        self.reg = reg
        self.clf = clf
        self.features = features
        self.horizon_bars = horizon_bars
        self.train_samples = train_samples
        self.train_days = train_days
        self.oof_mae = oof_mae
        self.oof_direction_accuracy = oof_direction_accuracy
        self.oof_opportunity_accuracy = oof_opportunity_accuracy


def _symbol_path(base, code, key):
    p=Path(base)/"entry"; p.mkdir(parents=True,exist_ok=True); return p/f"{code}_{key}.joblib"


def load_training_intraday(client, code, days=140):
    end=pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize(); start=end-pd.Timedelta(days=days)
    parts=[]; cur=start
    while cur<=end:
        ce=min(cur+pd.Timedelta(days=329),end)
        try:
            p=client.historical_candles(code,cur.date(),ce.date(),"5")
            if not p.empty: parts.append(p)
        except Exception: pass
        cur=ce+pd.Timedelta(days=1)
    if not parts:
        raise RuntimeError("無法取得5分鐘歷史K線")
    return pd.concat(parts,ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _features(d):
    x=d.copy().sort_values("date"); c=pd.to_numeric(x.close,errors="coerce"); o=pd.to_numeric(x.open,errors="coerce"); h=pd.to_numeric(x.high,errors="coerce"); l=pd.to_numeric(x.low,errors="coerce"); v=pd.to_numeric(x.volume,errors="coerce").fillna(0)
    rng=(h-l).replace(0,np.nan); x["body"]=(c-o)/rng; x["upper"]=(h-np.maximum(o,c))/rng; x["lower"]=(np.minimum(o,c)-l)/rng; x["ret1"]=c.pct_change(); x["ret3"]=c.pct_change(3); x["ret6"]=c.pct_change(6); x["ma3"]=c/c.rolling(3).mean()-1; x["ma6"]=c/c.rolling(6).mean()-1; x["ma12"]=c/c.rolling(12).mean()-1; x["vol6"]=v/v.rolling(6).mean(); typical=(h+l+c)/3; x["vwap"]=(typical*v).cumsum()/v.cumsum().replace(0,np.nan); x["vwap_gap"]=c/x["vwap"]-1; x["position12"]=(c-l.rolling(12).min())/(h.rolling(12).max()-l.rolling(12).min()).replace(0,np.nan)
    x["future_low"] = c.shift(-6).rolling(6,min_periods=1).min().shift(-5); x["future_low_return"]=x["future_low"]/c-1; x["opportunity"]=((x["future_low_return"]<=-0.002)&(x["future_low_return"]>=-0.02)).astype(float)
    return x


def fit_model(hist,horizon_bars=6):
    x=_features(hist); cols=["body","upper","lower","ret1","ret3","ret6","ma3","ma6","ma12","vol6","vwap_gap","position12"]
    train=x.dropna(subset=cols+['future_low_return']).copy(); X=train[cols].fillna(0); y=train.future_low_return; yc=train.opportunity.astype(int)
    reg=RandomForestRegressor(n_estimators=300,max_depth=10,min_samples_leaf=5,random_state=42,n_jobs=-1).fit(X,y)
    clf=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=5,class_weight='balanced',random_state=42,n_jobs=-1).fit(X,yc)
    pred=reg.predict(X); p=clf.predict(X); mae=float(np.mean(np.abs(y-pred))); acc=float(np.mean(yc==p)); return EntryModel(reg, clf, cols, horizon_bars, len(train), train.date.dt.date.nunique(), mae, acc, acc)


def save_saved_model(code,key,model): joblib.dump(model,_symbol_path(Path(__file__).resolve().parent/'models',code,key))
def load_saved_model(code,key):
    p=_symbol_path(Path(__file__).resolve().parent/'models',code,key)
    return joblib.load(p) if p.exists() else None


def predict_entry(model,k5,live_price):
    x=_features(k5); row=x.iloc[-1]; X=row[model.features].to_frame().T.fillna(0); low=float(model.reg.predict(X)[0]); opp=float(model.clf.predict_proba(X)[0,1]) if hasattr(model.clf,'predict_proba') else 0.5
    c=float(live_price); vwap=float(row.vwap) if np.isfinite(row.vwap) else c; atr=float((k5.high-k5.low).rolling(12).mean().iloc[-1]) if len(k5)>=12 else max(c*0.003,0.01); best=max(0.0,c*(1+low)); lower=best-0.4*atr; upper=best+0.4*atr
    trend=0; trend += 25 if row.ma3>0 else 0; trend += 25 if row.ma6>0 else 0; trend += 20 if row.ma12>0 else 0; trend += 20 if c>vwap else 0; trend += 10 if row.ret3>0 else 0
    score=float(np.clip(0.55*trend+45*opp,0,100)); over=bool(c>vwap+atr); action='適合回檔進場' if trend>=68 and score>=68 and not over else '上漲趨勢，但不追高' if trend>=55 and over else '等待回檔確認' if trend>=55 and score>=52 else '不建議進場'
    pattern='大陽線' if row.body>0.6 else '大陰線' if row.body<-0.6 else '下影線陽線' if row.lower>0.55 and row.body>0 else '上影線陽線' if row.upper>0.55 and row.body>0 else '十字線' if abs(row.body)<0.08 else '一般K線'
    conf=float(np.clip(0.5+abs(score-50)/100,0,0.95))
    support=float(k5.low.tail(12).min()); rebound=float(c+max(atr,0)*0.8)
    reason='；'.join(["趨勢偏弱" if trend<55 else "趨勢偏多", "價格偏離VWAP/日內高點" if over else "K線結構尚可"]) 
    return {'action':action,'decision_reason':reason,'reference_price':c,'best_entry_price':best,'entry_score':score,'confidence':conf,'trend_score':trend,'pattern_name':pattern,'opportunity_probability':opp,'vwap':vwap,'trend_state':'偏強／上漲' if trend>=68 else '區間震盪' if trend>=45 else '偏弱／下跌','pattern_score':float(row.body),'entry_lower':lower,'entry_upper':upper,'predicted_future_low':best,'overextended':over,'support_price':support,'predicted_rebound_price':rebound,'atr_price':atr,'volume_ratio_6':float(row.vol6) if np.isfinite(row.vol6) else 1.0,'feature_time':row.date}
