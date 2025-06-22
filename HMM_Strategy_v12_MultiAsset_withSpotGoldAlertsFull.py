#!/usr/bin/env python
# coding: utf-8

"""
📊 HMM Strategy v12 (Global Scaler): Multi-Asset & Two-Signal Mapping
— Uses a single StandardScaler fitted on all assets
— Stores last signal (BUY/SELL) in GCS last_signal.json
— Sends Telegram only when the BUY/SELL signal flips
"""

import os, json
from pathlib import Path

import nltk
nltk.download('vader_lexicon', quiet=True)

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import ta
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests
import joblib
from google.cloud import storage

# ─── CONFIG ───────────────────────────────────────────────────────────────
GCS_BUCKET      = "my-hmm-state"
STATE_FILE      = "last_signal.json"
BOT_TOKEN       = os.getenv("BOT_TOKEN")
CHAT_ID         = os.getenv("CHAT_ID", "1669179604")
BASE_URL        = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
START_DATE      = "2010-01-01"
END_DATE        = pd.Timestamp.today().strftime("%Y-%m-%d")
LOOKBACK_DAYS   = 60
assets          = {
    'SPY':'SPY','TSLA':'TSLA','BYD':'1211.HK',
    'GOLD':'GC=F','DBS':'D05.SI'
}
sia = SentimentIntensityAnalyzer()
feature_cols = ["LogReturn","MACD","MACD_diff","RSI","NewsSentiment","VIX","PCR","Volume_Z"]

# ─── GCS HELPERS ───────────────────────────────────────────────────────────
def download_last_signals():
    try:
        client = storage.Client()
        blob   = client.bucket(GCS_BUCKET).blob(STATE_FILE)
        return json.loads(blob.download_as_text()) if blob.exists() else {}
    except Exception:
        return {}

def upload_last_signals(signals):
    client = storage.Client()
    blob   = client.bucket(GCS_BUCKET).blob(STATE_FILE)
    blob.upload_from_string(json.dumps(signals))

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────
def compute_features(df):
    """Given raw df with price/volume, adds all signal columns in-place."""
    # 1) NewsSentiment
    titles = [e.title for e in feedparser.parse("https://finance.yahoo.com/news/rss").entries]
    df["NewsSentiment"] = np.mean([sia.polarity_scores(t)["compound"] for t in titles]) if titles else 0.0

    # 2) VIX z-score
    vix = yf.download("^VIX", start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [" ".join(c) for c in vix.columns]
    vcol = next(c for c in vix.columns if "Close" in c)
    vix_z = (vix[vcol] - vix[vcol].rolling(20).mean())/vix[vcol].rolling(20).std()
    df["VIX"] = vix_z.reindex(df.index).fillna(0)

    # 3) PCR z-score
    resp = requests.get("https://finance.yahoo.com/quote/%5EPCR/options", headers={"User-Agent":"Mozilla/5.0"})
    soup = BeautifulSoup(resp.text,"html.parser")
    val  = float(soup.select_one("td[data-test='PUT_CALL_RATIO-value']").text or 0)
    pcr  = pd.Series(val, index=df.index)
    df["PCR"] = (pcr - pcr.rolling(20).mean())/pcr.rolling(20).std()

    # 4) Returns & indicators
    close = next(c for c in df.columns if "Close" in c and not c.startswith("Adj"))
    df["LogReturn"] = np.log(df[close]/df[close].shift(1))
    macd = ta.trend.MACD(close=df[close])
    df["MACD"]      = macd.macd()
    df["MACD_diff"] = macd.macd_diff()
    df["RSI"]       = ta.momentum.RSIIndicator(close=df[close]).rsi()
    vol = next((c for c in df.columns if "Volume" in c),None)
    if vol:
        df["Volume_Z"] = (df[vol]-df[vol].rolling(20).mean())/df[vol].rolling(20).std()

    return df

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    # --- 1) Download full history & build global feature matrix
    all_feats = []
    for ticker in assets.values():
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns,pd.MultiIndex):
            df.columns = [" ".join(c) for c in df.columns]
        df = compute_features(df).dropna(subset=feature_cols)
        all_feats.append(df[feature_cols])
    global_df = pd.concat(all_feats)
    
    # --- 2) Fit one scaler over all assets
    scaler = StandardScaler().fit(global_df)
    joblib.dump(scaler, "global_scaler_v12.pkl")

    # --- 3) Train each HMM & record backtest cumulatives
    results = {}
    for name,ticker in assets.items():
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if isinstance(df.columns,pd.MultiIndex):
            df.columns = [" ".join(c) for c in df.columns]
        df = compute_features(df).dropna(subset=feature_cols)
        X  = scaler.transform(df[feature_cols])
        m  = GaussianHMM(3,"diag",n_iter=1000,tol=1e-4,random_state=42)
        m.fit(X)
        df["State"] = m.predict(X)
        pos = df.groupby("State")["LogReturn"].mean()>0
        pos_states = list(pos[pos].index)
        df["Pos"] = df["State"].isin(pos_states).astype(int)
        results[ticker] = {
            "model":m, "pos_states":pos_states,
            "back_bh":np.exp(df["LogReturn"].cumsum()).iat[-1],
            "back_hm":np.exp((df["LogReturn"]*df["Pos"]).cumsum()).iat[-1]
        }
        joblib.dump(m, f"hmm_{ticker.lower()}_v12.pkl")

    # --- 4) Load last signals & alert loop
    last_signals = download_last_signals()
    for name,ticker in assets.items():
        info = results[ticker]
        m, ps = info["model"], info["pos_states"]
        df2 = yf.download(ticker, period=f"{LOOKBACK_DAYS}d", interval="1d", progress=False)
        if isinstance(df2.columns,pd.MultiIndex):
            df2.columns = [" ".join(c) for c in df2.columns]
        df2 = compute_features(df2).dropna(subset=feature_cols)
        if len(df2)<2:
            print(f"{ticker}: skipping, insufficient lookback data")
            continue

        X2 = scaler.transform(df2[feature_cols])
        st = m.predict(X2)
        curr_sig = "BUY" if st[-1] in ps else "SELL"
        
        # Send if flipped
        if last_signals.get(ticker)!=curr_sig:
            price = df2[next(c for c in df2.columns if "Close" in c)].iat[-1]
            rh = info["back_hm"]/info["back_bh"]
            icon = "✅ BUY" if curr_sig=="BUY" else "🚫 SELL"
            msg = (
                f"📊 {ticker} Alert\n"
                f"Signal: {icon}\n"
                f"Price:  ${price:.2f}\n"
                f"BH vs HMM: {rh:.2f}×"
            )
            requests.post(BASE_URL, json={"chat_id":CHAT_ID,"text":msg})
            last_signals[ticker]=curr_sig
            print(f"{ticker}: sent {curr_sig}")
        else:
            print(f"{ticker}: no change ({curr_sig})")

    upload_last_signals(last_signals)


if __name__=="__main__":
    if not BOT_TOKEN:
        raise RuntimeError("Set BOT_TOKEN")
    main()
