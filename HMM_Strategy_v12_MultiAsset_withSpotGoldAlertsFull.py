#!/usr/bin/env python
# coding: utf-8

"""
📊 HMM Strategy v12: Multi-Asset (Spot Gold) & Two-Signal Mapping
— Uses one global StandardScaler across all assets
— Sends Telegram alerts on “BUY”/“SELL” *signal* changes only
— Persists last signals in GCS last_signal.json
"""

import os
import json
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

# ─── GCS helpers ────────────────────────────────────────────────
def download_last_signals(bucket_name, file_name='last_signal.json'):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(file_name)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"Error downloading last_signal.json from GCS: {e}")
    return {}

def upload_last_signals(signals, bucket_name, file_name='last_signal.json'):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        bucket.blob(file_name).upload_from_string(json.dumps(signals))
    except Exception as e:
        print(f"Error uploading last_signal.json to GCS: {e}")

# ─── Config ─────────────────────────────────────────────────────
BOT_TOKEN   = os.getenv("BOT_TOKEN")
CHAT_ID     = os.getenv("CHAT_ID", "1669179604")
BASE_URL    = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
GCS_BUCKET  = "my-hmm-state"

assets = {
    'SPY':  'SPY',
    'TSLA': 'TSLA',
    'BYD':  '1211.HK',
    'GOLD': 'GC=F',
    'DBS':  'D05.SI'
}
START_DATE = '2010-01-01'
END_DATE   = pd.Timestamp.today().strftime('%Y-%m-%d')
sia        = SentimentIntensityAnalyzer()

# ─── 1) Download last signals ────────────────────────────────────
last_signals = download_last_signals(GCS_BUCKET)

# ─── 2) TRAINING: gather features for all assets ────────────────
feature_cols = ['LogReturn','MACD','MACD_diff','RSI','NewsSentiment','VIX','PCR','Volume_Z']
all_features = []
results      = {}

for name, ticker in assets.items():
    # 2.1) News sentiment
    feed   = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    # 2.2) Price history
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    # 2.3) VIX Z-score
    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vcol = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vcol] - vix[vcol].rolling(20).mean()) /
                  vix[vcol].rolling(20).std()).fillna(0)

    # 2.4) Put/Call Ratio Z-score
    resp = requests.get(
        'https://finance.yahoo.com/quote/%5EPCR/options',
        headers={'User-Agent':'Mozilla/5.0'}
    )
    soup = BeautifulSoup(resp.text, 'html.parser')
    el   = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr  = float(el.text) if el and el.text.strip() else 0.0
    s1   = pd.Series(pcr, index=df.index)
    df['PCR'] = ((s1 - s1.rolling(20).mean()) /
                 s1.rolling(20).std()).fillna(0)

    # 2.5) Indicators & returns
    close_col = next(c for c in df.columns if 'Close' in c and not c.startswith('Adj'))
    vol_col   = next((c for c in df.columns if 'Volume' in c), None)

    df['LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
    macd = ta.trend.MACD(close=df[close_col])
    df['MACD']      = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['RSI']       = ta.momentum.RSIIndicator(close=df[close_col]).rsi()

    # ── **NEW**: always ensure Volume_Z exists
    if vol_col:
        df['Volume_Z'] = ((df[vol_col] -
                           df[vol_col].rolling(20).mean()) /
                          df[vol_col].rolling(20).std()).fillna(0)
    else:
        df['Volume_Z'] = 0.0

    # 2.6) drop any rows missing our core features
    df.dropna(subset=feature_cols, inplace=True)

    # 2.7) collect for global scaling
    all_features.append(df[feature_cols].values)

    # stash for later backtest
    results[ticker] = {
        'df':         df.copy(),
        'close_col':  close_col
    }

# 2.8) fit one StandardScaler on *all* assets together
X_all = np.vstack(all_features)
scaler = StandardScaler().fit(X_all)

# 2.9) train each HMM, identify positive regimes & full backtest
for ticker, info in results.items():
    df = info['df']
    X  = scaler.transform(df[feature_cols])
    model = GaussianHMM(n_components=3, covariance_type='diag',
                        n_iter=1000, tol=1e-4, random_state=42)
    model.fit(X)
    states = model.predict(X)

    # mean return per state → positive regimes
    df['HiddenState'] = states
    state_ret = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret>0].index.tolist()

    df['InPos'] = df['HiddenState'].isin(pos_states).astype(int)

    # cumulative returns
    cumM = np.exp(df['LogReturn'].cumsum()).iloc[-1]
    cumH = np.exp((df['LogReturn'] * df['InPos']).cumsum()).iloc[-1]

    info.update({
        'model':      model,
        'pos_states': pos_states,
        'cumM':       cumM,
        'cumH':       cumH
    })

    print(f"{ticker}: Buy & Hold → {cumM:.4f}, HMM → {cumH:.4f}")

# ─── 3) ALERT LOOP ────────────────────────────────────────────────
LOOKBACK = 60
for ticker, info in results.items():
    model      = info['model']
    pos_states = info['pos_states']
    close_col  = info['close_col']

    # recent window
    df2 = yf.download(ticker, period=f"{LOOKBACK}d", interval="1d", progress=False)
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [' '.join(c).strip() for c in df2.columns.values]
    if len(df2) < LOOKBACK//2:
        print(f"{ticker}: insufficient data for alerts")
        continue

    # recompute exactly same features:
    df2['NewsSentiment'] = news_score  # same global snapshot
    # VIX
    v2 = yf.download('^VIX', start=df2.index.min(), end=df2.index.max(), progress=False)
    if isinstance(v2.columns, pd.MultiIndex):
        v2.columns = [' '.join(c).strip() for c in v2.columns.values]
    vc = next(c for c in v2.columns if 'Close' in c)
    df2['VIX'] = ((v2[vc] - v2[vc].rolling(20).mean()) /
                   v2[vc].rolling(20).std()).fillna(0)
    # PCR
    resp = requests.get(
        'https://finance.yahoo.com/quote/%5EPCR/options',
        headers={'User-Agent':'Mozilla/5.0'}
    )
    soup = BeautifulSoup(resp.text, 'html.parser')
    el   = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr  = float(el.text) if el and el.text.strip() else 0.0
    spcr = pd.Series(pcr, index=df2.index)
    df2['PCR'] = ((spcr - spcr.rolling(20).mean()) /
                   spcr.rolling(20).std()).fillna(0)
    # returns & indicators
    df2['LogReturn'] = np.log(df2[close_col]/df2[close_col].shift(1))
    m2 = ta.trend.MACD(close=df2[close_col])
    df2['MACD']      = m2.macd()
    df2['MACD_diff'] = m2.macd_diff()
    df2['RSI']       = ta.momentum.RSIIndicator(close=df2[close_col]).rsi()
    if 'Volume' in df2:
        df2['Volume_Z'] = ((df2['Volume']-
                           df2['Volume'].rolling(20).mean())/
                           df2['Volume'].rolling(20).std()).fillna(0)
    else:
        df2['Volume_Z'] = 0.0

    df2.dropna(subset=feature_cols, inplace=True)
    if df2.empty:
        print(f"{ticker}: no complete feature set, skipping")
        continue

    # last two states → signal change?
    tail       = df2.iloc[-2:]
    X2         = scaler.transform(tail[feature_cols])
    prev_s, curr_s = model.predict(X2)[-2:]
    curr_sig   = "BUY" if curr_s in pos_states else "SELL"

    # prepare message
    price   = tail[close_col].iat[-1]
    date    = tail.index[-1].date()
    ratio   = "N/A"  # could compute cumH/cumM here if desired
    icon    = "✅ ENTER / BUY" if curr_sig=="BUY" else "🚫 EXIT / SELL"
    msg     = (
      f"📊 HMM v12 — {ticker}\n"
      f"Date: {date}\n"
      f"Prev→Curr: {prev_s} → {curr_s}\n"
      f"Signal:   {icon}\n"
      f"Price:    ${price:.2f}\n"
      f"BH vs HMM:{ratio}"
    )

    last = last_signals.get(ticker)
    if last != curr_sig:
        requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg})
        last_signals[ticker] = curr_sig
        upload_last_signals(last_signals, GCS_BUCKET)
        print(f"{ticker}: sent {curr_sig}")
    else:
        print(f"{ticker}: no change ({last}→{curr_sig})")

