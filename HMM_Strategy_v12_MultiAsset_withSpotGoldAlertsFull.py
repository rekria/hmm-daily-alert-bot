#!/usr/bin/env python
# coding: utf-8

"""
📊 HMM Strategy v12: Multi-Asset (Spot Gold) & Two-Signal Mapping
— Sends Telegram alerts with “BH vs HMM” ratio, per‐asset regime memory in GCS,
  full backtest ratio, and prints “Sent alert” or “No change” for each asset.
— Fits one global StandardScaler, then trains each asset’s HMM on scaled features.
— Recomputes identical features in the alert lookback window.
— Skips assets missing lookback features.
— Stores “BUY”/“SELL” in last_signal.json on GCS; only fires on signal change.
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

# ─── GCS storage ---------------------------------------------
def download_last_signals(bucket_name="my-hmm-state",
                          file_name="last_signal.json"):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(file_name)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"Error downloading last_signal.json from GCS: {e}")
    return {}

def upload_last_signals(signals,
                        bucket_name="my-hmm-state",
                        file_name="last_signal.json"):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(file_name)
        blob.upload_from_string(json.dumps(signals))
    except Exception as e:
        print(f"Error uploading last_signal.json to GCS: {e}")

# ─── Telegram config -----------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID  = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

GCS_BUCKET = "my-hmm-state"

# ─── Assets & Dates ------------------------------------------
assets = {
    'SPY':   'SPY',
    'TSLA':  'TSLA',
    'BYD':   '1211.HK',
    'GOLD':  'GC=F',    # Spot gold futures
    'DBS':   'D05.SI'
}
START_DATE = '2010-01-01'
END_DATE   = pd.Timestamp.today().strftime('%Y-%m-%d')

# NLTK sentiment
sia = SentimentIntensityAnalyzer()

# ─── Load last signals from GCS ------------------------------
last_signals = download_last_signals(GCS_BUCKET)

# ─── TRAINING LOOP: collect features for global scaler ───────
feature_cols = [
    'LogReturn','MACD','MACD_diff','RSI',
    'NewsSentiment','VIX','PCR','Volume_Z'
]
all_X   = []
results = {}

for name, ticker in assets.items():
    # 1) News Sentiment
    feed   = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound']
                          for t in titles]) if titles else 0.0

    # 2) Download history
    df = yf.download(ticker,
                     start=START_DATE,
                     end=END_DATE,
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    # 3) VIX Z-score
    vix = yf.download('^VIX',
                      start=df.index.min(),
                      end=df.index.max(),
                      progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vix_col = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vix_col] - vix[vix_col].rolling(20).mean()) /
                 vix[vix_col].rolling(20).std()).fillna(0)

    # 4) Put/Call Ratio Z-score
    resp = requests.get(
        'https://finance.yahoo.com/quote/%5EPCR/options',
        headers={'User-Agent':'Mozilla/5.0'}
    )
    soup = BeautifulSoup(resp.text, 'html.parser')
    el   = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr_val = float(el.text) if el and el.text.strip() else 0.0
    s1  = pd.Series(pcr_val, index=df.index)
    df['PCR'] = ((s1 - s1.rolling(20).mean()) /
                 s1.rolling(20).std()).fillna(0)

    # 5) Feature Engineering
    close_col = next(c for c in df.columns
                     if 'Close' in c and not c.startswith('Adj'))
    vol_col   = next((c for c in df.columns if 'Volume' in c), None)

    # 5a) Log returns
    df['LogReturn'] = np.log(df[close_col] /
                             df[close_col].shift(1))

    # 5b) MACD, MACD_diff
    macd = ta.trend.MACD(close=df[close_col])
    df['MACD']      = macd.macd()
    df['MACD_diff'] = macd.macd_diff()

    # 5c) RSI
    df['RSI'] = ta.momentum.RSIIndicator(
        close=df[close_col]).rsi()

    # 5d) Volume Z
    if vol_col:
        df['Volume_Z'] = ((df[vol_col] -
                           df[vol_col].rolling(20).mean()) /
                          df[vol_col].rolling(20).std()).fillna(0)
    else:
        df['Volume_Z'] = 0.0

    # drop any rows missing these features
    df.dropna(subset=feature_cols, inplace=True)

    # stash df & close column for later
    results[ticker] = {
        'df':        df,
        'close_col': close_col
    }

    # collect X for scaler
    all_X.append(df[feature_cols].values)

# 6) Fit one global scaler
X_all  = np.vstack(all_X)
scaler = StandardScaler().fit(X_all)

# 7) Train HMM per asset & full backtest
for ticker, info in results.items():
    df    = info['df']
    X     = scaler.transform(df[feature_cols])
    model = GaussianHMM(n_components=3,
                        covariance_type='diag',
                        n_iter=1000,
                        tol=1e-4,
                        random_state=42)
    model.fit(X)

    # assign hidden states
    states = model.predict(X)
    df['HiddenState'] = states

    # identify positive regimes by mean log-return
    state_ret  = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret > 0].index.tolist()
    df['InPos'] = df['HiddenState'].isin(pos_states).astype(int)

    # persist model & scaler locally if desired
    joblib.dump(model,
        f"hmm_{ticker.lower()}_v12_diag_2signal.pkl")
    joblib.dump(scaler,
        f"scaler_{ticker.lower()}_v12_diag_2signal.pkl")

    # compute full backtest cumulatives
    cumM  = np.exp(df['LogReturn'].cumsum()).iloc[-1]
    cumH  = np.exp((df['LogReturn'] * df['InPos']).cumsum()).iloc[-1]
    ratio = cumH / cumM if cumM else np.nan

    # store backtest results
    info.update({
        'model':      model,
        'pos_states': pos_states,
        'cumM':       cumM,
        'cumH':       cumH,
        'ratio':      ratio
    })

    print(f"{ticker}: Buy&Hold → {cumM:.4f}, "
          f"HMM → {cumH:.4f}, Ratio → {ratio:.2f}×")

# ─── ALERT LOOP ----------------------------------------------
LOOKBACK = 60

for ticker, info in results.items():
    df2 = yf.download(ticker,
                      period=f"{LOOKBACK}d",
                      interval="1d",
                      progress=False)
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [' '.join(c).strip()
                       for c in df2.columns.values]

    # skip if too little data
    if len(df2) < LOOKBACK // 2:
        print(f"{ticker}: Not enough data for alerts")
        continue

    # — recompute ALL exactly the same features on df2 —
    # NewsSentiment
    feed2   = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles2 = [e.title for e in feed2.entries]
    news2   = (np.mean([sia.polarity_scores(t)['compound']
                        for t in titles2])
               if titles2 else 0.0)
    df2['NewsSentiment'] = news2

    # VIX z-score
    v2 = yf.download('^VIX',
                     start=df2.index.min(),
                     end=df2.index.max(),
                     progress=False)
    if isinstance(v2.columns, pd.MultiIndex):
        v2.columns = [' '.join(c).strip()
                      for c in v2.columns.values]
    vc2 = next(c for c in v2.columns if 'Close' in c)
    df2['VIX'] = ((v2[vc2] -
                   v2[vc2].rolling(20).mean()) /
                  v2[vc2].rolling(20).std()).fillna(0)

    # PCR z-score
    resp2 = requests.get(
        'https://finance.yahoo.com/quote/%5EPCR/options',
        headers={'User-Agent':'Mozilla/5.0'}
    )
    soup2 = BeautifulSoup(resp2.text, 'html.parser')
    el2   = soup2.select_one(
        "td[data-test='PUT_CALL_RATIO-value']")
    pcr2  = float(el2.text) if el2 and el2.text.strip() else 0.0
    s2    = pd.Series(pcr2, index=df2.index)
    df2['PCR'] = ((s2 -
                   s2.rolling(20).mean()) /
                  s2.rolling(20).std()).fillna(0)

    # LogReturn, MACD, MACD_diff, RSI, Volume_Z
    close2 = info['close_col']
    df2['LogReturn'] = np.log(df2[close2] /
                              df2[close2].shift(1))

    macd2 = ta.trend.MACD(close=df2[close2])
    df2['MACD']      = macd2.macd()
    df2['MACD_diff'] = macd2.macd_diff()
    df2['RSI']       = ta.momentum.RSIIndicator(
                            close=df2[close2]).rsi()

    volc2 = next((c for c in df2.columns
                  if 'Volume' in c), None)
    if volc2:
        df2['Volume_Z'] = ((df2[volc2] -
                           df2[volc2].rolling(20).mean()) /
                          df2[volc2].rolling(20).std()).fillna(0)
    else:
        df2['Volume_Z'] = 0.0

    # drop any rows missing features
    df2.dropna(subset=feature_cols, inplace=True)
    if df2.empty:
        print(f"{ticker}: No features in lookback")
        continue

    # now get last two days’ states
    tail      = df2.iloc[-2:]
    X2        = scaler.transform(tail[feature_cols])
    prev_s, curr_s = info['model'].predict(X2)[-2:]
    signal    = "BUY" if curr_s in info['pos_states'] else "SELL"

    # build ratio text
    ratio_text = f"{info['ratio']:.2f}×"

    # prepare Telegram message
    price = tail[close2].iat[-1]
    date  = tail.index[-1].date()
    icon  = "✅ ENTER / BUY" if signal=="BUY" else "🚫 EXIT / SELL"

    msg = (
      f"📊 HMM v12 — {ticker}\n"
      f"Date: {date}\n"
      f"Prev→Curr: {prev_s} → {curr_s}\n"
      f"Signal:   {icon}\n"
      f"Price:    ${price:.2f}\n"
      f"BH vs HMM:{ratio_text}"
    )

    last_sig = last_signals.get(ticker)
    if last_sig != signal:
        # send Telegram alert
        requests.post(BASE_URL,
                      json={"chat_id": CHAT_ID,
                            "text":    msg})
        last_signals[ticker] = signal
        upload_last_signals(last_signals, GCS_BUCKET)
        print(f"{ticker}: Sent alert ({signal})")
    else:
        print(f"{ticker}: No change ({last_sig}→{signal})")
