#!/usr/bin/env python
# coding: utf-8

"""
📊 HMM Strategy v12 (Global Scaler): Multi-Asset & Two-Signal Mapping
— Uses a single StandardScaler fitted on all assets
— Stores last signal (BUY/SELL) in GCS last_signal.json
— Sends Telegram only when the BUY/SELL signal flips
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

# Telegram config
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# Persistence file
STATE_FILE = Path("last_state.json")
if STATE_FILE.exists():
    last_state = json.loads(STATE_FILE.read_text())
else:
    last_state = {}

# Assets & date range
assets = {
    'SPY': 'SPY',
    'TSLA': 'TSLA',
    'BYD': '1211.HK',
    'GOLD': 'GC=F',
    'DBS': 'D05.SI'
}
START_DATE = '2010-01-01'
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

sia = SentimentIntensityAnalyzer()

# TRAINING LOOP
results = {}
for name, ticker in assets.items():
    # 1) News sentiment
    feed = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    # 2) Price history
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    # 3) VIX z-score
    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vix_col = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vix_col] - vix[vix_col].rolling(20).mean()) / vix[vix_col].rolling(20).std()).fillna(0)

    # 4) PCR z-score
    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options', headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr_val = float(el.text) if el and el.text.strip() else 0.0
    series_pcr = pd.Series(pcr_val, index=df.index)
    df['PCR'] = ((series_pcr - series_pcr.rolling(20).mean()) / series_pcr.rolling(20).std()).fillna(0)

    # 5) Feature engineering
    close_col = next(c for c in df.columns if 'Close' in c and not c.startswith('Adj'))
    vol_col = next((c for c in df.columns if 'Volume' in c), None)
    df['LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
    macd = ta.trend.MACD(close=df[close_col])
    df['MACD'] = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(close=df[close_col]).rsi()
    if vol_col:
        df['Volume_Z'] = ((df[vol_col] - df[vol_col].rolling(20).mean()) / df[vol_col].rolling(20).std())

    features = ['LogReturn','MACD','MACD_diff','RSI','NewsSentiment','VIX','PCR']
    if 'Volume_Z' in df.columns:
        features.append('Volume_Z')
    df.dropna(subset=features, inplace=True)

    # 6) Train HMM
    scaler = StandardScaler().fit(df[features])
    X = scaler.transform(df[features])
    model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, tol=1e-4, random_state=42)
    model.fit(X)
    df['HiddenState'] = model.predict(X)

    # 7) Identify positive regimes
    state_ret = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret > 0].index.tolist()

    # Persist model & scaler
    joblib.dump(model, f'hmm_{ticker.lower()}_v12.pkl')
    joblib.dump(scaler, f'scaler_{ticker.lower()}_v12.pkl')

    # Backtest cumulative
    df['StratRet'] = df['LogReturn'] * df['HiddenState'].isin(pos_states)
    results[ticker] = {'model': model, 'scaler': scaler, 'features': features, 'pos_states': pos_states, 'close_col': close_col}

# ALERT LOOP
LOOKBACK = 60
for name, ticker in assets.items():
    info = results[ticker]
    model, scaler, features, pos_states, close_col = (
        info['model'], info['scaler'], info['features'], info['pos_states'], info['close_col']
    )
    df2 = yf.download(ticker, period=f"{LOOKBACK}d", interval='1d', progress=False)
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [' '.join(c).strip() for c in df2.columns.values]
    if len(df2) < LOOKBACK//2:
        print(f"{ticker}: not enough data for alert evaluation.")
        continue

    # recompute features on df2
    df2['NewsSentiment'] = np.mean([sia.polarity_scores(e.title)['compound'] for e in feedparser.parse('https://finance.yahoo.com/news/rss').entries]) or 0.0
    v2 = yf.download('^VIX', start=df2.index.min(), end=df2.index.max(), progress=False)
    if isinstance(v2.columns, pd.MultiIndex):
        v2.columns = [' '.join(c).strip() for c in v2.columns.values]
    vc2 = next(c for c in v2.columns if 'Close' in c)
    df2['VIX'] = ((v2[vc2] - v2[vc2].rolling(20).mean()) / v2[vc2].rolling(20).std()).fillna(0)
    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options', headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr = float(el.text) if el and el.text.strip() else 0.0
    series_pcr2 = pd.Series(pcr, index=df2.index)
    df2['PCR'] = ((series_pcr2 - series_pcr2.rolling(20).mean()) / series_pcr2.rolling(20).std()).fillna(0)
    df2['LogReturn'] = np.log(df2[close_col] / df2[close_col].shift(1))
    m2 = ta.trend.MACD(close=df2[close_col])
    df2['MACD'] = m2.macd()
    df2['MACD_diff'] = m2.macd_diff()
    df2['RSI'] = ta.momentum.RSIIndicator(close=df2[close_col]).rsi()
    vcol2 = next((c for c in df2.columns if 'Volume' in c), None)
    if vcol2:
        df2['Volume_Z'] = ((df2[vcol2] - df2[vcol2].rolling(20).mean()) / df2[vcol2].rolling(20).std())
    df2.dropna(subset=features, inplace=True)
    if df2.empty:
        print(f"{ticker}: not enough feature data for alert evaluation.")
        continue

    tail = df2.iloc[-2:]
    X2 = scaler.transform(tail[features])
    prev_s, curr_s = model.predict(X2)[-2:]
    sig = 'BUY' if curr_s in pos_states else 'SELL'

    # build message\    
    price = tail[close_col].iat[-1]
    date = tail.index[-1].date()
    msg = (
        f"📊 HMM v12 — {ticker}\n"
        f"Date: {date}\n"
        f"Prev→Curr: {prev_s} → {curr_s}\n"
        f"Signal: { '✅ ENTER / BUY' if sig=='BUY' else '🚫 EXIT / SELL'}\n"
        f"Price: ${price:.2f}\n"
        f"BH vs HMM: (calc externally)"
    )

    # debug-print full message both branches
    if last_state.get(ticker) != sig:
        print(f"{ticker}: would send → {msg.replace(chr(10), ' | ')}")
        requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg})
        last_state[ticker] = sig
        STATE_FILE.write_text(json.dumps(last_state))
        print(f"{ticker}: sent {sig}\n")
    else:
        print(f"{ticker}: no change ({sig}); would have sent → {msg.replace(chr(10), ' | ')}\n")
