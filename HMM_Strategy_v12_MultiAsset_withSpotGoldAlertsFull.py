#!/usr/bin/env python
# coding: utf-8
"""
📊 HMM Strategy v13: Multi-Asset & Two-Signal Mapping with BH vs HMM ratio
This script trains an HMM per asset (SPY, TSLA, BYD, GOLD, DBS), computes full-history
cumulative returns for both Buy-&-Hold and HMM strategy, and sends Telegram alerts at most once per day
only when the hidden state changes. The alert includes BH vs HMM performance ratio.
"""
import json
from pathlib import Path
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import ta
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import requests
import joblib

# ─── Persisted state file ────────────────────────────────────────────
STATE_FILE = Path("last_state.json")
if STATE_FILE.exists():
    last = json.loads(STATE_FILE.read_text())
    last_state = int(last.get("state", -1))
else:
    last_state = None
# ──────────────────────────────────────────────────────────────────────

# Telegram config\BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID   = os.environ.get("CHAT_ID") or "1669179604"
BASE_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# Assets and date range
assets = {
    'SPY': 'SPY',
    'TSLA': 'TSLA',
    'BYD': '1211.HK',
    'GOLD': 'GC=F',
    'DBS': 'D05.SI'
}
START_DATE = '2010-01-01'
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

# sentiment analyzer
sia = SentimentIntensityAnalyzer()

# training results storage
results = {}

# 1️⃣ Train & backtest each asset
for name, ticker in assets.items():
    # 1. News sentiment
    feed = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    # 2. Download history\    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    # 3. VIX z-score\    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vcol = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vcol] - vix[vcol].rolling(20).mean()) / vix[vcol].rolling(20).std()).fillna(0)

    # 4. Put/Call Ratio z-score
    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options', headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr_val = float(el.text) if el and el.text.strip() else 0.0
    series_pcr = pd.Series(pcr_val, index=df.index)
    df['PCR'] = ((series_pcr - series_pcr.rolling(20).mean()) / series_pcr.rolling(20).std()).fillna(0)

    # 5. Feature engineering
    close_col = next(c for c in df.columns if 'Close' in c and not c.startswith('Adj'))
    vol_col = next((c for c in df.columns if 'Volume' in c), None)
    df['LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
    macd = ta.trend.MACD(close=df[close_col])
    df['MACD'] = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(close=df[close_col]).rsi()
    if vol_col:
        df['Volume_Z'] = (df[vol_col] - df[vol_col].rolling(20).mean()) / df[vol_col].rolling(20).std()

    features = ['LogReturn','MACD','MACD_diff','RSI','NewsSentiment','VIX','PCR']
    if 'Volume_Z' in df.columns:
        features.append('Volume_Z')
    df.dropna(subset=features, inplace=True)

    # 6. Train HMM
    X = StandardScaler().fit_transform(df[features])
    model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, tol=1e-4, random_state=42)
    model.fit(X)
    df['HiddenState'] = model.predict(X)

    # 7. Identify positive regimes
    state_ret = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret>0].index.tolist()

    # 8. Compute full-history cumulative returns
    df['Position'] = df['HiddenState'].isin(pos_states).astype(int)
    df['StratRet'] = df['LogReturn'] * df['Position']
    df['CumulBH'] = np.exp(df['LogReturn'].cumsum())
    df['CumulHMM'] = np.exp(df['StratRet'].cumsum())
    bh_ret  = float(df['CumulBH'].iat[-1])
    hmm_ret = float(df['CumulHMM'].iat[-1])
    ratio   = hmm_ret/bh_ret if bh_ret!=0 else np.nan

    # Persist model, scaler, metrics
    scaler = StandardScaler().fit(df[features])
    joblib.dump(model, f'model_{ticker}_v13.pkl')
    joblib.dump(scaler, f'scaler_{ticker}_v13.pkl')
    results[ticker] = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'pos_states': pos_states,
        'close_col': close_col,
        'bh_ret': bh_ret,
        'hmm_ret': hmm_ret,
        'ratio': ratio
    }
    print(f"{ticker}: BH → {bh_ret:.4f}, HMM → {hmm_ret:.4f}, Ratio → {ratio:.2f}×")

# 2️⃣ Daily alert: fire once per run only when regime changes
for name, ticker in assets.items():
    info = results[ticker]
    model = info['model']
    scaler = info['scaler']
    features = info['features']
    pos_states = info['pos_states']
    close_col = info['close_col']

    df2 = yf.download(ticker, period='2d', interval='1d', progress=False)
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [' '.join(c).strip() for c in df2.columns.values]
    if len(df2)<2:
        print(f"{ticker}: insufficient recent data")
        continue

    # recompute features for df2
    # NewsSentiment
    df2['NewsSentiment'] = np.mean([sia.polarity_scores(e.title)['compound']
                                    for e in feedparser.parse('https://finance.yahoo.com/news/rss').entries]) or 0.0
    # VIX
    vix2 = yf.download('^VIX', start=df2.index.min(), end=df2.index.max(), progress=False)
    if isinstance(vix2.columns, pd.MultiIndex):
        vix2.columns = [' '.join(c).strip() for c in vix2.columns.values]
    vc2 = next(c for c in vix2.columns if 'Close' in c)
    df2['VIX'] = ((vix2[vc2] - vix2[vc2].rolling(20).mean()) / vix2[vc2].rolling(20).std()).fillna(0)
    # PCR
    resp2 = requests.get('https://finance.yahoo.com/quote/%5EPCR/options',
                         headers={'User-Agent':'Mozilla/5.0'})
    soup2 = BeautifulSoup(resp2.text, 'html.parser')
    el2 = soup2.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr2 = float(el2.text) if el2 and el2.text.strip() else 0.0
    df2['PCR'] = ((pd.Series(pcr2, index=df2.index) - pd.Series(pcr2, index=df2.index).rolling(20).mean()) /
                  pd.Series(pcr2, index=df2.index).rolling(20).std()).fillna(0)
    # price/volume
    df2['LogReturn'] = np.log(df2[close_col] / df2[close_col].shift(1))
    macd2 = ta.trend.MACD(close=df2[close_col])
    df2['MACD'] = macd2.macd()
    df2['MACD_diff'] = macd2.macd_diff()
    df2['RSI'] = ta.momentum.RSIIndicator(close=df2[close_col]).rsi()
    vol2 = next((c for c in df2.columns if 'Volume' in c), None)
    if vol2:
        df2['Volume_Z'] = (df2[vol2] - df2[vol2].rolling(20).mean()) / df2[vol2].rolling(20).std()

    df2.dropna(subset=features, inplace=True)
    if df2.empty:
        print(f"{ticker}: no feature rows")
        continue

    # predict last two states
    tail = df2.iloc[-2:]
    X2 = scaler.transform(tail[features].values)
    prev_s, curr_s = model.predict(X2)[-2:]

    # skip if unchanged
    if last_state is not None and int(curr_s)==last_state:
        print(f"{ticker}: state unchanged ({curr_s}), skip")
        continue

    # compose message
    price = tail[close_col].iat[-1]
    date  = tail.index[-1].date()
    ratio = info['ratio']
    signal= "✅ ENTER / BUY" if curr_s in pos_states else "🚫 EXIT / SELL"
    msg = (f"📊 HMM v13 Alert — {ticker}\n"
           f"Date: {date}\n"
           f"Prev→Curr: {prev_s}→{curr_s}\n"
           f"Signal: {signal}\n"
           f"Price: ${price:.2f}\n"
           f"BH vs HMM: {ratio:.2f}×")

    requests.post(BASE_URL, json={"chat_id":CHAT_ID, "text":msg})
    print(f"{ticker}: sent alert")

    # persist last state
    STATE_FILE.write_text(json.dumps({"state": int(curr_s)}))
    last_state = int(curr_s)





