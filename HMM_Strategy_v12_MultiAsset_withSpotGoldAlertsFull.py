#!/usr/bin/env python
# coding: utf-8

"""
HMM Multi-Asset v12 Telegram Bot: Signal-Change with GCS persistence.

- Stores last signal ("BUY"/"SELL") per asset in GCS last_signal.json.
- Only fires Telegram alert on a *signal* change, not regime number.
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

# --- GCS storage ---------------------------------------------
from google.cloud import storage

def download_last_signals(bucket_name, file_name='last_signal.json'):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    if blob.exists():
        return json.loads(blob.download_as_text())
    else:
        return {}

def upload_last_signals(last_signals, bucket_name, file_name='last_signal.json'):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(json.dumps(last_signals))

# --- Telegram config -----------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET:
    raise RuntimeError("Environment variable GCS_BUCKET_NAME not set")

# --- Assets --------------------------------------------------
assets = {
    'SPY':   'SPY',
    'TSLA':  'TSLA',
    'BYD':   '1211.HK',
    'GOLD':  'GC=F',
    'DBS':   'D05.SI'
}
START_DATE = '2010-01-01'
END_DATE   = pd.Timestamp.today().strftime('%Y-%m-%d')

sia = SentimentIntensityAnalyzer()

# --- Download persistent last signals ------------------------
last_signals = download_last_signals(GCS_BUCKET)

# --- TRAINING LOOP -------------------------------------------
results = {}
for name, ticker in assets.items():
    # 1) News Sentiment
    feed = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    # 2) Download price history
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    # 3) VIX Z-score
    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vix_col = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vix_col] - vix[vix_col].rolling(20).mean()) / vix[vix_col].rolling(20).std()).fillna(0)

    # 4) Put/Call Ratio Z-score
    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options',
                        headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr_val = float(el.text) if el and el.text.strip() else 0.0
    s1 = pd.Series(pcr_val, index=df.index)
    df['PCR'] = ((s1 - s1.rolling(20).mean()) / s1.rolling(20).std()).fillna(0)

    # 5) Feature Engineering
    close_col = next(c for c in df.columns if 'Close' in c and not c.startswith('Adj'))
    vol_col = next((c for c in df.columns if 'Volume' in c), None)
    df['LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
    macd = ta.trend.MACD(close=df[close_col])
    df['MACD'] = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(close=df[close_col]).rsi()
    if vol_col:
        df['Volume_Z'] = (df[vol_col] - df[vol_col].rolling(20).mean()) / df[vol_col].rolling(20).std()

    features = ['LogReturn', 'MACD', 'MACD_diff', 'RSI', 'NewsSentiment', 'VIX', 'PCR']
    if 'Volume_Z' in df.columns:
        features.append('Volume_Z')
    df.dropna(subset=features, inplace=True)

    # 6) Train HMM
    scaler = StandardScaler().fit(df[features])
    X = scaler.transform(df[features])
    model = GaussianHMM(n_components=3, covariance_type='diag',
                        n_iter=1000, tol=1e-4, random_state=42)
    model.fit(X)
    df['HiddenState'] = model.predict(X)

    # 7) Identify positive regimes
    state_ret = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret > 0].index.tolist()

    # Save model + scaler
    joblib.dump(model, f'hmm_{ticker.lower()}_v12_diag_2signal.pkl')
    joblib.dump(scaler, f'scaler_{ticker.lower()}_v12_diag_2signal.pkl')

    results[ticker] = {
        'model':      model,
        'scaler':     scaler,
        'features':   features,
        'pos_states': pos_states,
        'close_col':  close_col,
        'cum_market': np.exp(df['LogReturn'].cumsum()).iloc[-1],
        'cum_hmm':    np.exp((df['LogReturn'] * df['HiddenState'].isin(pos_states)).cumsum()).iloc[-1]
    }

    print(f"{ticker}: Buy&Hold → {results[ticker]['cum_market']:.4f}, HMM → {results[ticker]['cum_hmm']:.4f}")

# --- ALERT LOOP -------------------------------------------------------
LOOKBACK = 60
for name, ticker in assets.items():
    info = results[ticker]
    model      = info['model']
    scaler     = info['scaler']
    features   = info['features']
    pos_states = info['pos_states']
    close_col  = info['close_col']

    df2 = yf.download(ticker, period=f"{LOOKBACK}d", interval="1d", progress=False)
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [' '.join(c).strip() for c in df2.columns.values]
    if len(df2) < LOOKBACK // 2:
        print(f"{ticker}: Not enough data for alert evaluation.")
        continue

    # Recompute features
    df2['NewsSentiment'] = np.mean([sia.polarity_scores(e.title)['compound']
                                    for e in feedparser.parse('https://finance.yahoo.com/news/rss').entries]) or 0.0

    v2 = yf.download('^VIX', start=df2.index.min(), end=df2.index.max(), progress=False)
    if isinstance(v2.columns, pd.MultiIndex):
        v2.columns = [' '.join(c).strip() for c in v2.columns.values]
    vc2 = next(c for c in v2.columns if 'Close' in c)
    df2['VIX'] = ((v2[vc2] - v2[vc2].rolling(20).mean()) / v2[vc2].rolling(20).std()).fillna(0)

    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options',
                        headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr = float(el.text) if el and el.text.strip() else 0.0
    spcr = pd.Series(pcr, index=df2.index)
    df2['PCR'] = ((spcr - spcr.rolling(20).mean()) / spcr.rolling(20).std()).fillna(0)

    df2['LogReturn'] = np.log(df2[close_col] / df2[close_col].shift(1))
    m2 = ta.trend.MACD(close=df2[close_col])
    df2['MACD'] = m2.macd()
    df2['MACD_diff'] = m2.macd_diff()
    df2['RSI'] = ta.momentum.RSIIndicator(close=df2[close_col]).rsi()
    vcol2 = next((c for c in df2.columns if 'Volume' in c), None)
    if vcol2:
        df2['Volume_Z'] = (df2[vcol2] - df2[vcol2].rolling(20).mean()) / df2[vcol2].rolling(20).std()

    df2.dropna(subset=features, inplace=True)
    if df2.empty:
        print(f"{ticker}: Not enough feature data for alert evaluation.")
        continue

    tail = df2.iloc[-2:]
    X2 = scaler.transform(tail[features].values)
    prev_s, curr_s = model.predict(X2)[-2:]
    curr_signal = "BUY" if curr_s in pos_states else "SELL"

    # BH vs HMM ratio from full backtest
    cumM = info['cum_market']
    cumH = info['cum_hmm']
    ratio_text = "N/A" if cumM == 0 else f"{(cumH / cumM):.2f}×"

    signal_icon = "✅ ENTER / BUY" if curr_signal == "BUY" else "🚫 EXIT / SELL"
    price  = tail[close_col].iat[-1]
    date   = tail.index[-1].date()

    msg = (
        f"📊 HMM v12 Alert — {ticker}\n"
        f"Date: {date}\n"
        f"Prev→Curr: {prev_s} → {curr_s}\n"
        f"Signal:   {signal_icon}\n"
        f"Price:    ${price:.2f}\n"
        f"BH vs HMM:{ratio_text}"
    )

    last_for_ticker = last_signals.get(ticker)
    if last_for_ticker != curr_signal:
        requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg})
        last_signals[ticker] = curr_signal
        upload_last_signals(last_signals, GCS_BUCKET)
        print(f"{ticker}: Sent alert (Prev→Curr: {prev_s}→{curr_s}, Signal: {signal_icon}, Ratio: {ratio_text})")
    else:
        print(f"{ticker}: No signal change ({last_for_ticker}→{curr_signal}), no alert sent. (Signal would be: {signal_icon}, Ratio: {ratio_text})")

