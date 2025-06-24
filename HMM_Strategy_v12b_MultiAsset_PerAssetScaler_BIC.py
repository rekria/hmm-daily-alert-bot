# === HMM Strategy v12b — Cleaned for GitHub CI ===
# - Adds explicit auto_adjust=False to yfinance
# - Skips training for assets with insufficient data
# - Skips models with dead state transitions (zero-row transmat)

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

def download_last_signals(bucket_name="my-hmm-state", file_name="last_signal.json"):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(file_name)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"Error downloading last_signal.json from GCS: {e}")
    return {}

def upload_last_signals(signals, bucket_name="my-hmm-state", file_name="last_signal.json"):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(file_name)
        blob.upload_from_string(json.dumps(signals))
    except Exception as e:
        print(f"Error uploading last_signal.json to GCS: {e}")

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID  = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

GCS_BUCKET = "my-hmm-state"

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
last_signals = download_last_signals(GCS_BUCKET)

feature_cols = ['LogReturn','MACD','MACD_diff','RSI','NewsSentiment','VIX','PCR','Volume_Z']
results = {}

def compute_bic(model, X):
    logL = model.score(X)
    n_params = model.n_components**2 + 2 * model.n_components * X.shape[1] - 1
    return -2 * logL + n_params * np.log(len(X))

for name, ticker in assets.items():
    feed = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vix_col = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vix_col] - vix[vix_col].rolling(20).mean()) / vix[vix_col].rolling(20).std()).fillna(0)

    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options', headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el   = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr_val = float(el.text) if el and el.text.strip() else 0.0
    s1  = pd.Series(pcr_val, index=df.index)
    df['PCR'] = ((s1 - s1.rolling(20).mean()) / s1.rolling(20).std()).fillna(0)

    close_col = next(c for c in df.columns if 'Close' in c and not c.startswith('Adj'))
    vol_col   = next((c for c in df.columns if 'Volume' in c), None)
    df['LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
    macd = ta.trend.MACD(close=df[close_col])
    df['MACD'] = macd.macd()
    df['MACD_diff'] = macd.macd_diff()
    df['RSI'] = ta.momentum.RSIIndicator(close=df[close_col]).rsi()
    df['Volume_Z'] = ((df[vol_col] - df[vol_col].rolling(20).mean()) / df[vol_col].rolling(20).std()).fillna(0) if vol_col else 0.0

    df.dropna(subset=feature_cols, inplace=True)
    X = df[feature_cols].values
    if len(X) < 100:
        print(f"{ticker}: Skipped due to insufficient data")
        continue

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    best_bic = np.inf
    best_model = None
    best_n = None
    for n in range(2, 31):
        try:
            model = GaussianHMM(n_components=n, covariance_type='diag', n_iter=1000, tol=1e-4, random_state=42)
            model.fit(X_scaled)
            if np.any(model.transmat_.sum(axis=1) == 0):
                continue  # skip models with unvisited states
            bic = compute_bic(model, X_scaled)
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n
        except Exception as e:
            continue

    if best_model is None:
        print(f"{ticker}: No valid model found")
        continue

    model = best_model
    states = model.predict(X_scaled)
    df['HiddenState'] = states

    state_ret = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret > 0].index.tolist()
    df['InPos'] = df['HiddenState'].isin(pos_states).astype(int)

    joblib.dump(model, f"hmm_{ticker.lower()}_v12b_bic.pkl")
    joblib.dump(scaler, f"scaler_{ticker.lower()}_v12b_bic.pkl")

    cumM = np.exp(df['LogReturn'].cumsum()).iloc[-1]
    cumH = np.exp((df['LogReturn'] * df['InPos']).cumsum()).iloc[-1]
    ratio = cumH / cumM if cumM else np.nan

    results[ticker] = {
        'df': df,
        'close_col': close_col,
        'model': model,
        'scaler': scaler,
        'pos_states': pos_states,
        'cumM': cumM,
        'cumH': cumH,
        'ratio': ratio,
        'n_states': best_n
    }

    print(f"{ticker}: Best n={best_n}, Buy&Hold → {cumM:.4f}, HMM → {cumH:.4f}, Ratio → {ratio:.2f}")
