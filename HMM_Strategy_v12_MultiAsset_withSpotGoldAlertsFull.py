#!/usr/bin/env python
# coding: utf-8

# 
# # 📊 HMM Strategy v12: Multi-Asset (Spot Gold) & Two-Signal Mapping
# 
# This notebook trains the v12 HMM strategy for SPY, TSLA, BYD (1211.HK), Spot Gold (GC=F), and DBS (D05.SI),
# and sends Telegram alerts via HTTP API, with full feature recomputation.
# 

# In[1]:


# Install dependencies
get_ipython().system('pip install yfinance hmmlearn pandas numpy scikit-learn ta feedparser nltk beautifulsoup4 requests joblib matplotlib')
import nltk
nltk.download('vader_lexicon')


# In[2]:


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
import matplotlib.pyplot as plt

# Telegram config
BOT_TOKEN = "***REMOVED***"
CHAT_ID   = "1669179604"
BASE_URL  = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# Assets and date range
assets = {
    'SPY': 'SPY',
    'TSLA': 'TSLA',
    'BYD': '1211.HK',
    'GOLD': 'GC=F',  # Spot Gold futures
    'DBS': 'D05.SI'
}
START_DATE = '2010-01-01'
END_DATE = pd.Timestamp.today().strftime('%Y-%m-%d')

sia = SentimentIntensityAnalyzer()


# In[3]:


results = {}
for name, ticker in assets.items():
    # 1. News Sentiment
    feed = feedparser.parse('https://finance.yahoo.com/news/rss')
    titles = [e.title for e in feed.entries]
    news_score = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    # 2. Download price history
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(c).strip() for c in df.columns.values]
    df['NewsSentiment'] = news_score

    # 3. VIX Z-score
    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [' '.join(c).strip() for c in vix.columns.values]
    vix_col = next(c for c in vix.columns if 'Close' in c)
    df['VIX'] = ((vix[vix_col] - vix[vix_col].rolling(20).mean()) /
                 vix[vix_col].rolling(20).std()).fillna(0)

    # 4. Put/Call Ratio Z-score
    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options', headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr_val = float(el.text) if el and el.text.strip() else 0.0
    series_pcr = pd.Series(pcr_val, index=df.index)
    df['PCR'] = ((series_pcr - series_pcr.rolling(20).mean()) /
                 series_pcr.rolling(20).std()).fillna(0)

    # 5. Feature Engineering
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

    # 6. Train HMM
    X = StandardScaler().fit_transform(df[features])
    model = GaussianHMM(n_components=3, covariance_type='diag',
                        n_iter=1000, tol=1e-4, random_state=42)
    model.fit(X)
    df['HiddenState'] = model.predict(X)

    # 7. Identify positive regimes
    state_ret = df.groupby('HiddenState')['LogReturn'].mean()
    pos_states = state_ret[state_ret > 0].index.tolist()

    # Save model and scaler
    scaler = StandardScaler().fit(df[features])
    joblib.dump(model, f'hmm_{ticker.lower()}_v12_diag_2signal.pkl')
    joblib.dump(scaler, f'scaler_{ticker.lower()}_v12_diag_2signal.pkl')

    results[ticker] = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'pos_states': pos_states,
        'close_col': close_col
    }

    # 8. Print performance
    df['Position'] = df['HiddenState'].isin(pos_states).astype(int)
    df['StratRet'] = df['LogReturn'] * df['Position']
    df['CumulM'] = np.exp(df['LogReturn'].cumsum())
    df['CumulS'] = np.exp(df['StratRet'].cumsum())
    print(f"{ticker}: Buy & Hold -> {df['CumulM'].iloc[-1]:.4f}, HMM Strategy -> {df['CumulS'].iloc[-1]:.4f}")


# In[4]:


# Send Telegram alerts with a 60-day lookback
LOOKBACK = 60
for name, ticker in assets.items():
    info = results[ticker]
    model = info['model']
    scaler = info['scaler']
    features = info['features']
    pos_states = info['pos_states']
    close_col = info['close_col']

    df2 = yf.download(ticker, period=f"{LOOKBACK}d", interval="1d", progress=False)
    if isinstance(df2.columns, pd.MultiIndex):
        df2.columns = [' '.join(c).strip() for c in df2.columns.values]
    if len(df2) < LOOKBACK // 2:
        print(f"{ticker}: insufficient data")
        continue

    # Recompute features on df2
    df2['NewsSentiment'] = np.mean([sia.polarity_scores(e.title)['compound']
                                    for e in feedparser.parse('https://finance.yahoo.com/news/rss').entries]) or 0.0

    vix2 = yf.download('^VIX', start=df2.index.min(), end=df2.index.max(), progress=False)
    if isinstance(vix2.columns, pd.MultiIndex):
        vix2.columns = [' '.join(c).strip() for c in vix2.columns.values]
    vc2 = next(c for c in vix2.columns if 'Close' in c)
    df2['VIX'] = ((vix2[vc2] - vix2[vc2].rolling(20).mean()) / vix2[vc2].rolling(20).std()).fillna(0)

    resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options',
                        headers={'User-Agent':'Mozilla/5.0'})
    soup = BeautifulSoup(resp.text, 'html.parser')
    el = soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
    pcr = float(el.text) if el and el.text.strip() else 0.0
    df2['PCR'] = ((pd.Series(pcr, index=df2.index) - pd.Series(pcr, index=df2.index).rolling(20).mean()) /
                  pd.Series(pcr, index=df2.index).rolling(20).std()).fillna(0)

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

    tail = df2.iloc[-2:]
    X2 = scaler.transform(tail[features].values)
    prev_s, curr_s = model.predict(X2)[-2:]

    signal = "✅ ENTER / BUY" if curr_s in pos_states else "🚫 EXIT / SELL"
    price = tail[close_col].iloc[-1]
    date = tail.index[-1].date()
    msg = (f"📊 HMM v12 Alert — {ticker}\n"
           f"Date: {date}\n"
           f"Prev→Curr: {prev_s}→{curr_s}\n"
           f"Signal: {signal}\n"
           f"Price: ${price:.2f}")
    requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg})
    print(f"{ticker}: {signal}")


# In[ ]:




