# HMM Strategy v12d: Final Robust Version with Full Enhancements Retained
# Author: ChatGPT (for @rekria)

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import joblib
import warnings
from datetime import datetime

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from ta.trend import MACD
from ta.momentum import RSIIndicator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
from bs4 import BeautifulSoup
from google.cloud import storage

# ─── Config ──────────────────────────────────────────────────────────────
ASSETS = {
    'SPY': 'SPY', 'TSLA': 'TSLA', 'BYD': '1211.HK', 'GOLD': 'GC=F', 'DBS': 'D05.SI',
    'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'NVDA': 'NVDA',
    'META': 'META', 'NFLX': 'NFLX', 'ASML': 'ASML', 'TSM': 'TSM', 'BABA': 'BABA', 'BA': 'BA'
}
START_DATE = '2017-01-01'
END_DATE = None
STATE_RANGE = range(2, 30)
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10
ROLLING_HYBRID_WINDOW = 10
LOOKBACK = 60

FEATURE_COLS = ['LogReturn', 'MACD', 'MACD_diff', 'RSI', 'NewsSentiment', 'VIX', 'PCR', 'Volume_Z']
GCS_BUCKET = "my-hmm-state"
SIGNAL_LOG_FILE = "signal_log.csv"
BACKTEST_FILE = "backtest_summary.csv"

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID  = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# ─── GCS Utilities ───────────────────────────────────────────────────────
def download_last_signals(file_name='last_signal.json'):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(file_name)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception as e:
        print(f"Error downloading {file_name} from GCS: {e}")
    return {}

def upload_last_signals(signals, file_name='last_signal.json'):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(file_name)
        blob.upload_from_string(json.dumps(signals))
    except Exception as e:
        print(f"Error uploading {file_name} to GCS: {e}")

def append_signal_log(row_dict):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(SIGNAL_LOG_FILE)
        header = not blob.exists()
        local_file = "/tmp/signal_log.csv"
        df = pd.DataFrame([row_dict])
        df.to_csv(local_file, index=False, mode='a', header=header)
        blob.upload_from_filename(local_file)
    except Exception as e:
        print(f"Error appending to signal_log.csv: {e}")

def upload_backtest_summary(df):
    try:
        local_path = "/tmp/backtest_summary.csv"
        df.to_csv(local_path, index=False)
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(BACKTEST_FILE)
        blob.upload_from_filename(local_path)
    except Exception as e:
        print(f"Error uploading backtest_summary.csv to GCS: {e}")

# ─── Init ─────────────────────────────────────────────────────────────────
sia = SentimentIntensityAnalyzer()
last_signals = download_last_signals()
summary = []

# ─── Processing Loop ──────────────────────────────────────────────────────
for name, ticker in ASSETS.items():
    print(f"\n🔍 Processing: {ticker}")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        df['LogReturn'] = np.log(df['Close']).diff()
        df.dropna(inplace=True)
    except Exception as e:
        print(f"⚠️ Data download failed for {ticker}: {e}")
        continue

    titles = [e.title for e in feedparser.parse('https://finance.yahoo.com/news/rss').entries]
    df['NewsSentiment'] = np.mean([sia.polarity_scores(t)['compound'] for t in titles]) if titles else 0.0

    try:
        vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), auto_adjust=True, progress=False)
        vix_close = vix['Close'] if 'Close' in vix else vix.iloc[:, 0]
        df['VIX'] = ((vix_close - vix_close.rolling(20).mean()) / vix_close.rolling(20).std()).reindex(df.index).fillna(0)
    except:
        df['VIX'] = 0.0

    try:
        pcr_resp = requests.get('https://finance.yahoo.com/quote/%5EPCR/options', headers={'User-Agent':'Mozilla/5.0'})
        pcr_soup = BeautifulSoup(pcr_resp.text, 'html.parser')
        el = pcr_soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
        pcr_val = float(el.text) if el and el.text.strip() else 0.0
    except:
        pcr_val = 0.0
    df['PCR'] = ((pcr_val - df['LogReturn'].rolling(20).mean()) / df['LogReturn'].rolling(20).std()).fillna(0)

    df['MACD'] = MACD(df['Close']).macd()
    df['MACD_diff'] = MACD(df['Close']).macd_diff()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    df['Volume_Z'] = ((df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()).fillna(0)

    try:
        df.dropna(subset=FEATURE_COLS, inplace=True)
    except KeyError as e:
        print(f"⚠️ Skipping {ticker} — missing feature columns: {e}")
        continue

    best_model, best_bic, scaler_type, best_states = None, np.inf, None, 0
    for scale_type in ['per-asset', 'global']:
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[['LogReturn']] if scale_type=='per-asset' else df[['LogReturn']].values.reshape(-1,1))
            for n_states in STATE_RANGE:
                model = GaussianHMM(n_components=n_states, covariance_type='diag', n_iter=200)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X)
                bic = -2 * model.score(X) + n_states * np.log(len(X))
                if bic < best_bic:
                    best_model, best_bic = model, bic
                    scaler_type, best_states = scale_type, n_states
            break
        except Exception as e:
            print(f"⚠️ {scale_type} scaling failed for {ticker}: {e}")

    if best_model is None:
        print(f"❌ HMM failed — using hybrid fallback for {ticker}")
        df['Position'] = (df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean().iloc[-1] > 0).astype(int)
        regime_seq = [-1, -1]
        good_states = []
        sharpe = pd.Series()
    else:
        hidden = best_model.predict(X)
        df['HiddenState'] = hidden
        sharpe = df.groupby('HiddenState')['LogReturn'].mean() / df.groupby('HiddenState')['LogReturn'].std()
        durations = df.groupby('HiddenState').size()
        good_states = sharpe[(sharpe > SHARPE_THRESHOLD) & (durations > DURATION_THRESHOLD)].index.tolist()
        if not good_states:
            good_states = [sharpe.idxmax()]
        df['Position'] = df['HiddenState'].isin(good_states).astype(int)
        if df['Position'].sum() == 0:
            df['Position'] = (df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean().iloc[-1] > 0).astype(int)
        regime_seq = df['HiddenState'].iloc[-2:].tolist()

    df['StrategyReturn'] = df['LogReturn'] * df['Position']
    cumM = np.exp(df['LogReturn'].cumsum()).iloc[-1]
    cumH = np.exp(df['StrategyReturn'].cumsum()).iloc[-1]
    ratio = cumH / cumM if cumM else 1.0

    tail = df.iloc[-2:]
    prev_signal = "BUY" if tail['Position'].iloc[-2] else "SELL"
    curr_signal = "BUY" if tail['Position'].iloc[-1] else "SELL"
    last_data = last_signals.get(ticker, {})
    last_signal = last_data.get("signal")
    last_regime = last_data.get("regime", -999)
    curr_regime = regime_seq[-1] if regime_seq[-1] != -1 else None

    msg = (
        f"📊 HMM v12d — {ticker}\n"
        f"Prev→Curr Regime: {regime_seq[0]} → {regime_seq[1]}\n"
        f"Signal: {prev_signal} → {curr_signal}\n"
        f"Ratio: {ratio:.2f}×\n"
        f"Price: ${df['Close'].iloc[-1]:.2f}\n"
        f"States: {best_states} ({scaler_type or 'hybrid'})"
    )

    if last_signal != curr_signal or last_regime != curr_regime:
        requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg})
        last_signals[ticker] = {"signal": curr_signal, "regime": curr_regime}
        upload_last_signals(last_signals)
        append_signal_log({
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Ticker": ticker,
            "Signal": curr_signal,
            "Regime": curr_regime,
            "Price": round(df['Close'].iloc[-1], 2),
            "PrevSignal": prev_signal,
            "PrevRegime": regime_seq[0],
            "Ratio": round(ratio, 4)
        })
        print(f"✅ {ticker}: Alert sent ({curr_signal})")
    else:
        print(f"{ticker}: No signal/regime change ({curr_signal}, Regime {curr_regime})")

    summary.append({
        'Ticker': ticker,
        'BuyHoldReturn': round(cumM, 4),
        'HMMReturn': round(cumH, 4),
        'Ratio': round(ratio, 4),
        'ScalerType': scaler_type or 'hybrid',
        'NumUsedStates': best_states if best_model else 0,
        'PosRegimes': good_states if best_model else [],
        'RegimeSharpeMap': sharpe.round(2).to_dict() if best_model else {},
        'FallbackUsed': best_model is None
    })

# ─── Save Backtest Summary ───────────────────────────────────────────────
df_summary = pd.DataFrame(summary)
upload_backtest_summary(df_summary)
print("\n✅ Backtest summary uploaded to GCS")
