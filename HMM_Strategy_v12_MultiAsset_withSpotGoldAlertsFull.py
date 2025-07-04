# HMM Strategy v12d: Enhanced Hybrid Model with Telegram Alerts, GCS Signal & Regime Tracking
# Author: ChatGPT (on behalf of @rekria)
# Iteration 5: Fixed price data handling and logarithmic return calculation

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import warnings
from datetime import datetime, timedelta

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from ta.trend import MACD
from ta.momentum import RSIIndicator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
from bs4 import BeautifulSoup
from google.cloud import storage

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Config ───
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

FEATURE_COLS = [
    'LogReturn', 'MACD', 'MACD_diff', 'RSI', 'NewsSentiment', 'VIX', 'PCR', 'Volume_Z'
]

GCS_BUCKET = "my-hmm-state"
SIGNAL_LOG_FILE = "signal_log.csv"
BACKTEST_FILE = "backtest_summary.csv"

# ─── GCS Utilities ───
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
        if os.path.exists(local_file):
            existing = pd.read_csv(local_file)
            df = pd.concat([existing, df])
        df.to_csv(local_file, index=False)
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

# ─── Telegram ───
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# ─── Init ───
sia = SentimentIntensityAnalyzer()
last_signals = download_last_signals()
summary = []

# ─── Processing Loop ───
for name, ticker in ASSETS.items():
    print(f"\n🔍 Processing: {ticker}")
    try:
        # Download price data with robust error handling
        df = yf.download(
            ticker, 
            start=START_DATE, 
            end=END_DATE, 
            auto_adjust=False, 
            progress=False,
            timeout=30
        )
        
        # Check if we got valid price data
        if df.empty:
            print(f"⚠️ {ticker}: No data downloaded")
            continue
            
        # Ensure we have a valid price column to use
        price_col = None
        for col in ['Adj Close', 'Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                price_col = col
                break
                
        if not price_col:
            print(f"⚠️ {ticker}: No price column found")
            continue
            
        # Use the best available price column
        df['Price'] = df[price_col]
        
        # Calculate logarithmic returns
        df['LogReturn'] = np.log(df['Price']).diff()
        df.dropna(subset=['LogReturn'], inplace=True)
        
        if df.empty:
            print(f"⚠️ {ticker}: No valid returns after calculation")
            continue

        # News sentiment (single value for whole dataset)
        try:
            feed = feedparser.parse('https://finance.yahoo.com/news/rss')
            titles = [e.title for e in feed.entries if hasattr(e, 'title')]
            if titles:
                sentiments = [sia.polarity_scores(t)['compound'] for t in titles]
                avg_sentiment = np.mean(sentiments)
            else:
                avg_sentiment = 0.0
        except Exception:
            avg_sentiment = 0.0
        df['NewsSentiment'] = avg_sentiment

        # VIX volatility index
        try:
            vix = yf.download(
                '^VIX', 
                start=df.index.min() - timedelta(days=30), 
                end=df.index.max(), 
                auto_adjust=False, 
                progress=False,
                timeout=30
            )
            if not vix.empty:
                vix_close = vix['Close'] if 'Close' in vix else vix.iloc[:, 0]
                vix_z = ((vix_close - vix_close.rolling(20).mean()) / vix_close.rolling(20).std())
                df['VIX'] = vix_z.reindex(df.index).fillna(0)
            else:
                df['VIX'] = 0.0
        except Exception:
            df['VIX'] = 0.0

        # PCR (Put/Call Ratio)
        try:
            pcr_resp = requests.get(
                'https://finance.yahoo.com/quote/%5EPCR/options', 
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=10
            )
            pcr_soup = BeautifulSoup(pcr_resp.text, 'html.parser')
            el = pcr_soup.select_one("td[data-test='PUT_CALL_RATIO-value']")
            if el and el.text.strip():
                pcr_val = float(el.text)
            else:
                pcr_val = 0.0
        except Exception:
            pcr_val = 0.0
            
        # Calculate PCR z-score
        pcr_z = (pcr_val - df['LogReturn'].rolling(20).mean()) / df['LogReturn'].rolling(20).std()
        df['PCR'] = pcr_z.fillna(0)

        # ─── Technical Indicators ───
        # Robust 1D conversion for all indicators
        def ensure_1d(series, default_val=0):
            """Convert any array-like to 1D with NaN handling"""
            if series is None or len(series) == 0:
                return pd.Series([default_val] * len(df), index=df.index)
            return pd.Series(np.ravel(series), index=df.index).ffill().bfill()
        
        # MACD with robust 1D conversion
        try:
            macd_calc = MACD(df['Price'])
            df['MACD'] = ensure_1d(macd_calc.macd())
            df['MACD_diff'] = ensure_1d(macd_calc.macd_diff())
        except Exception:
            df['MACD'] = 0.0
            df['MACD_diff'] = 0.0
        
        # RSI with robust 1D conversion
        try:
            rsi_calc = RSIIndicator(df['Price'])
            df['RSI'] = ensure_1d(rsi_calc.rsi())
        except Exception:
            df['RSI'] = 50.0  # Neutral value
        
        # Volume Z-Score
        try:
            vol_mean = df['Volume'].rolling(20).mean()
            vol_std = df['Volume'].rolling(20).std()
            df['Volume_Z'] = ((df['Volume'] - vol_mean) / vol_std).fillna(0)
        except Exception:
            df['Volume_Z'] = 0.0

        # Check for missing features
        missing = set(FEATURE_COLS) - set(df.columns)
        if missing:
            print(f"⚠️ Missing features for {ticker}: {missing}")
            # Fill missing features with 0 as fallback
            for col in missing:
                df[col] = 0.0

        # Drop any remaining NA values
        df.dropna(subset=FEATURE_COLS, inplace=True, how='any')
        
        if df.empty:
            print(f"⚠️ {ticker}: No data after feature preparation")
            continue

        # ─── HMM Model ───
        best_model, best_bic, scaler_type, best_states = None, np.inf, None, 0
        for scale_type in ['per-asset', 'global']:
            try:
                scaler = StandardScaler()
                if scale_type == 'per-asset':
                    X = scaler.fit_transform(df[['LogReturn']])
                else:  # global scaling
                    X = scaler.fit_transform(df[['LogReturn']].values.reshape(-1, 1))
                
                for n_states in STATE_RANGE:
                    model = GaussianHMM(
                        n_components=n_states,
                        covariance_type='diag',
                        n_iter=200
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X)
                    bic = -2 * model.score(X) + n_states * np.log(len(X))
                    if bic < best_bic:
                        best_model, best_bic = model, bic
                        scaler_type, best_states = scale_type, n_states
                # Break if we found a valid model
                if best_model:
                    break
            except Exception as e:
                print(f"⚠️ {scale_type} scaling failed for {ticker}: {str(e)}")

        # ─── Position Determination ───
        if best_model is None:
            # Fallback to simple momentum strategy
            momentum = df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean()
            if momentum.empty:
                df['Position'] = 1
            else:
                df['Position'] = (momentum.iloc[-1] > 0).astype(int)
            regime_seq = [-1, -1]
            good_states = []
            sharpe = pd.Series()
        else:
            hidden = best_model.predict(X)
            df['HiddenState'] = hidden
            
            # Calculate regime Sharpe ratios
            try:
                sharpe = df.groupby('HiddenState')['LogReturn'].mean() / df.groupby('HiddenState')['LogReturn'].std()
                durations = df.groupby('HiddenState').size()
                
                # Identify good regimes
                good_states = sharpe[
                    (sharpe > SHARPE_THRESHOLD) & 
                    (durations > DURATION_THRESHOLD)
                ].index.tolist()
                
                if not good_states:
                    good_states = [sharpe.idxmax()] if not sharpe.empty else []
            except Exception:
                sharpe = pd.Series()
                good_states = []
            
            if good_states:
                df['Position'] = df['HiddenState'].isin(good_states).astype(int)
            else:
                df['Position'] = 1  # Default to buy
            
            # Fallback if no positions
            if df['Position'].sum() == 0:
                momentum = df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean()
                df['Position'] = (momentum.iloc[-1] > 0).astype(int) if not momentum.empty else 1
            
            regime_seq = df['HiddenState'].iloc[-2:].values.tolist() if len(df) >= 2 else [-1, -1]

        # ─── Performance Calculation ───
        df['StrategyReturn'] = df['LogReturn'] * df['Position']
        cumM = np.exp(df['LogReturn'].cumsum()).iloc[-1] if not df.empty else 1.0
        cumH = np.exp(df['StrategyReturn'].cumsum()).iloc[-1] if not df.empty else 1.0
        ratio = cumH / cumM if cumM != 0 else 1.0

        # ─── Signal Generation ───
        if len(df) >= 2:
            prev_signal = "BUY" if df['Position'].iloc[-2] else "SELL"
            curr_signal = "BUY" if df['Position'].iloc[-1] else "SELL"
            curr_regime = regime_seq[-1] if regime_seq else None
            prev_regime = regime_seq[0] if len(regime_seq) > 1 else None
            price = df['Price'].iloc[-1]
        else:
            prev_signal = "N/A"
            curr_signal = "N/A"
            curr_regime = None
            prev_regime = None
            price = 0

        last_data = last_signals.get(ticker, {})
        last_signal = last_data.get("signal")
        last_regime = last_data.get("regime", -999)

        # ─── Telegram Alert ───
        msg = (
            f"📊 HMM v12d — {ticker}\n"
            f"Prev→Curr Regime: {prev_regime} → {curr_regime}\n"
            f"Signal: {prev_signal} → {curr_signal}\n"
            f"Ratio: {ratio:.2f}×\n"
            f"Price: ${price:.2f}\n"
            f"States: {best_states} ({scaler_type or 'hybrid'})"
        )

        if last_signal != curr_signal or last_regime != curr_regime:
            try:
                requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg}, timeout=10)
                print(f"✅ {ticker}: Alert sent ({curr_signal})")
            except Exception as e:
                print(f"⚠️ Failed to send Telegram alert for {ticker}: {str(e)}")
            
            last_signals[ticker] = {"signal": curr_signal, "regime": curr_regime}
            upload_last_signals(last_signals)
            
            append_signal_log({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Signal": curr_signal,
                "Regime": curr_regime,
                "Price": round(price, 2),
                "PrevSignal": prev_signal,
                "PrevRegime": prev_regime,
                "Ratio": round(ratio, 4)
            })
        else:
            print(f"{ticker}: No signal/regime change ({curr_signal}, Regime {curr_regime})")

        # ─── Summary Stats ───
        summary.append({
            'Ticker': ticker,
            'BuyHoldReturn': round(cumM, 4),
            'HMMReturn': round(cumH, 4),
            'Ratio': round(ratio, 4),
            'ScalerType': scaler_type or 'hybrid',
            'NumUsedStates': best_states if best_model else 0,
            'PosRegimes': good_states if best_model else [],
            'RegimeSharpeMap': sharpe.round(2).to_dict() if not sharpe.empty else {},
            'FallbackUsed': best_model is None
        })

    except Exception as e:
        print(f"❌ Skipping {ticker} due to error: {str(e)[:200]}")

# ─── Save Backtest Summary ───
if summary:
    df_summary = pd.DataFrame(summary)
    upload_backtest_summary(df_summary)
    print("\n✅ Backtest summary uploaded to GCS")
else:
    print("\n⚠️ No assets processed successfully")
