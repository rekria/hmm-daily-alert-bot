# HMM Strategy v13: Robust Implementation with Data Validation and Model Improvements
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
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Enhanced Config ───
ASSETS = {
    'SPY': 'SPY', 'TSLA': 'TSLA', 'BYD': '1211.HK', 'GOLD': 'GC=F', 'DBS': 'D05.SI',
    'AAPL': 'AAPL', 'MSFT': 'MSFT', 'GOOGL': 'GOOGL', 'AMZN': 'AMZN', 'NVDA': 'NVDA',
    'META': 'META', 'NFLX': 'NFLX', 'ASML': 'ASML', 'TSM': 'TSM', 'BABA': 'BABA', 'BA': 'BA'
}
START_DATE = '2012-01-01'  # Extended history
END_DATE = None
MIN_DATA_POINTS = 250      # Minimum data points required
MAX_STATES = 5             # Reduced complexity
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10
ROLLING_HYBRID_WINDOW = 50 # Longer lookback for fallback
LOOKBACK = 60
UNIFORM_SIGNAL_THRESHOLD = 0.8  # 80% same signal triggers review

FEATURE_COLS = [
    'LogReturn', 'MACD', 'MACD_diff', 'RSI', 'Volume_Z'
]  # Removed problematic features

GCS_BUCKET = "my-hmm-state"
SIGNAL_LOG_FILE = "signal_log.csv"
BACKTEST_FILE = "backtest_summary.csv"

# ─── GCS Utilities (fixed indentation) ───
def download_last_signals(file_name='last_signal.json'):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(file_name)
        if blob.exists():
            content = blob.download_as_text()
            # Ensure we always get a dictionary
            data = json.loads(content)
            if isinstance(data, dict):
                return data
            else:
                print(f"⚠️ Downloaded {file_name} is not a dictionary. Returning empty dict.")
                return {}
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
if not isinstance(last_signals, dict):
    print("⚠️ Resetting last_signals to empty dictionary")
    last_signals = {}
summary = []
signal_counter = {'BUY': 0, 'SELL': 0}  # Track signal distribution

# ─── Temporal Feature Processing ───
def process_historical_vix(df):
    """Process VIX data with proper temporal alignment"""
    try:
        # Get VIX data with extended history
        vix_start = df.index.min() - timedelta(days=60)
        vix_end = df.index.max()
        vix = yf.download(
            '^VIX', 
            start=vix_start, 
            end=vix_end, 
            auto_adjust=False, 
            progress=False,
            timeout=60
        )
        
        if not vix.empty:
            # Simplify multi-index if needed
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
                
            vix_close = vix['Close'] if 'Close' in vix else vix.iloc[:, 0]
            
            # Calculate rolling metrics using only past data
            vix_mean = vix_close.expanding(min_periods=1).mean()
            vix_std = vix_close.expanding(min_periods=1).std()
            vix_z = (vix_close - vix_mean) / vix_std
            
            # Forward fill and backfill to handle missing values
            vix_z = vix_z.ffill().bfill()
            return vix_z.reindex(df.index).fillna(0)
    except Exception as e:
        print(f"⚠️ VIX processing failed: {str(e)}")
    return pd.Series(0, index=df.index)

# ─── Processing Loop ───
for name, ticker in ASSETS.items():
    print(f"\n🔍 Processing: {ticker}")
    try:
        # Download price data
        df = yf.download(
            ticker, 
            start=START_DATE, 
            end=END_DATE, 
            auto_adjust=False, 
            progress=False,
            timeout=60
        )
        
        # Simplify multi-index columns to single level
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Check data sufficiency
        if len(df) < MIN_DATA_POINTS:
            print(f"⚠️ {ticker}: Insufficient data ({len(df)} < {MIN_DATA_POINTS})")
            continue
            
        # Ensure we have a valid price column
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if price_col not in df.columns:
            print(f"⚠️ {ticker}: No price column found")
            continue
            
        df['Price'] = df[price_col]
        df['LogReturn'] = np.log(df['Price']).diff()
        df.dropna(subset=['LogReturn'], inplace=True)
        
        # Temporal VIX processing
        df['VIX'] = process_historical_vix(df)
        
        # ─── Technical Indicators ───
        # MACD
        try:
            macd_calc = MACD(df['Price'])
            df['MACD'] = macd_calc.macd().ffill().bfill().fillna(0)
            df['MACD_diff'] = macd_calc.macd_diff().ffill().bfill().fillna(0)
        except Exception:
            df['MACD'] = 0.0
            df['MACD_diff'] = 0.0
        
        # RSI
        try:
            rsi_calc = RSIIndicator(df['Price'])
            df['RSI'] = rsi_calc.rsi().ffill().bfill().fillna(50)
        except Exception:
            df['RSI'] = 50.0  # Neutral value
        
        # Volume Z-Score
        try:
            vol_mean = df['Volume'].expanding(min_periods=1).mean()
            vol_std = df['Volume'].expanding(min_periods=1).std()
            df['Volume_Z'] = ((df['Volume'] - vol_mean) / vol_std).fillna(0)
        except Exception:
            df['Volume_Z'] = 0.0

        # Ensure all required features exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        # Fill any remaining NA values
        df.fillna(0, inplace=True)
        
        # Use only recent data for modeling
        modeling_df = df.iloc[-LOOKBACK:] if len(df) > LOOKBACK else df
        
        # ─── HMM Model ───
        best_model, best_bic, best_states = None, np.inf, 0
        state_range = range(2, min(MAX_STATES, len(modeling_df) - 1) + 1)
        
        if not state_range:
            state_range = [2]
            
        for n_states in state_range:
            try:
                # Prepare features
                X = modeling_df[FEATURE_COLS].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=200,
                    random_state=42
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_scaled)
                
                # Calculate BIC with proper parameter count
                n_features = len(FEATURE_COLS)
                n_params = n_states * (n_states - 1) + 2 * n_states * n_features
                bic = -2 * model.score(X_scaled) + n_params * np.log(len(X_scaled))
                
                if bic < best_bic:
                    best_model, best_bic = model, bic
                    best_states = n_states
            except Exception as e:
                print(f"⚠️ HMM fitting failed for {n_states} states: {str(e)}")

        # ─── Position Determination ───
        if best_model is None:
            # Robust fallback strategy
            momentum = modeling_df['LogReturn'].rolling(
                ROLLING_HYBRID_WINDOW, min_periods=1).mean()
            df['Position'] = (momentum > 0).astype(int)
            print(f"⚠️ Using fallback strategy for {ticker}")
            regime_seq = [-1, -1]
            good_states = []
            sharpe = pd.Series()
        else:
            # Predict with best model
            X = modeling_df[FEATURE_COLS].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            hidden = best_model.predict(X_scaled)
            modeling_df['HiddenState'] = hidden
            
            # Calculate regime performance
            try:
                returns = modeling_df.groupby('HiddenState')['LogReturn']
                sharpe = returns.mean() / returns.std()
                durations = modeling_df.groupby('HiddenState').size()
                
                good_states = sharpe[
                    (sharpe > SHARPE_THRESHOLD) & 
                    (durations > DURATION_THRESHOLD)
                ].index.tolist()
            except Exception:
                sharpe = pd.Series()
                good_states = []
            
            # Create position vector
            modeling_df['Position'] = modeling_df['HiddenState'].isin(good_states).astype(int)
            df = df.join(modeling_df[['Position']], how='left')
            df['Position'].ffill(inplace=True)
            df['Position'].fillna(1, inplace=True)  # Default to buy
            
            regime_seq = modeling_df['HiddenState'].iloc[-2:].values.tolist() if len(modeling_df) >= 2 else [-1, -1]

        # ─── Signal Generation ───
        current_signal = "BUY" if df['Position'].iloc[-1] else "SELL"
        prev_signal = "BUY" if df['Position'].iloc[-2] else "SELL" if len(df) >= 2 else "N/A"
        curr_regime = regime_seq[-1] if regime_seq else None
        prev_regime = regime_seq[0] if len(regime_seq) > 1 else None
        price = df['Price'].iloc[-1]
        
        # Track signal distribution
        signal_counter[current_signal] += 1

        # Robust signal checking
        last_data = last_signals.get(ticker, {})
        last_signal = last_data.get("signal") if isinstance(last_data, dict) else None
        last_regime = last_data.get("regime", -999) if isinstance(last_data, dict) else -999

        # ─── Telegram Alert ───
        msg = (
            f"📊 HMM v13 — {ticker}\n"
            f"Regime: {prev_regime} → {curr_regime}\n"
            f"Signal: {prev_signal} → {current_signal}\n"
            f"Price: ${price:.2f}\n"
            f"States: {best_states}"
        )

        if last_signal != current_signal or last_regime != curr_regime:
            try:
                requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg}, timeout=10)
                print(f"✅ {ticker}: Alert sent ({current_signal})")
            except Exception as e:
                print(f"⚠️ Failed to send Telegram alert for {ticker}: {str(e)}")
            
            # Update last signals
            last_signals[ticker] = {"signal": current_signal, "regime": curr_regime}
            upload_last_signals(last_signals)
            
            append_signal_log({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Signal": current_signal,
                "Regime": curr_regime,
                "Price": round(price, 2),
                "PrevSignal": prev_signal,
                "PrevRegime": prev_regime
            })

        # ─── Performance Calculation ───
        df['StrategyReturn'] = df['LogReturn'] * df['Position']
        cumM = np.exp(df['LogReturn'].sum())
        cumH = np.exp(df['StrategyReturn'].sum())
        ratio = cumH / cumM if cumM != 0 else 1.0

        # ─── Summary Stats ───
        summary.append({
            'Ticker': ticker,
            'BuyHoldReturn': round(cumM, 4),
            'HMMReturn': round(cumH, 4),
            'Ratio': round(ratio, 4),
            'NumUsedStates': best_states,
            'FallbackUsed': best_model is None
        })

    except Exception as e:
        print(f"❌ Skipping {ticker} due to error: {str(e)}")

# ─── Uniform Signal Check ───
total_assets = len(summary)
if total_assets > 0:
    max_signal = max(signal_counter.values())
    uniform_ratio = max_signal / total_assets
    
    if uniform_ratio >= UNIFORM_SIGNAL_THRESHOLD:
        dominant_signal = 'BUY' if signal_counter['BUY'] > signal_counter['SELL'] else 'SELL'
        warning_msg = (
            f"⚠️ MARKET WARNING: {uniform_ratio:.0%} assets show {dominant_signal} signals\n"
            f"This may indicate systemic market conditions rather than asset-specific signals\n"
            f"Recommend additional fundamental analysis before taking positions"
        )
        try:
            requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": warning_msg}, timeout=10)
            print(f"✅ Sent uniform signal warning")
        except Exception as e:
            print(f"⚠️ Failed to send uniform signal warning: {str(e)}")

# ─── Save Backtest Summary ───
if summary:
    df_summary = pd.DataFrame(summary)
    upload_backtest_summary(df_summary)
    print("\n✅ Backtest summary uploaded to GCS")
else:
    print("\n⚠️ No assets processed successfully")
