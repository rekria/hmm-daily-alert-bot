# HMM Strategy v13: Final Convergence Optimization
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import warnings
import time
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
START_DATE = '2012-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
MIN_DATA_POINTS = 250
MAX_STATES = 5
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10
ROLLING_HYBRID_WINDOW = 50
LOOKBACK = 60
UNIFORM_SIGNAL_THRESHOLD = 0.8

FEATURE_COLS = [
    'LogReturn', 'MACD', 'MACD_diff', 'RSI', 'Volume_Z'
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
            content = blob.download_as_text()
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
signal_counter = {'BUY': 0, 'SELL': 0}

# ─── Temporal Feature Processing ───
def process_historical_vix(df):
    try:
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
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
                
            vix_close = vix['Close'] if 'Close' in vix else vix.iloc[:, 0]
            
            vix_mean = vix_close.expanding(min_periods=1).mean()
            vix_std = vix_close.expanding(min_periods=1).std()
            vix_z = (vix_close - vix_mean) / vix_std
            
            vix_z = vix_z.ffill().bfill()
            return vix_z.reindex(df.index).fillna(0)
    except Exception as e:
        print(f"⚠️ VIX processing failed: {str(e)}")
    return pd.Series(0, index=df.index)

# ─── Robust Data Download ───
def download_asset_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            asset = yf.Ticker(ticker)
            df = asset.history(
                start=start_date, 
                end=end_date, 
                auto_adjust=True,
                actions=False,
                timeout=30
            )
            
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
                
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for {ticker}: {str(e)}")
            time.sleep(2)
    
    try:
        print(f"⚠️ Using fallback download for {ticker}")
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            auto_adjust=True, 
            progress=False,
            timeout=60
        )
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            return df
    except Exception as e:
        print(f"⚠️ Fallback download failed for {ticker}: {str(e)}")
    
    return pd.DataFrame()

# ─── Optimized HMM Training ───
def train_hmm_model(X, n_states):
    """Train HMM with final convergence optimization"""
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=500,           # Increased iterations
            tol=1e-3,             # Looser tolerance (0.001)
            init_params='stmc',    # Initialize all parameters
            random_state=42,
            verbose=False          # Disable verbose output
        )
        model.fit(X)
        return model
    except Exception as e:
        print(f"⚠️ HMM training failed: {str(e)}")
    return None

# ─── Processing Loop ───
for name, ticker in ASSETS.items():
    print(f"\n🔍 Processing: {ticker}")
    try:
        # Download price data
        df = download_asset_data(ticker, START_DATE, END_DATE)
        
        if df.empty:
            print(f"⚠️ {ticker}: No data downloaded")
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < MIN_DATA_POINTS:
            print(f"⚠️ {ticker}: Insufficient data ({len(df)} < {MIN_DATA_POINTS})")
            continue
            
        price_col = 'Close'
        if price_col not in df.columns:
            print(f"⚠️ {ticker}: No price column found")
            continue
            
        df['Price'] = df[price_col]
        df['LogReturn'] = np.log(df['Price']).diff()
        df.dropna(subset=['LogReturn'], inplace=True)
        
        df['VIX'] = process_historical_vix(df)
        
        # ─── Technical Indicators ───
        try:
            macd_calc = MACD(df['Price'])
            df['MACD'] = macd_calc.macd().ffill().bfill().fillna(0)
            df['MACD_diff'] = macd_calc.macd_diff().ffill().bfill().fillna(0)
        except Exception:
            df['MACD'] = 0.0
            df['MACD_diff'] = 0.0
        
        try:
            rsi_calc = RSIIndicator(df['Price'])
            df['RSI'] = rsi_calc.rsi().ffill().bfill().fillna(50)
        except Exception:
            df['RSI'] = 50.0
        
        try:
            vol_mean = df['Volume'].expanding(min_periods=1).mean()
            vol_std = df['Volume'].expanding(min_periods=1).std()
            df['Volume_Z'] = ((df['Volume'] - vol_mean) / vol_std).fillna(0)
        except Exception:
            df['Volume_Z'] = 0.0

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df.fillna(0, inplace=True)
        
        modeling_df = df.iloc[-LOOKBACK:].copy()
        
        # ─── HMM Model ───
        best_model, best_bic, best_states = None, np.inf, 0
        state_range = range(2, min(MAX_STATES, len(modeling_df) - 1) + 1)
        
        if not state_range:
            state_range = [2]
            
        # Prepare features once
        X = modeling_df[FEATURE_COLS].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
            
        for n_states in state_range:
            try:
                model = train_hmm_model(X_scaled, n_states)
                if model is None:
                    continue
                
                # Calculate BIC
                n_features = len(FEATURE_COLS)
                n_params = n_states * (n_states - 1) + 2 * n_states * n_features
                bic = -2 * model.score(X_scaled) + n_params * np.log(len(X_scaled))
                
                if bic < best_bic:
                    best_model, best_bic = model, bic
                    best_states = n_states
            except Exception as e:
                # Non-fatal error, continue with next state
                continue

        # ─── Position Determination ───
        if best_model is None:
            momentum = modeling_df['LogReturn'].rolling(
                ROLLING_HYBRID_WINDOW, min_periods=1).mean()
            modeling_df['Position'] = (momentum > 0).astype(int)
            print(f"⚠️ Using fallback strategy for {ticker}")
            regime_seq = [-1, -1]
            good_states = []
        else:
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
                good_states = []
            
            modeling_df['Position'] = modeling_df['HiddenState'].isin(good_states).astype(int)
            regime_seq = modeling_df['HiddenState'].iloc[-2:].values.tolist() if len(modeling_df) >= 2 else [-1, -1]

        # Merge position back
        df = df.join(modeling_df[['Position']], how='left')
        df['Position'].ffill(inplace=True)
        df['Position'].fillna(1, inplace=True)

        # ─── Signal Generation ───
        current_signal = "BUY" if df['Position'].iloc[-1] else "SELL"
        prev_signal = "BUY" if df['Position'].iloc[-2] else "SELL" if len(df) >= 2 else "N/A"
        curr_regime = regime_seq[-1] if regime_seq else None
        prev_regime = regime_seq[0] if len(regime_seq) > 1 else None
        price = df['Price'].iloc[-1]
        
        signal_counter[current_signal] += 1

        last_data = last_signals.get(ticker, {})
        last_signal = last_data.get("signal", "")
        last_regime = last_data.get("regime", -999)

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
            f"This may indicate systemic market conditions\n"
            f"Recommend fundamental analysis confirmation"
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
