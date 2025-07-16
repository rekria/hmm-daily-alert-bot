# HMM Strategy v14: Enhanced Profitability
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
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google.cloud import storage

# Suppress all warnings
warnings.filterwarnings("ignore")

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
DURATION_THRESHOLD = 10
ROLLING_HYBRID_WINDOW = 50
UNIFORM_SIGNAL_THRESHOLD = 0.8
SPY_CORR_THRESHOLD = 0.7

# Sector-specific thresholds
TECH_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'NFLX', 'ASML', 'TSM']
COMMODITY_TICKERS = ['GC=F']
FINANCIAL_TICKERS = ['D05.SI']

# Enhanced features
FEATURE_COLS = [
    'LogReturn', 'MACD', 'MACD_diff', 'RSI', 'Volume_Z', 
    'BB_Width', 'ATR', 'Stochastic', 'VIX_Z'
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
                return {}
    except Exception:
        return {}

def upload_last_signals(signals, file_name='last_signal.json'):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(file_name)
        blob.upload_from_string(json.dumps(signals))
    except Exception:
        pass

def append_signal_log(row_dict):
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(SIGNAL_LOG_FILE)
        local_file = "/tmp/signal_log.csv"
        df = pd.DataFrame([row_dict])
        if os.path.exists(local_file):
            existing = pd.read_csv(local_file)
            df = pd.concat([existing, df])
        df.to_csv(local_file, index=False)
        blob.upload_from_filename(local_file)
    except Exception:
        pass

def upload_backtest_summary(df):
    try:
        local_path = "/tmp/backtest_summary.csv"
        df.to_csv(local_path, index=False)
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(BACKTEST_FILE)
        blob.upload_from_filename(local_path)
    except Exception:
        pass

# ─── Telegram ───
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Environment variable BOT_TOKEN not set")
CHAT_ID = os.getenv("CHAT_ID", "1669179604")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# ─── Init ───
sia = SentimentIntensityAnalyzer()
last_signals = download_last_signals() or {}
summary = []
signal_counter = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
signals_updated = False

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
            return vix_close.reindex(df.index).ffill().bfill().fillna(0)
    except Exception:
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
                
        except Exception:
            time.sleep(2)
    
    try:
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
    except Exception:
        pass
    
    return pd.DataFrame()

# ─── Optimized HMM Training ───
def train_hmm_model(X, n_states):
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=100,
            tol=0.1,
            init_params='stmc',
            random_state=42,
            verbose=False
        )
        model.fit(X)
        return model
    except Exception:
        return None

# ─── Processing Loop ───
for name, ticker in ASSETS.items():
    try:
        # Download price data
        df = download_asset_data(ticker, START_DATE, END_DATE)
        
        if df.empty:
            continue
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if len(df) < MIN_DATA_POINTS:
            continue
            
        price_col = 'Close'
        if price_col not in df.columns:
            continue
            
        df['Price'] = df[price_col]
        df['LogReturn'] = np.log(df['Price']).diff()
        df.dropna(subset=['LogReturn'], inplace=True)
        
        # Add overnight gap feature
        df['Overnight'] = np.log(df['Open'] / df['Close'].shift(1))
        df['Gap_Size'] = df['Overnight'].abs()
        
        # Calculate volatility for dynamic lookback
        volatility = df['LogReturn'].std() * np.sqrt(252)  # Annualized vol
        if volatility > 0.3:   LOOKBACK = 90
        elif volatility < 0.15: LOOKBACK = 30
        else:                  LOOKBACK = 60
        
        # VIX processing
        vix_series = process_historical_vix(df)
        df['VIX'] = vix_series
        df['VIX_Z'] = (vix_series - vix_series.mean()) / vix_series.std() if vix_series.std() > 0 else 0
        
        # ─── Enhanced Technical Indicators ───
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
            
        try:
            # Bollinger Bands Width
            bb = BollingerBands(df['Price'])
            df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            df['BB_Width'].fillna(0, inplace=True)
            
            # Average True Range
            atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
            df['ATR'] = atr.average_true_range().fillna(0)
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
            df['Stochastic'] = stoch.stoch().fillna(50)
        except Exception:
            df['BB_Width'] = 0.0
            df['ATR'] = 0.0
            df['Stochastic'] = 50.0

        # Ensure all features exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        df.fillna(0, inplace=True)
        
        # Correlation filter (reduce exposure to SPY-correlated assets)
        position_adjustment = 1.0
        if ticker != 'SPY':
            spy_data = download_asset_data('SPY', START_DATE, END_DATE)
            if not spy_data.empty and not df.empty:
                spy_returns = spy_data['Close'].pct_change().dropna()
                asset_returns = df['Price'].pct_change().dropna()
                
                # Align indices
                common_index = spy_returns.index.intersection(asset_returns.index)
                if len(common_index) > 10:  # Minimum data points
                    correlation = pd.Series(spy_returns.loc[common_index]).corr(
                        pd.Series(asset_returns.loc[common_index])
                    
                    if abs(correlation) > SPY_CORR_THRESHOLD:
                        position_adjustment = 0.7  # Reduce exposure
        
        modeling_df = df.iloc[-LOOKBACK:].copy()
        
        # ─── HMM Model ───
        best_model, best_bic, best_states = None, np.inf, 0
        state_range = range(2, min(MAX_STATES, len(modeling_df) - 1) + 1)
        
        if not state_range:
            state_range = [2]
            
        # Prepare features
        X = modeling_df[FEATURE_COLS].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
            
        for n_states in state_range:
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

        # ─── Position Determination ───
        if best_model is None:
            momentum = modeling_df['LogReturn'].rolling(
                ROLLING_HYBRID_WINDOW, min_periods=1).mean()
            modeling_df['Position'] = (momentum > 0).astype(int)
            regime_seq = [-1, -1]
        else:
            hidden = best_model.predict(X_scaled)
            modeling_df['HiddenState'] = hidden
            
            # Set sector-specific Sharpe threshold
            if ticker in TECH_TICKERS:
                sharpe_threshold = 0.15
            elif ticker in COMMODITY_TICKERS + FINANCIAL_TICKERS:
                sharpe_threshold = 0.05
            else:
                sharpe_threshold = 0.1
            
            # Enhanced regime selection
            try:
                state_returns = modeling_df.groupby('HiddenState')['LogReturn']
                state_durations = modeling_df.groupby('HiddenState').size()
                
                # More sophisticated regime scoring
                regime_scores = []
                for state in state_returns.groups:
                    returns = state_returns.get_group(state)
                    score = (returns.mean() * 3) - returns.std()  # Reward return, penalize volatility
                    regime_scores.append((state, score))
                
                # Sort by best score and select top regimes
                regime_scores.sort(key=lambda x: x[1], reverse=True)
                good_states = [state for state, score in regime_scores 
                              if score > sharpe_threshold and 
                              state_durations[state] > DURATION_THRESHOLD][:2]  # Max 2 best states
                
                # Probabilistic position sizing
                probs = best_model.predict_proba(X_scaled)[-1]
                position_strength = sum(probs[state] for state in good_states if state < len(probs))
                modeling_df['Position'] = position_strength
                
            except Exception:
                good_states = []
                modeling_df['Position'] = 1.0  # Default to full position
            
            regime_seq = modeling_df['HiddenState'].iloc[-2:].values.tolist() if len(modeling_df) >= 2 else [-1, -1]

        # Risk management: Drawdown protection
        modeling_df['TrailingMax'] = modeling_df['Price'].cummax()
        modeling_df['Drawdown'] = (modeling_df['Price'] - modeling_df['TrailingMax']) / modeling_df['TrailingMax']
        modeling_df.loc[modeling_df['Drawdown'] < -0.08, 'Position'] *= 0.5
        modeling_df.loc[modeling_df['Drawdown'] < -0.15, 'Position'] = 0
        
        # Apply gap protection
        modeling_df.loc[modeling_df['Gap_Size'] > 0.03, 'Position'] *= 0.7
        
        # Apply volatility-based position sizing
        asset_volatility = modeling_df['LogReturn'].std() * np.sqrt(252)
        if asset_volatility > 0.25:
            modeling_df['Position'] *= 0.8
        elif asset_volatility < 0.15:
            modeling_df['Position'] *= 1.2
        modeling_df['Position'] = np.clip(modeling_df['Position'], 0, 1.5)  # Cap at 150%
        
        # Apply correlation adjustment
        modeling_df['Position'] *= position_adjustment
        
        # Hybrid momentum integration
        mom_strength = RSIIndicator(modeling_df['Price']).rsi() / 50  # 0-2 scale
        modeling_df['Position'] = (modeling_df['Position'] * 0.7) + (mom_strength * 0.3)
        modeling_df['Position'] = np.clip(modeling_df['Position'], 0, 1.5)  # Final clip
        
        # Merge position back
        df = df.join(modeling_df[['Position']], how='left')
        df['Position'].ffill(inplace=True)
        df['Position'].fillna(1, inplace=True)

        # ─── Signal Generation ───
        current_position = df['Position'].iloc[-1]
        current_signal = "BUY" if current_position > 0.7 else "SELL" if current_position < 0.3 else "HOLD"
        prev_position = df['Position'].iloc[-2] if len(df) >= 2 else current_position
        prev_signal = "BUY" if prev_position > 0.7 else "SELL" if prev_position < 0.3 else "HOLD"
        
        curr_regime = regime_seq[-1] if regime_seq else None
        prev_regime = regime_seq[0] if len(regime_seq) > 1 else None
        price = df['Price'].iloc[-1]
        
        # Count signals
        if current_signal == "BUY":
            signal_counter['BUY'] += 1
        elif current_signal == "SELL":
            signal_counter['SELL'] += 1
        else:
            signal_counter['HOLD'] += 1

        last_data = last_signals.get(ticker, {})
        last_signal = last_data.get("signal", "")
        last_regime = last_data.get("regime", -999)

        # ─── Telegram Alert ───
        msg = (
            f"📊 HMM v14 — {ticker}\n"
            f"Regime: {prev_regime} → {curr_regime}\n"
            f"Signal: {prev_signal} → {current_signal}\n"
            f"Position: {current_position:.0%}\n"
            f"Price: ${price:.2f}\n"
            f"States: {best_states}\n"
            f"Volatility: {asset_volatility:.1%}"
        )

        if last_signal != current_signal or last_regime != curr_regime:
            try:
                requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": msg}, timeout=10)
            except Exception:
                pass
            
            last_signals[ticker] = {"signal": current_signal, "regime": curr_regime}
            signals_updated = True
            
            append_signal_log({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Signal": current_signal,
                "Position": round(current_position, 2),
                "Regime": curr_regime,
                "Price": round(price, 2),
                "PrevSignal": prev_signal,
                "PrevRegime": prev_regime,
                "Volatility": round(asset_volatility, 4)
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
            'FallbackUsed': best_model is None,
            'FinalPosition': round(current_position, 2),
            'Volatility': round(asset_volatility, 4),
            'Sector': "Tech" if ticker in TECH_TICKERS else 
                     "Commodity" if ticker in COMMODITY_TICKERS else 
                     "Financial" if ticker in FINANCIAL_TICKERS else "Other"
        })

    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

# ─── Save FINAL state if any changes occurred ───
if signals_updated:
    upload_last_signals(last_signals)

# ─── Uniform Signal Check ───
total_assets = len(summary)
if total_assets > 0:
    max_signal = max(signal_counter.get('BUY', 0), signal_counter.get('SELL', 0))
    uniform_ratio = max_signal / total_assets
    
    if uniform_ratio >= UNIFORM_SIGNAL_THRESHOLD:
        dominant_signal = 'BUY' if signal_counter.get('BUY', 0) > signal_counter.get('SELL', 0) else 'SELL'
        warning_msg = (
            f"⚠️ MARKET WARNING: {uniform_ratio:.0%} assets show {dominant_signal} signals\n"
            f"This may indicate systemic market conditions"
        )
        try:
            requests.post(BASE_URL, json={"chat_id": CHAT_ID, "text": warning_msg}, timeout=10)
        except Exception:
            pass

# ─── Save Backtest Summary ───
if summary:
    df_summary = pd.DataFrame(summary)
    upload_backtest_summary(df_summary)
