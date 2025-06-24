import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# --- SETTINGS ---
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
TICKERS = ['SPY', 'TSLA', '1211.HK', 'GC=F', 'D05.SI']
STATE_RANGE = range(2, 11)  # BIC search from 2 to 10 states
PLOT = False  # Set to True if you want charts (won’t display on GitHub Actions)

# --- BIC-based HMM Selection ---
def select_hmm_by_bic(X, state_range):
    best_bic = np.inf
    best_model = None
    best_n = None
    for n_states in state_range:
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000, random_state=42)
            model.fit(X)
            bic = model.score(X) * -2 + n_states * np.log(len(X)) * X.shape[1]
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_n = n_states
        except:
            continue
    return best_model, best_n

# --- Analysis ---
results = []

for ticker in TICKERS:
    print(f"\n🔍 Processing: {ticker}")

    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
    if df.isnull().values.any() or len(df) < 250:
        print(f"⚠️ Not enough data for {ticker}, skipping.")
        continue

    df = df[['Close']].copy()
    df['LogReturn'] = np.log(df['Close']).diff()
    df.dropna(inplace=True)

    # Per-asset scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[['LogReturn']])
    
    # BIC Model selection
    model, n_selected = select_hmm_by_bic(scaled, STATE_RANGE)
    if not model:
        print(f"❌ Failed to fit HMM for {ticker}")
        continue
    print(f"✅ Selected {n_selected} states using BIC")

    hidden_states = model.predict(scaled)
    df['HiddenState'] = hidden_states
    state_returns = df.groupby('HiddenState')['LogReturn'].mean()
    positive_states = state_returns[state_returns > 0].index.tolist()
    df['Position'] = df['HiddenState'].isin(positive_states).astype(int)
    df['StrategyReturn'] = df['LogReturn'] * df['Position']
    df['CumulMarket'] = np.exp(df['LogReturn'].cumsum())
    df['CumulStrategy'] = np.exp(df['StrategyReturn'].cumsum())

    # Performance
    buyhold = df['CumulMarket'].iloc[-1]
    hmm_perf = df['CumulStrategy'].iloc[-1]
    ratio = hmm_perf / buyhold if buyhold > 0 else 0
    results.append((ticker, buyhold, hmm_perf, ratio))

    print(f"{ticker}: Buy&Hold → {buyhold:.4f}, HMM → {hmm_perf:.4f}, Ratio → {ratio:.2f}×")

    if PLOT:
        plt.figure(figsize=(10, 5))
        plt.plot(df['CumulMarket'], label='Buy & Hold')
        plt.plot(df['CumulStrategy'], label='HMM Strategy')
        plt.title(f"{ticker} - HMM vs Buy & Hold")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.show()

# --- Summary Output ---
print("\n📊 Final Comparison:")
for t, b, h, r in results:
    print(f"{t}: Buy&Hold → {b:.4f}, HMM → {h:.4f}, Ratio → {r:.2f}×")

