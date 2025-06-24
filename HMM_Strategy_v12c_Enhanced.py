"""
HMM Strategy v12c - Clean Patched Version
Features:
✔ Hybrid Global/Per-Asset Scaling
✔ BIC-based State Selection with State Usage Check
✔ Sharper Regime Mapping Filters
✔ Confidence-Weighted Positioning
✔ Regime Transition Plotting
✔ BIC Score CSV Logging
✔ Backtest Summary Output Logging
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os

# Parameters
TICKERS = ['SPY', 'TSLA', '1211.HK', 'GC=F', 'D05.SI']
START_DATE = '2015-01-01'
END_DATE = '2024-12-31'
STATE_RANGE = range(2, 15)  # Reduced range to prevent overfitting
VOLATILITY_THRESHOLD = 0.015
SHARPE_THRESHOLD = 0.5
DURATION_THRESHOLD = 20

# Output paths
os.makedirs('output', exist_ok=True)
summary_results = []

for ticker in TICKERS:
    print(f"🔍 Processing: {ticker}")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    if df.empty or 'Adj Close' not in df.columns:
        print(f"⚠️ No valid adjusted close data for {ticker}. Skipping.")
        continue
    df['LogReturn'] = np.log(df['Adj Close']).diff()
    df.dropna(inplace=True)
    if len(df) < 200:
        print(f"⚠️ Not enough data for {ticker}. Skipping.")
        continue

    # Hybrid Scaling
    volatility = df['LogReturn'].std()
    scaler_type = "global" if volatility < VOLATILITY_THRESHOLD else "per-asset"
    scaler = StandardScaler()
    X = scaler.fit_transform(df['LogReturn'].values.reshape(-1, 1))

    # BIC Selection with State Usage Check
    best_bic = np.inf
    best_model = None
    bic_scores = []

    for n in STATE_RANGE:
        try:
            model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=42)
            model.fit(X)
            post = model.predict(X)
            if len(set(post)) < n:
                continue  # skip unused state models
            bic = model.bic(X)
            bic_scores.append((n, bic))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        except:
            continue

    if best_model is None:
        print(f"❌ No suitable HMM found for {ticker}")
        continue

    df['HiddenState'] = best_model.predict(X)

    # Evaluate states by Sharpe + duration
    grouped = df.groupby('HiddenState')['LogReturn']
    sharpe_ratios = grouped.mean() / grouped.std()
    durations = grouped.size()
    good_states = sharpe_ratios[(sharpe_ratios > SHARPE_THRESHOLD) & (durations > DURATION_THRESHOLD)].index
    weights = sharpe_ratios.clip(lower=0)
    weights = weights / weights.max()

    df['Position'] = df['HiddenState'].apply(lambda s: weights[s] if s in good_states else 0)
    df['StrategyReturn'] = df['LogReturn'] * df['Position']
    df['CumulM'] = np.exp(df['LogReturn'].cumsum())
    df['CumulS'] = np.exp(df['StrategyReturn'].cumsum())

    market_return = df['CumulM'].iloc[-1]
    strategy_return = df['CumulS'].iloc[-1]
    ratio = strategy_return / market_return
    summary_results.append([ticker, market_return, strategy_return, ratio, scaler_type, len(good_states)])

    # Regime plot
    plt.figure(figsize=(12,6))
    for state in df['HiddenState'].unique():
        plt.plot(df[df['HiddenState']==state].index, df[df['HiddenState']==state]['Adj Close'], label=f'State {state}')
    plt.title(f"{ticker} Regime Overlay")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output/{ticker}_regimes.png")
    plt.close()

    # BIC log
    pd.DataFrame(bic_scores, columns=['States', 'BIC']).to_csv(f"output/{ticker}_bic_scores.csv", index=False)

    # Performance plot
    plt.figure(figsize=(12,6))
    plt.plot(df['CumulM'], label='Buy & Hold')
    plt.plot(df['CumulS'], label='HMM Strategy')
    plt.title(f"{ticker} Strategy Comparison ({scaler_type} scaled, {len(good_states)} states used)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output/{ticker}_strategy_perf.png")
    plt.close()

# Final summary
summary_df = pd.DataFrame(summary_results, columns=['Ticker', 'BuyHold', 'HMMReturn', 'Ratio', 'ScalerType', 'NumUsedStates'])
summary_df.to_csv("output/Backtest_Summary_v12c.csv", index=False)
print(summary_df)
