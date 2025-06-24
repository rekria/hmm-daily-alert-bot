import os
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Parameters
TICKERS = ['SPY', 'TSLA', '1211.HK', 'GC=F', 'D05.SI']
START_DATE = '2015-01-01'
END_DATE = None
STATE_RANGE = range(2, 14)
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10
OUTPUT_DIR = "output_v12c"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_sharpe(series):
    return series.mean() / (series.std() + 1e-9)

summary = []

for ticker in TICKERS:
    print(f"\n🔍 Processing: {ticker}")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
    if df.empty or 'Adj Close' not in df.columns:
        print(f"{ticker}: Data not available or missing 'Adj Close'. Skipping.")
        continue

    df['LogReturn'] = np.log(df['Adj Close']).diff()
    df.dropna(inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[['LogReturn']])

    bic_scores = []
    best_bic = np.inf
    best_model = None
    best_states = None

    for n_states in STATE_RANGE:
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200)
            model.fit(X)
            states = model.predict(X)
            if len(set(states)) < n_states:
                continue  # Skip if not all states are used
            bic = -2 * model.score(X) + n_states * np.log(len(X))
            bic_scores.append((n_states, bic))
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_states = states
        except:
            continue

    if best_model is None:
        print(f"{ticker}: No suitable HMM model found.")
        continue

    df['HiddenState'] = best_states
    df['Regime'] = best_states

    sharpe_ratios = df.groupby('Regime')['LogReturn'].apply(calculate_sharpe)
    durations = df.groupby('Regime').size()

    good_states = sharpe_ratios[(sharpe_ratios > SHARPE_THRESHOLD) & (durations > DURATION_THRESHOLD)].index.tolist()

    if len(good_states) == 0:
        print(f"{ticker} - No states passed filters. Sharpe ratios:\n{sharpe_ratios}")
        # ✅ Fallback 1: Use top-1 Sharpe state
        good_states = sharpe_ratios.sort_values(ascending=False).head(1).index.tolist()

    if len(good_states) == 0:
        # ✅ Fallback 2: Hybrid rule if no states at all
        df['Position'] = (df['LogReturn'].rolling(10).mean().iloc[-1] > 0).astype(int)
        print(f"{ticker} - Fallback to hybrid signal")
    else:
        norm_sharpe = sharpe_ratios[good_states] / sharpe_ratios[good_states].sum()
        state_weights = dict(zip(good_states, norm_sharpe))
        df['Position'] = df['Regime'].map(state_weights).fillna(0)

    df['StrategyReturn'] = df['LogReturn'] * df['Position']
    df['CumulM'] = np.exp(df['LogReturn'].cumsum())
    df['CumulS'] = np.exp(df['StrategyReturn'].cumsum())

    ratio = df['CumulS'].iloc[-1] / df['CumulM'].iloc[-1]
    scaler_type = 'per-asset' if ticker in ['TSLA', '1211.HK'] else 'global'
    num_states_used = len(good_states) if isinstance(good_states, list) else 1

    summary.append({
        'Ticker': ticker,
        'BuyHold': round(df['CumulM'].iloc[-1], 6),
        'HMMReturn': round(df['CumulS'].iloc[-1], 6),
        'Ratio': round(ratio, 6),
        'ScalerType': scaler_type,
        'NumUsedStates': num_states_used
    })

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df['CumulM'], label='Buy & Hold')
    plt.plot(df['CumulS'], label='HMM Strategy')
    plt.title(f'{ticker} - Buy & Hold vs HMM Strategy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{ticker}_performance.png'))
    plt.close()

# Save summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "Backtest_Summary_v12c.csv"), index=False)
print(summary_df)
