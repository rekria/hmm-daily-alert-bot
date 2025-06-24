# HMM_Strategy_v12d_Hybrid_20250624_1243.py
# Hybrid model combining BIC state selection + v12c robust logic
# GitHub Actions compatible script

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# PARAMETERS
TICKERS = ['SPY', 'TSLA', '1211.HK', 'GC=F', 'D05.SI']
START_DATE = '2018-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
SCALERS = ['global', 'per-asset']
STATE_RANGE = range(2, 15)
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10

# FETCH VIX ONCE
VIX = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)['Close']

# FUNCTION: Fit HMM with best BIC

def fit_hmm_bic(X):
    best_bic = np.inf
    best_model = None
    for n in STATE_RANGE:
        try:
            model = GaussianHMM(n_components=n, covariance_type="full", random_state=42, n_iter=500)
            model.fit(X)
            logL = model.score(X)
            n_params = n**2 + 2 * n * X.shape[1] - 1
            bic = -2 * logL + n_params * np.log(len(X))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        except:
            continue
    return best_model

# FUNCTION: Signal from good states

def signal_from_good_states(df, model):
    hidden_states = model.predict(df[['LogReturn', 'VIX_Change']])
    df['State'] = hidden_states

    positions = pd.Series(0, index=df.index)
    good_states = []
    sharpe_ratios = {}

    for s in np.unique(hidden_states):
        mask = df['State'] == s
        if mask.sum() < DURATION_THRESHOLD:
            continue
        returns = df['LogReturn'][mask]
        if returns.std() == 0:
            continue
        sharpe = returns.mean() / returns.std()
        sharpe_ratios[s] = sharpe
        if sharpe > SHARPE_THRESHOLD:
            good_states.append(s)

    if not good_states:
        print(f"No states passed filters. Sharpe ratios:\n{sharpe_ratios}")
        if sharpe_ratios:
            good_states = pd.Series(sharpe_ratios).sort_values(ascending=False).head(1).index.tolist()
        else:
            print("Fallback: using hybrid rolling mean rule")
            if df['LogReturn'].rolling(10).mean().iloc[-1] > 0:
                positions[:] = 1
            return positions, 0

    positions[df['State'].isin(good_states)] = 1
    return positions, len(good_states)

# MAIN LOOP

summary = []

for ticker in TICKERS:
    print(f"\n🔍 Processing: {ticker}")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        df = df[['Close']].rename(columns={'Close': 'Price'})
        df['LogReturn'] = np.log(df['Price']).diff()
        df['VIX'] = VIX.reindex(df.index).fillna(method='ffill')
        df['VIX_Change'] = df['VIX'].pct_change().fillna(0)
        df.dropna(inplace=True)

        best_result = None

        for scale_type in SCALERS:
            df_scaled = df.copy()
            if scale_type == 'global':
                df_scaled[['LogReturn', 'VIX_Change']] = (df_scaled[['LogReturn', 'VIX_Change']] - df_scaled[['LogReturn', 'VIX_Change']].mean()) / df_scaled[['LogReturn', 'VIX_Change']].std()
            else:
                df_scaled[['LogReturn', 'VIX_Change']] = df_scaled[['LogReturn', 'VIX_Change']].apply(lambda x: (x - x.mean()) / x.std())

            model = fit_hmm_bic(df_scaled[['LogReturn', 'VIX_Change']])
            if model is None:
                continue

            position, used_states = signal_from_good_states(df_scaled, model)
            df_scaled['Position'] = position.shift().fillna(0)
            df_scaled['Strategy'] = df_scaled['LogReturn'] * df_scaled['Position']
            perf = df_scaled[['LogReturn', 'Strategy']].cumsum().apply(np.exp)

            hmm_return = perf['Strategy'].iloc[-1]
            buyhold = perf['LogReturn'].iloc[-1]
            ratio = hmm_return / buyhold

            if best_result is None or ratio > best_result['Ratio']:
                best_result = {'Ticker': ticker, 'BuyHold': buyhold, 'HMMReturn': hmm_return, 'Ratio': ratio, 'ScalerType': scale_type, 'NumUsedStates': used_states}

        if best_result:
            summary.append(best_result)

    except Exception as e:
        print(f"❌ {ticker} failed: {e}")

# FINAL OUTPUT
print("\n    Ticker    BuyHold  HMMReturn     Ratio ScalerType  NumUsedStates")
for row in summary:
    print(f"{row['Ticker']:>8} {row['BuyHold']:10.6f} {row['HMMReturn']:11.6f} {row['Ratio']:10.6f} {row['ScalerType']:>10} {row['NumUsedStates']:>14}")
