
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

# Configuration
TICKERS = ['SPY', 'TSLA', '1211.HK', 'GC=F', 'D05.SI']
START_DATE = "2017-01-01"
END_DATE = None
STATE_RANGE = range(2, 30)
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10
ROLLING_HYBRID_WINDOW = 10

SCALERS = {
    'per-asset': lambda df: StandardScaler().fit_transform(df[['LogReturn']].dropna()),
    'global': lambda df: StandardScaler().fit_transform(df[['LogReturn']].dropna().values.reshape(-1, 1))
}

summary = []

for ticker in TICKERS:
    print(f"\n🔍 Processing: {ticker}")
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        df['LogReturn'] = np.log(df['Adj Close']).diff()
        df.dropna(inplace=True)
    except Exception as e:
        print(f"⚠️ Data download failed for {ticker}: {e}")
        continue

    best_model = None
    best_bic = np.inf
    scaler_type = 'per-asset'

    for scaler_option in ['per-asset', 'global']:
        try:
            X = SCALERS[scaler_option](df)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for n_states in STATE_RANGE:
                    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200)
                    model.fit(X)
                    bic = -2 * model.score(X) + n_states * np.log(len(X))
                    if bic < best_bic:
                        best_bic = bic
                        best_model = model
                        best_states = n_states
                        scaler_type = scaler_option
            break  # if successful, skip fallback
        except Exception as e:
            print(f"⚠️ Error using {scaler_option} scaler on {ticker}: {e}")
            continue

    if best_model is None:
        print(f"❌ HMM fitting failed for {ticker}. Using hybrid fallback.")
        try:
            df['Position'] = 1 if df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean().iloc[-1] > 0 else 0
            df['StrategyReturn'] = df['LogReturn'] * df['Position']
            buyhold = np.exp(df['LogReturn'].cumsum())[-1]
            strategy = np.exp(df['StrategyReturn'].cumsum())[-1]
            summary.append({
                'Ticker': ticker, 'BuyHold': buyhold, 'HMMReturn': strategy,
                'Ratio': strategy / buyhold, 'ScalerType': 'hybrid', 'NumUsedStates': 0
            })
            continue
        except Exception as e:
            print(f"🔥 Hybrid fallback failed for {ticker}: {e}")
            summary.append({
                'Ticker': ticker, 'BuyHold': 1.0, 'HMMReturn': 1.0,
                'Ratio': 1.0, 'ScalerType': 'failed', 'NumUsedStates': 0
            })
            continue

    hidden_states = best_model.predict(X)
    df['HiddenState'] = hidden_states
    sharpe_ratios = df.groupby('HiddenState')['LogReturn'].mean() / df.groupby('HiddenState')['LogReturn'].std()
    durations = df.groupby('HiddenState').size()

    good_states = sharpe_ratios[(sharpe_ratios > SHARPE_THRESHOLD) & (durations > DURATION_THRESHOLD)].index.tolist()

    if len(good_states) == 0:
        print(f"⚠️ No states passed filters for {ticker}. Sharpe ratios:\n{sharpe_ratios.round(2)}")
        good_states = sharpe_ratios.sort_values(ascending=False).head(1).index

    df['Position'] = df['HiddenState'].isin(good_states).astype(int)
    df['StrategyReturn'] = df['LogReturn'] * df['Position']

    if df['Position'].sum() == 0:
        print(f"⚠️ Final strategy dark for {ticker}. Applying hybrid fallback.")
        df['Position'] = 1 if df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean().iloc[-1] > 0 else 0
        df['StrategyReturn'] = df['LogReturn'] * df['Position']

    buyhold = np.exp(df['LogReturn'].cumsum())[-1]
    strategy = np.exp(df['StrategyReturn'].cumsum())[-1]

    summary.append({
        'Ticker': ticker, 'BuyHold': buyhold, 'HMMReturn': strategy,
        'Ratio': strategy / buyhold, 'ScalerType': scaler_type, 'NumUsedStates': best_states
    })

# Final output
results = pd.DataFrame(summary)
print(results.to_string(index=False))
