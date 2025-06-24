
import yfinance as yf
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Configurations ----------------------------- #
TICKERS = ["SPY", "TSLA", "1211.HK", "GC=F", "D05.SI"]
START_DATE = "2018-01-01"
END_DATE = None  # default is today

SCALERS = ["per-asset", "global"]
STATE_RANGE = range(2, 15)
SHARPE_THRESHOLD = 0.1
DURATION_THRESHOLD = 10
ROLLING_HYBRID_WINDOW = 10
MAX_BIC_RETRIES = 3

# ----------------------------- Main Script ----------------------------- #
def compute_sharpe(series):
    return np.mean(series) / np.std(series) * np.sqrt(252) if np.std(series) > 0 else 0

def try_fit_hmm(X, n_states):
    for _ in range(MAX_BIC_RETRIES):
        try:
            model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
            model.fit(X)
            if np.isnan(model.transmat_).any(): continue
            return model, model.score(X)
        except Exception:
            continue
    return None, -np.inf

all_results = []

for ticker in TICKERS:
    print(f"🔍 Processing: {ticker}")
    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

    if df.empty:
        print(f"⚠️ No data for {ticker}, skipping.")
        continue

    scaler_type_used = None
    model = None
    best_model = None
    best_score = -np.inf
    best_states = 0
    X_final = None

    for scaler_type in SCALERS:
        try:
            df = df.copy()
            df['LogReturn'] = np.log(df['Adj Close']).diff().dropna()

            if scaler_type == "per-asset":
                scaler = StandardScaler()
                X = scaler.fit_transform(df['LogReturn'].dropna().values.reshape(-1, 1))
            else:
                all_returns = []
                for t in TICKERS:
                    d = yf.download(t, start=START_DATE, end=END_DATE, progress=False)
                    if not d.empty:
                        d['LogReturn'] = np.log(d['Adj Close']).diff()
                        all_returns.extend(d['LogReturn'].dropna().values)
                scaler = StandardScaler()
                X = scaler.fit_transform(df['LogReturn'].dropna().values.reshape(-1, 1))

            for n_states in STATE_RANGE:
                m, score = try_fit_hmm(X, n_states)
                if m and score > best_score:
                    best_model = m
                    best_score = score
                    best_states = n_states
                    X_final = X

            if best_model:
                scaler_type_used = scaler_type
                break

        except Exception as e:
            print(f"⚠️ Error using {scaler_type} scaler on {ticker}: {e}")

    if not best_model:
        print(f"❌ HMM fitting failed for {ticker}. Using hybrid fallback.")
        df['Position'] = 0
        if df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean().iloc[-1] > 0:
            df['Position'] = 1
        hmm_return = np.exp((df['LogReturn'] * df['Position']).cumsum()).iloc[-1]
        buyhold_return = np.exp(df['LogReturn'].cumsum()).iloc[-1]
        all_results.append([ticker, buyhold_return, hmm_return, hmm_return / buyhold_return, "hybrid-fallback", 0])
        continue

    hidden_states = best_model.predict(X_final)
    df = df.iloc[-len(hidden_states):].copy()
    df['HiddenState'] = hidden_states
    df['LogReturn'] = np.log(df['Adj Close']).diff()

    state_metrics = df.groupby('HiddenState').agg({
        'LogReturn': ['mean', 'std', 'count']
    })
    state_metrics.columns = ['Mean', 'Std', 'Count']
    state_metrics['Sharpe'] = state_metrics['Mean'] / state_metrics['Std'] * np.sqrt(252)
    good_states = state_metrics[
        (state_metrics['Sharpe'] > SHARPE_THRESHOLD) & 
        (state_metrics['Count'] > DURATION_THRESHOLD)
    ].index.tolist()

    if len(good_states) == 0:
        print(f"{ticker} - No states passed filters. Sharpe ratios:\n{state_metrics['Sharpe']}")
        good_states = state_metrics['Sharpe'].sort_values(ascending=False).head(1).index.tolist()

    if len(good_states) == 0:
        df['Position'] = 0
        if df['LogReturn'].rolling(ROLLING_HYBRID_WINDOW).mean().iloc[-1] > 0:
            df['Position'] = 1
    else:
        df['Position'] = df['HiddenState'].isin(good_states).astype(int)

    df['StrategyReturn'] = df['LogReturn'] * df['Position']
    buyhold_return = np.exp(df['LogReturn'].cumsum()).iloc[-1]
    hmm_return = np.exp(df['StrategyReturn'].cumsum()).iloc[-1]
    ratio = hmm_return / buyhold_return

    all_results.append([ticker, buyhold_return, hmm_return, ratio, scaler_type_used, best_states])

summary = pd.DataFrame(all_results, columns=[
    "Ticker", "BuyHold", "HMMReturn", "Ratio", "ScalerType", "NumUsedStates"
])
print(summary)
