# HMM Strategy v12d: Enhanced Hybrid Model with Telegram Alerts
# Debug Version - Extensive LogReturn troubleshooting
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")

# ─── Config ───
# TEST WITH JUST ONE ASSET FIRST
ASSETS = {'SPY': 'SPY'}  # Test with one asset first

START_DATE = '2017-01-01'
END_DATE = None

# ─── Debugging Functions ───
def debug_dataframe(df, ticker, stage):
    """Print detailed DataFrame info for debugging"""
    print(f"\n🔧 [{ticker}] DEBUG AT STAGE: {stage}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns present: {list(df.columns)}")
    
    if not df.empty:
        print("First 2 rows:")
        print(df.head(2))
        print("Last 2 rows:")
        print(df.tail(2))
        
        if 'Price' in df.columns:
            print(f"Price stats: min={df['Price'].min()}, max={df['Price'].max()}, nulls={df['Price'].isnull().sum()}")
    else:
        print("DataFrame is EMPTY")

# ─── Processing Loop ───
for name, ticker in ASSETS.items():
    print(f"\n🔍 STARTING: {ticker}")
    try:
        # 1. Download price data
        print(f"Downloading data for {ticker}...")
        df = yf.download(
            ticker, 
            start=START_DATE, 
            end=END_DATE, 
            auto_adjust=False, 
            progress=False,
            timeout=60
        )
        debug_dataframe(df, ticker, "AFTER DOWNLOAD")
        
        # Check if we got valid price data
        if df.empty:
            print(f"⚠️ {ticker}: No data downloaded")
            continue
            
        # 2. Select price column
        price_col = None
        for col in ['Adj Close', 'Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                price_col = col
                break
                
        if not price_col:
            print(f"⚠️ {ticker}: No price column found")
            debug_dataframe(df, ticker, "NO PRICE COLUMN")
            continue
            
        print(f"Using price column: {price_col}")
        df['Price'] = df[price_col]
        debug_dataframe(df, ticker, "AFTER PRICE SELECTION")
        
        # 3. Check for NaN values in price
        if df['Price'].isnull().all():
            print(f"⚠️ {ticker}: All prices are NaN")
            debug_dataframe(df, ticker, "ALL PRICE NAN")
            continue
            
        # 4. Check data length
        if len(df) < 2:
            print(f"⚠️ {ticker}: Insufficient data points ({len(df)})")
            continue
            
        # 5. Attempt to create LogReturn
        print("Attempting to create LogReturn...")
        try:
            # Create a temporary column to verify calculation
            df['PriceLog'] = np.log(df['Price'])
            df['LogReturn'] = df['PriceLog'].diff()
            
            print("LogReturn creation succeeded!")
            print(f"LogReturn sample: {df['LogReturn'].head(3).tolist()}")
            
            # Clean up temporary column
            df.drop('PriceLog', axis=1, inplace=True, errors='ignore')
        except Exception as e:
            print(f"❌ LogReturn creation FAILED: {str(e)}")
            print("Troubleshooting info:")
            print(f"Price values sample: {df['Price'].head(3).tolist()}")
            print(f"Log values sample: {np.log(df['Price'].head(3)).tolist()}")
            print(f"Any negative prices? {any(df['Price'] <= 0)}")
            print(f"Any zero prices? {any(df['Price'] == 0)}")
            continue
        
        # 6. Drop NA returns
        before_count = len(df)
        df.dropna(subset=['LogReturn'], inplace=True)
        after_count = len(df)
        print(f"Dropped {before_count - after_count} rows with NA returns")
        
        if df.empty:
            print(f"⚠️ {ticker}: No valid returns after calculation")
            continue

        # 7. Continue with feature creation (simplified)
        print("Creating other features...")
        try:
            # Add mock features for testing
            df['MACD'] = 0.0
            df['MACD_diff'] = 0.0
            df['RSI'] = 50.0
            df['NewsSentiment'] = 0.0
            df['VIX'] = 0.0
            df['PCR'] = 0.0
            df['Volume_Z'] = 0.0
            
            print("All features created successfully!")
            debug_dataframe(df, ticker, "AFTER FEATURE CREATION")
            
        except Exception as e:
            print(f"⚠️ Feature creation failed: {str(e)}")
            continue

        # 8. Final check for LogReturn
        if 'LogReturn' not in df.columns:
            print("❌ CRITICAL: LogReturn disappeared after feature creation!")
            debug_dataframe(df, ticker, "LOG RETURN DISAPPEARED")
        else:
            print("✅ SUCCESS: All features present including LogReturn")
            print("Model would proceed from here...")

    except Exception as e:
        print(f"❌ TOP-LEVEL ERROR: {str(e)}")

print("\n⚠️ Debugging complete")
