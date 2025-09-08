import pandas as pd
import numpy as np

# -----------------------------
# STEP 1: Load Raw Feature Mart
# -----------------------------
df = pd.read_csv("META_Complete_FeatureMart_2020_2024.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# -----------------------------
# STEP 2: Remove Target Leakage Columns
# -----------------------------
leak_cols = ['LogReturn', 'Target']
df = df.drop(columns=[col for col in leak_cols if col in df.columns], errors='ignore')

# -----------------------------
# STEP 3: Fill Missing Macro Data (monthly/weekly)
# -----------------------------
macro_features = [
    'CPI', 'Money_Supply_M2', 'Industrial_Production',
    'Federal_Funds_Rate', 'Unemployment_Rate'
]
df[macro_features] = df[macro_features].fillna(method='ffill')

# -----------------------------
# STEP 4: Fill Remaining Gaps
# -----------------------------
df = df.fillna(method='bfill').fillna(method='ffill')

# -----------------------------
# STEP 5: Add Engineered Features
# -----------------------------
# Time-based
df['WeekOfYear'] = df.index.isocalendar().week.astype(int)
df['IsMonthStart'] = df.index.is_month_start.astype(int)

# Interactions
df['RSI_OBV'] = df['RSI_14'] * df['OBV']
df['MACD_Diff'] = df['MACD'] - df['MACD_signal']

# Lag features
df['Lag3_Close'] = df['Close'].shift(3)
df['Lag5_Close'] = df['Close'].shift(5)
df['Lag10_Close'] = df['Close'].shift(10)

# Rolling features
df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
df['Volatility_20'] = df['Close'].rolling(window=20).std()
df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

# -----------------------------
# STEP 6: Create Target (Next-Day Log Return)
# -----------------------------
df['Target_LogReturn'] = np.log(df['Close'].shift(-1) / df['Close'])
df['Target_LogReturn'] = df['Target_LogReturn'].fillna(method='ffill')

# -----------------------------
# STEP 7: Patch Missing Early Dates
# -----------------------------
full_index = pd.date_range(start="2020-01-01", end=df.index[-1], freq="D")
df = df.reindex(full_index)
df.index.name = "Date"

# -----------------------------
# STEP 8: Final Cleanup and Export
# -----------------------------
df = df.dropna(how="all")  # remove any fully blank rows
df.to_csv("Final_META_FeatureMart_2020_2024.csv")

print("✅ Feature mart generated successfully: Final_META_FeatureMart_2020_2024.csv")
