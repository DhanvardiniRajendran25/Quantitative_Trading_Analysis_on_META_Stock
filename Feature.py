# ================================================
# META Stock Feature Mart: End-to-End Implementation (2020-2024)
# ================================================

# ✅ PART 1: Data Gathering & Feature Engineering
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import joblib
from pandas_datareader import data as web

# Download META stock data
data = yf.download("META", start="2020-01-01", end="2024-12-31", group_by="ticker")
meta = data.copy()
if isinstance(meta.columns, pd.MultiIndex):
    meta.columns = meta.columns.get_level_values(1)
meta.dropna(inplace=True)

# Technical Indicators
meta['SMA_14'] = ta.sma(meta['Close'], length=14)
meta['EMA_14'] = ta.ema(meta['Close'], length=14)
meta['RSI_14'] = ta.rsi(meta['Close'], length=14)
meta.ta.macd(close='Close', append=True)
meta.ta.obv(append=True)
meta.ta.bbands(close='Close', append=True)
meta['CDL_DOJI'] = talib.CDLDOJI(meta['Open'], meta['High'], meta['Low'], meta['Close'])
meta['CDL_HAMMER'] = talib.CDLHAMMER(meta['Open'], meta['High'], meta['Low'], meta['Close'])
meta['CDL_ENGULFING'] = talib.CDLENGULFING(meta['Open'], meta['High'], meta['Low'], meta['Close'])

# Engineered Features
meta['LogReturn'] = np.log(meta['Close'] / meta['Close'].shift(1))
meta['Target'] = meta['LogReturn'].shift(-1)
meta['Lag1_Return'] = meta['LogReturn'].shift(1)
meta['Volatility_10'] = meta['LogReturn'].rolling(10).std()
meta['Momentum_5'] = meta['Close'] - meta['Close'].shift(5)
meta['MACD_Diff'] = meta['MACD_12_26_9'] - meta['MACDs_12_26_9']
meta['DayOfWeek'] = meta.index.dayofweek
meta['Month'] = meta.index.month

# ✅ PART 2: External Macro & Sentiment Features
start = "2020-01-01"
end = "2024-12-31"
fred_series = {
    'UNRATE': 'Unemployment_Rate',
    'CPIAUCSL': 'CPI',
    'M2SL': 'Money_Supply_M2',
    'INDPRO': 'Industrial_Production',
    'FEDFUNDS': 'Federal_Funds_Rate',
    'T10Y3M': 'Yield_Spread',
    'VIXCLS': 'VIX'
}

macro = pd.DataFrame()
for code, name in fred_series.items():
    df = web.DataReader(code, 'fred', start, end)
    df.rename(columns={code: name}, inplace=True)
    macro = pd.concat([macro, df], axis=1)

# ADS Index (uploaded Excel file)
ads = pd.read_excel("ADS_Index_Most_Current_Vintage.xlsx", sheet_name="Sheet1", skiprows=1)
ads.columns = ['Date', 'ADS_Index']
ads['Date'] = pd.to_datetime(ads['Date'], format="%Y:%m:%d")
ads.set_index("Date", inplace=True)
macro = macro.join(ads, how='outer')

# Fama-French Factors + Momentum + RMW + CMA
ff3 = pd.read_csv("F-F_Research_Data_Factors_daily.CSV", skiprows=3)
ff3.rename(columns={'Unnamed: 0': 'Date', 'Mkt-RF': 'Mkt_RF', 'RF': 'RiskFree'}, inplace=True)
ff3['Date'] = pd.to_datetime(ff3['Date'], format='%Y%m%d')
ff3.set_index("Date", inplace=True)
ff3 = ff3[["Mkt_RF", "SMB", "HML", "RiskFree"]] / 100

mom = pd.read_csv("F-F_Momentum_Factor_daily.CSV", skiprows=13)
mom.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
mom['Date'] = pd.to_datetime(mom['Date'], format='%Y%m%d')
mom.set_index("Date", inplace=True)
mom = mom[["Mom"]] / 100

ff5 = pd.read_csv("F-F_Research_Data_5_Factors_2x3_daily.CSV", skiprows=3)
ff5.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
ff5['Date'] = pd.to_datetime(ff5['Date'], format='%Y%m%d')
ff5.set_index("Date", inplace=True)
ff5 = ff5[["RMW", "CMA"]] / 100

macro = macro.join(ff3, how='outer').join(mom, how='outer').join(ff5, how='outer')
macro.index = pd.to_datetime(macro.index)
macro = macro.sort_index()
macro = macro.loc["2020-01-01":"2024-12-31"]

# ✅ PART 3: Preprocessing
meta.reset_index(inplace=True)
macro.reset_index(inplace=True)
final_df = pd.merge(meta, macro, on="Date", how="inner")
final_df = final_df.set_index("Date").dropna()
final_df.to_csv("META_Complete_FeatureMart_2020_2024.csv")

# ✅ PART 4: Analysis & Modeling
y = final_df["Target"]
X = final_df.drop(columns=["Target", "LogReturn"])
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Linear Regression (OLS)
X_const = sm.add_constant(X_scaled)
model = sm.OLS(y, X_const).fit()
y_pred = model.predict(X_const)

# Evaluation
print("R²:", r2_score(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("MAE:", mean_absolute_error(y, y_pred))

# Feature Importance Plot
importance = pd.Series(np.abs(model.params[1:]), index=X.columns)
top_features = importance.sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 20 Important Features (Linear Regression Coefficients)")
plt.tight_layout()
plt.savefig("top_features_barplot.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(X.corr(), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Save Model
joblib.dump(model, "meta_ols_model.pkl")
print("✅ Feature Mart, Modeling, and Export Complete")
