# Quantitative Trading Analysis on META Stock

<img width="1352" height="842" alt="image" src="https://github.com/user-attachments/assets/f85e83f6-17d3-478a-8f28-2cc50c05251d" />

## 📌 Overview

This project is an **end-to-end quantitative trading pipeline** for Meta (META) stock. It builds a **feature mart (2020–2024)** combining:

* **Technical indicators** (SMA, EMA, RSI, MACD, ATR, OBV, Bollinger Bands, etc.)
* **Macroeconomic factors** (Fama-French 5 factors, ADS Index, FRED series like yields, VIX, oil, FX, CPI)
* **Google Trends signals** (Meta, Facebook, Instagram, WhatsApp, Threads)
* **News sentiment features** (headline polarity using TextBlob + NewsAPI).

On top of this enriched dataset, the project applies **feature engineering, regression models, and trading signal generation** to backtest strategies and evaluate predictive performance.

---

## 🛠️ Tech Stack

* **Python** (pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, plotly)
* **Finance APIs**: yFinance, FRED API, PyTrends, NewsAPI
* **TA Library**: `ta` (technical indicators)
* **Stats/Econometrics**: statsmodels, arch
* **Visualization**: matplotlib, seaborn, plotly

---

## 🧩 Key Components

### 1. Data Ingestion

* Historical OHLCV data (2020–2024) via Yahoo Finance.
* Macro data:

  * Fama-French 5-Factor CSV
  * ADS Business Conditions Index
  * FRED series (yields, VIX, oil, FX, inflation, crypto, etc.).
* Google Trends (Meta ecosystem keywords).
* News sentiment (TextBlob on NewsAPI headlines).

### 2. Feature Engineering

* **Technical Indicators**: SMA/EMA, MACD, RSI, ATR, Bollinger Bands, OBV, CCI, ROC, MFI.
* **Lagged Features**: Past 1–5 day closes and returns.
* **Rolling Stats**: 20-day volatility, max, min.
* **Buy/Sell Signals**: RSI + MACD rules, composite “Strong Buy” scores.
* **Macro + Trends Merge**: Combined into a **META\_FeatureMart\_2020\_2024\_FULL.csv**.

### 3. Modeling

* Feature correlation analysis.
* Regression models with feature selection:

  * **OLS (baseline)**
  * **Ridge Regression**
  * **Lasso Regression**
  * **ElasticNet Regression**.
* Evaluation metrics: **RMSE, MAE, R²**.

### 4. Backtesting & Evaluation

* Generated trading signals from buy/sell conditions.
* Evaluated strategy performance with:

  * CAGR
  * Sharpe Ratio
  * Sortino Ratio
  * Max Drawdown
* Visualized with **equity curves, trade markers, heatmaps**.

---

## 📊 Project Architecture

<img width="1138" height="819" alt="image" src="https://github.com/user-attachments/assets/2eb4936e-685b-4462-9367-19d2bd94fd97" />


---

## ⚙️ Functional Description

### 📊 Models Implemented

* **Baseline:** OLS, AR(1), CAPM, Fama-French 3-Factor
* **Regularized ML:** ElasticNet, Ridge, Lasso
* **Tree-Based ML:** Random Forest, Gradient Boosting
* **Dimensionality Reduction:** PCA + OLS, Factor Augmentation, PLS
* **Advanced Econometrics:** GARCH(1,1), Kalman Filter

### 🔑 Trading Signals

* Buy / Sell / Hold classification from predicted returns
* Candlestick patterns + RSI confirmation
* Momentum vs. Mean Reversion strategies

### 📈 Evaluation Metrics

* RMSE, R² for predictive accuracy
* CAGR, Sharpe ratio, Sortino ratio
* Max Drawdown analysis
* Confusion matrix for signal accuracy

---

## 🔑 Key Achievements

* Built a **comprehensive feature mart (70+ features)** combining technical, macro, sentiment, and trend data.
* Compared **12+ regression & ML models**.
* Implemented **multiple regression models** achieving **R² > 0.99** on test data.
* Designed **signal-based strategies** (RSI + MACD confirmation) with interpretable buy/sell markers.
* Generated **Buy-Hold-Sell signals** with >59% accuracy on test data.
* Backtested strategies with visualization of **equity curves & trade signals**.
* Quantified risk using Sharpe, Sortino, and drawdown metrics.
* Delivered rich **visualizations** for patterns, signals, and portfolio equity.

---

## 📊 Impact

* Demonstrated how combining **technical, macroeconomic, and factor-based features** improves predictive power.
* Created a reproducible framework for **quant trading research**.
* Highlighted trade-offs between interpretability (OLS, AR) and performance (Random Forest, Gradient Boosting).
* Served as a **blueprint for future stock modeling pipelines** beyond META.

---

## 📬 Contact

For questions or collaboration, please contact **Dhanvardini Rajendran**.
