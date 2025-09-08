# ==========================================
# META Feature Mart Analysis (Correlation + Regression)
# Works on local & Google Colab
# ==========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# ------------------------------------------
# 1. Load Feature Mart
# ------------------------------------------
file_path = "META_Complete_FeatureMart_2020_2024.csv"  # Update if needed
mart = pd.read_csv(file_path, parse_dates=['Date'])
mart.set_index("Date", inplace=True)

# ------------------------------------------
# 2. Prepare Features & Target
# ------------------------------------------
# Drop columns we don't want to include in X
X = mart.drop(columns=["Target", "LogReturn"], errors='ignore')
y = mart["Target"].copy()

# Optional: drop rows with any NaN values
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns=["Target"])
y = data["Target"]

# ------------------------------------------
# 3. Correlation Analysis
# ------------------------------------------
correlations = X.corrwith(y)
sorted_corr = correlations.abs().sort_values(ascending=False)

print("\n🔹 Feature-Target Correlation Analysis")
print("Mean correlation:", sorted_corr.mean())
print("Max correlation:", sorted_corr.idxmax(), sorted_corr.max())
print("Min correlation:", sorted_corr.idxmin(), sorted_corr.min())

# Heatmap of all features
plt.figure(figsize=(14, 10))
sns.heatmap(X.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# ------------------------------------------
# 4. Linear Regression
# ------------------------------------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
corr_yyhat = np.corrcoef(y, y_pred)[0, 1]

print("\n📊 Linear Regression Performance")
print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)
print("Correlation between y and y_pred:", corr_yyhat)

# ------------------------------------------
# 5. Hard Thresholding (p-values)
# ------------------------------------------
X_with_const = sm.add_constant(X)
ols_model = sm.OLS(y, X_with_const).fit()
pvals = ols_model.pvalues.drop("const", errors='ignore')
significant_count = (pvals <= 0.05).sum()

print("\n🔎 Hard Thresholding Results")
print(f"Total features: {len(pvals)}")
print(f"Significant features (p <= 0.05): {significant_count}")

# ------------------------------------------
# 6. Feature Importance Plot + Export
# ------------------------------------------
results_df = pd.DataFrame({
    "feature": ols_model.params.index,
    "coefficient": ols_model.params.values,
    "p_value": ols_model.pvalues.values
}).set_index("feature").drop("const", errors='ignore')

# Top N by absolute coefficient
top_n = 15
top_features = results_df.reindex(results_df["coefficient"].abs().sort_values(ascending=False).index).head(top_n)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_features.reset_index(), x="coefficient", y="feature", palette="viridis")
plt.title(f"Top {top_n} Most Important Features by Coefficient Magnitude")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.savefig("top_features_barplot.png")
plt.show()

# Export
top_features.reset_index().to_csv("top_important_features.csv", index=False)
print("\n✅ Exported top_important_features.csv")
print("✅ Exported correlation_heatmap.png and top_features_barplot.png")
print("✅ Analysis Complete")
