import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

# Load data from Excel file
file_path = r"C:\Users\Farah Abbasi\OneDrive\Desktop\sop\DataSheet_Final.xlsx"
df = pd.read_excel(file_path)

# Convert columns to numeric and drop rows with missing values in 'Peak Voltage (V)' and 'Peak Current (mA)'
df["Peak Voltage (V)"] = pd.to_numeric(df["Peak Voltage (V)"], errors="coerce")
df["Peak Current (mA)"] = pd.to_numeric(df["Peak Current (mA)"], errors="coerce")
df = df.dropna(subset=["Peak Voltage (V)", "Peak Current (mA)"])

# Define features and target
X = df[["Peak Voltage (V)"]]
y = df["Peak Current (mA)"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a scaler for models that require feature scaling (Bayesian Ridge & SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------- Train Bayesian Ridge --------------------------
bayes_model = BayesianRidge()
bayes_model.fit(X_train_scaled, y_train)
y_pred_bayes = bayes_model.predict(X_test_scaled)

# -------------------------- Train Random Forest --------------------------
# Random Forest doesn't require scaling, so we use the original features
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# -------------------------- Train SVR --------------------------
# Using scaled features improves SVR performance
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)

# -------------------------- Evaluation Metrics --------------------------
def print_metrics(model_name, y_true, y_pred, extra_info=""):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    n = len(y_true)
    p = X.shape[1]
    r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    print(f"--- {model_name} ---")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {r2_adj:.4f}")
    print(f"MSE: {mse:.4f}")
    if extra_info:
        print(extra_info)
    print("\n")

# Print metrics for each model
print_metrics("Bayesian Ridge Regression", y_test, y_pred_bayes, 
              f"Coefficients: {bayes_model.coef_}\nIntercept: {bayes_model.intercept_}")
print_metrics("Random Forest Regressor", y_test, y_pred_rf, 
              "Model Type: RandomForestRegressor does not support coef_")
print_metrics("SVR", y_test, y_pred_svr, 
              "Model Type: SVR does not support coef_")

# -------------------------- Plot Predicted vs. Actual --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bayesian Ridge Plot
axes[0].scatter(y_test, y_pred_bayes, color="blue", alpha=0.6)
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
axes[0].set_xlabel("Actual Peak Current (mA)")
axes[0].set_ylabel("Predicted Peak Current (mA)")
axes[0].set_title("Bayesian Ridge Regression")

# Random Forest Plot
axes[1].scatter(y_test, y_pred_rf, color="green", alpha=0.6)
axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
axes[1].set_xlabel("Actual Peak Current (mA)")
axes[1].set_ylabel("Predicted Peak Current (mA)")
axes[1].set_title("Random Forest Regressor")

# SVR Plot
axes[2].scatter(y_test, y_pred_svr, color="purple", alpha=0.6)
axes[2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
axes[2].set_xlabel("Actual Peak Current (mA)")
axes[2].set_ylabel("Predicted Peak Current (mA)")
axes[2].set_title("SVR")
plt.tight_layout()
plt.show()

# -------------------------- Residual Plots --------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Residuals for Bayesian Ridge
residuals_bayes = y_test - y_pred_bayes
axes[0].scatter(y_test, residuals_bayes, color="blue", alpha=0.6)
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_xlabel("Actual Peak Current (mA)")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Bayesian Ridge Residuals")

# Residuals for Random Forest
residuals_rf = y_test - y_pred_rf
axes[1].scatter(y_test, residuals_rf, color="green", alpha=0.6)
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_xlabel("Actual Peak Current (mA)")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Random Forest Residuals")

# Residuals for SVR
residuals_svr = y_test - y_pred_svr
axes[2].scatter(y_test, residuals_svr, color="purple", alpha=0.6)
axes[2].axhline(y=0, color="red", linestyle="--")
axes[2].set_xlabel("Actual Peak Current (mA)")
axes[2].set_ylabel("Residuals")
axes[2].set_title("SVR Residuals")

plt.tight_layout()
plt.show()
