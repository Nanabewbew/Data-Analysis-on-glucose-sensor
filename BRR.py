import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = r"C:\Users\Farah Abbasi\OneDrive\Desktop\sop\DataSheet_Final.xlsx"
df = pd.read_excel(file_path)

required_columns = ["Peak Current (mA)", "SR"]
for col in required_columns:
    if col not in df.columns:
        print(f"Column '{col}' not found in dataset.")
        exit()

df["Ratio of functional groups/C-C"] = pd.to_numeric(df["Ratio of functional groups/C-C"], errors="coerce")
df["Peak Current (mA)"] = pd.to_numeric(df["Peak Current (mA)"], errors="coerce")

df = df.dropna(subset=["Peak Current (mA)", "SRs"])

X = df[["Ratio of functional groups/C-C"]]
y = df["Peak Current (mA)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = BayesianRidge()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print(f" Model Coefficients: {model.coef_}")
print(f" Intercept: {model.intercept_}")

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual Peak Current (mA)")
plt.ylabel("Predicted Peak Current (mA)")
plt.title("Bayesian Ridge Regression: Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, color="purple", alpha=0.6)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Actual Peak Current (mA)")
plt.ylabel("Residuals (Errors)")
plt.title("Residual Plot: Bayesian Ridge Regression")
plt.grid(True)
plt.show()