import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

file_path = r"C:\Users\Farah Abbasi\OneDrive\Desktop\sop\DataSheet_Final.xlsx"
df = pd.read_excel(file_path)

df_selected = df[['Peak Current (mA)', 'SR']].dropna()
df_selected = df_selected.apply(pd.to_numeric, errors='coerce').dropna()

X = df_selected[['SR']].values
y = df_selected['Peak Current (mA)'].values

degree = 2
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

r2 = r2_score(y, y_pred)
n = X_poly.shape[0]
p = X_poly.shape[1] - 1 
r2_adj = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)



print("\n--- Polynomial Regression (Degree 2) ---")
print(f"R² Score: {r2:.4f}")
print(f"Adjusted R²: {r2_adj:.4f}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.scatter(X, y_pred, color='red', marker='x', label='Predicted data')
plt.xlabel('SR')
plt.ylabel('Peak Current (mA)')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.grid(True)
plt.show()
