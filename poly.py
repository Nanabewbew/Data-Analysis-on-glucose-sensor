import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load your data (replace 'your_file.xlsx' with the actual filename)
file_path = r"C:\Users\Farah Abbasi\OneDrive\Desktop\sop\DataSheet_Final.xlsx"
df = pd.read_excel(file_path)

# Extract relevant columns and clean data
df_selected = df[['Peak Current (mA)', 'SR']].dropna()
df_selected = df_selected.apply(pd.to_numeric, errors='coerce').dropna()

# Define independent (X) and dependent (y) variables
X = df_selected[['SR']].values
y = df_selected['Peak Current (mA)'].values

# Transform to polynomial features (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Generate predictions
y_pred = model.predict(X_poly)
r2 = model.score(X_poly, y)
print(f"R² Score: {r2:.4f}")

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.scatter(X, y_pred, color='red', marker='x', label='Predicted data')
plt.xlabel('SR')
plt.ylabel('Peak Current (mA)')
plt.title('Polynomial Regression (Degree 2)')
plt.legend()
plt.show()
