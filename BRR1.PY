import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# Read the Excel file
df = pd.read_excel('C:/Users/Farah Abbasi/OneDrive/Desktop/sop/DataSheetSOP.xlsx')

# Standardize column names (strip whitespace and lower-case)
df.columns = [col.strip().lower() for col in df.columns]
print("Columns available:", df.columns.tolist())

# Convert the dependent variable 'contact angle of functional group with glucose solution (degree)' to numeric
df['contact angle of functional group with glucose solution (degree)'] = pd.to_numeric(df['contact angle of functional group with glucose solution (degree)'], errors='coerce')

# Ensure the 'interferons' column is treated as a string
df['interferons'] = df['interferons'].astype(str)

# Create dummy variables for the 'interferons' column
interferons_dummies = pd.get_dummies(df['interferons'], drop_first=True)

# Force the dummy columns to be numeric (float)
interferons_dummies = interferons_dummies.astype(float)

# Concatenate the dependent variable and dummy variables
data = pd.concat([df['contact angle of functional group with glucose solution (degree)'], interferons_dummies], axis=1)

# Drop rows with missing values (if any conversion failed)
data = data.dropna()

# Define dependent and independent variables
y = data['contact angle of functional group with glucose solution (degree)']
X = data.drop(columns=['contact angle of functional group with glucose solution (degree)'])

# Add a constant term for the intercept
X = sm.add_constant(X)

# Force all columns in X to numeric (float)
X = X.astype(float)

# Check data types for debugging
print("X data types:")
print(X.dtypes)
print("y data type:", y.dtype)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

# Construct and print the regression equation
coef_details = model.params
eq_terms = [f"{coef_details['const']:.4f}"]
for col in X.columns:
    if col != 'const':
        eq_terms.append(f"{coef_details[col]:+.4f}*{col}")
equation = "contact angle of functional group with glucose solution (degree) = " + " ".join(eq_terms)
print("\nRegression Equation:")
print(equation)
import matplotlib.pyplot as plt

# Compute predicted values from the regression model
y_pred = model.fittedvalues

# Create a scatter plot of observed vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.5, label='Data points')
plt.xlabel("Observed Contact Angle (degree)")
plt.ylabel("Predicted Contact Angle (degree)")
plt.title("Observed vs. Predicted Contact Angle of Base Material")
# Plot a 45° line for reference
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='45° reference')
plt.legend()
plt.grid(True)
plt.show()
