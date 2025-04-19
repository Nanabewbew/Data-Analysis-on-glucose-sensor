import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_path = r"C:\Users\Farah Abbasi\OneDrive\Desktop\sop\DataSheet_Final.xlsx"

df = pd.read_excel(file_path, sheet_name="Sheet1", engine='openpyxl')

object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].astype(str)  
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    median_val = df[col].median()

    if pd.isna(median_val):
        df[col].fillna(0, inplace=True)
    else:
        df[col].fillna(median_val, inplace=True)


corr_matrix = df[numeric_cols].corr(method='pearson')


plt.figure(figsize=(16, 12))  
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.title("Pearson Correlation Heatmap")
plt.tight_layout() 
plt.savefig("pearson_correlation_heatmap_all_columns.png")
plt.show()
