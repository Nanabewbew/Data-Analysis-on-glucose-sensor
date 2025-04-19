import pandas as pd
import numpy as np

df = pd.read_excel("C:\\Users\\Farah Abbasi\\OneDrive\\Desktop\\sop\\SOP_DataSheet.xlsx")  
df.loc[df["Base material"] == "Ti3C2", "Surface Area of Base material (m2/g)"] = 18
df.loc[df["Base material"] == "Ti3C2", "Base material conductivity (S/m)"] = 850
df.loc[df["Base material"] == "Ti3C2", "Base material electroneghativity"] = 5.79
df.loc[df["Base material"] == "TiO₂", "Base material electroneghativity"] = 5.79
df.loc[df["Base material"] == "C", "Surface Area of Base material (m2/g)"] = 2.73
df.loc[df["Base material"] == "C", "Base material conductivity (S/m)"] = 55
df.loc[df["Base material"] == "C", "Base material electroneghativity"] = 2.55
df.loc[df["Base material"] == "TiO₂", "Surface Area of Base material (m2/g)"] = 18
df.loc[df["Base material"] == "TiO₂", "Base material conductivity (S/m)"] = 2.3 * np.exp(-6)

condition = (df["Material"] == "EG0.1") & (df["Functional groups"] == "C=O")
df.loc[condition, "Functional Group Conductivity"] = 13.427326  

condition = (df["Material"] == "EG0.1") & (df["Functional groups"] == "C-O")
df.loc[condition, "Functional Group Conductivity"] = 37.84781

condition = (df["Material"] == "EG0.5") & (df["Functional groups"] == "C-O")
df.loc[condition, "Functional Group Conductivity"] = 17.11798 

condition = (df["Material"] == "EG0.5") & (df["Functional groups"] == "C=O")
df.loc[condition, "Functional Group Conductivity"] = 10.612272 

condition = (df["Material"] == "EG0.5") & (df["Functional groups"] == "C-N")
df.loc[condition, "Functional Group Conductivity"] = 343.6 

condition = (df["Material"] == "EG1") & (df["Functional groups"] == "C-O")
df.loc[condition, "Functional Group Conductivity"] = 37.121062 

condition = (df["Material"] == "EG1") & (df["Functional groups"] == "C=O")
df.loc[condition, "Functional Group Conductivity"] = 19.21942 

condition = (df["Material"] == "EG1") & (df["Functional groups"] == "C-N")
df.loc[condition, "Functional Group Conductivity"] = 315.4 

col_b = df["Base material"]
col_c = df["Functional groups"]
col_d = df["Ratio of functional groups/C-C"]

ti3c2_mask = col_b == "Ti3C2"
df.loc[ti3c2_mask & (col_c == "Cl"), "Functional Group Conductivity"] = col_d * 0.01
df.loc[ti3c2_mask & (col_c == "Al"), "Functional Group Conductivity"] = col_d * 3.8e7
df.loc[ti3c2_mask & (col_c == "O"),  "Functional Group Conductivity"] = col_d * 43.78
df.loc[ti3c2_mask & (col_c == "OS"), "Functional Group Conductivity"] = col_d * 1.2e7
df.loc[ti3c2_mask & (col_c == "N"),  "Functional Group Conductivity"] = col_d * 2000


tio2_mask = col_b == "TiO₂"
df.loc[tio2_mask & (col_c == "Ti"), "Functional Group Conductivity"] = col_d * 2.38e6

ti3c2_mask = col_b == "C"
df.loc[ti3c2_mask & (col_c == "C-O"), "Functional group electronegativity (S/m)"] = col_d * 3.44
df.loc[ti3c2_mask & (col_c == "C=O"), "Functional group electronegativity (S/m)"] = col_d * 3.44
df.loc[ti3c2_mask & (col_c == "C-N"), "Functional group electronegativity (S/m)"] = col_d * 3.04

ti3c2_mask = col_b == "Ti3C2"
df.loc[ti3c2_mask & (col_c == "Cl"), "Functional group electronegativity (S/m)"] = col_d * 3.16
df.loc[ti3c2_mask & (col_c == "Al"), "Functional group electronegativity (S/m)"] = col_d * 1.61
df.loc[ti3c2_mask & (col_c == "O"),  "Functional group electronegativity (S/m)"] = col_d * 3.44
df.loc[ti3c2_mask & (col_c == "OS"), "Functional group electronegativity (S/m)"] = col_d * 2.2
df.loc[ti3c2_mask & (col_c == "N"),  "Functional group electronegativity (S/m)"] = col_d * 3.04

tio2_mask = col_b == "TiO₂"
df.loc[tio2_mask & (col_c == "Ti"), "Functional Group electronegativity (S/m)"] = col_d * 1.54

tio2_mask = col_b == "TiO₂"
df.loc[tio2_mask & (col_c == "Ti"), "Functional group atomic radii (pm)"] = col_d * 147

ti3c2_mask = col_b == "C"
df.loc[ti3c2_mask & (col_c == "C-O"), "Functional group atomic radii (pm)"] = col_d * 48
df.loc[ti3c2_mask & (col_c == "C=O"), "Functional group atomic radii (pm)"] = col_d * 48
df.loc[ti3c2_mask & (col_c == "C-N"), "Functional group atomic radii (pm)"] = col_d * 74

ti3c2_mask = col_b == "Ti3C2"
df.loc[ti3c2_mask & (col_c == "Cl"), "Functional group atomic radii (pm)"] = col_d * 99
df.loc[ti3c2_mask & (col_c == "Al"), "Functional group atomic radii (pm)"] = col_d * 143
df.loc[ti3c2_mask & (col_c == "O"),  "Functional group atomic radii (pm)"] = col_d * 48
df.loc[ti3c2_mask & (col_c == "OS"), "Functional group atomic radii (pm)"] = col_d * 136
df.loc[ti3c2_mask & (col_c == "N"),  "Functional group atomic radii (pm)"] = col_d * 74

df.loc[df["Base material"] == "Ti3C2", "Contact ange of base material with glucose solution (degree)"] = 148.7
df.loc[df["Base material"] == "TiO₂", "Contact ange of base material with glucose solution (degree)"] = 72
df.loc[df["Base material"] == "C", "Contact ange of base material with glucose solution (degree)"] = 150.3

tio2_mask = col_b == "TiO₂"
df.loc[tio2_mask & (col_c == "Ti"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 70

ti3c2_mask = col_b == "C"
df.loc[ti3c2_mask & (col_c == "C-O"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 150.3
df.loc[ti3c2_mask & (col_c == "C=O"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 150.3
df.loc[ti3c2_mask & (col_c == "C-N"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 103

ti3c2_mask = col_b == "Ti3C2"
df.loc[ti3c2_mask & (col_c == "Cl"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 152.9
df.loc[ti3c2_mask & (col_c == "Al"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 143.7
df.loc[ti3c2_mask & (col_c == "O"),  "Contact angle of functional group with glucose solution (degree)"] = col_d * 150.3
df.loc[ti3c2_mask & (col_c == "OS"), "Contact angle of functional group with glucose solution (degree)"] = col_d * 151.4
df.loc[ti3c2_mask & (col_c == "N"),  "Contact angle of functional group with glucose solution (degree)"] = col_d * 103

df.loc[df["Base material"] == "Ti3C2", "Interlayer Spacing of base material (nm)"] = 1.3
df.loc[df["Base material"] == "TiO₂", "Interlayer Spacing of base material (nm)"] = 0.35
df.loc[df["Base material"] == "C", "Interlayer Spacing of base material (nm)"] = 0.335

df.to_excel("DataSheetSOP.xlsx", index=False)
