
import pandas as pd
from pathlib import Path

# Locate the original dataset
csv_path = Path("data/raw/epcg23.csv")

if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")

print("Loading NSCG 2023 dataset...")

# Load the data
df = pd.read_csv(csv_path, low_memory=False)

print("\n--- DATASET SIZE ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes.value_counts())

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isna().sum().sort_values(ascending=False).head(20))

# Inspect candidate project variables
candidates = [
    "AGE", "N2BAMED", "BAACYR",
    "INDCODE", "SALARY", "HRSWK"
]

print("\n--- CANDIDATE VARIABLES ---")

for col in candidates:
    if col in df.columns:
        print(f"\n{col}")
        print("Data type:", df[col].dtype)
        print("Distinct values:", df[col].nunique())
        print("Most common values:")
        print(df[col].value_counts(dropna=False).head(10))
    else:
        print(f"\n{col}: NOT FOUND")

print("\nInspection complete.")
