
import pandas as pd
from pathlib import Path

data_path = Path("data/raw/epcg23.csv")
df = pd.read_csv(data_path, low_memory=False)

# Variables selected for initial investigation
variables = [
    "AGE", "N2BAMED", "BAACYR",
    "INDCODE", "SALARY", "HRSWK", "ACTRD"
]

print("SPECIAL-CODE INVESTIGATION")
print("=" * 60)

for col in variables:
    print(f"\n{col}")
    print("-" * 30)

    print("Data type:", df[col].dtype)
    print("Actual missing values:", df[col].isna().sum())
    print("Most frequent values:")
    print(df[col].value_counts(dropna=False).head(15))

print("\nInvestigation complete.")
