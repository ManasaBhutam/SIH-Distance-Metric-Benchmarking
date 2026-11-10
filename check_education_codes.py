


import pandas as pd
import os

# 1. Load original NSCG dataset
file_path = "data/raw/epcg23.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found: {file_path}\n"
        "Check your CSV file location."
    )

print("Loading NSCG dataset...")
df = pd.read_csv(file_path, low_memory=False)

print("Dataset loaded:", df.shape)

# 2. Check required columns
required_columns = ["BAIND", "N2BAMED"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in dataset")

# 3. Correct BAIND
df["BAIND"] = df["BAIND"].astype("string").str.strip()

# 4. Correct N2BAMED
df["N2BAMED"] = df["N2BAMED"].astype("string").str.strip()

df["N2BAMED"] = df["N2BAMED"].replace({
    "999989": "NOT_APPLICABLE",
    "999999": "UNKNOWN_MISSING"
})

df["N2BAMED"] = df["N2BAMED"].fillna("UNKNOWN_MISSING")

# 5. Print distributions
print("\n========== BAIND DISTRIBUTION ==========")
print(df["BAIND"].value_counts(dropna=False))

print("\n========== N2BAMED DISTRIBUTION ==========")
print(df["N2BAMED"].value_counts(dropna=False).head(20))

# 6. Save corrected education data
os.makedirs("data/processed", exist_ok=True)

df[["BAIND", "N2BAMED"]].to_csv(
    "data/processed/education_codes_corrected.csv",
    index=False
)

print("\nSUCCESS: Corrected education codes saved!")
