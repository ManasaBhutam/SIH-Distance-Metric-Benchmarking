
import pandas as pd
from pathlib import Path

input_path = Path("data/raw/epcg23.csv")
output_path = Path("data/processed/epcg23_cleaned.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_path, low_memory=False)

print("Original dataset:", df.shape)

# Preserve the original special codes in the raw CSV.
# Convert confirmed logical skips / special values to missing
# only in the analysis copy.
clean = df.copy()

# Salary: confirmed logical skip
clean.loc[clean["SALARY"].eq(9999998), "SALARY"] = pd.NA

# Industry: special code aligned with respondents not working.
# Exclude from numerical distance as a real industry value.
clean.loc[clean["INDCODE"].eq(9998), "INDCODE"] = pd.NA

# Hours worked: special code aligned with respondents not working.
clean.loc[clean["HRSWK"].eq(98), "HRSWK"] = pd.NA

# Education codes: retain original codes until their exact labels
# are confirmed in the official codebook.

clean.to_csv(output_path, index=False)

print("Cleaned analysis copy saved to:", output_path)
print("Cleaned dataset:", clean.shape)
print("Original record count preserved:", len(df) == len(clean))
print("\nRemaining special-code counts:")
for col, code in [
    ("SALARY", 9999998),
    ("INDCODE", 9998),
    ("HRSWK", 98),
    ("BAACYR", 9998),
    ("N2BAMED", 999989),
]:
    if col in clean.columns:
        print(f"{col} = {code}: {clean[col].eq(code).sum()}")

print("\nOriginal CSV was not modified.")
