
import pandas as pd
from pathlib import Path

data_path = Path("data/raw/epcg23.csv")
dictionary_path = Path("data/raw/Dpcg23.xlsx")

df = pd.read_csv(data_path, low_memory=False)
dictionary = pd.read_excel(dictionary_path)

variables = [
    "AGE", "N2BAMED", "BAACYR",
    "INDCODE", "SALARY", "HRSWK",
    "ACTRD", "WTSURVY"
]

print("VARIABLE DEFINITIONS")
print("=" * 70)

for variable in variables:
    match = dictionary[
        dictionary["SAS_NAME"].astype(str).str.strip()
        == variable
    ]

    if match.empty:
        print(f"\n{variable}: Definition not found")
        continue

    row = match.iloc[0]
    print(f"\nVariable: {variable}")
    print("Description:", row["DESCRIPTION"])
    print("Format:", row["SAS_FMT"])

    print("Most common values in CSV:")
    print(
        df[variable]
        .value_counts(dropna=False)
        .head(12)
        .to_string()
    )

print("\nVariable verification complete.")
