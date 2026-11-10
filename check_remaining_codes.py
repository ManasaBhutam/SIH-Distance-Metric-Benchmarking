
import pandas as pd

path = "data/raw/epcg23.csv"

columns = [
    "LFSTAT",
    "WRKG",
    "INDCODE",
    "HRSWK",
    "BAACYR",
    "N2BAMED"
]

df = pd.read_csv(path, usecols=columns, low_memory=False)

checks = {
    "INDCODE = 9998": ("INDCODE", 9998),
    "HRSWK = 98": ("HRSWK", 98),
    "BAACYR = 9998": ("BAACYR", 9998),
    "N2BAMED = 999989": ("N2BAMED", 999989),
}

for label, (column, code) in checks.items():
    mask = df[column].eq(code)

    print(f"\n{'=' * 55}")
    print(f"{label}: {int(mask.sum())} records")

    print("\nBy labor force status:")
    print(pd.crosstab(df.loc[mask, "LFSTAT"], df.loc[mask, "WRKG"]))

print("\nNo original data has been modified.")
