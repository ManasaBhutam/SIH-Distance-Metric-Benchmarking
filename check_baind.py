
import pandas as pd

path = "data/raw/epcg23.csv"

columns = [
    "LFSTAT", "WRKG", "BAIND", "BADG", "BAACYR",
    "BAYR", "N2BAMED", "N2BAMEDX"
]

df = pd.read_csv(path, usecols=columns, low_memory=False)

print("BAIND values across the full dataset:")
print(df["BAIND"].value_counts(dropna=False))

print("\nBAIND by labor force status:")
print(pd.crosstab(df["LFSTAT"], df["BAIND"], margins=True))

print("\nBAIND by BADG:")
print(pd.crosstab(df["BAIND"], df["BADG"], margins=True))

print("\nBAIND by BAACYR special code:")
print(pd.crosstab(df["BAIND"], df["BAACYR"].eq(9998), margins=True))

print("\nN2BAMED special-code counts:")
print(df["N2BAMED"].value_counts(dropna=False).head(15))

print("\nOriginal CSV was not modified.")
