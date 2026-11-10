
import pandas as pd

path = "data/raw/epcg23.csv"

columns = [
    "LFSTAT", "WRKG", "LOOKWK", "NWLAY",
    "INDCODE", "SALARY", "HRSWK", "ACTRD"
]

df = pd.read_csv(path, usecols=columns, low_memory=False)

print("LABOR FORCE STATUS")
print(df["LFSTAT"].value_counts(dropna=False).sort_index())

print("\nWORKING INDICATOR")
print(df["WRKG"].value_counts(dropna=False))

print("\nLABOR FORCE STATUS BY SALARY SPECIAL CODE")
print(
    pd.crosstab(
        df["LFSTAT"],
        df["SALARY"].eq(9999998),
        margins=True
    )
)

print("\nLABOR FORCE STATUS BY ACTRD")
print(pd.crosstab(df["LFSTAT"], df["ACTRD"], margins=True))

print("\nNo original data has been modified.")
