
import pandas as pd
from pathlib import Path

path = Path("data/raw/epcg23.csv")

columns = [
    "AGE", "N2BAMED", "BAACYR",
    "INDCODE", "SALARY", "HRSWK", "ACTRD"
]

df = pd.read_csv(path, usecols=columns, low_memory=False)

# Investigate rows with the repeated special-looking values
checks = {
    "BAACYR_9998": df["BAACYR"].eq(9998),
    "INDCODE_9998": df["INDCODE"].eq(9998),
    "SALARY_9999998": df["SALARY"].eq(9999998),
    "HRSWK_98": df["HRSWK"].eq(98),
    "ACTRD_L": df["ACTRD"].eq("L"),
}

print("SPECIAL-CODE COUNTS")
for name, mask in checks.items():
    print(f"{name}: {mask.sum()}")

print("\nPAIRWISE OVERLAP COUNTS")
names = list(checks)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        overlap = checks[names[i]] & checks[names[j]]
        print(f"{names[i]} + {names[j]}: {overlap.sum()}")

print("\nAGE VALUES WHEN SALARY IS LOGICALLY SKIPPED")
print(
    df.loc[checks["SALARY_9999998"], "AGE"]
    .value_counts()
    .head(10)
)

print("\nReport complete. Original CSV was not modified.")
