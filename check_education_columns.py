
import pandas as pd

path = "data/raw/epcg23.csv"

df = pd.read_csv(path, nrows=5, low_memory=False)

keywords = ["EDUC", "DEG", "BA", "MA", "PHD", "N2BAMED", "BAACYR"]

print("Possible education-related columns:\n")

for column in df.columns:
    if any(word in column.upper() for word in keywords):
        print(column)

print("\nTotal columns:", len(df.columns))
