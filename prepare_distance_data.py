
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv("data/processed/epcg23_cleaned.csv", low_memory=False)

# Select useful features for the initial comparison
numeric_cols = ["AGE", "SALARY", "HRSWK"]
categorical_cols = ["LFSTAT", "ACTRD", "BAIND", "INDCODE", "N2BAMED"]

features = df[numeric_cols + categorical_cols].copy()

# Convert special codes in education variables to missing.
# These codes are associated with unavailable/not-applicable
# education information; confirm exact labels against the codebook.
features["BAIND"] = features["BAIND"].replace(["M", "N"], pd.NA)
features["N2BAMED"] = features["N2BAMED"].replace(999989, pd.NA)

# Scale numeric features for Euclidean and Cosine calculations
features[numeric_cols] = features[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)
features[numeric_cols] = features[numeric_cols].fillna(
    features[numeric_cols].median()
)

scaler = StandardScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

# Save the prepared feature table
out = Path("data/processed/distance_features.csv")
features.to_csv(out, index=False)

print("Feature dataset saved:", out)
print("Rows:", features.shape[0])
print("Features:", features.shape[1])
print("\nColumns:", list(features.columns))
print("\nMissing values:")
print(features.isna().sum())
print("\nReady for the next step: distance calculation.")
