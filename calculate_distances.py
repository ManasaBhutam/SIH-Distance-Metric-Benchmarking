
import numpy as np
import pandas as pd
import gower

from pathlib import Path
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
df = pd.read_csv(
    "data/processed/epcg23_cleaned.csv",
    low_memory=False
)

numeric = ["AGE", "SALARY", "HRSWK"]
categorical = ["LFSTAT", "ACTRD", "BAIND", "INDCODE", "N2BAMED"]
columns = numeric + categorical

# Use the same reproducible sample for all three metrics
sample = df[columns].sample(
    n=min(3000, len(df)),
    random_state=42
).reset_index(drop=True)

# Prepare numeric columns
for col in numeric:
    sample[col] = pd.to_numeric(sample[col], errors="coerce")
    sample[col] = sample[col].fillna(sample[col].median())

# Preserve categorical values; don't treat their codes as numbers
for col in categorical:
    sample[col] = sample[col].astype("string").fillna("Unknown").astype(str)

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

# Gower: mixed numeric and categorical data
gower_input = sample.copy()

# Mark categorical columns explicitly
cat_features = [False] * len(numeric) + [True] * len(categorical)

print("Calculating Gower distance...")
gower_matrix = gower.gower_matrix(
    gower_input,
    cat_features=np.array(cat_features)
)

# Convert square matrix to condensed pairwise distances
gower_dist = gower_matrix[
    np.triu_indices_from(gower_matrix, k=1)
]
# Standardize numeric features for Euclidean and Cosine
scaled_numeric = StandardScaler().fit_transform(sample[numeric])

# One-hot encode categorical features
encoded_cat = pd.get_dummies(
    sample[categorical],
    dtype=float
).to_numpy()

# Combine numeric and categorical features
X = np.hstack([scaled_numeric, encoded_cat])

print("Calculating Euclidean distance...")
euclidean_dist = pdist(X, metric="euclidean")

print("Calculating Cosine distance...")
cosine_dist = pdist(X, metric="cosine")

# Save pairwise distances efficiently
np.savez_compressed(
    out / "distance_results.npz",
    gower=gower_dist,
    euclidean=euclidean_dist,
    cosine=cosine_dist
)

# Save sample identifiers so results can be traced
sample[columns].to_csv(out / "distance_sample.csv", index=False)

print("\nDistance calculations completed!")
print("Sample size:", len(sample))
print("Number of unique pairs:", len(euclidean_dist))
print("Saved: data/processed/distance_results.npz")
print("Saved: data/processed/distance_sample.csv")


def pdist_from_square(matrix):
    return matrix[np.triu_indices_from(matrix, k=1)]
