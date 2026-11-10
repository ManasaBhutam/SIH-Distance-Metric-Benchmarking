
import pandas as pd
import numpy as np
import gower

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load existing project data
sample = pd.read_csv("data/processed/distance_sample.csv")

# Features used in original Gower calculation
numeric_cols = ["AGE", "SALARY", "HRSWK"]

categorical_cols = [
    "LFSTAT", "ACTRD", "BAIND",
    "INDCODE", "N2BAMED"
]

missing_cols = [
    "SALARY_MISSING",
    "HRSWK_MISSING",
    "INDCODE_MISSING"
]

# Original Gower model
def run_gower(data, categorical):
    numeric = data[numeric_cols].copy()
    cat = data[categorical].copy()

    combined = pd.concat(
        [numeric, cat], axis=1
    )

    cat_features = np.array(
        [False] * len(numeric_cols)
        + [True] * len(categorical)
    )

    D = gower.gower_matrix(
        combined,
        cat_features=cat_features
    )

    model = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average"
    )

    labels = model.fit_predict(D)

    score = silhouette_score(
        D,
        labels,
        metric="precomputed"
    )

    sizes = pd.Series(labels).value_counts()

    return score, sizes.min(), sizes.max()

# Test 1: Original features including missingness indicators
original_cols = categorical_cols + missing_cols

original = run_gower(sample, original_cols)

# Test 2: Remove missingness indicators
without_missing = run_gower(
    sample,
    categorical_cols
)

# Display results
results = pd.DataFrame([
    {
        "Model": "Original with missingness indicators",
        "Silhouette": original[0],
        "Smallest Cluster": original[1],
        "Largest Cluster": original[2]
    },
    {
        "Model": "Without missingness indicators",
        "Silhouette": without_missing[0],
        "Smallest Cluster": without_missing[1],
        "Largest Cluster": without_missing[2]
    }
])

print("\nMISSINGNESS INFLUENCE ANALYSIS")
print(results.to_string(index=False))

results.to_csv(
    "data/processed/missingness_influence_results.csv",
    index=False
)

print("\nResults saved successfully!")
