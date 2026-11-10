
# ============================================================
# SIH PROJECT - ALTERNATIVE GOWER CLUSTERING
# AGE + EDUCATION FEATURES
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "alternative_clustering"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_DIR / "distance_sample.csv")
old_labels = pd.read_csv(DATA_DIR / "clustering_labels.csv")

if len(df) != len(old_labels):
    raise ValueError("Sample and label row counts do not match.")

if "Gower_k2" not in old_labels.columns:
    raise ValueError("Original Gower_k2 labels not found.")

# Only use these three features
features = ["AGE", "BAIND", "N2BAMED"]

for col in features:
    if col not in df.columns:
        raise ValueError(f"Missing feature: {col}")

age = pd.to_numeric(df["AGE"], errors="coerce").to_numpy()

if np.isnan(age).any():
    raise ValueError("AGE contains missing values. Handle them first.")

baind = df["BAIND"].astype("string").str.strip().to_numpy()
field = df["N2BAMED"].astype("string").str.strip().to_numpy()

# Treat unknown and blank categories as missing
def valid_categories(values):
    values = np.asarray(values, dtype=object)
    return np.array([
        pd.notna(x) and str(x).lower() not in
        ["unknown", "nan", "none", ""]
        for x in values
    ])

valid_baind = valid_categories(baind)
valid_field = valid_categories(field)

n = len(df)

print("Records:", n)
print("Features:", features)

# ------------------------------------------------------------
# 3. CALCULATE GOWER DISTANCE
# ------------------------------------------------------------

print("\nCalculating alternative Gower distances...")

distance_sum = np.zeros((n, n), dtype=np.float64)
feature_count = np.zeros((n, n), dtype=np.float64)

# Numerical feature: AGE
age_range = age.max() - age.min()

if age_range == 0:
    raise ValueError("AGE has no variation.")

age_distance = np.abs(
    age[:, None] - age[None, :]
) / age_range

distance_sum += age_distance
feature_count += 1

# Categorical feature: BAIND
baind_distance = (
    baind[:, None] != baind[None, :]
).astype(float)

baind_valid = (
    valid_baind[:, None] & valid_baind[None, :]
)

distance_sum += baind_distance * baind_valid
feature_count += baind_valid

# Categorical feature: N2BAMED
field_distance = (
    field[:, None] != field[None, :]
).astype(float)

field_valid = (
    valid_field[:, None] & valid_field[None, :]
)

distance_sum += field_distance * field_valid
feature_count += field_valid

# Average over available features for each pair
distance_matrix = np.divide(
    distance_sum,
    feature_count,
    out=np.zeros_like(distance_sum),
    where=feature_count > 0
)

np.fill_diagonal(distance_matrix, 0)

print("Distance matrix:", distance_matrix.shape)

# ------------------------------------------------------------
# 4. PERFORM HIERARCHICAL CLUSTERING
# ------------------------------------------------------------

print("\nRunning hierarchical clustering...")

results = []
label_results = pd.DataFrame(index=df.index)

label_results["Original_Gower_k2"] = (
    old_labels["Gower_k2"].values
)

profile_rows = []

for k in [2, 3, 4]:

    model = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average"
    )

    new_labels = model.fit_predict(distance_matrix)

    silhouette = silhouette_score(
        distance_matrix,
        new_labels,
        metric="precomputed"
    )

    counts = np.bincount(new_labels)

    balance = counts.min() / counts.max()

    ari = adjusted_rand_score(
        old_labels["Gower_k2"].values,
        new_labels
    )

    results.append({
        "Features": "AGE, BAIND, N2BAMED",
        "Clusters": k,
        "Silhouette": silhouette,
        "Smallest Cluster": counts.min(),
        "Largest Cluster": counts.max(),
        "Balance Ratio": balance,
        "ARI vs Original Gower k2": ari
    })

    label_results[f"Alternative_Gower_k{k}"] = new_labels

    for cluster in range(k):

        group = df.loc[new_labels == cluster]

        profile_rows.append({
            "Clusters": k,
            "Cluster": cluster,
            "Respondents": len(group),
            "Percentage": len(group) / n * 100,
            "Median Age": pd.to_numeric(
                group["AGE"], errors="coerce"
            ).median(),
            "BAIND Distribution": (
                group["BAIND"]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(5)
                .to_dict()
            ),
            "N2BAMED Distribution": (
                group["N2BAMED"]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(5)
                .to_dict()
            )
        })

# ------------------------------------------------------------
# 5. SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "alternative_clustering_summary.csv",
    index=False
)

label_results.to_csv(
    OUTPUT_DIR / "alternative_clustering_labels.csv",
    index=False
)

profiles_df = pd.DataFrame(profile_rows)

profiles_df.to_csv(
    OUTPUT_DIR / "alternative_cluster_profiles.csv",
    index=False
)

# ------------------------------------------------------------
# 6. DISPLAY RESULTS
# ------------------------------------------------------------

print("\nALTERNATIVE CLUSTERING RESULTS")

print(
    results_df.round(4).to_string(index=False)
)

print("\nCLUSTER PROFILES")

print(
    profiles_df.to_string(index=False)
)

print("\nEXPERIMENT COMPLETED")

print("Output folder:", OUTPUT_DIR)

print("Generated files:")
print("- alternative_clustering_summary.csv")
print("- alternative_clustering_labels.csv")
print("- alternative_cluster_profiles.csv")
