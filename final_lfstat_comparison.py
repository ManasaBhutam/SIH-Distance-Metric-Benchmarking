
import os
import gc
import warnings

import numpy as np
import pandas as pd
import gower

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

DATA_FILE = os.path.join(DATA_DIR, "distance_sample.csv")

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "final_lfstat_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. FEATURES
# ============================================================

ALL_FEATURES = [
    "AGE",
    "SALARY",
    "HRSWK",
    "LFSTAT",
    "ACTRD",
    "BAIND",
    "INDCODE",
    "N2BAMED"
]

NUMERIC_FEATURES = [
    "AGE",
    "SALARY",
    "HRSWK"
]

CATEGORICAL_FEATURES = [
    "LFSTAT",
    "ACTRD",
    "BAIND",
    "INDCODE",
    "N2BAMED"
]

# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 65)
print("FINAL FEATURE INFLUENCE AND MODEL COMPARISON")
print("=" * 65)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

missing = [col for col in ALL_FEATURES if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

data = df[ALL_FEATURES].copy()

for col in NUMERIC_FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

for col in CATEGORICAL_FEATURES:
    data[col] = data[col].astype(object)
    data[col] = data[col].where(data[col].notna(), np.nan)

print(f"Dataset rows: {len(data)}")
print(f"Features: {len(ALL_FEATURES)}")

# ============================================================
# 4. GOWER CLUSTERING FUNCTION
# ============================================================

def run_gower_model(feature_data, model_name):

    print(f"\nCalculating distances: {model_name}")

    distance_matrix = gower.gower_matrix(feature_data)

    distance_matrix = np.asarray(
        distance_matrix,
        dtype=np.float64
    )

    distance_matrix = (
        distance_matrix + distance_matrix.T
    ) / 2

    np.fill_diagonal(distance_matrix, 0)

    condensed = squareform(
        distance_matrix,
        checks=False
    )

    print(f"Running hierarchical clustering: {model_name}")

    tree = linkage(
        condensed,
        method="average"
    )

    labels = fcluster(
        tree,
        t=2,
        criterion="maxclust"
    )

    silhouette = silhouette_score(
        distance_matrix,
        labels,
        metric="precomputed"
    )

    counts = pd.Series(labels).value_counts()

    balance = counts.min() / counts.max()

    print(f"Silhouette: {silhouette:.4f}")
    print(f"Cluster balance: {balance:.4f}")

    print("Cluster sizes:")
    print(counts.sort_index().to_string())

    del condensed, tree
    gc.collect()

    return labels, silhouette, balance, distance_matrix


# ============================================================
# 5. MODEL A - ALL EIGHT FEATURES
# ============================================================

print("\n" + "=" * 65)
print("MODEL A: ALL EIGHT FEATURES")
print("=" * 65)

labels_a, silhouette_a, balance_a, distances_a = (
    run_gower_model(
        data,
        "Model A - All Features"
    )
)

# ============================================================
# 6. MODEL B - WITHOUT LFSTAT
# ============================================================

print("\n" + "=" * 65)
print("MODEL B: WITHOUT LFSTAT")
print("=" * 65)

features_b = [
    col for col in ALL_FEATURES
    if col != "LFSTAT"
]

data_b = data[features_b].copy()

labels_b, silhouette_b, balance_b, distances_b = (
    run_gower_model(
        data_b,
        "Model B - Without LFSTAT"
    )
)

# ============================================================
# 7. COMPARE BOTH MODELS
# ============================================================

ari = adjusted_rand_score(
    labels_a,
    labels_b
)

comparison = pd.DataFrame([
    {
        "Model": "Model A - All Features",
        "Feature Count": len(ALL_FEATURES),
        "Silhouette Score": silhouette_a,
        "Cluster Balance": balance_a,
        "ARI vs Model A": 1.0
    },
    {
        "Model": "Model B - Without LFSTAT",
        "Feature Count": len(features_b),
        "Silhouette Score": silhouette_b,
        "Cluster Balance": balance_b,
        "ARI vs Model A": ari
    }
])

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))
print(f"\nARI between models: {ari:.4f}")

# ============================================================
# 8. SAVE CLUSTER LABELS
# ============================================================

labels_df = pd.DataFrame({
    "Model_A_All_Features": labels_a,
    "Model_B_Without_LFSTAT": labels_b
})

labels_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_cluster_labels.csv"),
    index=False
)

# ============================================================
# 9. CLUSTER SIZE ANALYSIS
# ============================================================

size_results = []

for model_name, labels in [
    ("Model A - All Features", labels_a),
    ("Model B - Without LFSTAT", labels_b)
]:

    counts = pd.Series(labels).value_counts().sort_index()

    for cluster, count in counts.items():

        size_results.append({
            "Model": model_name,
            "Cluster": int(cluster),
            "Count": int(count),
            "Percentage": round(
                count / len(labels) * 100,
                2
            )
        })

size_df = pd.DataFrame(size_results)

size_df.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_sizes.csv"),
    index=False
)

# ============================================================
# 10. NUMERIC CLUSTER PROFILES
# ============================================================

numeric_results = []

for model_name, labels in [
    ("Model A - All Features", labels_a),
    ("Model B - Without LFSTAT", labels_b)
]:

    for cluster in sorted(np.unique(labels)):

        cluster_data = data.loc[labels == cluster]

        for feature in NUMERIC_FEATURES:

            values = cluster_data[feature]

            numeric_results.append({
                "Model": model_name,
                "Cluster": int(cluster),
                "Feature": feature,
                "Valid Count": int(values.notna().sum()),
                "Missing Percentage": round(
                    values.isna().mean() * 100,
                    2
                ),
                "Mean": values.mean(),
                "Median": values.median(),
                "Standard Deviation": values.std()
            })

numeric_df = pd.DataFrame(numeric_results)

numeric_df.to_csv(
    os.path.join(OUTPUT_DIR, "numeric_cluster_profiles.csv"),
    index=False
)

# ============================================================
# 11. CATEGORICAL CLUSTER PROFILES
# ============================================================

categorical_results = []

for model_name, labels in [
    ("Model A - All Features", labels_a),
    ("Model B - Without LFSTAT", labels_b)
]:

    for cluster in sorted(np.unique(labels)):

        cluster_data = data.loc[labels == cluster]

        for feature in CATEGORICAL_FEATURES:

            percentages = (
                cluster_data[feature]
                .fillna("Missing")
                .astype(str)
                .value_counts(normalize=True) * 100
            )

            for category, percentage in percentages.items():

                categorical_results.append({
                    "Model": model_name,
                    "Cluster": int(cluster),
                    "Feature": feature,
                    "Category": category,
                    "Percentage": round(percentage, 2)
                })

categorical_df = pd.DataFrame(categorical_results)

categorical_df.to_csv(
    os.path.join(OUTPUT_DIR, "categorical_cluster_profiles.csv"),
    index=False
)

# ============================================================
# 12. FEATURE MISSINGNESS ANALYSIS
# ============================================================

missingness_results = []

for feature in ALL_FEATURES:

    missingness_results.append({
        "Feature": feature,
        "Missing Count": int(data[feature].isna().sum()),
        "Missing Percentage": round(
            data[feature].isna().mean() * 100,
            2
        )
    })

missingness_df = pd.DataFrame(missingness_results)

missingness_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_missingness.csv"),
    index=False
)

# ============================================================
# 13. LFSTAT ASSOCIATION WITH MODEL A
# ============================================================

lfstat_results = pd.crosstab(
    pd.Series(labels_a, name="Cluster"),
    data["LFSTAT"].fillna("Missing").astype(str),
    normalize="index"
) * 100

lfstat_results.to_csv(
    os.path.join(OUTPUT_DIR, "lfstat_model_a_distribution.csv")
)

# ============================================================
# 14. SAVE FINAL REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison_report.txt"
)

with open(report_file, "w", encoding="utf-8") as report:

    report.write("FINAL FEATURE INFLUENCE ANALYSIS\n")
    report.write("=" * 60 + "\n\n")

    report.write(f"Number of records: {len(data)}\n\n")

    report.write("MODEL A: ALL EIGHT FEATURES\n")
    report.write(f"Silhouette Score: {silhouette_a:.4f}\n")
    report.write(f"Cluster Balance: {balance_a:.4f}\n\n")

    report.write("MODEL B: WITHOUT LFSTAT\n")
    report.write(f"Silhouette Score: {silhouette_b:.4f}\n")
    report.write(f"Cluster Balance: {balance_b:.4f}\n\n")

    report.write(f"ARI BETWEEN MODELS: {ari:.4f}\n\n")

    report.write("CLUSTER SIZES\n")
    report.write(size_df.to_string(index=False))
    report.write("\n\n")

    report.write("FEATURE MISSINGNESS\n")
    report.write(missingness_df.to_string(index=False))
    report.write("\n\n")

    report.write("LFSTAT DISTRIBUTION IN MODEL A\n")
    report.write(lfstat_results.to_string())
    report.write("\n\n")

    report.write("INTERPRETATION GUIDANCE\n")
    report.write(
        "Silhouette measures cluster separation using each model's "
        "own distance matrix. Cluster balance is the smallest cluster "
        "size divided by the largest cluster size. ARI measures "
        "agreement between the two cluster assignments. A low ARI "
        "indicates substantial changes in assignments. These measures "
        "should be interpreted together with cluster profiles, "
        "missing-data patterns, and the research objective.\n\n"
    )

    report.write(
        "Model B still contains employment-related variables such as "
        "SALARY, HRSWK, ACTRD and INDCODE. Therefore, removing LFSTAT "
        "does not completely remove employment-related information.\n"
    )

# ============================================================
# 15. CLEANUP AND FINISH
# ============================================================

del distances_a, distances_b
gc.collect()

print("\n" + "=" * 65)
print("FINAL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nResults folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nNext step: inspect model_comparison.csv and cluster_sizes.csv.")
