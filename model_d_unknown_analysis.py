
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

# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

DATA_FILE = os.path.join(DATA_DIR, "distance_sample.csv")

LABEL_FILE = os.path.join(
    DATA_DIR,
    "employment_feature_experiment",
    "three_model_cluster_labels.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "model_d_unknown_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 2. LOAD DATA
# =====================================================

print("=" * 65)
print("MODEL D: UNKNOWN CATEGORY SENSITIVITY ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_FILE)
saved_labels = pd.read_csv(LABEL_FILE)

if len(df) != len(saved_labels):
    raise ValueError("Dataset and saved labels have different row counts.")

required = ["AGE", "BAIND", "N2BAMED"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing feature: {col}")

model_c_column = "Model C - Demographic and Education"

if model_c_column not in saved_labels.columns:
    raise ValueError("Original Model C labels were not found.")

# =====================================================
# 3. PREPARE FEATURES
# =====================================================

data = df[required].copy()

data["AGE"] = pd.to_numeric(
    data["AGE"],
    errors="coerce"
)

for col in ["BAIND", "N2BAMED"]:
    data[col] = data[col].astype(object)
    data[col] = data[col].where(data[col].notna(), np.nan)

# Identify literal Unknown values, ignoring case and spaces.
unknown_mask = (
    data["N2BAMED"]
    .astype("string")
    .str.strip()
    .str.casefold()
    .eq("unknown")
    .fillna(False)
)

# Exclude Unknown and actual missing N2BAMED values.
eligible_mask = (
    data["N2BAMED"].notna()
    & ~unknown_mask
    & data["AGE"].notna()
    & data["BAIND"].notna()
)

eligible_data = data.loc[eligible_mask].copy()

original_c_labels = saved_labels.loc[
    eligible_mask,
    model_c_column
].to_numpy()

print("\nDATASET COUNTS")
print(f"Total records: {len(data)}")
print(f"Unknown N2BAMED: {int(unknown_mask.sum())}")
print(
    "Actual missing N2BAMED:",
    int(data["N2BAMED"].isna().sum())
)
print(f"Eligible records: {len(eligible_data)}")
print(f"Excluded records: {len(data) - len(eligible_data)}")

if len(eligible_data) < 3:
    raise ValueError("Not enough eligible records for clustering.")

if eligible_data["N2BAMED"].nunique() < 2:
    raise ValueError(
        "N2BAMED has fewer than two categories after filtering."
    )

# =====================================================
# 4. RUN MODEL D
# =====================================================

print("\nRunning Gower distance calculation...")

distance_matrix = gower.gower_matrix(eligible_data)

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

print("Running average-linkage hierarchical clustering...")

tree = linkage(
    condensed,
    method="average"
)

model_d_labels = fcluster(
    tree,
    t=2,
    criterion="maxclust"
)

silhouette = silhouette_score(
    distance_matrix,
    model_d_labels,
    metric="precomputed"
)

counts = pd.Series(model_d_labels).value_counts()

balance = counts.min() / counts.max()

print("\nMODEL D RESULTS")
print(f"Eligible records: {len(eligible_data)}")
print(f"Silhouette score: {silhouette:.4f}")
print(f"Cluster balance: {balance:.4f}")
print("\nCluster sizes:")
print(counts.sort_index())

# =====================================================
# 5. COMPARE WITH ORIGINAL MODEL C
# =====================================================

ari = adjusted_rand_score(
    original_c_labels,
    model_d_labels
)

print("\nMODEL C VS MODEL D")
print(f"ARI on eligible respondents: {ari:.4f}")

comparison = pd.DataFrame([
    {
        "Model": "Model C - Original full sample",
        "Records": len(data),
        "Silhouette": np.nan,
        "Cluster Balance": np.nan
    },
    {
        "Model": "Model D - Known N2BAMED only",
        "Records": len(eligible_data),
        "Silhouette": silhouette,
        "Cluster Balance": balance
    }
])

comparison["ARI vs Model C on eligible records"] = [
    np.nan,
    ari
]

comparison.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_comparison.csv"),
    index=False
)

# =====================================================
# 6. SAVE CLUSTER LABELS
# =====================================================

labels_output = pd.DataFrame({
    "Original Model C Label": original_c_labels,
    "Model D Label": model_d_labels
})

labels_output.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_cluster_labels.csv"),
    index=False
)

# =====================================================
# 7. CLUSTER SIZE PROFILES
# =====================================================

size_rows = []

for cluster, count in counts.sort_index().items():

    size_rows.append({
        "Cluster": int(cluster),
        "Count": int(count),
        "Percentage": round(
            count / len(eligible_data) * 100,
            2
        )
    })

size_df = pd.DataFrame(size_rows)

size_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_cluster_sizes.csv"),
    index=False
)

# =====================================================
# 8. AGE PROFILES
# =====================================================

age_rows = []

for cluster in sorted(np.unique(model_d_labels)):

    values = eligible_data.loc[
        model_d_labels == cluster,
        "AGE"
    ]

    age_rows.append({
        "Cluster": int(cluster),
        "Count": len(values),
        "Mean Age": values.mean(),
        "Median Age": values.median(),
        "Minimum Age": values.min(),
        "Maximum Age": values.max(),
        "Age Standard Deviation": values.std()
    })

age_df = pd.DataFrame(age_rows)

age_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_age_profiles.csv"),
    index=False
)

# =====================================================
# 9. BAIND PROFILES
# =====================================================

baind_counts = pd.crosstab(
    pd.Series(model_d_labels, name="Cluster"),
    eligible_data["BAIND"].astype(str)
)

baind_percentages = pd.crosstab(
    pd.Series(model_d_labels, name="Cluster"),
    eligible_data["BAIND"].astype(str),
    normalize="index"
) * 100

baind_counts.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_baind_counts.csv")
)

baind_percentages.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_baind_percentages.csv")
)

# =====================================================
# 10. N2BAMED PROFILES
# =====================================================

n2bamed_counts = pd.crosstab(
    pd.Series(model_d_labels, name="Cluster"),
    eligible_data["N2BAMED"].astype(str)
)

n2bamed_percentages = pd.crosstab(
    pd.Series(model_d_labels, name="Cluster"),
    eligible_data["N2BAMED"].astype(str),
    normalize="index"
) * 100

n2bamed_counts.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_n2bamed_counts.csv")
)

n2bamed_percentages.to_csv(
    os.path.join(OUTPUT_DIR, "model_d_n2bamed_percentages.csv")
)

# =====================================================
# 11. SAVE FINAL REPORT
# =====================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "model_d_final_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("MODEL D: UNKNOWN CATEGORY SENSITIVITY ANALYSIS\n")
    report.write("=" * 65 + "\n\n")

    report.write(f"Original records: {len(data)}\n")
    report.write(f"Eligible records: {len(eligible_data)}\n")
    report.write(f"Excluded records: {len(data) - len(eligible_data)}\n\n")

    report.write("MODEL D RESULTS\n")
    report.write(f"Silhouette score: {silhouette:.4f}\n")
    report.write(f"Cluster balance: {balance:.4f}\n")
    report.write(f"ARI vs Model C on eligible records: {ari:.4f}\n\n")

    report.write("CLUSTER SIZES\n")
    report.write(size_df.to_string(index=False))

    report.write("\n\nAGE PROFILES\n")
    report.write(age_df.to_string(index=False))

    report.write("\n\nBAIND DISTRIBUTION (%)\n")
    report.write(baind_percentages.to_string())

    report.write("\n\nN2BAMED DISTRIBUTION (%)\n")
    report.write(n2bamed_percentages.to_string())

    report.write(
        "\n\nINTERPRETATION NOTES\n"
        "Model D excludes records with Unknown or missing N2BAMED, "
        "and records with missing AGE or BAIND.\n"
        "ARI compares Model D labels with the original Model C labels "
        "for the same eligible respondents.\n"
        "Model C and Model D use different populations, so their "
        "silhouette scores should not be interpreted as a direct "
        "like-for-like comparison.\n"
        "Category meanings must be verified using the official NSCG "
        "codebook. Excluding Unknown records can introduce selection "
        "bias, so the filtered results describe only eligible records.\n"
    )

# =====================================================
# 12. FINISH
# =====================================================

del distance_matrix, condensed, tree
gc.collect()

print("\n" + "=" * 65)
print("MODEL D ANALYSIS COMPLETED")
print("=" * 65)

print(f"\nResults saved in:\n{OUTPUT_DIR}")

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)
