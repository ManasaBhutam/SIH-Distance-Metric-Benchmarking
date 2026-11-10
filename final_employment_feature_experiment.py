
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
    "employment_feature_experiment"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. DEFINE FEATURES AND MODELS
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

MODELS = {
    "Model A - All 8 Features": ALL_FEATURES,

    "Model B - Without LFSTAT": [
        "AGE",
        "SALARY",
        "HRSWK",
        "ACTRD",
        "BAIND",
        "INDCODE",
        "N2BAMED"
    ],

    "Model C - Demographic and Education": [
        "AGE",
        "BAIND",
        "N2BAMED"
    ]
}

# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("EMPLOYMENT FEATURE INFLUENCE EXPERIMENT")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

missing_columns = [
    col for col in ALL_FEATURES
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

data = df[ALL_FEATURES].copy()

for col in NUMERIC_FEATURES:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

for col in CATEGORICAL_FEATURES:
    data[col] = data[col].astype(object)
    data[col] = data[col].where(
        data[col].notna(),
        np.nan
    )

print(f"Records loaded: {len(data)}")
print(f"Original features: {len(ALL_FEATURES)}")

# ============================================================
# 4. GOWER CLUSTERING FUNCTION
# ============================================================

def run_gower_model(feature_data, model_name):

    print(f"\nCalculating Gower distances: {model_name}")

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

    print("Performing hierarchical clustering...")

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

    cluster_sizes = {
        int(cluster): int(count)
        for cluster, count in counts.sort_index().items()
    }

    print(f"Silhouette score: {silhouette:.4f}")
    print(f"Cluster balance: {balance:.4f}")
    print(f"Cluster sizes: {cluster_sizes}")

    del distance_matrix, condensed, tree
    gc.collect()

    return labels, silhouette, balance, cluster_sizes


# ============================================================
# 5. RUN ALL THREE MODELS
# ============================================================

model_results = {}
labels_by_model = {}
summary_rows = []

for model_name, selected_features in MODELS.items():

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    model_data = data[selected_features].copy()

    labels, silhouette, balance, cluster_sizes = (
        run_gower_model(
            model_data,
            model_name
        )
    )

    labels_by_model[model_name] = labels

    model_results[model_name] = {
        "features": selected_features,
        "silhouette": silhouette,
        "balance": balance,
        "cluster_sizes": cluster_sizes
    }

    summary_rows.append({
        "Model": model_name,
        "Feature Count": len(selected_features),
        "Features Used": ", ".join(selected_features),
        "Silhouette Score": silhouette,
        "Cluster Balance": balance,
        "Smallest Cluster": min(cluster_sizes.values()),
        "Largest Cluster": max(cluster_sizes.values())
    })

    del model_data
    gc.collect()

# ============================================================
# 6. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("COMPARING CLUSTER ASSIGNMENTS")
print("=" * 70)

model_names = list(MODELS.keys())

ari_rows = []

for i in range(len(model_names)):

    for j in range(i, len(model_names)):

        model_1 = model_names[i]
        model_2 = model_names[j]

        ari = adjusted_rand_score(
            labels_by_model[model_1],
            labels_by_model[model_2]
        )

        ari_rows.append({
            "Model 1": model_1,
            "Model 2": model_2,
            "Adjusted Rand Index": ari
        })

        print(
            f"{model_1} vs {model_2}: ARI = {ari:.4f}"
        )

# Save summary
summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "three_model_comparison.csv"
    ),
    index=False
)

# Save pairwise ARI
ari_df = pd.DataFrame(ari_rows)

ari_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_ari_comparison.csv"
    ),
    index=False
)

# ============================================================
# 7. SAVE CLUSTER LABELS
# ============================================================

labels_df = pd.DataFrame(labels_by_model)

labels_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "three_model_cluster_labels.csv"
    ),
    index=False
)

# ============================================================
# 8. CLUSTER SIZE ANALYSIS
# ============================================================

size_rows = []

for model_name, labels in labels_by_model.items():

    counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    for cluster, count in counts.items():

        size_rows.append({
            "Model": model_name,
            "Cluster": int(cluster),
            "Count": int(count),
            "Percentage": round(
                count / len(labels) * 100,
                2
            )
        })

size_df = pd.DataFrame(size_rows)

size_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "three_model_cluster_sizes.csv"
    ),
    index=False
)

# ============================================================
# 9. NUMERIC CLUSTER PROFILES
# ============================================================

numeric_rows = []

for model_name, labels in labels_by_model.items():

    selected_features = MODELS[model_name]

    for cluster in sorted(np.unique(labels)):

        cluster_data = data.loc[labels == cluster]

        for feature in NUMERIC_FEATURES:

            if feature not in selected_features:
                continue

            values = cluster_data[feature]

            numeric_rows.append({
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

numeric_df = pd.DataFrame(numeric_rows)

numeric_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "three_model_numeric_profiles.csv"
    ),
    index=False
)

# ============================================================
# 10. CATEGORICAL CLUSTER PROFILES
# ============================================================

categorical_rows = []

for model_name, labels in labels_by_model.items():

    selected_features = MODELS[model_name]

    for cluster in sorted(np.unique(labels)):

        cluster_data = data.loc[labels == cluster]

        for feature in CATEGORICAL_FEATURES:

            if feature not in selected_features:
                continue

            percentages = (
                cluster_data[feature]
                .fillna("Missing")
                .astype(str)
                .value_counts(normalize=True) * 100
            )

            for category, percentage in percentages.items():

                categorical_rows.append({
                    "Model": model_name,
                    "Cluster": int(cluster),
                    "Feature": feature,
                    "Category": category,
                    "Percentage": round(
                        percentage,
                        2
                    )
                })

categorical_df = pd.DataFrame(categorical_rows)

categorical_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "three_model_categorical_profiles.csv"
    ),
    index=False
)

# ============================================================
# 11. FEATURE MISSINGNESS
# ============================================================

missingness_rows = []

for feature in ALL_FEATURES:

    missingness_rows.append({
        "Feature": feature,
        "Missing Count": int(data[feature].isna().sum()),
        "Missing Percentage": round(
            data[feature].isna().mean() * 100,
            2
        )
    })

missingness_df = pd.DataFrame(missingness_rows)

missingness_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_missingness.csv"
    ),
    index=False
)

# ============================================================
# 12. FINAL REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "final_experiment_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "EMPLOYMENT FEATURE INFLUENCE EXPERIMENT\n"
    )
    report.write("=" * 65 + "\n\n")

    report.write(
        f"Number of records: {len(data)}\n\n"
    )

    report.write("MODEL SUMMARY\n")
    report.write("-" * 65 + "\n")
    report.write(
        summary_df.to_string(index=False)
    )

    report.write("\n\nPAIRWISE MODEL AGREEMENT (ARI)\n")
    report.write("-" * 65 + "\n")
    report.write(
        ari_df.to_string(index=False)
    )

    report.write("\n\nCLUSTER SIZES\n")
    report.write("-" * 65 + "\n")
    report.write(
        size_df.to_string(index=False)
    )

    report.write("\n\nFEATURE MISSINGNESS\n")
    report.write("-" * 65 + "\n")
    report.write(
        missingness_df.to_string(index=False)
    )

    report.write("\n\nINTERPRETATION GUIDANCE\n")
    report.write("-" * 65 + "\n")

    report.write(
        "Model A uses all eight original features.\n"
        "Model B excludes LFSTAT.\n"
        "Model C uses AGE, BAIND and N2BAMED only.\n\n"
    )

    report.write(
        "Silhouette scores measure separation within each model's "
        "own distance matrix. They are not directly comparable as "
        "proof that one feature set is universally better.\n\n"
    )

    report.write(
        "Cluster balance is the smallest cluster size divided by "
        "the largest cluster size. Higher values indicate more "
        "balanced cluster sizes.\n\n"
    )

    report.write(
        "ARI measures agreement between two cluster assignments. "
        "A low or negative ARI indicates little agreement.\n\n"
    )

    report.write(
        "Model C removes the selected employment-related variables, "
        "but the interpretation of BAIND and N2BAMED should still "
        "follow the NSCG codebook. Missing values and logical skips "
        "must not automatically be interpreted as meaningful groups.\n\n"
    )

    report.write(
        "These experiments use a 3,000-record sample and two-cluster "
        "hierarchical clustering. Findings should be validated before "
        "generalizing to the full NSCG population.\n"
    )

# ============================================================
# 13. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL THREE-MODEL COMPARISON")
print("=" * 70)

print(summary_df.to_string(index=False))

print("\nPAIRWISE ARI RESULTS")
print(ari_df.to_string(index=False))

print("\nCLUSTER SIZES")
print(size_df.to_string(index=False))

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nAll results saved in:\n{OUTPUT_DIR}")

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

gc.collect()
