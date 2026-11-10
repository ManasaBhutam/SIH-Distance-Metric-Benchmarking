
# ============================================================
# SIH PROJECT
# GOWER FEATURE INFLUENCE ANALYSIS
# LEAVE-ONE-FEATURE-OUT METHOD
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_FILE = BASE_DIR / "data" / "processed" / "distance_sample.csv"

LABELS_FILE = BASE_DIR / "data" / "processed" / "clustering_labels.csv"

OUTPUT_DIR = BASE_DIR / "feature_influence_results"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "N2BAMED",
    "SALARY_MISSING",
    "HRSWK_MISSING",
    "INDCODE_MISSING"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ORIGINAL_LABEL_COLUMN = "Gower_k2"

N_CLUSTERS = 2


# ============================================================
# 2. CALCULATE GOWER DISTANCE
# ============================================================

def calculate_gower(data, selected_features):

    numeric = [
        col for col in selected_features
        if col in NUMERIC_FEATURES
    ]

    categorical = [
        col for col in selected_features
        if col in CATEGORICAL_FEATURES
    ]

    work = data[selected_features].copy()

    # Convert numeric features
    for col in numeric:

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

        work[col] = work[col].fillna(
            work[col].median()
        )

    # Convert categorical features
    for col in categorical:

        work[col] = (
            work[col]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    # Numeric columns first, categorical columns next
    ordered_data = work[numeric + categorical]

    categorical_flags = np.array(
        [False] * len(numeric)
        + [True] * len(categorical)
    )

    distance_matrix = gower.gower_matrix(
        ordered_data,
        cat_features=categorical_flags
    )

    distance_matrix = np.asarray(
        distance_matrix,
        dtype=float
    )

    distance_matrix = (
        distance_matrix + distance_matrix.T
    ) / 2

    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


# ============================================================
# 3. PERFORM CLUSTERING AND EVALUATION
# ============================================================

def evaluate_clustering(distance_matrix, original_labels):

    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="precomputed",
        linkage="average"
    )

    new_labels = model.fit_predict(distance_matrix)

    # Compare new clusters with original Gower clusters
    ari = adjusted_rand_score(
        original_labels,
        new_labels
    )

    silhouette = silhouette_score(
        distance_matrix,
        new_labels,
        metric="precomputed"
    )

    cluster_sizes = pd.Series(
        new_labels
    ).value_counts()

    smallest = int(cluster_sizes.min())
    largest = int(cluster_sizes.max())

    balance_ratio = smallest / largest

    return (
        new_labels,
        ari,
        silhouette,
        smallest,
        largest,
        balance_ratio
    )


# ============================================================
# 4. MAIN FEATURE INFLUENCE ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("GOWER FEATURE INFLUENCE ANALYSIS")
    print("=" * 60)

    # Load sample data
    sample = pd.read_csv(
        SAMPLE_FILE,
        low_memory=False
    )

    # Load original clustering labels
    labels_df = pd.read_csv(
        LABELS_FILE
    )

    print("\nDataset loaded successfully.")

    print("Sample rows:", len(sample))
    print("Sample columns:", len(sample.columns))

    # Check required features
    missing_features = [
        col for col in ALL_FEATURES
        if col not in sample.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )

    if ORIGINAL_LABEL_COLUMN not in labels_df.columns:

        raise ValueError(
            f"Column {ORIGINAL_LABEL_COLUMN} "
            "not found in clustering_labels.csv"
        )

    if len(sample) != len(labels_df):

        raise ValueError(
            "Sample and labels have different row counts."
        )

    original_labels = labels_df[
        ORIGINAL_LABEL_COLUMN
    ].to_numpy()

    if pd.isna(original_labels).any():

        raise ValueError(
            "Original labels contain missing values."
        )

    print("\nOriginal cluster sizes:")

    print(
        pd.Series(original_labels)
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # BASELINE CHECK
    # --------------------------------------------------------

    print("\nCalculating original Gower baseline...")

    full_distance = calculate_gower(
        sample,
        ALL_FEATURES
    )

    (
        baseline_labels,
        baseline_ari,
        baseline_silhouette,
        _,
        _,
        _
    ) = evaluate_clustering(
        full_distance,
        original_labels
    )

    print(
        "Baseline ARI:",
        round(baseline_ari, 4)
    )

    print(
        "Baseline Silhouette:",
        round(baseline_silhouette, 4)
    )

    if baseline_ari < 0.99:

        print("\nWARNING:")
        print(
            "Recomputed baseline does not closely match "
            "the original labels."
        )

        print(
            "Check sample row order, preprocessing "
            "and Gower implementation before interpreting results."
        )

    else:

        print("\nBaseline verification successful.")

    del full_distance

    # --------------------------------------------------------
    # REMOVE ONE FEATURE AT A TIME
    # --------------------------------------------------------

    results = []

    for removed_feature in ALL_FEATURES:

        print("\n" + "-" * 55)

        print(
            "Removing feature:",
            removed_feature
        )

        selected_features = [
            col for col in ALL_FEATURES
            if col != removed_feature
        ]

        # Recalculate Gower distance
        distance_matrix = calculate_gower(
            sample,
            selected_features
        )

        # Perform clustering
        (
            new_labels,
            ari,
            silhouette,
            smallest,
            largest,
            balance
        ) = evaluate_clustering(
            distance_matrix,
            original_labels
        )

        # Save results
        results.append({

            "Removed_Feature": removed_feature,

            "ARI": ari,

            "Silhouette": silhouette,

            "Silhouette_Change": (
                silhouette - baseline_silhouette
            ),

            "Smallest_Cluster": smallest,

            "Largest_Cluster": largest,

            "Balance_Ratio": balance

        })

        # Save individual cluster labels
        label_output = pd.DataFrame({

            "Original_Gower_k2": original_labels,

            "New_Labels": new_labels

        })

        label_output.to_csv(
            OUTPUT_DIR /
            f"labels_without_{removed_feature}.csv",
            index=False
        )

        print("ARI:", round(ari, 4))

        print(
            "Silhouette:",
            round(silhouette, 4)
        )

        print("Smallest cluster:", smallest)

        print("Largest cluster:", largest)

        print(
            "Balance ratio:",
            round(balance, 4)
        )

        del distance_matrix

    # ========================================================
    # 5. SAVE FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    # Lowest ARI means the greatest change in cluster assignments
    results_df = results_df.sort_values(
        by="ARI",
        ascending=True
    )

    results_df.to_csv(
        OUTPUT_DIR / "feature_influence_summary.csv",
        index=False
    )

    # ========================================================
    # 6. GENERATE GRAPH
    # ========================================================

    plt.figure(figsize=(11, 6))

    plt.barh(
        results_df["Removed_Feature"],
        results_df["ARI"]
    )

    plt.xlabel(
        "Adjusted Rand Index (ARI)"
    )

    plt.ylabel(
        "Feature Removed"
    )

    plt.title(
        "Gower Clustering Feature Influence Analysis"
    )

    plt.xlim(0, 1.05)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_influence_ari.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # 7. DISPLAY FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)

    print("FINAL FEATURE INFLUENCE RESULTS")

    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\nAnalysis completed successfully!")

    print(
        "\nResults saved in:",
        OUTPUT_DIR
    )

    print("\nGenerated files:")

    print("- feature_influence_summary.csv")

    print("- feature_influence_ari.png")

    print("- Individual cluster label CSV files")

    print("\nInterpretation:")

    print(
        "Lower ARI means removing that feature "
        "changed the original clustering more."
    )

    print(
        "This is sensitivity analysis, not causal "
        "feature importance."
    )


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()