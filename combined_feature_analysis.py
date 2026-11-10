

# ============================================================
# SIH PROJECT
# COMBINED FEATURE REMOVAL ANALYSIS
# GOWER DISTANCE + HIERARCHICAL CLUSTERING
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

OUTPUT_DIR = BASE_DIR / "combined_feature_results"
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
# 2. DEFINE FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    "Employment_Features": [
        "LFSTAT",
        "ACTRD",
        "INDCODE"
    ],

    "Salary_and_Working_Hours": [
        "SALARY",
        "HRSWK"
    ],

    "Education_Features": [
        "BAIND",
        "N2BAMED"
    ],

    "Missingness_Indicators": [
        "SALARY_MISSING",
        "HRSWK_MISSING",
        "INDCODE_MISSING"
    ]

}


# ============================================================
# 3. CALCULATE GOWER DISTANCE
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

    if not selected_features:
        raise ValueError("No features available.")

    work = data[selected_features].copy()

    # Numeric preprocessing
    for col in numeric:

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce"
        )

        work[col] = work[col].fillna(
            work[col].median()
        )

    # Categorical preprocessing
    for col in categorical:

        work[col] = (
            work[col]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    # Arrange numeric features first
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
# 4. CLUSTERING AND EVALUATION
# ============================================================

def evaluate_clustering(distance_matrix, original_labels):

    model = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        metric="precomputed",
        linkage="average"
    )

    new_labels = model.fit_predict(distance_matrix)

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

    balance = smallest / largest

    return (
        new_labels,
        ari,
        silhouette,
        smallest,
        largest,
        balance
    )


# ============================================================
# 5. MAIN ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 65)

    print("COMBINED FEATURE REMOVAL ANALYSIS")

    print("=" * 65)

    # Load dataset
    sample = pd.read_csv(
        SAMPLE_FILE,
        low_memory=False
    )

    labels_df = pd.read_csv(
        LABELS_FILE
    )

    print("\nDataset loaded successfully.")

    print("Sample rows:", len(sample))

    print("Sample columns:", len(sample.columns))

    # Validate input
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
            "Original Gower_k2 labels not found."
        )

    if len(sample) != len(labels_df):

        raise ValueError(
            "Sample and label row counts do not match."
        )

    original_labels = labels_df[
        ORIGINAL_LABEL_COLUMN
    ].to_numpy()

    if pd.isna(original_labels).any():

        raise ValueError(
            "Original labels contain missing values."
        )

    # ========================================================
    # BASELINE CALCULATION
    # ========================================================

    print("\nCalculating original baseline...")

    baseline_distance = calculate_gower(
        sample,
        ALL_FEATURES
    )

    (
        baseline_labels,
        baseline_ari,
        baseline_silhouette,
        baseline_smallest,
        baseline_largest,
        baseline_balance
    ) = evaluate_clustering(
        baseline_distance,
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

        raise ValueError(
            "Baseline verification failed. "
            "Check sample order and preprocessing."
        )

    print("Baseline verification successful.")

    del baseline_distance

    # Store baseline
    results = []

    results.append({

        "Experiment": "Original_Baseline",

        "Removed_Features": "None",

        "Remaining_Features": len(ALL_FEATURES),

        "ARI": baseline_ari,

        "Silhouette": baseline_silhouette,

        "Silhouette_Change": 0.0,

        "Smallest_Cluster": baseline_smallest,

        "Largest_Cluster": baseline_largest,

        "Balance_Ratio": baseline_balance

    })

    # ========================================================
    # REMOVE FEATURE GROUPS
    # ========================================================

    for group_name, removed_features in FEATURE_GROUPS.items():

        print("\n" + "-" * 65)

        print("Experiment:", group_name)

        print("Removing:", removed_features)

        selected_features = [
            col for col in ALL_FEATURES
            if col not in removed_features
        ]

        print(
            "Remaining features:",
            len(selected_features)
        )

        # Calculate new Gower distance
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

        # Save individual labels
        label_df = pd.DataFrame({

            "Original_Gower_k2": original_labels,

            "New_Cluster_Labels": new_labels

        })

        label_df.to_csv(
            OUTPUT_DIR / f"{group_name}_labels.csv",
            index=False
        )

        # Store results
        results.append({

            "Experiment": group_name,

            "Removed_Features": ", ".join(removed_features),

            "Remaining_Features": len(selected_features),

            "ARI": ari,

            "Silhouette": silhouette,

            "Silhouette_Change": (
                silhouette - baseline_silhouette
            ),

            "Smallest_Cluster": smallest,

            "Largest_Cluster": largest,

            "Balance_Ratio": balance

        })

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
    # SAVE FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "combined_feature_summary.csv",
        index=False
    )

    # ========================================================
    # GENERATE ARI GRAPH
    # ========================================================

    plot_df = results_df[
        results_df["Experiment"] != "Original_Baseline"
    ]

    plt.figure(figsize=(11, 6))

    plt.barh(
        plot_df["Experiment"],
        plot_df["ARI"]
    )

    plt.xlabel(
        "Adjusted Rand Index"
    )

    plt.ylabel(
        "Removed Feature Group"
    )

    plt.title(
        "Combined Feature Removal - Gower Clustering"
    )

    plt.xlim(0, 1.05)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "combined_feature_ari.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # GENERATE SILHOUETTE GRAPH
    # ========================================================

    plt.figure(figsize=(11, 6))

    plt.bar(
        results_df["Experiment"],
        results_df["Silhouette"]
    )

    plt.xlabel(
        "Experiment"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "Silhouette Comparison - Combined Feature Removal"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "combined_feature_silhouette.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 65)

    print("FINAL COMBINED FEATURE ANALYSIS")

    print("=" * 65)

    print(
        results_df.to_string(index=False)
    )

    print("\nAnalysis completed successfully!")

    print(
        "\nResults saved in:",
        OUTPUT_DIR
    )

    print("\nGenerated files:")

    print("- combined_feature_summary.csv")

    print("- combined_feature_ari.png")

    print("- combined_feature_silhouette.png")

    print("- Individual experiment label CSV files")


# ============================================================
# 6. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
