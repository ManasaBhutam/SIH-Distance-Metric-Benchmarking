
# ============================================================
# SIH PROJECT
# EMPLOYED-ONLY CLUSTERING VALIDATION
# GOWER DISTANCE + HIERARCHICAL CLUSTERING
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_FILE = BASE_DIR / "data" / "processed" / "distance_sample.csv"

OUTPUT_DIR = BASE_DIR / "employed_only_results"
OUTPUT_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "AGE",
    "SALARY",
    "HRSWK"
]

CATEGORICAL_FEATURES = [
    "BAIND",
    "INDCODE",
    "N2BAMED"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLUSTER_COUNTS = [2, 3, 4, 5, 6]

# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SAMPLE_FILE}"
        )

    data = pd.read_csv(
        SAMPLE_FILE,
        low_memory=False
    )

    required = ["LFSTAT"] + FEATURES

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("Original sample size:", len(data))

    # LFSTAT = 1 identifies employed respondents
    employed = data[
        data["LFSTAT"].astype(str).str.strip() == "1"
    ].copy()

    employed.reset_index(drop=True, inplace=True)

    if len(employed) <= max(CLUSTER_COUNTS):
        raise ValueError("Not enough employed respondents.")

    print("Employed respondents:", len(employed))
    print("Excluded respondents:", len(data) - len(employed))

    return employed


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

def prepare_features(data):

    features = data[FEATURES].copy()

    for col in NUMERIC_FEATURES:

        features[col] = pd.to_numeric(
            features[col],
            errors="coerce"
        )

        if features[col].isna().any():

            raise ValueError(
                f"{col} contains missing values in the "
                "employed-only sample. Investigate these "
                "before clustering; do not silently impute."
            )

    for col in CATEGORICAL_FEATURES:

        features[col] = (
            features[col]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    return features


# ============================================================
# 4. CALCULATE GOWER DISTANCE
# ============================================================

def calculate_gower(features):

    print("\nCalculating Gower distance...")

    categorical_flags = np.array(
        [False] * len(NUMERIC_FEATURES)
        + [True] * len(CATEGORICAL_FEATURES)
    )

    distance_matrix = gower.gower_matrix(
        features,
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

    print("Gower distance calculated.")

    return distance_matrix


# ============================================================
# 5. CLUSTERING EVALUATION
# ============================================================

def evaluate_clustering(distance_matrix):

    results = []
    labels_collection = {}

    print("\nEvaluating cluster counts...")

    for k in CLUSTER_COUNTS:

        print(f"\nTesting k = {k}")

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )

        labels = model.fit_predict(
            distance_matrix
        )

        silhouette = silhouette_score(
            distance_matrix,
            labels,
            metric="precomputed"
        )

        sizes = pd.Series(labels).value_counts()

        smallest = int(sizes.min())
        largest = int(sizes.max())

        balance = smallest / largest

        results.append({

            "Clusters": k,

            "Silhouette": silhouette,

            "Smallest_Cluster": smallest,

            "Largest_Cluster": largest,

            "Balance_Ratio": balance

        })

        labels_collection[k] = labels

        print("Silhouette:", round(silhouette, 4))
        print("Smallest cluster:", smallest)
        print("Largest cluster:", largest)
        print("Balance ratio:", round(balance, 4))

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "employed_clustering_summary.csv",
        index=False
    )

    labels_df = pd.DataFrame(labels_collection)

    labels_df.columns = [
        f"k_{k}" for k in labels_collection
    ]

    labels_df.to_csv(
        OUTPUT_DIR / "employed_clustering_labels.csv",
        index=False
    )

    return results_df, labels_collection


# ============================================================
# 6. GENERATE CLUSTER PROFILES
# ============================================================

def generate_profiles(features, labels_collection):

    print("\nGenerating cluster profiles...")

    all_profiles = []

    for k, labels in labels_collection.items():

        profile_data = features.copy()
        profile_data["Cluster"] = labels

        for cluster in sorted(profile_data["Cluster"].unique()):

            group = profile_data[
                profile_data["Cluster"] == cluster
            ]

            profile = {
                "Number_of_Clusters": k,
                "Cluster": int(cluster),
                "Size": len(group)
            }

            for col in NUMERIC_FEATURES:

                profile[f"{col}_Mean"] = group[col].mean()
                profile[f"{col}_Median"] = group[col].median()

            for col in CATEGORICAL_FEATURES:

                modes = group[col].mode()

                profile[f"{col}_Most_Common"] = (
                    modes.iloc[0] if len(modes) else "Unknown"
                )

            all_profiles.append(profile)

    profiles_df = pd.DataFrame(all_profiles)

    profiles_df.to_csv(
        OUTPUT_DIR / "employed_cluster_profiles.csv",
        index=False
    )

    return profiles_df


# ============================================================
# 7. GENERATE GRAPHS
# ============================================================

def generate_graphs(results_df):

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_df["Clusters"],
        results_df["Silhouette"],
        marker="o"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Employed-Only Gower Clustering")
    plt.xticks(CLUSTER_COUNTS)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "employed_silhouette_comparison.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(9, 5))

    plt.plot(
        results_df["Clusters"],
        results_df["Balance_Ratio"],
        marker="o"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Cluster Balance Ratio")
    plt.title("Employed-Only Cluster Balance")
    plt.xticks(CLUSTER_COUNTS)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "employed_cluster_balance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. FINAL REPORT
# ============================================================

def generate_report(
    employed,
    results_df,
    profiles_df
):

    report_file = (
        OUTPUT_DIR / "employed_only_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "EMPLOYED-ONLY CLUSTERING VALIDATION\n"
        )

        file.write("=" * 55 + "\n\n")

        file.write(
            f"Employed respondents: {len(employed)}\n"
        )

        file.write(
            f"Features used: {FEATURES}\n\n"
        )

        file.write("CLUSTERING RESULTS\n")

        file.write(
            results_df.to_string(index=False)
        )

        file.write("\n\nCLUSTER PROFILES\n")

        file.write(
            profiles_df.to_string(index=False)
        )

        file.write(
            "\n\nINTERPRETATION NOTE\n"
            "This experiment studies subgroups among "
            "respondents with LFSTAT = 1. Silhouette "
            "and cluster balance should be considered "
            "together. These exploratory clusters do "
            "not establish causal relationships.\n"
        )

    print("\nReport saved:", report_file)


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("EMPLOYED-ONLY CLUSTERING VALIDATION")
    print("=" * 65)

    employed = load_data()

    features = prepare_features(employed)

    distance_matrix = calculate_gower(features)

    results_df, labels_collection = (
        evaluate_clustering(distance_matrix)
    )

    profiles_df = generate_profiles(
        features,
        labels_collection
    )

    generate_graphs(results_df)

    generate_report(
        employed,
        results_df,
        profiles_df
    )

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nFinal results:")
    print(results_df.to_string(index=False))

    print("\nOutput folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
