
# ============================================================
# SIH PROJECT
# COMPARATIVE ANALYSIS OF GOWER, EUCLIDEAN AND COSINE DISTANCE
# DATASET: NATIONAL SURVEY OF COLLEGE GRADUATES 2023
# ============================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gower
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "SIH Distance Metric Comparison"

INPUT_FILE = Path("data/raw/epcg23.csv")
OUTPUT_DIR = Path("data/processed")
PLOT_DIR = OUTPUT_DIR / "plots"

SAMPLE_SIZE = 3000
RANDOM_STATE = 42

CLUSTER_COUNTS = [2, 3, 4, 5, 6]

NUMERIC_COLS = [
    "AGE",
    "SALARY",
    "HRSWK"
]

CATEGORICAL_COLS = [
    "LFSTAT",
    "ACTRD",
    "BAIND",
    "INDCODE",
    "N2BAMED"
]

REQUIRED_COLS = NUMERIC_COLS + CATEGORICAL_COLS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset():

    print("\n[1/9] LOADING DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    missing = [
        col for col in REQUIRED_COLS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Dataset loaded successfully.")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    return df


# ============================================================
# 3. INITIAL DATA CLEANING
# ============================================================

def clean_dataset(df):

    print("\n[2/9] CLEANING DATA")

    data = df[REQUIRED_COLS].copy()

    # Confirmed logical skip for salary
    data["SALARY"] = pd.to_numeric(
        data["SALARY"],
        errors="coerce"
    )

    data.loc[
        data["SALARY"] == 9999998,
        "SALARY"
    ] = np.nan

    # Special values aligned with respondents not working.
    # Exact codebook labels still require final verification.
    data["INDCODE"] = data["INDCODE"].replace(
        9998,
        np.nan
    )

    data["HRSWK"] = pd.to_numeric(
        data["HRSWK"],
        errors="coerce"
    )

    data.loc[
        data["HRSWK"] == 98,
        "HRSWK"
    ] = np.nan

    data["AGE"] = pd.to_numeric(
        data["AGE"],
        errors="coerce"
    )

    
# Correct education variable preprocessing

    # Correct education variable preprocessing

    data["BAIND"] = (
        data["BAIND"]
        .astype("string")
        .str.strip()
    )

    data["N2BAMED"] = (
        data["N2BAMED"]
        .astype("string")
        .str.strip()
    )

    data["N2BAMED"] = data["N2BAMED"].replace({
        "999989": "NOT_APPLICABLE",
        "999999": "UNKNOWN_MISSING"
    })

    data["N2BAMED"] = data["N2BAMED"].fillna(
        "UNKNOWN_MISSING"
    )

    print("Initial cleaning completed.")
    print("Rows retained:", len(data))

    return data


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def prepare_features(data):

    print("\n[3/9] PREPARING FEATURES")

    features = data.copy()

    # Missingness indicators preserve information about
    # variables that are unavailable for some respondents.

    features["SALARY_MISSING"] = (
        features["SALARY"].isna().astype(str)
    )

    features["HRSWK_MISSING"] = (
        features["HRSWK"].isna().astype(str)
    )

    features["INDCODE_MISSING"] = (
        features["INDCODE"].isna().astype(str)
    )

    categorical_features = (
        CATEGORICAL_COLS
        + [
            "SALARY_MISSING",
            "HRSWK_MISSING",
            "INDCODE_MISSING"
        ]
    )

    # Fill numerical missing values using medians.
    for col in NUMERIC_COLS:

        median = features[col].median()

        if pd.isna(median):
            raise ValueError(
                f"Column {col} contains no valid numeric values."
            )

        features[col] = features[col].fillna(median)

    # Convert categorical values into strings.
    for col in CATEGORICAL_COLS:

        features[col] = (
            features[col]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    print("Numeric features:", NUMERIC_COLS)
    print("Categorical features:", categorical_features)

    return features, categorical_features


# ============================================================
# 5. REPRODUCIBLE SAMPLING
# ============================================================

def create_sample(features):

    print("\n[4/9] CREATING SAMPLE")

    n = min(SAMPLE_SIZE, len(features))

    sample = features.sample(
        n=n,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print("Sample size:", len(sample))

    sample.to_csv(
        OUTPUT_DIR / "distance_sample.csv",
        index=False
    )

    return sample


# ============================================================
# 6. CALCULATE DISTANCE METRICS
# ============================================================

def calculate_distances(sample, categorical_features):

    print("\n[5/9] CALCULATING DISTANCES")

    # --------------------------------------------------------
    # GOWER DISTANCE
    # --------------------------------------------------------

    print("Calculating Gower distance...")

    numeric_data = sample[NUMERIC_COLS].copy()

    categorical_data = sample[
        categorical_features
    ].copy()

    gower_input = pd.concat(
        [
            numeric_data.reset_index(drop=True),
            categorical_data.reset_index(drop=True)
        ],
        axis=1
    )

    cat_features = np.array(
        [False] * len(NUMERIC_COLS)
        + [True] * len(categorical_features)
    )

    gower_matrix = gower.gower_matrix(
        gower_input,
        cat_features=cat_features
    )

    gower_dist = gower_matrix[
        np.triu_indices_from(
            gower_matrix,
            k=1
        )
    ]

    del gower_matrix

    # --------------------------------------------------------
    # EUCLIDEAN AND COSINE
    # --------------------------------------------------------

    print("Preparing encoded features...")

    scaler = StandardScaler()

    scaled_numeric = scaler.fit_transform(
        numeric_data
    )

    encoded_categorical = pd.get_dummies(
        categorical_data,
        dtype=float
    ).to_numpy()

    X = np.hstack(
        [
            scaled_numeric,
            encoded_categorical
        ]
    )

    print("Encoded matrix:", X.shape)

    print("Calculating Euclidean distance...")

    euclidean_dist = pdist(
        X,
        metric="euclidean"
    )

    print("Calculating Cosine distance...")

    cosine_dist = pdist(
        X,
        metric="cosine"
    )

    cosine_dist = np.nan_to_num(
        cosine_dist,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Save distance matrices in compressed format.
    np.savez_compressed(
        OUTPUT_DIR / "distance_results.npz",
        gower=gower_dist,
        euclidean=euclidean_dist,
        cosine=cosine_dist
    )

    print("All distance calculations completed.")

    return {
        "Gower": gower_dist,
        "Euclidean": euclidean_dist,
        "Cosine": cosine_dist
    }


# ============================================================
# 7. DISTANCE COMPARISON
# ============================================================

def compare_distances(distances):

    print("\n[6/9] COMPARING DISTANCES")

    summary = []

    for name, values in distances.items():

        row = {
            "Metric": name,
            "Minimum": np.min(values),
            "Mean": np.mean(values),
            "Median": np.median(values),
            "Maximum": np.max(values),
            "Standard Deviation": np.std(values)
        }

        summary.append(row)

        print("\n", name)

        for key, value in row.items():

            if key != "Metric":
                print(
                    f"{key}: {value:.6f}"
                )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "distance_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # SPEARMAN CORRELATION
    # --------------------------------------------------------

    print("\nSpearman Rank Correlation")

    names = list(distances.keys())

    correlation_rows = []

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            corr, _ = spearmanr(
                distances[names[i]],
                distances[names[j]]
            )

            correlation_rows.append({
                "Metric A": names[i],
                "Metric B": names[j],
                "Correlation": corr
            })

            print(
                f"{names[i]} vs {names[j]}: {corr:.6f}"
            )

    correlation_df = pd.DataFrame(
        correlation_rows
    )

    correlation_df.to_csv(
        OUTPUT_DIR / "distance_correlations.csv",
        index=False
    )

    # Distance distribution plot
    plt.figure(figsize=(10, 6))

    for name, values in distances.items():

        plt.hist(
            values,
            bins=50,
            alpha=0.45,
            label=name,
            density=True
        )

    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Comparison of Distance Distributions")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "distance_distribution.png",
        dpi=300
    )

    plt.close()

    return summary_df, correlation_df


# ============================================================
# 8. HIERARCHICAL CLUSTERING
# ============================================================

def evaluate_clustering(distances):

    print("\n[7/9] EVALUATING CLUSTERING")

    results = []

    all_labels = {}

    for metric_name, condensed in distances.items():

        print(
            f"\nEvaluating {metric_name}..."
        )

        D = squareform(condensed)

        D = (D + D.T) / 2

        np.fill_diagonal(D, 0)

        for k in CLUSTER_COUNTS:

            model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage="average"
            )

            labels = model.fit_predict(D)

            score = silhouette_score(
                D,
                labels,
                metric="precomputed"
            )

            cluster_sizes = pd.Series(
                labels
            ).value_counts()

            results.append({
                "Metric": metric_name,
                "Clusters": k,
                "Silhouette": score,
                "Smallest Cluster": cluster_sizes.min(),
                "Largest Cluster": cluster_sizes.max()
            })

            all_labels[
                f"{metric_name}_k{k}"
            ] = labels

            print(
                f"k={k} | "
                f"Silhouette={score:.4f} | "
                f"Smallest={cluster_sizes.min()} | "
                f"Largest={cluster_sizes.max()}"
            )

        del D

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "clustering_evaluation.csv",
        index=False
    )

    labels_df = pd.DataFrame(all_labels)

    labels_df.to_csv(
        OUTPUT_DIR / "clustering_labels.csv",
        index=False
    )

    # --------------------------------------------------------
    # SILHOUETTE COMPARISON PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for metric in results_df["Metric"].unique():

        subset = results_df[
            results_df["Metric"] == metric
        ]

        plt.plot(
            subset["Clusters"],
            subset["Silhouette"],
            marker="o",
            label=metric
        )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Clustering Performance Comparison")
    plt.xticks(CLUSTER_COUNTS)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "clustering_comparison.png",
        dpi=300
    )

    plt.close()

    return results_df, labels_df


# ============================================================
# 9. CLUSTER PROFILING
# ============================================================

def generate_cluster_profiles(sample, labels_df):

    print("\n[8/9] GENERATING CLUSTER PROFILES")

    numeric = NUMERIC_COLS

    categorical = CATEGORICAL_COLS

    profiles = []

    for metric in [
        "Gower",
        "Euclidean",
        "Cosine"
    ]:

        column = f"{metric}_k2"

        data = sample.copy()

        data["Cluster"] = labels_df[column]

        print(
            f"\n{metric} Cluster Profiles"
        )

        print(
            data["Cluster"].value_counts().sort_index()
        )

        for cluster in sorted(
            data["Cluster"].unique()
        ):

            group = data[
                data["Cluster"] == cluster
            ]

            profile = {
                "Metric": metric,
                "Cluster": int(cluster),
                "Size": len(group)
            }

            for col in numeric:

                profile[
                    f"{col}_median"
                ] = group[col].median()

            for col in categorical:

                modes = group[col].mode(
                    dropna=True
                )

                profile[
                    f"{col}_most_common"
                ] = (
                    modes.iloc[0]
                    if len(modes)
                    else "Unknown"
                )

            profiles.append(profile)

    profile_df = pd.DataFrame(profiles)

    profile_df.to_csv(
        OUTPUT_DIR / "cluster_profiles.csv",
        index=False
    )

    print("Cluster profiles saved.")

    return profile_df


# ============================================================
# 10. FINAL REPORT
# ============================================================

def generate_final_report(
    df,
    sample,
    summary_df,
    correlation_df,
    clustering_df,
    profile_df
):

    print("\n[9/9] GENERATING FINAL REPORT")

    report_path = OUTPUT_DIR / "final_project_report.txt"

    best_rows = (
        clustering_df
        .sort_values(
            "Silhouette",
            ascending=False
        )
        .groupby("Metric")
        .head(1)
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SIH PROJECT: DISTANCE METRIC COMPARISON\n"
        )

        file.write("=" * 55 + "\n\n")

        file.write(
            f"Original dataset rows: {len(df)}\n"
        )

        file.write(
            f"Original dataset columns: {df.shape[1]}\n"
        )

        file.write(
            f"Sample size: {len(sample)}\n\n"
        )

        file.write(
            "DISTANCE SUMMARY\n"
        )

        file.write(
            summary_df.to_string(index=False)
        )

        file.write("\n\nSPEARMAN CORRELATIONS\n")

        file.write(
            correlation_df.to_string(index=False)
        )

        file.write("\n\nBEST CLUSTERING RESULTS\n")

        file.write(
            best_rows.to_string(index=False)
        )

        file.write("\n\nCLUSTER PROFILES\n")

        file.write(
            profile_df.to_string(index=False)
        )

        file.write(
            "\n\nNote: Results are exploratory. "
            "Validate codebook definitions, missing-data "
            "handling, cluster stability and application "
            "performance before making final conclusions."
        )

    print(
        "Final report saved:",
        report_path
    )


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 65)

    print(PROJECT_NAME)

    print("=" * 65)

    # Step 1
    df = load_dataset()

    # Step 2
    cleaned = clean_dataset(df)

    # Save cleaned analysis copy
    cleaned.to_csv(
        OUTPUT_DIR / "cleaned_dataset.csv",
        index=False
    )

    # Step 3
    features, categorical_features = prepare_features(
        cleaned
    )

    # Step 4
    sample = create_sample(features)

    # Step 5
    distances = calculate_distances(
        sample,
        categorical_features
    )

    # Step 6
    summary_df, correlation_df = compare_distances(
        distances
    )

    # Step 7
    clustering_df, labels_df = evaluate_clustering(
        distances
    )

    # Step 8
    profile_df = generate_cluster_profiles(
        sample,
        labels_df
    )

    # Step 9
    generate_final_report(
        df,
        sample,
        summary_df,
        correlation_df,
        clustering_df,
        profile_df
    )

    print("\n" + "=" * 65)

    print("SIH PROJECT EXECUTION COMPLETED SUCCESSFULLY")

    print("=" * 65)

    print("\nOutput directory:", OUTPUT_DIR)

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.iterdir()):

        if file.is_file():
            print(" -", file.name)

    print("\nGenerated plots:")

    for file in sorted(PLOT_DIR.iterdir()):

        if file.is_file():
            print(" -", file.name)

    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()
