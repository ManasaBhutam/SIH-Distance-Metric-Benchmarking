
import pandas as pd
import numpy as np
import gower

from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE = Path(__file__).resolve().parent

DATA_FILE = BASE / "data" / "processed" / "distance_sample.csv"
OUTPUT = BASE / "education_sensitivity_results"
OUTPUT.mkdir(exist_ok=True)

NUMERIC = ["AGE", "SALARY", "HRSWK"]

# Model A: Includes education
FEATURES_A = [
    "AGE", "SALARY", "HRSWK",
    "BAIND", "INDCODE", "N2BAMED"
]

# Model B: Excludes education
FEATURES_B = [
    "AGE", "SALARY", "HRSWK", "INDCODE"
]


def calculate_gower(data, features):

    numeric = [c for c in features if c in NUMERIC]
    categorical = [c for c in features if c not in NUMERIC]

    work = data[features].copy()

    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")

        if work[col].isna().any():
            raise ValueError(
                f"{col} contains missing values. "
                "Investigate before clustering."
            )

    for col in categorical:
        work[col] = (
            work[col].astype("string")
            .fillna("Unknown").astype(str)
        )

    ordered = work[numeric + categorical]

    flags = np.array(
        [False] * len(numeric) +
        [True] * len(categorical)
    )

    D = gower.gower_matrix(
        ordered,
        cat_features=flags
    )

    D = np.asarray(D, dtype=float)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)

    return D


def run_model(data, features, model_name):

    print(f"\nRunning {model_name}...")

    D = calculate_gower(data, features)

    results = []
    labels_output = {}

    for k in [2, 3, 4, 5, 6]:

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )

        labels = model.fit_predict(D)

        score = silhouette_score(
            D, labels, metric="precomputed"
        )

        sizes = pd.Series(labels).value_counts()

        results.append({
            "Model": model_name,
            "Clusters": k,
            "Silhouette": score,
            "Smallest_Cluster": int(sizes.min()),
            "Largest_Cluster": int(sizes.max()),
            "Balance_Ratio": sizes.min() / sizes.max()
        })

        labels_output[f"k_{k}"] = labels

        print(
            f"k={k} | Silhouette={score:.4f} | "
            f"Smallest={sizes.min()} | "
            f"Largest={sizes.max()}"
        )

    pd.DataFrame(labels_output).to_csv(
        OUTPUT / f"{model_name}_labels.csv",
        index=False
    )

    del D

    return results


def main():

    print("=" * 60)
    print("EDUCATION FEATURE SENSITIVITY TEST")
    print("=" * 60)

    data = pd.read_csv(DATA_FILE, low_memory=False)

    # Select employed respondents
    employed = data[
        data["LFSTAT"].astype(str).str.strip() == "1"
    ].copy().reset_index(drop=True)

    print("Employed respondents:", len(employed))

    # Inspect education categories
    print("\nBAIND distribution:")
    print(employed["BAIND"].value_counts(dropna=False))

    print("\nN2BAMED distribution:")
    print(employed["N2BAMED"].value_counts(dropna=False).head(15))

    # Inspect original employed-only k=2 cluster composition
    original_labels = pd.read_csv(
        BASE / "employed_only_results" /
        "employed_clustering_labels.csv"
    )

    if len(original_labels) != len(employed):
        raise ValueError("Employed labels and data rows do not match.")

    employed["Original_Cluster"] = original_labels["k_2"].values

    print("\nBAIND by original cluster:")
    print(pd.crosstab(
        employed["Original_Cluster"],
        employed["BAIND"],
        normalize="index"
    ).round(3) * 100)

    print("\nN2BAMED unknown percentage by original cluster:")
    unknown = (
        employed["N2BAMED"].astype(str).str.strip().isin(
            ["Unknown", "nan", "<NA>", ""]
        )
    )

    print(
        employed.assign(N2BAMED_Unknown=unknown)
        .groupby("Original_Cluster")["N2BAMED_Unknown"]
        .mean().mul(100).round(2)
    )

    # Model A: education included
    results_a = run_model(
        employed,
        FEATURES_A,
        "Model_A_Education_Included"
    )

    # Model B: education excluded
    results_b = run_model(
        employed,
        FEATURES_B,
        "Model_B_Education_Excluded"
    )

    results = pd.DataFrame(results_a + results_b)

    results.to_csv(
        OUTPUT / "education_sensitivity_summary.csv",
        index=False
    )

    # Compare k=2 labels between the two models
    labels_a = pd.read_csv(
        OUTPUT / "Model_A_Education_Included_labels.csv"
    )["k_2"]

    labels_b = pd.read_csv(
        OUTPUT / "Model_B_Education_Excluded_labels.csv"
    )["k_2"]

    ari = adjusted_rand_score(labels_a, labels_b)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(results.to_string(index=False))

    print(
        "\nARI between Model A and Model B:",
        round(ari, 4)
    )

    print("\nResults saved in:", OUTPUT)


if __name__ == "__main__":
    main()
