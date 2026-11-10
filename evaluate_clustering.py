
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path("data/processed/distance_results.npz")
OUTPUT_DIR = Path("data/processed")

CLUSTER_COUNTS = [2, 3, 4, 5, 6]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load previously calculated distances
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Run the distance analysis first: {INPUT_FILE}"
    )

saved = np.load(INPUT_FILE)

distance_vectors = {
    "Gower": saved["gower"],
    "Euclidean": saved["euclidean"],
    "Cosine": saved["cosine"],
}

results = []
cluster_labels = {}

# --------------------------------------------------
# Cluster and evaluate each distance metric
# --------------------------------------------------

for metric_name, condensed in distance_vectors.items():

    print(f"\nEvaluating {metric_name} distance...")

    # Convert condensed pairwise distances to square matrix
    D = squareform(condensed)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)

    for k in CLUSTER_COUNTS:

        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average",
        )

        labels = model.fit_predict(D)

        # Silhouette score using the same distance metric
        score = silhouette_score(
            D,
            labels,
            metric="precomputed",
        )

        cluster_labels[f"{metric_name}_k{k}"] = labels

        cluster_sizes = pd.Series(labels).value_counts()

        results.append({
            "Metric": metric_name,
            "Clusters": k,
            "Silhouette Score": score,
            "Smallest Cluster": int(cluster_sizes.min()),
            "Largest Cluster": int(cluster_sizes.max()),
        })

        print(
            f"k={k} | silhouette={score:.4f} | "
            f"smallest cluster={cluster_sizes.min()} | "
            f"largest cluster={cluster_sizes.max()}"
        )

    del D

# --------------------------------------------------
# Save evaluation results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "clustering_evaluation.csv",
    index=False,
)

# Save cluster assignments for each metric and k
labels_df = pd.DataFrame(cluster_labels)

labels_df.to_csv(
    OUTPUT_DIR / "clustering_labels.csv",
    index=False,
)

# --------------------------------------------------
# Show best exploratory configuration per metric
# --------------------------------------------------

print("\n" + "=" * 55)
print("BEST SILHOUETTE SCORE PER METRIC")
print("=" * 55)

for metric_name in distance_vectors:
    subset = results_df[results_df["Metric"] == metric_name]
    best = subset.loc[subset["Silhouette Score"].idxmax()]

    print(
        f"{metric_name}: k={int(best['Clusters'])}, "
        f"silhouette={best['Silhouette Score']:.4f}"
    )

print("\nSaved:")
print("data/processed/clustering_evaluation.csv")
print("data/processed/clustering_labels.csv")
print("\nClustering evaluation completed.")
