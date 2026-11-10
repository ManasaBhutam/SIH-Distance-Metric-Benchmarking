
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# File locations
DISTANCE_FILE = "data/processed/distance_results.npz"
OUTPUT_FILE = "data/processed/clustering_stability_results.csv"

SAMPLE_SIZE = 2000
REPEATS = 5
K_VALUES = [2, 3, 4, 5]

# Load compressed distance arrays
data = np.load(DISTANCE_FILE)

distance_names = ["gower", "euclidean", "cosine"]

results = []

def reconstruct_matrix(values):
    """Reconstruct a symmetric matrix from condensed distances."""

    n = int((1 + np.sqrt(1 + 8 * len(values))) / 2)

    if n * (n - 1) // 2 != len(values):
        raise ValueError("Distance array length is invalid.")

    matrix = np.zeros((n, n), dtype=np.float64)

    upper = np.triu_indices(n, k=1)
    matrix[upper] = values
    matrix[(upper[1], upper[0])] = values

    return matrix


for distance_name in distance_names:

    print(f"\nTesting {distance_name.upper()}")

    distance_matrix = reconstruct_matrix(data[distance_name])
    n = len(distance_matrix)

    rng = np.random.default_rng(42)

    for k in K_VALUES:

        silhouettes = []
        balances = []
        repeat_labels = []

        for repeat in range(REPEATS):

            indices = rng.choice(
                n,
                size=min(SAMPLE_SIZE, n),
                replace=False
            )

            subset_matrix = distance_matrix[
                np.ix_(indices, indices)
            ]

            model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage="average"
            )

            labels = model.fit_predict(subset_matrix)

            silhouette = silhouette_score(
                subset_matrix,
                labels,
                metric="precomputed"
            )

            counts = np.bincount(labels)

            silhouettes.append(silhouette)
            balances.append(counts.min() / counts.max())

            repeat_labels.append((indices, labels))

        # Compare cluster assignments on shared records
        ari_values = []

        for i in range(REPEATS):
            for j in range(i + 1, REPEATS):

                indices_a, labels_a = repeat_labels[i]
                indices_b, labels_b = repeat_labels[j]

                common, pos_a, pos_b = np.intersect1d(
                    indices_a,
                    indices_b,
                    return_indices=True
                )

                if len(common) >= k + 1:

                    ari = adjusted_rand_score(
                        labels_a[pos_a],
                        labels_b[pos_b]
                    )

                    ari_values.append(ari)

        results.append({
            "Distance": distance_name,
            "Clusters": k,
            "Mean Silhouette": np.mean(silhouettes),
            "Silhouette Std": np.std(silhouettes),
            "Mean Balance Ratio": np.mean(balances),
            "Mean Stability ARI": (
                np.mean(ari_values) if ari_values else np.nan
            )
        })

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\nCLUSTERING STABILITY RESULTS")
print(results_df.round(4).to_string(index=False))

print("\nSaved:", OUTPUT_FILE)
