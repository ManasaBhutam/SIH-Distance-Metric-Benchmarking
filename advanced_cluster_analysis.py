
# ============================================================
# SIH PROJECT - ADVANCED CLUSTER INTERPRETATION
# GOWER DISTANCE | K = 2
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "cluster_interpretation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = DATA_DIR / "distance_sample.csv"
LABEL_FILE = DATA_DIR / "clustering_labels.csv"

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("\nLoading project data...")

sample = pd.read_csv(SAMPLE_FILE)
labels = pd.read_csv(LABEL_FILE)

label_column = "Gower_k2"

if label_column not in labels.columns:
    raise ValueError(
        f"{label_column} not found. Available columns: "
        f"{labels.columns.tolist()}"
    )

if len(sample) != len(labels):
    raise ValueError(
        "Sample and cluster label row counts do not match. "
        "Cannot safely combine these files."
    )

df = sample.copy()
df["Cluster"] = labels[label_column].values

print("Records:", len(df))
print("Features:", len(sample.columns))
print("Clusters:", sorted(df["Cluster"].dropna().unique()))

# ------------------------------------------------------------
# 3. CLUSTER SIZE ANALYSIS
# ------------------------------------------------------------

cluster_sizes = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Respondents")
)

cluster_sizes["Percentage"] = (
    cluster_sizes["Respondents"] / len(df) * 100
)

cluster_sizes.to_csv(
    OUTPUT_DIR / "cluster_sizes.csv",
    index=False
)

print("\nCLUSTER SIZES")
print(cluster_sizes.to_string(index=False))

# ------------------------------------------------------------
# 4. NUMERICAL FEATURE ANALYSIS
# ------------------------------------------------------------

numeric_features = [
    col for col in ["AGE", "SALARY", "HRSWK"]
    if col in df.columns
]

numeric_rows = []

for feature in numeric_features:

    for cluster in sorted(df["Cluster"].dropna().unique()):

        values = pd.to_numeric(
            df.loc[df["Cluster"] == cluster, feature],
            errors="coerce"
        )

        numeric_rows.append({
            "Feature": feature,
            "Cluster": cluster,
            "Valid Records": values.notna().sum(),
            "Missing Records": values.isna().sum(),
            "Mean": values.mean(),
            "Median": values.median(),
            "Standard Deviation": values.std()
        })

numeric_profile = pd.DataFrame(numeric_rows)

numeric_profile.to_csv(
    OUTPUT_DIR / "numeric_cluster_profiles.csv",
    index=False
)

print("\nNUMERICAL FEATURE PROFILES")
print(numeric_profile.round(2).to_string(index=False))

# ------------------------------------------------------------
# 5. CATEGORICAL FEATURE ANALYSIS
# ------------------------------------------------------------

categorical_features = [
    col for col in [
        "LFSTAT",
        "ACTRD",
        "BAIND",
        "INDCODE",
        "N2BAMED"
    ]
    if col in df.columns
]

categorical_rows = []

for feature in categorical_features:

    for cluster in sorted(df["Cluster"].dropna().unique()):

        group = df[df["Cluster"] == cluster]

        values = (
            group[feature]
            .fillna("Missing")
            .astype(str)
        )

        counts = values.value_counts()

        for category, count in counts.head(10).items():

            categorical_rows.append({
                "Feature": feature,
                "Cluster": cluster,
                "Category": category,
                "Respondents": count,
                "Percentage": count / len(group) * 100
            })

categorical_profile = pd.DataFrame(categorical_rows)

categorical_profile.to_csv(
    OUTPUT_DIR / "categorical_cluster_profiles.csv",
    index=False
)

print("\nCATEGORICAL FEATURE PROFILES")
print(categorical_profile.round(2).to_string(index=False))

# ------------------------------------------------------------
# 6. CREATE NUMERICAL COMPARISON CHARTS
# ------------------------------------------------------------

for feature in numeric_features:

    plot_data = df.copy()

    plot_data[feature] = pd.to_numeric(
        plot_data[feature],
        errors="coerce"
    )

    if plot_data[feature].notna().sum() == 0:
        continue

    plt.figure(figsize=(8, 5))

    for cluster in sorted(df["Cluster"].dropna().unique()):

        values = plot_data.loc[
            plot_data["Cluster"] == cluster,
            feature
        ].dropna()

        plt.hist(
            values,
            bins=30,
            alpha=0.5,
            label=f"Cluster {cluster}"
        )

    plt.title(f"{feature} Distribution by Cluster")
    plt.xlabel(feature)
    plt.ylabel("Number of Respondents")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{feature.lower()}_comparison.png",
        dpi=200
    )

    plt.close()

# ------------------------------------------------------------
# 7. CREATE CLUSTER SIZE CHART
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    cluster_sizes["Cluster"].astype(str),
    cluster_sizes["Respondents"]
)

plt.title("Gower Clustering - Cluster Sizes (k=2)")
plt.xlabel("Cluster")
plt.ylabel("Number of Respondents")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cluster_size_comparison.png",
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# 8. GENERATE SUMMARY REPORT
# ------------------------------------------------------------

report_file = OUTPUT_DIR / "cluster_interpretation_report.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("SIH PROJECT - CLUSTER INTERPRETATION\n")
    report.write("Gower Distance - Two Clusters\n")
    report.write("=" * 55 + "\n\n")

    report.write(f"Total records analysed: {len(df)}\n\n")

    report.write("CLUSTER SIZES\n")
    report.write(cluster_sizes.to_string(index=False))
    report.write("\n\n")

    report.write("NUMERICAL FEATURE PROFILES\n")
    report.write(numeric_profile.round(2).to_string(index=False))
    report.write("\n\n")

    report.write("CATEGORICAL FEATURE PROFILES\n")
    report.write(
        categorical_profile.round(2).to_string(index=False)
    )
    report.write("\n\n")

    report.write(
        "INTERPRETATION NOTE:\n"
        "Compare the feature distributions and category "
        "percentages before assigning meanings to clusters. "
        "Do not assume that a cluster represents a particular "
        "type of respondent without supporting evidence.\n"
    )

# ------------------------------------------------------------
# 9. FINISH
# ------------------------------------------------------------

print("\n" + "=" * 55)
print("CLUSTER INTERPRETATION COMPLETED")
print("=" * 55)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nNext: Share the generated CSV files for interpretation.")
