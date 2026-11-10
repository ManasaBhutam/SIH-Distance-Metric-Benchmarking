
import os
import numpy as np
import pandas as pd

# ======================================================
# 1. PATHS
# ======================================================

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
    "model_c_cluster_inspection"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# 2. LOAD DATA
# ======================================================

print("=" * 65)
print("MODEL C SMALLER CLUSTER INSPECTION")
print("=" * 65)

df = pd.read_csv(DATA_FILE)
labels_df = pd.read_csv(LABEL_FILE)

if len(df) != len(labels_df):
    raise ValueError("Dataset and cluster labels have different row counts.")

if "Model C - Demographic and Education" not in labels_df.columns:
    raise ValueError("Model C cluster labels not found.")

df["Model_C_Cluster"] = labels_df[
    "Model C - Demographic and Education"
].values

# ======================================================
# 3. IDENTIFY SMALLER AND LARGER CLUSTERS
# ======================================================

cluster_counts = df["Model_C_Cluster"].value_counts()

small_cluster = cluster_counts.idxmin()
large_cluster = cluster_counts.idxmax()

small_df = df[df["Model_C_Cluster"] == small_cluster].copy()
large_df = df[df["Model_C_Cluster"] == large_cluster].copy()

print("\nCluster sizes:")
print(cluster_counts.sort_index())

print(f"\nSmaller cluster: {small_cluster}")
print(f"Smaller cluster records: {len(small_df)}")

print(f"\nLarger cluster: {large_cluster}")
print(f"Larger cluster records: {len(large_df)}")

# ======================================================
# 4. AGE COMPARISON
# ======================================================

print("\n" + "=" * 65)
print("AGE COMPARISON")
print("=" * 65)

age_results = []

for cluster_id, group in [
    (small_cluster, small_df),
    (large_cluster, large_df)
]:

    age = pd.to_numeric(group["AGE"], errors="coerce")

    result = {
        "Cluster": cluster_id,
        "Count": len(group),
        "Valid Age Count": int(age.notna().sum()),
        "Missing Age Percentage": round(
            age.isna().mean() * 100, 2
        ),
        "Mean Age": age.mean(),
        "Median Age": age.median(),
        "Minimum Age": age.min(),
        "Maximum Age": age.max(),
        "Age Standard Deviation": age.std()
    }

    age_results.append(result)
    print(result)

age_df = pd.DataFrame(age_results)

age_df.to_csv(
    os.path.join(OUTPUT_DIR, "age_comparison.csv"),
    index=False
)

# ======================================================
# 5. BAIND DISTRIBUTION
# ======================================================

print("\n" + "=" * 65)
print("BAIND DISTRIBUTION")
print("=" * 65)

baind_table = pd.crosstab(
    df["Model_C_Cluster"],
    df["BAIND"].fillna("Missing").astype(str),
    normalize="index"
) * 100

baind_counts = pd.crosstab(
    df["Model_C_Cluster"],
    df["BAIND"].fillna("Missing").astype(str)
)

print("\nBAIND percentages:")
print(baind_table.round(2))

print("\nBAIND counts:")
print(baind_counts)

baind_table.to_csv(
    os.path.join(OUTPUT_DIR, "baind_percentages.csv")
)

baind_counts.to_csv(
    os.path.join(OUTPUT_DIR, "baind_counts.csv")
)

# ======================================================
# 6. N2BAMED DISTRIBUTION
# ======================================================

print("\n" + "=" * 65)
print("N2BAMED DISTRIBUTION")
print("=" * 65)

n2bamed_counts = pd.crosstab(
    df["Model_C_Cluster"],
    df["N2BAMED"].fillna("Missing").astype(str)
)

n2bamed_percentages = pd.crosstab(
    df["Model_C_Cluster"],
    df["N2BAMED"].fillna("Missing").astype(str),
    normalize="index"
) * 100

print("\nN2BAMED counts:")
print(n2bamed_counts.to_string())

print("\nN2BAMED percentages:")
print(n2bamed_percentages.round(2).to_string())

n2bamed_counts.to_csv(
    os.path.join(OUTPUT_DIR, "n2bamed_counts.csv")
)

n2bamed_percentages.to_csv(
    os.path.join(OUTPUT_DIR, "n2bamed_percentages.csv")
)

# ======================================================
# 7. MISSING VALUE COMPARISON
# ======================================================

print("\n" + "=" * 65)
print("MISSING VALUE COMPARISON")
print("=" * 65)

features = ["AGE", "BAIND", "N2BAMED"]

missing_results = []

for cluster_id, group in [
    (small_cluster, small_df),
    (large_cluster, large_df)
]:

    for feature in features:

        missing_count = group[feature].isna().sum()

        missing_results.append({
            "Cluster": cluster_id,
            "Feature": feature,
            "Missing Count": int(missing_count),
            "Missing Percentage": round(
                group[feature].isna().mean() * 100, 2
            )
        })

missing_df = pd.DataFrame(missing_results)

print(missing_df.to_string(index=False))

missing_df.to_csv(
    os.path.join(OUTPUT_DIR, "missing_value_comparison.csv"),
    index=False
)

# ======================================================
# 8. SMALL CLUSTER CATEGORY DETAILS
# ======================================================

print("\n" + "=" * 65)
print("SMALLER CLUSTER CATEGORY DETAILS")
print("=" * 65)

category_results = []

for feature in ["BAIND", "N2BAMED"]:

    counts = (
        small_df[feature]
        .fillna("Missing")
        .astype(str)
        .value_counts(dropna=False)
    )

    for category, count in counts.items():

        category_results.append({
            "Feature": feature,
            "Category": category,
            "Count": int(count),
            "Percentage": round(
                count / len(small_df) * 100, 2
            )
        })

small_category_df = pd.DataFrame(category_results)

print(small_category_df.to_string(index=False))

small_category_df.to_csv(
    os.path.join(OUTPUT_DIR, "small_cluster_categories.csv"),
    index=False
)

# ======================================================
# 9. FINAL REPORT
# ======================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "model_c_inspection_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write("MODEL C SMALLER CLUSTER INSPECTION\n")
    report.write("=" * 60 + "\n\n")

    report.write(f"Smaller cluster ID: {small_cluster}\n")
    report.write(f"Smaller cluster size: {len(small_df)}\n")
    report.write(
        f"Smaller cluster percentage: "
        f"{len(small_df) / len(df) * 100:.2f}%\n\n"
    )

    report.write("AGE COMPARISON\n")
    report.write(age_df.to_string(index=False))

    report.write("\n\nBAIND DISTRIBUTION\n")
    report.write(baind_table.round(2).to_string())

    report.write("\n\nN2BAMED DISTRIBUTION\n")
    report.write(n2bamed_counts.to_string())

    report.write("\n\nMISSING VALUE COMPARISON\n")
    report.write(missing_df.to_string(index=False))

    report.write("\n\nSMALLER CLUSTER CATEGORIES\n")
    report.write(small_category_df.to_string(index=False))

    report.write(
        "\n\nInterpretation caution:\n"
        "Category codes must be checked against the official NSCG "
        "codebook before assigning meanings. Missing values and "
        "special codes must not automatically be treated as real "
        "education categories. This descriptive analysis alone "
        "does not establish why the cluster formed.\n"
    )

print("\n" + "=" * 65)
print("CLUSTER INSPECTION COMPLETED")
print("=" * 65)

print(f"\nResults saved in:\n{OUTPUT_DIR}")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)
