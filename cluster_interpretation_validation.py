
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

SAMPLE_FILE = BASE_DIR / "data" / "processed" / "distance_sample.csv"
LABEL_FILE = BASE_DIR / "data" / "processed" / "clustering_labels.csv"

OUTPUT_DIR = BASE_DIR / "cluster_validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

sample = pd.read_csv(SAMPLE_FILE, low_memory=False)
labels = pd.read_csv(LABEL_FILE)

if len(sample) != len(labels):
    raise ValueError("Sample and labels row counts do not match.")

if "Gower_k2" not in labels.columns:
    raise ValueError("Gower_k2 column not found.")

data = sample.copy()
data["Cluster"] = labels["Gower_k2"].values

print("=" * 60)
print("CLUSTER INTERPRETATION VALIDATION")
print("=" * 60)

print("\nCluster sizes:")
print(data["Cluster"].value_counts().sort_index())

# =====================================================
# 1. CATEGORICAL FEATURE DISTRIBUTIONS
# =====================================================

categorical_features = [
    "LFSTAT",
    "ACTRD",
    "BAIND",
    "INDCODE",
    "SALARY_MISSING",
    "HRSWK_MISSING",
    "INDCODE_MISSING"
]

categorical_results = []
chi_square_results = []

for feature in categorical_features:

    print("\n" + "-" * 55)
    print("Feature:", feature)

    table = pd.crosstab(
        data["Cluster"],
        data[feature],
        normalize="index"
    ) * 100

    print("\nPercentage distribution:")
    print(table.round(2))

    table.to_csv(
        OUTPUT_DIR / f"{feature}_distribution.csv"
    )

    categorical_results.append(
        table.stack().rename("Percentage").reset_index()
        .assign(Feature=feature)
    )

    # Chi-square association test
    count_table = pd.crosstab(
        data["Cluster"],
        data[feature]
    )

    if count_table.shape[0] > 1 and count_table.shape[1] > 1:

        chi2, p_value, dof, expected = chi2_contingency(
            count_table
        )

        chi_square_results.append({
            "Feature": feature,
            "Chi_Square": chi2,
            "Degrees_of_Freedom": dof,
            "P_Value": p_value
        })

# =====================================================
# 2. NUMERIC FEATURE COMPARISON
# =====================================================

numeric_features = [
    "AGE",
    "SALARY",
    "HRSWK"
]

numeric_results = []

for feature in numeric_features:

    print("\n" + "-" * 55)
    print("Numeric feature:", feature)

    profile = data.groupby("Cluster")[feature].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    print(profile.round(2))

    profile.to_csv(
        OUTPUT_DIR / f"{feature}_profile.csv"
    )

    numeric_results.append(
        profile.reset_index().assign(Feature=feature)
    )

    group0 = data.loc[
        data["Cluster"] == 0, feature
    ].dropna()

    group1 = data.loc[
        data["Cluster"] == 1, feature
    ].dropna()

    if len(group0) > 0 and len(group1) > 0:

        statistic, p_value = mannwhitneyu(
            group0,
            group1,
            alternative="two-sided"
        )

        print("Mann-Whitney U:", statistic)
        print("P-value:", p_value)

# =====================================================
# 3. CHI-SQUARE TEST SUMMARY
# =====================================================

chi_df = pd.DataFrame(chi_square_results)

chi_df.to_csv(
    OUTPUT_DIR / "categorical_association_tests.csv",
    index=False
)

# =====================================================
# 4. SAVE COMBINED REPORT
# =====================================================

if categorical_results:

    categorical_df = pd.concat(
        categorical_results,
        ignore_index=True
    )

    categorical_df.to_csv(
        OUTPUT_DIR / "all_categorical_distributions.csv",
        index=False
    )

if numeric_results:

    numeric_df = pd.concat(
        numeric_results,
        ignore_index=True
    )

    numeric_df.to_csv(
        OUTPUT_DIR / "all_numeric_profiles.csv",
        index=False
    )

# =====================================================
# 5. FINAL SUMMARY
# =====================================================

report_file = OUTPUT_DIR / "cluster_validation_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("CLUSTER INTERPRETATION VALIDATION REPORT\n")
    f.write("=" * 55 + "\n\n")

    f.write("Cluster sizes:\n")
    f.write(
        data["Cluster"].value_counts()
        .sort_index().to_string()
    )

    f.write("\n\nCATEGORICAL ASSOCIATION TESTS\n")
    f.write(chi_df.to_string(index=False))

    f.write("\n\nNUMERIC FEATURE PROFILES\n")

    for feature in numeric_features:

        profile = data.groupby("Cluster")[feature].agg(
            ["count", "mean", "median", "std"]
        )

        f.write(f"\n{feature}\n")
        f.write(profile.to_string())

    f.write(
        "\n\nInterpretation note: Statistical association "
        "does not establish causation. Missing and logically "
        "skipped values must be distinguished from observed values."
    )

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)

print("\nResults saved in:", OUTPUT_DIR)
print("Report:", report_file)
