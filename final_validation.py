
import pandas as pd
import numpy as np
import gower

from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

NUMERIC = ["AGE", "SALARY", "HRSWK"]

CATEGORICAL = [
    "LFSTAT", "ACTRD", "BAIND", "INDCODE", "N2BAMED"
]

MISSINGNESS = [
    "SALARY_MISSING",
    "HRSWK_MISSING",
    "INDCODE_MISSING"
]

# --------------------------------------------------
# 2. LOAD EXISTING PROJECT RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL SIH PROJECT VALIDATION")
print("=" * 65)

sample = pd.read_csv(OUTPUT / "distance_sample.csv")
labels_df = pd.read_csv(OUTPUT / "clustering_labels.csv")

original_labels = labels_df["Gower_k2"].to_numpy()

print("\nDataset sample:", sample.shape)
print("Original clusters loaded successfully.")

# --------------------------------------------------
# 3. EDUCATION DATA VALIDATION
# --------------------------------------------------

print("\n[1/6] EDUCATION VALIDATION")

cleaned = pd.read_csv(OUTPUT / "cleaned_dataset.csv")

education_results = {
    "Total records": len(cleaned),
    "BAIND missing": int(cleaned["BAIND"].isna().sum()),
    "N2BAMED missing": int(cleaned["N2BAMED"].isna().sum()),
    "NOT_APPLICABLE": int(
        (cleaned["N2BAMED"] == "NOT_APPLICABLE").sum()
    ),
    "UNKNOWN_MISSING": int(
        (cleaned["N2BAMED"] == "UNKNOWN_MISSING").sum()
    ),
    "Remaining 999989": int(
        (cleaned["N2BAMED"].astype(str) == "999989").sum()
    ),
    "Remaining 999999": int(
        (cleaned["N2BAMED"].astype(str) == "999999").sum()
    )
}

for key, value in education_results.items():
    print(f"{key}: {value}")

# --------------------------------------------------
# 4. GOWER CLUSTERING FUNCTION
# --------------------------------------------------

def run_model(name, categorical_columns):

    print(f"\nRunning: {name}")

    numeric_data = sample[NUMERIC].copy()

    categorical_data = sample[
        categorical_columns
    ].copy()

    combined = pd.concat(
        [
            numeric_data.reset_index(drop=True),
            categorical_data.reset_index(drop=True)
        ],
        axis=1
    )

    categorical_flags = np.array(
        [False] * len(NUMERIC)
        + [True] * len(categorical_columns)
    )

    distance_matrix = gower.gower_matrix(
        combined,
        cat_features=categorical_flags
    )

    distance_matrix = (
        distance_matrix + distance_matrix.T
    ) / 2

    np.fill_diagonal(distance_matrix, 0)

    model = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average"
    )

    labels = model.fit_predict(distance_matrix)

    silhouette = silhouette_score(
        distance_matrix,
        labels,
        metric="precomputed"
    )

    sizes = pd.Series(labels).value_counts()

    ari = adjusted_rand_score(
        original_labels,
        labels
    )

    result = {
        "Model": name,
        "Silhouette": silhouette,
        "Smallest Cluster": int(sizes.min()),
        "Largest Cluster": int(sizes.max()),
        "ARI vs Original": ari
    }

    print("Silhouette:", round(silhouette, 4))
    print("Smallest cluster:", sizes.min())
    print("Largest cluster:", sizes.max())
    print("ARI:", round(ari, 4))

    del distance_matrix

    return result, labels


# --------------------------------------------------
# 5. RUN ALL VALIDATION EXPERIMENTS
# --------------------------------------------------

print("\n[2/6] RUNNING FEATURE VALIDATION")

experiments = []

# A. Original model
result, baseline_labels = run_model(
    "Original: All features",
    CATEGORICAL + MISSINGNESS
)
experiments.append(result)

# B. Remove missingness indicators
result, labels_no_missing = run_model(
    "Without missingness indicators",
    CATEGORICAL
)
experiments.append(result)

# C. Remove education features
without_education = [
    col for col in CATEGORICAL
    if col not in ["BAIND", "N2BAMED"]
]

result, labels_no_education = run_model(
    "Without education features",
    without_education + MISSINGNESS
)
experiments.append(result)

# D. Remove employment features
without_employment = [
    col for col in CATEGORICAL
    if col not in ["LFSTAT", "ACTRD", "INDCODE"]
]

result, labels_no_employment = run_model(
    "Without employment features",
    without_employment + MISSINGNESS
)
experiments.append(result)

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

print("\n[3/6] SAVING VALIDATION RESULTS")

results_df = pd.DataFrame(experiments)

results_df.to_csv(
    OUTPUT / "final_validation_results.csv",
    index=False
)

labels_output = pd.DataFrame({
    "Original_Gower": original_labels,
    "Without_Missingness": labels_no_missing,
    "Without_Education": labels_no_education,
    "Without_Employment": labels_no_employment
})

labels_output.to_csv(
    OUTPUT / "final_validation_labels.csv",
    index=False
)

# --------------------------------------------------
# 7. EMPLOYMENT CLUSTER VALIDATION
# --------------------------------------------------

print("\n[4/6] EMPLOYMENT VALIDATION")

employment_table = pd.crosstab(
    pd.Series(original_labels, name="Cluster"),
    sample["LFSTAT"]
)

employment_table.to_csv(
    OUTPUT / "final_employment_validation.csv"
)

print(employment_table)

# --------------------------------------------------
# 8. MISSINGNESS VALIDATION
# --------------------------------------------------

print("\n[5/6] MISSINGNESS VALIDATION")

missing_table = sample[
    MISSINGNESS
].copy()

missing_table["Cluster"] = original_labels

missing_percent = (
    missing_table
    .groupby("Cluster")[MISSINGNESS]
    .agg(lambda x: (
        x.astype(str).str.lower() == "true"
    ).mean() * 100)
)

missing_percent.to_csv(
    OUTPUT / "final_missingness_validation.csv"
)

print(missing_percent.round(2))

# --------------------------------------------------
# 9. GENERATE FINAL REPORT
# --------------------------------------------------

print("\n[6/6] GENERATING FINAL REPORT")

report_path = OUTPUT / "final_validation_report.txt"

with open(report_path, "w", encoding="utf-8") as f:

    f.write("SIH PROJECT - FINAL VALIDATION REPORT\n")
    f.write("=" * 55 + "\n\n")

    f.write("DATASET INFORMATION\n")
    f.write(f"Sample size: {len(sample)}\n\n")

    f.write("EDUCATION VALIDATION\n")
    for key, value in education_results.items():
        f.write(f"{key}: {value}\n")

    f.write("\nFEATURE INFLUENCE COMPARISON\n")
    f.write(results_df.to_string(index=False))

    f.write("\n\nEMPLOYMENT VALIDATION\n")
    f.write(employment_table.to_string())

    f.write("\n\nMISSINGNESS VALIDATION (%)\n")
    f.write(missing_percent.round(2).to_string())

    f.write(
        "\n\nINTERPRETATION NOTES\n"
        "1. Higher silhouette does not automatically mean "
        "better clustering.\n"
        "2. Cluster balance and ARI must also be considered.\n"
        "3. ARI measures agreement with the original labels, "
        "not external ground-truth accuracy.\n"
        "4. Feature-removal experiments indicate sensitivity "
        "to the selected variables.\n"
        "5. Results are exploratory and based on the "
        "existing 3000-record sample.\n"
    )

print("\n" + "=" * 65)
print("FINAL VALIDATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nValidation results:")
print(results_df.to_string(index=False))

print("\nSaved files:")
print("- final_validation_results.csv")
print("- final_validation_labels.csv")
print("- final_employment_validation.csv")
print("- final_missingness_validation.csv")
print("- final_validation_report.txt")

print("\nAll files saved in:", OUTPUT)
