
from pathlib import Path

import numpy as np
import pandas as pd
import gower

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/raw/epcg23.csv")
OUTPUT_DIR = Path("data/processed")

SAMPLE_SIZE = 3000
RANDOM_STATE = 42

NUMERIC_COLS = ["AGE", "SALARY", "HRSWK"]

CATEGORICAL_COLS = [
    "LFSTAT",
    "ACTRD",
    "BAIND",
    "INDCODE",
    "N2BAMED",
]

REQUIRED_COLS = NUMERIC_COLS + CATEGORICAL_COLS


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("\n" + "=" * 60)
print("SIH: GOWER, EUCLIDEAN AND COSINE DISTANCE ANALYSIS")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE.resolve()}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, low_memory=False)

missing_columns = [
    col for col in REQUIRED_COLS if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("\nOriginal dataset shape:", df.shape)

# Work only on a copy. Never overwrite the raw CSV.
data = df[REQUIRED_COLS].copy()


# ============================================================
# 3. CLEAN SPECIAL VALUES IN THE ANALYSIS COPY
# ============================================================

# Confirmed logical-skip codes from the investigated dataset.
# These are not real salary, industry, or hours measurements.

data["SALARY"] = pd.to_numeric(
    data["SALARY"], errors="coerce"
).replace(9999998, np.nan)

data["INDCODE"] = data["INDCODE"].replace(9998, np.nan)

data["HRSWK"] = pd.to_numeric(
    data["HRSWK"], errors="coerce"
).replace(98, np.nan)

data["AGE"] = pd.to_numeric(
    data["AGE"], errors="coerce"
)

# Keep education indicator categories unchanged.
# Convert the documented special field-of-study code into
# an explicit unavailable category, not a real field of study.
data["N2BAMED"] = data["N2BAMED"].replace(
    999989, np.nan
)

print("\nRows retained:", len(data))
print("Special codes handled in analysis copy.")


# ============================================================
# 4. CREATE MISSINGNESS INDICATORS
# ============================================================

# Missingness may itself carry information, especially when
# respondents are not employed. Preserve that information.

data["SALARY_MISSING"] = data["SALARY"].isna().astype(str)
data["HRSWK_MISSING"] = data["HRSWK"].isna().astype(str)
data["INDCODE_MISSING"] = data["INDCODE"].isna().astype(str)

MISSINGNESS_COLS = [
    "SALARY_MISSING",
    "HRSWK_MISSING",
    "INDCODE_MISSING",
]

ALL_CATEGORICAL = CATEGORICAL_COLS + MISSINGNESS_COLS


# ============================================================
# 5. TAKE A REPRODUCIBLE SAMPLE
# ============================================================

n = min(SAMPLE_SIZE, len(data))

sample = data.sample(
    n=n,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nSample size:", len(sample))
print("Numeric features:", NUMERIC_COLS)
print("Categorical features:", ALL_CATEGORICAL)


# ============================================================
# 6. PREPARE NUMERIC FEATURES
# ============================================================

numeric_data = sample[NUMERIC_COLS].copy()

# Median imputation is applied to the sample for this
# preliminary comparison. Missingness indicators are retained.
for col in NUMERIC_COLS:
    median_value = numeric_data[col].median()

    if pd.isna(median_value):
        raise ValueError(
            f"Cannot impute {col}: all sampled values are missing."
        )

    numeric_data[col] = numeric_data[col].fillna(median_value)

# Scale numerical features for Euclidean and Cosine.
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_data)


# ============================================================
# 7. PREPARE CATEGORICAL FEATURES
# ============================================================

categorical_data = sample[ALL_CATEGORICAL].copy()

for col in CATEGORICAL_COLS:
    categorical_data[col] = (
        categorical_data[col]
        .astype("string")
        .fillna("Unknown")
        .astype(str)
    )

for col in MISSINGNESS_COLS:
    categorical_data[col] = categorical_data[col].astype(str)


# ============================================================
# 8. CALCULATE GOWER DISTANCE
# ============================================================

print("\nCalculating Gower distance...")

gower_input = pd.concat(
    [numeric_data.reset_index(drop=True),
     categorical_data.reset_index(drop=True)],
    axis=1,
)

# Numeric features first, then categorical features.
cat_features = np.array(
    [False] * len(NUMERIC_COLS)
    + [True] * len(ALL_CATEGORICAL)
)

gower_matrix = gower.gower_matrix(
    gower_input,
    cat_features=cat_features,
)

# Store only unique pairs, not the full square matrix.
upper_triangle = np.triu_indices_from(
    gower_matrix, k=1
)

gower_dist = gower_matrix[upper_triangle]

del gower_matrix


# ============================================================
# 9. ENCODE FEATURES FOR EUCLIDEAN AND COSINE
# ============================================================

print("Encoding categorical features...")

encoded_categorical = pd.get_dummies(
    categorical_data,
    dtype=float
).to_numpy()

X = np.hstack([
    scaled_numeric,
    encoded_categorical,
])

print("Encoded feature matrix shape:", X.shape)


# ============================================================
# 10. CALCULATE EUCLIDEAN DISTANCE
# ============================================================

print("Calculating Euclidean distance...")

euclidean_dist = pdist(
    X,
    metric="euclidean",
)


# ============================================================
# 11. CALCULATE COSINE DISTANCE
# ============================================================

print("Calculating Cosine distance...")

cosine_dist = pdist(
    X,
    metric="cosine",
)

# Handle any undefined cosine distances if a pair contains
# zero vectors after encoding.
cosine_dist = np.nan_to_num(
    cosine_dist,
    nan=0.0,
    posinf=1.0,
    neginf=0.0,
)


# ============================================================
# 12. SAVE DISTANCES AND SAMPLE
# ============================================================

distance_file = OUTPUT_DIR / "distance_results.npz"
sample_file = OUTPUT_DIR / "distance_sample.csv"

np.savez_compressed(
    distance_file,
    gower=gower_dist,
    euclidean=euclidean_dist,
    cosine=cosine_dist,
)

# Save features used for the comparison.
sample_output = pd.concat(
    [
        sample[REQUIRED_COLS].reset_index(drop=True),
        categorical_data[MISSINGNESS_COLS].reset_index(drop=True),
    ],
    axis=1,
)

sample_output.to_csv(sample_file, index=False)

print("\nDistance results saved:", distance_file)
print("Sample saved:", sample_file)


# ============================================================
# 13. DISTANCE SUMMARY STATISTICS
# ============================================================

distances = {
    "Gower": gower_dist,
    "Euclidean": euclidean_dist,
    "Cosine": cosine_dist,
}

summary_rows = []

print("\n" + "=" * 60)
print("DISTANCE SUMMARY")
print("=" * 60)

for name, values in distances.items():
    row = {
        "Metric": name,
        "Min": float(np.min(values)),
        "Mean": float(np.mean(values)),
        "Median": float(np.median(values)),
        "Max": float(np.max(values)),
        "Std": float(np.std(values)),
    }

    summary_rows.append(row)

    print(f"\n{name}")
    for key, value in row.items():
        if key != "Metric":
            print(f"  {key}: {value:.6f}")

summary_df = pd.DataFrame(summary_rows)

summary_file = OUTPUT_DIR / "distance_summary.csv"
summary_df.to_csv(summary_file, index=False)


# ============================================================
# 14. COMPARE PAIRWISE DISTANCE RANKINGS
# ============================================================

print("\n" + "=" * 60)
print("SPEARMAN RANK CORRELATIONS")
print("=" * 60)

correlation_rows = []
metric_names = list(distances.keys())

for i in range(len(metric_names)):
    for j in range(i + 1, len(metric_names)):
        name_a = metric_names[i]
        name_b = metric_names[j]

        corr, _ = spearmanr(
            distances[name_a],
            distances[name_b],
        )

        correlation_rows.append({
            "Metric A": name_a,
            "Metric B": name_b,
            "Spearman Correlation": float(corr),
        })

        print(
            f"{name_a} vs {name_b}: {corr:.6f}"
        )

correlation_df = pd.DataFrame(correlation_rows)

correlation_file = OUTPUT_DIR / "distance_correlations.csv"
correlation_df.to_csv(correlation_file, index=False)


# ============================================================
# 15. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Original respondents:", len(df))
print("Sample respondents:", n)
print("Unique respondent pairs:", len(euclidean_dist))
print("Number of input features:", len(REQUIRED_COLS))
print("\nGenerated files:")

for file in [
    distance_file,
    sample_file,
    summary_file,
    correlation_file,
]:
    print(" -", file)

print("\nThe original CSV was not modified.")
print("Review codebook assumptions before final reporting.")
