## 🚀 Live Project Demo

**Live Dashboard:** [[text](https://sih-distance-metric-benchmarking-xnfnyz2khtmgvhcy8bdayy.streamlit.app/)]

Built with Python, Streamlit, Pandas, NumPy, Scikit-learn, and Plotly.
# SIH Distance Metric Benchmarking

### Comparative Study of Gower, Euclidean and Cosine Distance Metrics for Similarity-Based Grouping of Mixed-Format Alumni Profile Data

## 📌 Project Overview

This project benchmarks three distance metrics — Gower, Euclidean and Cosine — to evaluate their suitability for similarity-based grouping of mixed-format alumni profile data.

The project uses the National Survey of College Graduates (NSCG) 2023 dataset and applies hierarchical clustering to compare the behavior of different distance metrics.

An interactive Streamlit dashboard presents the experimental results, validation analysis and 3D cluster visualization.

## 🎯 Objectives

1. Consolidate mixed-format education, career and employment information into unified feature representations.
2. Benchmark Gower, Euclidean and Cosine distance metrics.
3. Evaluate clustering performance using silhouette scores and cluster sizes.
4. Validate metric choices through feature sensitivity experiments.
5. Present results through an interactive dashboard.

## 📂 Dataset

**Dataset:** National Survey of College Graduates (NSCG) 2023

- Original records: 94,606
- Original columns: 548
- Experimental sample: 3,000 records
- Selected features: 8

### Selected Features

| Feature | Description |
|---|---|
| AGE | Respondent age |
| SALARY | Salary information |
| HRSWK | Weekly working hours |
| LFSTAT | Labour force status |
| ACTRD | Activity-related information |
| BAIND | Bachelor's degree indicator |
| INDCODE | Industry code |
| N2BAMED | Education-related information |

**Note:** The original dataset and large intermediate files are not included in this GitHub repository. Obtain the dataset separately from its official source and place the required files in the appropriate project directories.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib

## 📊 Methodology

1. Dataset collection
2. Data preprocessing and cleaning
3. Feature selection
4. Distance calculation
5. Hierarchical clustering
6. Silhouette score evaluation
7. Cluster profile analysis
8. Feature sensitivity validation
9. Interactive dashboard development

## 📈 Experimental Results

### K = 2 Clustering Results

| Distance Metric | Silhouette Score | Cluster Sizes |
|---|---:|---|
| Gower | 0.6343 | 2504 / 496 |
| Euclidean | 0.6840 | 2988 / 12 |
| Cosine | 0.5309 | 2504 / 496 |

### Distance Correlation

| Metric Pair | Spearman Correlation |
|---|---:|
| Gower vs Euclidean | 0.801438 |
| Gower vs Cosine | 0.883909 |
| Euclidean vs Cosine | 0.938344 |

### Key Finding

Euclidean achieved the highest silhouette score, but generated a highly imbalanced clustering result.

Gower produced more balanced clusters and is a defensible candidate for mixed-format data.

## 🔬 Final Validation

Feature sensitivity experiments produced the following results:

| Experiment | Silhouette | ARI |
|---|---:|---:|
| Original: All features | 0.634288 | 1.0 |
| Without missingness indicators | 0.396497 | 1.0 |
| Without education features | 0.740803 | 1.0 |
| Without employment features | 0.667237 | 1.0 |

ARI = 1.0 indicates identical cluster assignments compared with the original model in these experiments. These are sensitivity analyses, not independent external validation.

## 🖥️ Dashboard Features

The interactive Streamlit dashboard includes:

- Project Overview
- Distance Comparison
- Correlation Analysis
- Clustering Analysis
- Stability Validation
- Cluster Profiles
- Dataset Explorer
- Final Validation
- SIH Presentation
- Interactive 3D Cluster Visualization
- Final Insights
- Downloads

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ManasaBhutam/SIH-Distance-Metric-Benchmarking.git
```

### 2. Open the project folder

```bash
cd SIH-Distance-Metric-Benchmarking
```

### 3. Install dashboard dependencies

```bash
pip install streamlit pandas numpy plotly
```

Install any additional dependencies required by the project's analysis scripts.

### 4. Prepare the data

Place the required dataset and generated output files in their expected directories:

```text
data/
├── raw/
└── processed/
```

### 5. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## ⚠️ Limitations

- The main clustering experiments use a sample of 3,000 records.
- Clusters strongly reflect employment status and employment-related missingness.
- The results do not establish independent alumni or professional groups.
- Independent validation using manually curated reference groups is still required.

## 🚀 Future Scope

- Validate clusters using manually curated reference groups.
- Explore additional mixed-data clustering algorithms.
- Improve feature selection strategies.
- Evaluate larger samples.
- Improve the interpretation of alumni similarity groups.

## 🏁 Conclusion

This project demonstrates the importance of selecting appropriate distance metrics for mixed-format data.

Gower provides a practical and defensible approach for this dataset. However, metric selection should consider silhouette score, cluster balance, feature sensitivity and independent validation together.

---

**Project:** SIH Distance Metric Benchmarking  
**Dataset:** NSCG 2023  
**Repository:** [GitHub Project](https://github.com/ManasaBhutam/SIH-Distance-Metric-Benchmarking)