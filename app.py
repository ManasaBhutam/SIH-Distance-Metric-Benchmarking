# ============================================================
# SIH INTERACTIVE DASHBOARD
# GOWER vs EUCLIDEAN vs COSINE DISTANCE
# ============================================================

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SIH Distance Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
PLOT_DIR = DATA_DIR / "plots"


# ============================================================
# 2. DASHBOARD DESIGN
# ============================================================


st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #101c32 50%,
        #172b46 100%
    );
    color: #f1f5f9;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */
h1, h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 750 !important;
}

/* Paragraphs */
p, label, span {
    color: #cbd5e1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0b1424,
        #14243b
    );
    border-right: 1px solid #263c59;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(36, 59, 92, 0.95),
        rgba(17, 32, 54, 0.95)
    );
    border: 1px solid rgba(96, 165, 250, 0.3);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

/* Metric hover effect */
div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.2);
}

/* Metric values */
div[data-testid="stMetricValue"] {
    color: #60a5fa;
    font-weight: 800;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 650;
    transition: 0.25s ease;
}

/* Button hover */
.stButton button:hover,
.stDownloadButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
    color: white;
    border: none;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #334b69;
    border-radius: 12px;
    overflow: hidden;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #172b46;
    border-color: #3b82f6;
    border-radius: 9px;
}

/* Input fields */
input, textarea {
    background-color: #172b46 !important;
    color: white !important;
}

/* Section separators */
hr {
    border-color: #334b69;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    padding: 25px;
    font-size: 13px;
}

/* Sidebar scrolling */
section[data-testid="stSidebar"] {
    min-width: 270px;
}

section[data-testid="stSidebar"] > div {
    height: 100vh;
    overflow-y: auto;
}

/* Compact sidebar navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 2px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding-top: 3px;
    padding-bottom: 3px;
    font-size: 13px;
}

/* Sidebar information */
section[data-testid="stSidebar"] h3 {
    margin-top: 12px;
    font-size: 17px;
}

/* Main heading */
.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #60a5fa;
    margin-top: 15px;
    margin-bottom: 10px;
    line-height: 1.35;
    overflow-wrap: anywhere;
}

/* Main page spacing */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Make all sidebar text visible */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #f1f5f9 !important;
}

/* Bright bullet points */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li::marker {
    color: #38bdf8 !important;
}

/* Highlight section headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #60a5fa !important;
}

/* Footer visibility */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)



# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    files = {
        "summary": "distance_summary.csv",
        "correlations": "distance_correlations.csv",
        "evaluation": "clustering_evaluation.csv",
        "profiles": "cluster_profiles.csv",
        "sample": "distance_sample.csv",
        "labels": "clustering_labels.csv",
        "cleaned": "cleaned_dataset.csv",
        "stability": "clustering_stability_results.csv",
        "validation": "final_validation_results.csv",
        "validation_labels": "final_validation_labels.csv",
        "employment_validation": "final_employment_validation.csv",
        "missingness_validation": "final_missingness_validation.csv",
    }

    data = {}

    for key, filename in files.items():

        path = DATA_DIR / filename

        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = None

    return data


data = load_data()


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

required = [
    "summary",
    "correlations",
    "evaluation",
    "profiles",
    "sample",
    "labels"
]

missing_files = [
    name for name in required
    if data[name] is None
]

if missing_files:

    st.error(
        "Some project output files are missing: "
        + ", ".join(missing_files)
    )

    st.info(
        "Run python sih_complete_project.py first."
    )

    st.stop()


summary = data["summary"]
correlations = data["correlations"]
evaluation = data["evaluation"]
profiles = data["profiles"]
sample = data["sample"]
labels = data["labels"]


# ============================================================
# 5. SIDEBAR
# ============================================================


st.sidebar.title("📊 SIH Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Distance Comparison",
        "Correlation Analysis",
        "Clustering Analysis",
        "Stability Validation",
        "Cluster Profiles",
        "Dataset Explorer",
        "Final Validation",
        "SIH Presentation",
        "3D Cluster Visualization",
        "Final Insights",
        "Downloads"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### 📌 Project Information")

st.sidebar.markdown("""
**Dataset:** NSCG 2023

**Total Respondents:** 94,606

**Sample Size:** 3,000

**Distance Metrics:**
- Gower
- Euclidean
- Cosine

**Clustering:** Hierarchical

**Features:**
- AGE
- SALARY
- HRSWK
- LFSTAT
- ACTRD
- BAIND
- INDCODE
- N2BAMED
""")

st.sidebar.divider()

st.sidebar.caption(
    "SIH Distance Metric Comparison Project"
)

# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    for column in candidates:
        if column in df.columns:
            return column

    return None


def display_metric_table(df):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def csv_download(df, filename, label):

    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 7. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">'
        'SIH Distance Metrics Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Comparative analysis of Gower, Euclidean '
        'and Cosine distance metrics'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Respondents",
        "94,606"
    )

    c2.metric(
        "Sample Size",
        f"{len(sample):,}"
    )

    c3.metric(
        "Distance Metrics",
        "3"
    )

    c4.metric(
        "Unique Pairs",
        f"{len(sample) * (len(sample)-1)//2:,}"
    )

    st.markdown(
        '<div class="section-title">'
        'Project Introduction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    This project compares three distance metrics:
    Gower, Euclidean and Cosine.

    The objective is to understand how different
    distance measurements represent similarities
    and differences between respondents in a mixed-type
    dataset containing numerical and categorical variables.

    Hierarchical clustering is used to evaluate the
    grouping behavior of the distance metrics.
    """)

    st.markdown("### Selected Features")

    feature_list = [
        "AGE",
        "SALARY",
        "HRSWK",
        "LFSTAT",
        "ACTRD",
        "BAIND",
        "INDCODE",
        "N2BAMED"
    ]

    st.write(", ".join(feature_list))

    st.markdown("### Distance Metric Summary")

    display_metric_table(summary)

    st.markdown("### Clustering Comparison")

    fig = px.line(
        evaluation,
        x="Clusters",
        y="Silhouette",
        color="Metric",
        markers=True,
        title="Silhouette Score by Number of Clusters"
    )

    fig.update_layout(
        xaxis_title="Number of Clusters",
        yaxis_title="Silhouette Score",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # ============================================================
# 8. DISTANCE COMPARISON PAGE
# ============================================================

elif page == "Distance Comparison":

    st.title("📏 Distance Metric Comparison")

    st.write(
        "Compare the statistical properties of "
        "Gower, Euclidean and Cosine distances."
    )

    display_metric_table(summary)

    st.markdown("### Mean Distance Comparison")

    metric_col = find_column(
        summary,
        ["Metric", "Distance_Metric", "Distance"]
    )

    mean_col = find_column(
        summary,
        ["Mean", "mean", "Mean_Distance"]
    )

    if metric_col and mean_col:

        fig = px.bar(
            summary,
            x=metric_col,
            y=mean_col,
            color=metric_col,
            title="Mean Distance by Metric",
            text_auto=".4f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Distance Distribution")

    if "Metric" in summary.columns:

        for metric in summary["Metric"].unique():

            row = summary[
                summary["Metric"] == metric
            ]

            st.markdown(f"**{metric}**")

            st.dataframe(
                row,
                use_container_width=True,
                hide_index=True
            )

    st.info(
        "The three metrics use different distance scales. "
        "Their raw mean values should not be interpreted "
        "as directly comparable measures of performance."
    )


# ============================================================
# 9. CORRELATION ANALYSIS PAGE
# ============================================================

elif page == "Correlation Analysis":

    st.title("🔗 Distance Correlation Analysis")

    st.write(
        "This section shows how similarly the three "
        "distance metrics rank respondent pairs."
    )

    display_metric_table(correlations)

    st.markdown("### Correlation Visualization")

    correlation_col = find_column(
        correlations,
        [
            "Spearman_Correlation",
            "Correlation",
            "Spearman",
            "Correlation_Value"
        ]
    )

    if correlation_col:

        fig = px.bar(
            correlations,
            y=correlation_col,
            title="Spearman Rank Correlation",
            text_auto=".4f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Interpretation")

    st.write("""
    - A correlation closer to +1 indicates stronger agreement
      between the distance rankings.
    - A lower correlation indicates differences in how
      distance metrics rank respondent pairs.
    - Correlation measures agreement, not clustering quality.
    """)


# ============================================================
# 10. CLUSTERING ANALYSIS PAGE
# ============================================================

elif page == "Clustering Analysis":

    st.title("📊 Clustering Analysis")

    st.write(
        "Evaluate clustering performance using "
        "silhouette scores and cluster sizes."
    )

    display_metric_table(evaluation)

    st.markdown("### Silhouette Score Comparison")

    metric_col = find_column(
        evaluation,
        ["Metric", "Distance_Metric"]
    )

    cluster_col = find_column(
        evaluation,
        ["Clusters", "K", "n_clusters"]
    )

    silhouette_col = find_column(
        evaluation,
        ["Silhouette", "Silhouette_Score", "silhouette"]
    )

    if metric_col and cluster_col and silhouette_col:

        fig = px.line(
            evaluation,
            x=cluster_col,
            y=silhouette_col,
            color=metric_col,
            markers=True,
            title="Silhouette Score by Metric and Cluster Count"
        )

        fig.update_layout(
            xaxis_title="Number of Clusters",
            yaxis_title="Silhouette Score",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Could not identify all required columns "
            "for the silhouette chart. Check the evaluation CSV."
        )

    st.markdown("### Select a Distance Metric")

    if metric_col:

        selected_metric = st.selectbox(
            "Choose metric",
            evaluation[metric_col].dropna().unique()
        )

        filtered = evaluation[
            evaluation[metric_col] == selected_metric
        ]

        st.markdown(
            f"**Results for {selected_metric}**"
        )

        display_metric_table(filtered)

    st.info("""
    A higher silhouette score generally indicates better
    separation between clusters.

    However, cluster sizes must also be considered.
    A high score with a very small cluster may indicate
    an imbalanced grouping.
    """)
    # ============================================================
# 11. STABILITY VALIDATION PAGE
# ============================================================

elif page == "Stability Validation":

    st.title("🔬 Clustering Stability Validation")

    stability = data["stability"]

    if stability is None:

        st.warning(
            "Stability validation file was not found."
        )

    else:

        st.write("""
        This section displays the stability results
        generated during repeated clustering experiments.
        """)

        display_metric_table(stability)

        numeric_columns = stability.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Select numeric result",
                numeric_columns
            )

            fig = px.histogram(
                stability,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# 12. CLUSTER PROFILES PAGE
# ============================================================

elif page == "Cluster Profiles":

    st.title("👥 Cluster Profiles")

    st.write("""
    Explore the characteristics of respondents
    belonging to each cluster.
    """)

    display_metric_table(profiles)

    st.markdown("### Explore Profile Columns")

    selected_column = st.selectbox(
        "Select a profile column",
        profiles.columns
    )

    st.dataframe(
        profiles[[selected_column]],
        use_container_width=True
    )

    st.info("""
    Cluster profiles summarize the features of each group.
    Interpret imputed values and missingness-related
    features carefully.
    """)


# ============================================================
# 13. DATASET EXPLORER PAGE
# ============================================================

elif page == "Dataset Explorer":

    st.title("📂 Dataset Explorer")

    cleaned = data["cleaned"]

    if cleaned is None:

        st.warning(
            "Cleaned dataset file was not found."
        )

    else:

        st.write(
            "Explore the cleaned dataset used in the project."
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Total Rows",
            f"{len(cleaned):,}"
        )

        c2.metric(
            "Total Columns",
            f"{len(cleaned.columns):,}"
        )

        st.markdown("### Preview Dataset")

        rows = st.slider(
            "Number of rows to display",
            min_value=5,
            max_value=100,
            value=10
        )

        st.dataframe(
            cleaned.head(rows),
            use_container_width=True
        )

        st.markdown("### Column Information")

        column_info = pd.DataFrame({
            "Column": cleaned.columns,
            "Data Type": cleaned.dtypes.astype(str).values,
            "Missing Values": cleaned.isnull().sum().values
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

        st.markdown("### Select a Column")

        selected_column = st.selectbox(
            "Choose column",
            cleaned.columns
        )

        st.write(
            cleaned[selected_column]
            .value_counts(dropna=False)
            .head(20)
        )


# ============================================================
# 14. FINAL VALIDATION PAGE
# ============================================================

elif page == "Final Validation":

    st.title("✅ Final Validation")

    st.write("""
    This section evaluates how feature selection,
    employment information and missingness indicators
    influence the Gower clustering results.
    """)

    validation = data["validation"]

    if validation is None:

        st.warning(
            "Final validation results were not found. "
            "Check final_validation_results.csv "
            "inside data/processed."
        )

    else:

        st.markdown("### Feature Influence Analysis")

        display_metric_table(validation)

        metric_col = find_column(
            validation,
            ["Model", "Feature_Set", "Experiment", "Test"]
        )

        silhouette_col = find_column(
            validation,
            ["Silhouette", "Silhouette_Score", "silhouette"]
        )

        if metric_col and silhouette_col:

            fig = px.bar(
                validation,
                x=metric_col,
                y=silhouette_col,
                color=metric_col,
                text_auto=".4f",
                title="Silhouette Score Across Validation Experiments"
            )

            fig.update_layout(
                xaxis_title="Validation Experiment",
                yaxis_title="Silhouette Score",
                xaxis_tickangle=-25
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("### Validation Interpretation")

        st.write("""
        The validation experiments compare the original
        Gower model with models created after removing
        selected feature groups.

        These experiments help identify whether
        missingness, education and employment features
        influence clustering quality.

        These are sensitivity analyses, not causal
        feature-importance measurements.
        """)

    st.divider()

    # --------------------------------------------------------
    # EMPLOYMENT VALIDATION
    # --------------------------------------------------------

    st.markdown("### Employment Validation")

    employment = data["employment_validation"]

    if employment is not None:

        display_metric_table(employment)

    else:

        st.warning(
            "Employment validation CSV was not found."
        )

    st.write("""
    The employment validation results help examine
    how employment status is distributed across
    the generated clusters.
    """)

    st.divider()

    # --------------------------------------------------------
    # MISSINGNESS VALIDATION
    # --------------------------------------------------------

    st.markdown("### Missingness Validation")

    missingness = data["missingness_validation"]

    if missingness is not None:

        display_metric_table(missingness)

    else:

        st.warning(
            "Missingness validation CSV was not found."
        )

    st.write("""
    Missingness validation examines whether
    missing-value indicators are associated with
    the clustering structure.
    """)

    st.divider()

    # --------------------------------------------------------
    # VALIDATION LABELS
    # --------------------------------------------------------

    st.markdown("### Validation Cluster Labels")

    validation_labels = data["validation_labels"]

    if validation_labels is not None:

        st.dataframe(
            validation_labels.head(100),
            use_container_width=True
        )

        st.caption(
            "Showing the first 100 validation records."
        )

    else:

        st.warning(
            "Validation labels file was not found."
        )

# ============================================================
# SIH PRESENTATION PAGE
# ============================================================

elif page == "SIH Presentation":

    st.title("🏆 SIH Project Presentation")

    st.write(
        "Comparative Study of Distance Metrics "
        "for Mixed-Format Alumni Profile Data"
    )

    st.divider()

    # PROJECT OBJECTIVES

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    1. Consolidate mixed-format alumni information
       into unified feature representations.

    2. Benchmark Gower, Euclidean and Cosine
       distance metrics.

    3. Evaluate clustering results using
       silhouette scores and cluster sizes.

    4. Validate metric behavior using
       feature-removal and sensitivity experiments.
    """)

    st.divider()

    # DISTANCE COMPARISON

    st.subheader("📊 Distance Metric Comparison")

    metric_names = [
        "Gower",
        "Euclidean",
        "Cosine"
    ]

    mean_distances = [
        0.386461,
        3.495766,
        0.586027
    ]

    fig = px.bar(
        x=metric_names,
        y=mean_distances,
        color=metric_names,
        text_auto=".4f",
        title="Mean Distance Values"
    )

    fig.update_layout(
        xaxis_title="Distance Metric",
        yaxis_title="Mean Distance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Raw distance values have different scales "
        "and should not be directly compared as performance scores."
    )

    st.divider()

    # SILHOUETTE COMPARISON

    st.subheader("📈 Clustering Performance (K = 2)")

    silhouette_values = [
        0.634288,
        0.684027,
        0.530943
    ]

    cluster_sizes = [
        "2504 / 496",
        "2988 / 12",
        "2504 / 496"
    ]

    result_df = pd.DataFrame({
        "Metric": metric_names,
        "Silhouette Score": silhouette_values,
        "Cluster Sizes": cluster_sizes
    })

    c1, c2, c3 = st.columns(3)

    c1.metric("Gower", "0.6343")
    c2.metric("Euclidean", "0.6840")
    c3.metric("Cosine", "0.5309")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        result_df,
        x="Metric",
        y="Silhouette Score",
        color="Metric",
        text_auto=".4f",
        title="Silhouette Score Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # FINAL RECOMMENDATION

    st.subheader("🏆 Project Recommendation")

    st.markdown("""
    **Gower Distance is a defensible candidate
    for mixed-format data.**

    Reasons:

    - It supports numerical and categorical features.
    - It produced more balanced clusters than Euclidean
      in the K = 2 experiment.
    - Its silhouette score was 0.6343.

    Euclidean obtained a higher silhouette score of
    0.6840, but its cluster sizes were highly imbalanced
    (2988 and 12).

    Therefore, silhouette score alone is not sufficient
    to select the best metric.
    """)

    st.warning("""
    **Important limitation:** The Gower clusters strongly
    reflected employment status and employment-related
    missingness. These clusters should not automatically
    be interpreted as independent alumni groups.
    """)

    st.divider()

    # CONCLUSION

    st.subheader("📝 Final Conclusion")

    st.write("""
    This project demonstrates that distance metric
    selection significantly affects similarity-based
    grouping.

    Gower provides a practical approach for mixed-format
    data, while Euclidean and Cosine provide useful
    comparison baselines.

    Further validation with independently curated
    reference groups is needed before treating the
    clusters as meaningful alumni segments.
    """)

    st.success(
        "Project analysis and final validation completed."
    )
    
# ============================================================
# 3D CLUSTER VISUALIZATION
# ============================================================

elif page == "3D Cluster Visualization":

    st.title("🌐 Interactive 3D Cluster Visualization")

    st.write(
        "Explore respondent groups using AGE, SALARY "
        "and HRSWK with interactive 3D visualization."
    )

    # Load sampled features
    sample_path = DATA_DIR / "distance_sample.csv"

    if sample_path.exists():

        sample_df = pd.read_csv(sample_path)

        # Select clustering metric
        metric = st.selectbox(
            "Select Distance Metric",
            ["Gower", "Euclidean", "Cosine"]
        )

        k = st.selectbox(
            "Select Number of Clusters",
            [2, 3, 4, 5, 6]
        )

        cluster_column = f"{metric}_k{k}"

        if cluster_column in labels.columns:

            if len(sample_df) == len(labels):

                plot_df = sample_df[
                    ["AGE", "SALARY", "HRSWK"]
                ].copy()

                plot_df["Cluster"] = labels[
                    cluster_column
                ].astype(str).values

                # Interactive 3D scatter plot
                fig = px.scatter_3d(
                    plot_df,
                    x="AGE",
                    y="SALARY",
                    z="HRSWK",
                    color="Cluster",
                    title=f"{metric} Clustering (K={k})",
                    opacity=0.75,
                    hover_data=["Cluster"]
                )

                fig.update_traces(
                    marker=dict(size=3)
                )

                fig.update_layout(
                    height=700,
                    template="plotly_dark",
                    scene=dict(
                        xaxis_title="AGE",
                        yaxis_title="SALARY",
                        zaxis_title="HRSWK"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Cluster statistics
                st.subheader("Cluster Distribution")

                counts = plot_df["Cluster"].value_counts()

                distribution = pd.DataFrame({
                    "Cluster": counts.index,
                    "Respondents": counts.values
                })

                st.dataframe(
                    distribution,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    distribution.set_index("Cluster")
                )

                st.success(
                    f"{metric} clustering with K={k} "
                    "visualized successfully."
                )

            else:

                st.error(
                    "Sample and cluster label row counts do not match."
                )

        else:

            st.error(
                f"Cluster column {cluster_column} not found."
            )

    else:

        st.error(
            "distance_sample.csv was not found "
            "inside data/processed."
        )
        
# ============================================================
# FINAL INSIGHTS AND RESEARCH FINDINGS
# ============================================================

elif page == "Final Insights":

    st.title("🏆 Final Insights & Research Findings")

    st.write(
        "Key findings from benchmarking Gower, "
        "Euclidean and Cosine distance metrics."
    )

    st.divider()

    # 1. MAIN FINDINGS

    st.subheader("📊 Key Experimental Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Gower Silhouette", "0.6343")
    c2.metric("Euclidean Silhouette", "0.6840")
    c3.metric("Cosine Silhouette", "0.5309")

    st.caption(
        "Results are from the K=2 clustering experiments."
    )

    st.divider()

    # 2. CLUSTER BALANCE

    st.subheader("👥 Cluster Size Comparison")

    cluster_data = pd.DataFrame({
        "Metric": [
            "Gower",
            "Euclidean",
            "Cosine"
        ],
        "Cluster 1": [
            2504,
            2988,
            2504
        ],
        "Cluster 2": [
            496,
            12,
            496
        ]
    })

    fig = px.bar(
        cluster_data,
        x="Metric",
        y=["Cluster 1", "Cluster 2"],
        barmode="group",
        title="Cluster Size Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        cluster_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # 3. WHY GOWER?

    st.subheader("🎯 Why Consider Gower Distance?")

    st.markdown("""
    - Supports mixed numerical and categorical data.
    - Does not require every feature to be numerical.
    - Produced two comparatively balanced clusters
      in our K=2 experiment.
    - Achieved a silhouette score of 0.6343.
    """)

    st.info(
        "Gower is a defensible candidate for this dataset. "
        "Euclidean achieved a higher silhouette score, "
        "but produced a very small cluster of only 12 respondents."
    )

    st.divider()

    # 4. VALIDATION FINDINGS

    st.subheader("🔬 Feature Sensitivity Results")

    validation_data = pd.DataFrame({
        "Experiment": [
            "Original: All features",
            "Without missingness indicators",
            "Without education features",
            "Without employment features"
        ],
        "Silhouette": [
            0.634288,
            0.396497,
            0.740803,
            0.667237
        ],
        "ARI": [
            1.0,
            1.0,
            1.0,
            1.0
        ]
    })

    st.dataframe(
        validation_data,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        validation_data,
        x="Experiment",
        y="Silhouette",
        color="Experiment",
        title="Feature Sensitivity Analysis"
    )

    fig.update_layout(
        xaxis_tickangle=-25
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "ARI = 1.0 indicates identical cluster assignments "
        "to the original model in these experiments. "
        "These are sensitivity tests, not external validation."
    )

    st.divider()

    # 5. LIMITATIONS

    st.subheader("⚠️ Project Limitations")

    st.markdown("""
    1. The clustering was evaluated on a sample of 3,000 records.

    2. The clusters strongly reflected employment status
       and employment-related missingness.

    3. The results do not establish independent alumni
       or professional groups.

    4. External validation using manually curated
       reference groups is still needed.
    """)

    st.divider()

    # 6. FUTURE SCOPE

    st.subheader("🚀 Future Improvements")

    st.markdown("""
    - Validate clusters using manually curated reference groups.
    - Explore additional mixed-data clustering algorithms.
    - Test different feature-selection strategies.
    - Evaluate results on larger samples.
    - Improve the interpretation of alumni similarity groups.
    """)

    st.divider()

    # 7. FINAL CONCLUSION

    st.subheader("🏁 Final Conclusion")

    st.success("""
    Gower distance provides a practical and defensible
    approach for measuring similarity in mixed-format
    alumni profile data.

    However, metric selection should consider cluster
    balance, silhouette score, feature sensitivity
    and independent validation together.
    """)

    st.markdown(
        "**Thank you! — SIH Distance Metric Benchmarking Project**"
    )

# ============================================================
    # 15. DOWNLOADS PAGE
# ============================================================
    

elif page == "Downloads":

    st.title("📥 Download Project Files")

    st.write("""
    Download the datasets, analysis results,
    validation results and reports generated
    by the SIH project.
    """)

    st.markdown("### Main Project Results")

    download_files = {
        "Distance Summary": "distance_summary.csv",
        "Distance Correlations": "distance_correlations.csv",
        "Clustering Evaluation": "clustering_evaluation.csv",
        "Clustering Labels": "clustering_labels.csv",
        "Cluster Profiles": "cluster_profiles.csv",
        "Distance Sample": "distance_sample.csv",
        "Cleaned Dataset": "cleaned_dataset.csv",
        "Clustering Stability": "clustering_stability_results.csv",
        "Final Project Report": "final_project_report.txt",
        "Distance Results": "distance_results.npz",
    }

    for label, filename in download_files.items():

        file_path = DATA_DIR / filename

        if file_path.exists():

            with open(file_path, "rb") as file:

                st.download_button(
                    label=f"Download {label}",
                    data=file.read(),
                    file_name=filename,
                    use_container_width=True
                )

        else:

            st.warning(
                f"{filename} is not available."
            )

    st.divider()

    st.markdown("### Final Validation Files")

    validation_files = {
        "Feature Influence Results":
            "final_validation_results.csv",

        "Validation Cluster Labels":
            "final_validation_labels.csv",

        "Employment Validation":
            "final_employment_validation.csv",

        "Missingness Validation":
            "final_missingness_validation.csv",

        "Validation Report":
            "final_validation_report.txt",

        "Missingness Influence Results":
            "missingness_influence_results.csv",
    }

    for label, filename in validation_files.items():

        file_path = DATA_DIR / filename

        if file_path.exists():

            with open(file_path, "rb") as file:

                st.download_button(
                    label=f"Download {label}",
                    data=file.read(),
                    file_name=filename,
                    use_container_width=True
                )

        else:

            st.caption(
                f"{filename} is not available."
            )

    st.divider()

    st.markdown("### Project Plots")

    if PLOT_DIR.exists():

        plot_files = list(PLOT_DIR.glob("*"))

        if plot_files:

            for plot_file in plot_files:

                if plot_file.suffix.lower() in [
                    ".png", ".jpg", ".jpeg"
                ]:

                    with open(plot_file, "rb") as file:

                        st.download_button(
                            label=f"Download {plot_file.name}",
                            data=file.read(),
                            file_name=plot_file.name,
                            use_container_width=True
                        )

        else:

            st.info("No plot files found.")

    else:

        st.info("Plots folder was not found.")


# ============================================================
# 16. DASHBOARD FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>SIH Distance Metric Benchmarking Project</b><br>
        Gower | Euclidean | Cosine<br>
        NSCG 2023 Dataset<br>
        Comparative Analysis and Final Validation
    </div>
    """,
    unsafe_allow_html=True
)