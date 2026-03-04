import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Robust Quantile Scoring
# -----------------------------
def _quantile_score(series, higher_is_better=True, use_rank=False):
    values = series.rank(method="first") if use_rank else series

    try:
        codes = pd.qcut(values, q=5, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(3, index=series.index, dtype="int64")

    if codes.isna().all():
        return pd.Series(3, index=series.index, dtype="int64")

    n_bins = int(codes.max()) + 1
    if n_bins <= 1:
        return pd.Series(3, index=series.index, dtype="int64")

    scaled = (codes.astype(float) / (n_bins - 1) * 4).round()
    scores = (1 + scaled).astype(int)

    if not higher_is_better:
        scores = 6 - scores

    return scores.astype("int64")


# -----------------------------
# Date Parsing Helper
# -----------------------------
def _parse_invoice_dates(series):
    # Try mixed parsing first for datasets with inconsistent date formats.
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except TypeError:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


# -----------------------------
# Create RFM Features
# -----------------------------
def create_rfm(df):
    df = df.copy()
    df["InvoiceDate"] = _parse_invoice_dates(df["InvoiceDate"])
    df = df.dropna(subset=["InvoiceDate"])

    if df.empty:
        raise ValueError("No valid InvoiceDate values after parsing.")

    reference_date = df["InvoiceDate"].max()

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Revenue": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"].replace(0, pd.NA)
    rfm["AOV"] = rfm["AOV"].fillna(0)

    return rfm


# -----------------------------
# RFM Scoring
# -----------------------------
def score_rfm(rfm):
    rfm["R_Score"] = _quantile_score(rfm["Recency"], higher_is_better=False)
    rfm["F_Score"] = _quantile_score(
        rfm["Frequency"], higher_is_better=True, use_rank=True
    )
    rfm["M_Score"] = _quantile_score(rfm["Monetary"], higher_is_better=True)

    return rfm


# -----------------------------
# Rule-Based Segmentation
# -----------------------------
def segment_customers(rfm):

    def rfm_segment(row):
        if row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
            return "Champions"
        elif row["F_Score"] >= 4 and row["R_Score"] >= 3:
            return "Loyal"
        elif row["R_Score"] <= 2 and row["M_Score"] >= 4:
            return "At Risk"
        elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
            return "Hibernating"
        else:
            return "Others"

    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)

    return rfm


# -----------------------------
# MACHINE LEARNING CLUSTERING
# -----------------------------
def apply_kmeans(rfm, n_clusters=4):

    if rfm.empty:
        raise ValueError("RFM table is empty; cannot cluster.")

    features = rfm[["Recency", "Frequency", "Monetary"]]

    # Scale data before clustering.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    cluster_count = min(max(1, n_clusters), len(rfm))

    if cluster_count == 1:
        rfm["Cluster"] = 0
        return rfm, None, scaler

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(scaled_features)

    return rfm, kmeans, scaler


def save_cluster_scatter_plot(rfm, static_dir):
    os.makedirs(static_dir, exist_ok=True)
    output_path = os.path.join(static_dir, "customer_clusters.png")

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        rfm["Frequency"],
        rfm["Monetary"],
        c=rfm["Cluster"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5
    )
    plt.title("Customer Cluster Scatter Plot")
    plt.xlabel("Frequency")
    plt.ylabel("Revenue")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


# -----------------------------
# FULL PIPELINE
# -----------------------------
def full_pipeline(df, static_dir="static"):

    rfm = create_rfm(df)
    rfm = score_rfm(rfm)
    rfm = segment_customers(rfm)
    rfm, model, scaler = apply_kmeans(rfm)
    save_cluster_scatter_plot(rfm, static_dir)

    return rfm
