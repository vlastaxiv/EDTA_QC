import os
import pandas as pd
import joblib
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import importlib
from functools import lru_cache
import importlib
from pathlib import Path

# === Paths for Data Files ===
# Project root and data directory
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
EXAMPLE_DATA_PATH = DATA_DIR / "example_new_samples.csv"

# === Constants for Upload ===
REFERENCE_GENES = ["GAPDH", "GUSB", "PPIB"]
UPLOAD_COLUMNS = [
    "sample",
    "BTG3",
    "CD69",
    "CXCR1",
    "CXCR2",
    "FCGR3A",
    "GAPDH",   # reference
    "GUSB",    # reference
    "JUN",
    "PPIB",    # reference
    "STEAP4"]


# === Cached Data Loaders ===
@lru_cache(maxsize=1)
def get_training_data() -> pd.DataFrame:
    """
    Load and cache the fixed training dataset.
    """
    return pd.read_csv(TRAIN_DATA_PATH)

@lru_cache(maxsize=1)
def get_example_data() -> pd.DataFrame:
    """
    Load and cache the example new-samples dataset.
    """
    return pd.read_csv(EXAMPLE_DATA_PATH)


# === Pipeline Loader ===
def load_pipeline(pipeline_path: str):
    """
    Load and return an sklearn Pipeline that already
    contains scaler, PCA, and the SVM model (with probability=True).
    """
    return joblib.load(pipeline_path)

def read_and_unify(file_buffer: io.BytesIO) -> pd.DataFrame:
    """
    Load raw data from CSV or Excel and unify decimal separators.
    """
    file_buffer.seek(0)
    filename = getattr(file_buffer, "name", "").lower()

    df = None
    # Excel files
    if filename.endswith((".xlsx", ".xls")):
        # Zkontrolujeme, zda je nainstalován openpyxl
        if importlib.util.find_spec("openpyxl") is None:
            raise ValueError("Excel support requires the 'openpyxl' library. Please install it via `pip install openpyxl`.")
        try:
            df = pd.read_excel(file_buffer, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Cannot read Excel file: {e}")
    else:
        # Treat file as CSV
        file_buffer.seek(0)
        try:
            df = pd.read_csv(file_buffer, sep=None, engine="python")
        except Exception as e:
            raise ValueError(f"Cannot read CSV file: {e}")

    # Normalize decimals and convert numeric columns
    df = df.replace({',': '.'}, regex=True)

    # Attempt numeric conversion; ignore errors
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df



def validate_columns_presence(df: pd.DataFrame, required_cols: list[str])-> None:
    """
    Ensure that df contains all required columns, in any order.
    """
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

def check_missing(df: pd.DataFrame):
    """
    Check for missing values and raise an error if any are found.
    """
    total_missing = int(df.isna().sum().sum())
    if total_missing:
        raise ValueError(f"Data contain {total_missing} missing values. Please clean your input.")

def get_new_data(use_example: bool, new_file) -> pd.DataFrame:
    """
    Return either the example dataset (if use_example=True) or the uploaded file (CSV/XLSX)
    processed through unified reader, validation, and missing-value check.

    Raises:
        ValueError: if neither source is provided.
    """
    if use_example:
        df = get_example_data()
    elif new_file is not None:
        df = read_and_unify(new_file)
    else:
        raise ValueError("Please upload a CSV or Excel file or select the example dataset.")
    validate_columns_presence(df, UPLOAD_COLUMNS)
    check_missing(df)
    return df


# === Validation for Pipeline Data ===
def validate_data(df: pd.DataFrame, required_cols: list) -> None:
    """
    Ensure the DataFrame contains the required columns. Raise ValueError if missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# === Visualization Utilities ===
def create_pca_chart(pipeline, df_train: pd.DataFrame, df_new: pd.DataFrame, expected_cols: list):
    """
    Generate an Altair PCA projection chart for training vs. new samples.
    """
    scaler = pipeline.named_steps['scaler']
    pca = pipeline.named_steps['pca']
    X_train = scaler.transform(df_train[expected_cols])
    X_new = scaler.transform(df_new[expected_cols])
    pc_train = pca.transform(X_train)[:, :2]
    pc_new = pca.transform(X_new)[:, :2]
    df_plot = pd.DataFrame(np.vstack([pc_train, pc_new]), columns=['PC1', 'PC2'])
    df_plot['Set'] = ['Train'] * len(pc_train) + ['Test'] * len(pc_new)
    chart = alt.Chart(df_plot).mark_circle(size=60).encode(x='PC1', y='PC2', color='Set').properties(width=600, height=400, title='PCA Projection: Training vs. New Samples')
    return chart

def create_matplotlib_decision_plot(
    pipeline,
    df_train: pd.DataFrame,
    df_new: pd.DataFrame,
    expected_cols: list,
    threshold: float,
    fnr: int
) -> plt.Figure:
    """
    Generate a matplotlib figure showing decision boundary at given threshold.
    """
    scaler = pipeline.named_steps['scaler']
    pca = pipeline.named_steps['pca']
    model = pipeline.named_steps['model']

    # PCA transform
    X_train_pca = pca.transform(scaler.transform(df_train[expected_cols]))[:, :2]
    X_unknown_pca = pca.transform(scaler.transform(df_new[expected_cols]))[:, :2]

    # Decision scores for train & unknown
    y_train_orig = (model.decision_function(pca.transform(scaler.transform(df_train[expected_cols]))) > threshold).astype(int)
    y_unk_orig = (model.decision_function(pca.transform(scaler.transform(df_new[expected_cols]))) > threshold).astype(int)

    # Grid for boundary
    x_min, x_max = X_train_pca[:,0].min()-1, X_train_pca[:,0].max()+1
    y_min, y_max = X_train_pca[:,1].min()-1, X_train_pca[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    Z = model.decision_function(grid).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[threshold], colors='k', linewidths=1)
    ax.scatter(*X_train_pca.T, c=y_train_orig, cmap=mcolors.ListedColormap(["#eb9696","#5ECEAC"]), marker='x', label='Training')
    ax.scatter(*X_unknown_pca.T, c=y_unk_orig, cmap=mcolors.ListedColormap(["#eb9696","#5ECEAC"]),
               edgecolors='k', label='Predictions')
    ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
    ax.set_title(f'SVM Decision Boundary for FNR {fnr}%')
    ax.legend(loc='upper left'); ax.grid(True)

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    return fig

# === FNR mapping and styling ===
FNR_TO_THRESHOLD = {
    0: -0.556,
    1: -0.567,
    2: -0.578,
    3: -0.589,
    4: -0.600,
    5: -0.611,
    6: -0.622,
    7: -0.633,
    8: -0.644,
    9: -0.655,
}

def highlight_pred(val):
    """Return CSS style string based on the prediction value."""
    if val == 'Sample quality is OK.':
        return 'background-color: white'
    else:
        return 'background-color: #eb9696'


def get_fnr_explanation() -> list[str]:
    """
    Returns the three key points explaining FNR behavior.
    """
    return [ 
        "**Increasing FNR**, "
        "shifts the decision threshold toward the OK class, permitting a limited fraction of samples with borderline‐altered gene expression to be classified as good quality samples.  "
        "Because our altered-expression criteria are very stringent, these marginal cases pose minimal risk.",

        "**FNR = 0 %**  The strictest setting. "
        "The decision threshold sits at the extreme edge of the altered distribution, "
        "so everything else—even borderline cases—is classified as OK.",
        
        "**Maximum FNR = 9 %**  The highest FNR that still maintains 100 % specificity on the training set "
        "(i.e. no altered training sample is classified wrongly)."
    ]


def normalize_qpcr(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Perform ΔCt normalization on raw Cq (Ct) data:
    - Keeps the 'sample' column for identification.
    - Compute the arithmetic mean of the reference genes for each sample.
    - For each target gene in feature_cols, compute ΔCt = Ct_gene - Ct_ref_mean.
    Returns a DataFrame containing the 'sample' column plus all normalized features.
    """
    df_norm = df.copy()
    # Compute mean Cq of reference genes
    df_norm["Ct_ref_mean"] = df_norm[REFERENCE_GENES].mean(axis=1)

    # Apply –ΔCt to each feature column
    for gene in feature_cols:
        df_norm[gene] = (df_norm[gene] - df_norm["Ct_ref_mean"])

    # Drop helper column
    df_norm = df_norm.drop(columns=["Ct_ref_mean"])

    # Build final column order: always include 'sample' if present
    cols = []
    if "sample" in df_norm.columns:
        cols.append("sample")
    cols += [gene for gene in feature_cols if gene in df_norm.columns]

    return df_norm[cols]

