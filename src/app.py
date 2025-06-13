import streamlit as st

from pathlib import Path

# Core helpers: pipeline, data loading & validation, charting
from utils import (
    load_pipeline,
    get_training_data,
    get_example_data,
    read_and_unify,
    validate_columns_presence,
    check_missing,
    normalize_qpcr,
    create_pca_chart,
    create_matplotlib_decision_plot,
    FNR_TO_THRESHOLD,
    highlight_pred,
    get_fnr_explanation,
    UPLOAD_COLUMNS
)


# Data & plotting
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# File I/O
import io

# ── Project root and data/model paths ──
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"

TRAIN_DATA_PATH   = DATA_DIR / "train_data.csv"
EXAMPLE_DATA_PATH = DATA_DIR / "example_new_samples.csv"  # nebo jak se soubor jmenuje
PIPELINE_PATH     = MODELS_DIR / "final_pipeline_prob.joblib"
# ── End of paths ──

# Global CSS adjustments
st.markdown(
    """
    <style>
    /* Left-align titles and markdown */
    h1, h2, h3, .stMarkdown {
        text-align: left !important;
    }
    /* Ensure DataFrame containers are full-width */
    .element-container {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Application title
st.title("EDTA Blood RNA Quality Control")
st.markdown(
    "This application evaluates whether gene expression profiles, derived from RNA isolated from EDTA-tube blood,"
    "meet quality standards using a trained Support Vector Machine (SVM) model."
)

# Load pipeline and training dataset
try:
    pipeline = load_pipeline(str(PIPELINE_PATH))
    df_train = get_training_data()
    st.success("Pipeline and training data loaded successfully.")
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Expected feature columns
try:
    expected_cols = list(pipeline.feature_names_in_)
except Exception:
    expected_cols = list(pipeline.named_steps['model'].feature_names_in_)

# Sidebar: upload & settings
st.sidebar.header("Upload New Samples for QC")
with st.sidebar.expander("ℹ️ Detailed upload instructions"):
    st.caption(
        "- Upload either an Excel (*.xlsx) or CSV file\n"
        "- Upload raw Cq data (prior normalization)\n"
        "- Required columns: sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4\n"
        "- Ensure no missing values are present"
    )
uploaded_file = st.sidebar.file_uploader(
    "Upload your samples (CSV or XLSX)", type=["csv", "xlsx"], label_visibility="visible"
)
use_example = st.sidebar.checkbox("Use built-in example dataset", value=False)
if use_example:
    st.sidebar.info("Using the built-in example dataset. Uncheck to upload your own file.")

# Sidebar: decision boundary adjustment
st.sidebar.markdown("---")
st.sidebar.header("Optional Decision Boundary Adjustment")
with st.sidebar.expander("ℹ️ Why to adjust FNR?"):
    for line in get_fnr_explanation():
        st.caption(line)
fnr = st.sidebar.selectbox(
    "Select false negative rate (FNR) %",
    list(FNR_TO_THRESHOLD.keys()),
    index=5,
    key="fnr_selectbox"
)
threshold = FNR_TO_THRESHOLD[fnr]

# Sidebar: about link
st.sidebar.markdown("---")
st.sidebar.header("About This Application")
with st.sidebar.expander("ℹ️ SVM parameters and performance"):
    st.caption(
        "- Kernel: Polynomial\n"
        "- Polynomial degree: 2\n"
        "- Regularization parameter (C): 0.1\n"
        "- Coef0: 1\n"
        "- Gamma: Scale\n"
        "- Total support vectors (SV): 25\n"
        "- Training samples: 220\n"
        "- Test samples: 23\n"
        "- SV (% of training data): 11.4%\n"
        "- Cross-validation AUC (5-fold): 1.0 ± 0"
    )
  
st.sidebar.markdown("[ℹ️ Full documentation](https://github.com/vlastaxiv/EDTA_QC#readme)")

# Load and prepare data
if use_example:
    # Built-in example dataset
    df_new = get_example_data()
    # Validate against model features 
    from utils import validate_data  # ensure function is imported
    validate_data(df_new, expected_cols)

elif uploaded_file:
    try:
        df_raw = read_and_unify(uploaded_file)
        validate_columns_presence(df_raw, UPLOAD_COLUMNS)
        check_missing(df_raw)
        df_new = normalize_qpcr(df_raw, expected_cols)
    except ValueError as err:
        st.warning(f"⚠️ File error: {err}. Please check your file and try again.")
        st.stop()

else:
    st.info("Please upload a file or select the example dataset.")
    st.stop()

# Data preview
st.write("## Normalized Data Using Reference Genes ")

# Drop any 'group' or 'groups' columns if present
df_display = df_new.drop(columns=["group", "groups"], errors="ignore").copy()

# Round float values and reset index
float_cols = df_display.select_dtypes(include=["float"]).columns
df_display[float_cols] = df_display[float_cols].round(1)
df_display.index = range(1, len(df_display) + 1)
st.dataframe(df_display, use_container_width=True)

# PCA Projection: Training vs. New Samples
chart = create_pca_chart(pipeline, df_train, df_new, expected_cols)
st.altair_chart(chart, use_container_width=True)

# Display current decision threshold
st.markdown(
    f"**Current decision threshold:** {threshold:.3f} (FNR = {fnr}%). Samples with decision score > decision threshold are classified as OK."
)

# Prediction step
if st.button("Run Prediction"):
    # Identify the first column name
    first_col = df_new.columns[0]

    # Preprocess input data
    X_input_df = pd.DataFrame(df_new[expected_cols], columns=expected_cols)
    decision_scores = pipeline.decision_function(X_input_df)
 
    # “Thresholded” predictions (use your chosen threshold to control FNR)
    preds = (decision_scores > threshold).astype(int)
    
    # generate prediction labels based on thresholded decision scores
    pred_labels = [
        "Sample quality is OK." if p == 1 else
        "Do not use this sample. Its gene expression has been altered in EDTA tube."
        for p in preds
    ]

    # Plot decision boundary
    fig = create_matplotlib_decision_plot(
        pipeline, df_train, df_new, expected_cols, threshold, fnr
    )
    st.pyplot(fig)

    # Display results table
    st.write("## Quality Assessment")


    # Create results DataFrame
    df_result = pd.DataFrame({
        'sample': range(1, len(preds) + 1),
        first_col: df_new[first_col].values,
        'decision_score': np.round(decision_scores, 1),
        'prediction': pred_labels
    })

    # Remove trailing zeros from decision score
    df_result['decision_score'] = df_result['decision_score'] \
        .apply(lambda x: f"{x:.1f}".rstrip('0').rstrip('.'))

    # Style the results table
    def highlight_pred(val):
        return 'background-color: white' if val == 'Sample quality is OK.' \
               else 'background-color: #eb9696'

    styled = (
        df_result.style
                 .map(highlight_pred, subset=['prediction'])
                 .set_properties(subset=['decision_score'], **{'text-align': 'left'})
    )

    # Table visualization
    st.dataframe(styled, use_container_width=True)

    # Summary of predictions
    total = len(preds)
    ok_count = sum(preds)
    altered_count = total - ok_count
    st.markdown(f"**Support vector model prediction results with FNR = {fnr}% and FPR = 0%:**")
    st.markdown(f"{ok_count} samples OK ({ok_count/total*100:.1f}%), {altered_count} samples with altered gene expression ({altered_count/total*100:.1f}%)"
    )

    # Export results to Excel
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Predictions')
    excel_buffer.seek(0)
    st.download_button(
        "Download Results (.xlsx)",
        data=excel_buffer,
        file_name="predictions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
