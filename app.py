import streamlit as st

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

# Title
st.title("EDTA Blood RNA QC")
st.markdown(
    "This application checks whether gene expression profiles—derived from RNA isolated from EDTA-tube blood—"
    "meet quality standards, via a trained Support Vector Machine model (SVM)."
)

# 1) Load pipeline and training data
try:
    pipeline = load_pipeline(
        "models/scaler.joblib",
        "models/pca_model.joblib",
        "models/final_svm_poly_model.joblib"
    )
    df_train = get_training_data()
    st.success("Pipeline and training data loaded successfully.")
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# 2) Expected feature columns
try:
    expected_cols = list(pipeline.feature_names_in_)
except Exception:
    expected_cols = list(pipeline.named_steps['model'].feature_names_in_)

# Sidebar: Upload & settings
st.sidebar.header("Upload New Samples for QC")
with st.sidebar.expander("ℹ️ Detailed upload instructions"):
    st.markdown(
        "- You can either upload an Excel (*.xlsx) or a CSV file\n"
        "- Upload raw Cq data (no normalization)\n"
        "- Required columns: sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4\n"
        "- No missing values allowed"
    )
uploaded_file = st.sidebar.file_uploader(
    "Your samples (CSV or XLSX)", type=["csv", "xlsx"], label_visibility="visible"
)
use_example = st.sidebar.checkbox("Built-in example dataset", value=False)
if use_example:
    st.sidebar.info("Using built-in example dataset. Uncheck to upload your own file.")

# Sidebar: FNR selection
st.sidebar.markdown("---")
st.sidebar.header("Optional Change of Decision Boundary")
with st.sidebar.expander("ℹ️ Why change FNR?"):
    for line in get_fnr_explanation():
        st.markdown(line)
fnr = st.sidebar.selectbox(
    "Select False Negative Rate (FNR) %",
    list(FNR_TO_THRESHOLD.keys()),
    index=6
)
threshold = FNR_TO_THRESHOLD[fnr]

# Sidebar: About link
st.sidebar.markdown("---")
st.sidebar.header("About This Application")
st.sidebar.markdown("[📘 Full documentation](https://…/YOUR_REPO#readme)")

# 3) Load and Prepare Data
if use_example:
    # A) Built-in example path
    df_new = get_example_data()
    # Validate against model features only
    from utils import validate_data  # ensure import
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

# Preview
st.write("## Your Data Normalized with Reference Genes ")
# Drop any group columns if present
df_display = df_new.drop(columns=["group", "groups"], errors="ignore").copy()
# Round floats and reindex
float_cols = df_display.select_dtypes(include=["float"]).columns
df_display[float_cols] = df_display[float_cols].round(1)
df_display.index = range(1, len(df_display) + 1)
st.dataframe(df_display, use_container_width=True)

# 5) PCA Projection: Training vs New Samples
chart = create_pca_chart(pipeline, df_train, df_new, expected_cols)
st.altair_chart(chart, use_container_width=True)

# 6) Show current threshold usage
st.markdown(
    f"**Current prediction threshold:** {threshold:.3f} (FNR = {fnr}%). Samples with decision_score > threshold are classified as OK."
)

# 7) Prediction step
if st.button("Run Prediction"):
    # Určíme, který sloupec je ten „první“ (např. identifikátor vzorku)
    first_col = df_new.columns[0]

    # 1) Připravíme data pro predikci
    X_input_df = pd.DataFrame(df_new[expected_cols], columns=expected_cols)
    decision_scores = pipeline.decision_function(X_input_df)
    preds = (decision_scores > threshold).astype(int)
    pred_labels = [
        "Sample quality is OK." if p == 1 else
        "Do not use this sample. Its gene expression has been altered in EDTA tube."
        for p in preds
    ]

    # 2) Vykreslíme rozhodovací plochu
    fig = create_matplotlib_decision_plot(
        pipeline, df_train, df_new, expected_cols, threshold, fnr
    )
    st.pyplot(fig)

    # 3) Nadpis výsledků
    st.write("## Quality Assessment")

    # 4) Vytvoření výsledné tabulky
    df_result = pd.DataFrame({
        'sample': range(1, len(preds) + 1),
        first_col: df_new[first_col].values,
        'decision_score': np.round(decision_scores, 1),
        'prediction': pred_labels
    })

    # 5) Odebrání zbytečných nul
    df_result['decision_score'] = df_result['decision_score'] \
        .apply(lambda x: f"{x:.1f}".rstrip('0').rstrip('.'))

    # 6) Stylování: bílá pro OK, červená pro altered
    def highlight_pred(val):
        return 'background-color: white' if val == 'Sample quality is OK.' \
               else 'background-color: #eb9696'

    styled = (
        df_result.style
                 .map(highlight_pred, subset=['prediction'])
                 .set_properties(subset=['decision_score'], **{'text-align': 'left'})
    )

    # 7) Zobrazíme stylovanou tabulku
    st.dataframe(styled, use_container_width=True)

    # 8) Shrnutí
    total = len(preds)
    ok_count = sum(preds)
    altered_count = total - ok_count
    st.markdown(
        f"**Results summary:** {ok_count} samples OK ({ok_count/total*100:.1f}%), "
        f"{altered_count} altered ({altered_count/total*100:.1f}%)"
    )

    # 9) Tlačítko pro stažení Excelu
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
