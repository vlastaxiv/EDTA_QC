
# EDTA Blood RNA QC

A Streamlit app for automated quality control of EDTA-tube blood RNA profiles using a custom-trained Support Vector Machine (SVM) model.
The model was developed on internally generated experimental data, combining ΔCt normalization, PCA, and SVM classification to assess sample quality.
Features include interactive PCA plots, adjustable False Negative Rate (FNR) thresholds, and Excel report export.

---

## 📋 Table of Contents

**A. Application usage**

1. Clone the Repository  
2. Installation  
3. Usage  
4. Repository Structure  
5. Input Data & Data Format  
6. Error Handling  
7. Contributions  
8. Citation  
9. Contact

**B. Model Development**

10. Model Development & Jupyter Notebooks  
11. Model files

----

# A. Application usage

## 🔽 Clone the Repository

```bash
git clone https://github.com/vlastaxiv/EDTA_QC.git
cd EDTA_QC

---

## 🚀 Installation

**Requirements**

- Python 3.13.2
- Git
- Conda (recommended) or venv
- Python packages listed in requirements.txt (installed automatically during setup)

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate predikce_env
```

### pip + venv

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```
---

## 🎬 Usage

1. **Start the app** 

```bash
streamlit run src/app.py
   ```

2. **Built-in example**  
   - Check **Built-in example dataset** in the sidebar to instantly load demo data.

3. **Upload your own data**  
   - Click **Browse files** and select a CSV or Excel (.xlsx) file containing these columns (any order):  
     ```
     sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4
     ```
   - No missing values allowed. Decimal commas will be auto-converted to dots.

4. **Adjust the False Negative Rate (FNR)**  
   - Use the slider to shift the decision threshold:
     - **0 %** is strictest.
     - **Increasing FNR** permits more borderline samples to pass.
     - **Max 9 %** keeps 100 % specificity on training data.

5. **Run Prediction**  
   - Click **Run Prediction**.  
   - View SVM decision-boundary plot.  
   - Inspect the styled results table (white = OK, red = altered).  
   - Download an Excel report.

---

## 📁 Repository Structure

The repository contains the following key components:

- `src/` — Streamlit app and utility scripts
- `models/` — saved trained model pipeline (`final_pipeline_prob.joblib`)
- `notebooks/` — Jupyter notebooks used for model development and evaluation
- `data/` — input dataset used for model training (`SVM_training_data.csv`)
- `data_for_testing/` — example files for user testing and upload validation
- `requirements.txt` — list of required Python packages
- `environment.yml` — conda environment specification
- `README.md` — project documentation (this file)
   
---
## 📊 Input Data & Data Format
---

###  Input file format for prediction

- File format: **CSV** or **Excel (.xlsx)**
- Each row corresponds to one sample.
- **⚠ All values must be raw Cq values as directly exported from the qPCR machine (unprocessed, not normalized).**
- Required columns:
sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4
- No missing values allowed.
- Decimal commas will be automatically converted to dots.

### Reference genes used for normalization

- `GAPDH, GUSB, PPIB`

### ΔCt normalization (performed automatically in the Streamlit app)

1. Compute mean Cq of reference genes for each sample.
2. ΔCt = (Cq_gene – Cq_ref_mean)
3. Final value = –ΔCt

### Data included in repository

- data/SVM_training_data.csv — processed input data (-ΔCt values) used for model training.
- data_for_testing/ — example files for user testing and error handling demonstration.

---

## ❌ Error Handling

- The app validates input files before processing.
- File-format or validation errors display a friendly `⚠️` warning in the main panel.
- Common errors handled:
  - Missing required columns
  - Non-numeric or invalid Cq values
  - Missing values
  - Incorrect file format (only CSV or XLSX accepted)

---

## Contributions

This repository is not open for external contributions. For any issues or questions, please contact the author directly.

---

## 📑 Citation

If you use this code or parts of this pipeline in your own work or publication, please cite the associated article:

[Full citation will be added here once published]

Alternatively, you can refer to this repository as:

Korenková V. EDTA Blood RNA QC – Streamlit app for EDTA blood RNA quality control using SVM classification. GitHub repository: https://github.com/vlastaxiv/EDTA_QC

---

## ✉️ Contact

For questions or support, please contact: `ctrnacta@yahoo.com`.  

_Last updated: 2025-06-17_

----

# B. Model Development


## 🔬 Model Development in Jupyter Notebooks

The SVM model was developed using blood RNA profiles obtained from EDTA tubes.

The model development pipeline included two main stages:

### 1️⃣ Model selection and training 

- Input data consisted of already processed `-ΔCt` values.
- Autoscaling was applied using `StandardScaler` (zero mean and unit variance).
- Dimensionality reduction was performed using Principal Component Analysis (PCA), retaining 2 components.
- Support Vector Machine (SVM) with polynomial kernel (`degree=2`) was used.
- Hyperparameters (`C`, `coef0`) were optimized using `GridSearchCV` and cross-validation.
- The optimized model, together with scaler and PCA transformation, was combined into a single pipeline.
- The complete pipeline was saved as `final_pipeline_prob.joblib` for deployment in the Streamlit app.

### 2️⃣ Decision threshold evaluation 

- The trained model was evaluated at various False Negative Rate (FNR) thresholds.
- Decision boundary shifts were analyzed to balance sensitivity and specificity.
- These FNR adjustments are implemented as interactive slider options in the Streamlit app.


### 📓 Available Jupyter notebooks

| Notebook | Description |
|----------|-------------|
| `SVM_model_training_final.ipynb` | Model training, hyperparameter optimization with GridSearchCV, and pipeline export |
| `SVM_model_optimalization.ipynb` | Evaluation of decision thresholds (FNR) using the trained model |

> These notebooks reproduce the full model development pipeline and can be adapted for retraining on new datasets.

---

### 📦 Model files

The trained model pipeline is stored in the models/ directory:

   - final_pipeline_prob.joblib — complete pipeline containing StandardScaler, PCA transformation, and trained SVM classifier.

This file is automatically loaded by the Streamlit app to perform predictions on new data.

---
