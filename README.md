
# EDTA Blood RNA QC

A Streamlit app for quality control of EDTA-tube blood RNA profiles using a trained SVM model.  
Interactive PCA plots, adjustable thresholds (FNR), ΔCt normalization, and Excel report download.

---

## 📋 Table of Contents

1. [Clone the Repository](#-clone-the-repository)  
2. [Installation](#-installation)  
   - [Conda (recommended)](#conda-recommended)  
   - [pip + venv](#pip--venv)  
3. [Run Tests](#-run-tests)  
4. [Usage](#-usage)  
5. [Data Format](#-data-format)  
6. [Error Handling](#-error-handling)  
7. [Contributing](#-contributing)  
8. [License](#-license)  
9. [Contact](#-contact)  

---

## 🔽 Clone the Repository

```bash
git clone https://github.com/vlastaxiv/EDTA_QC.git
cd EDTA_QC
```

---

## 🚀 Installation

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

## 🧪 Run Tests

(Optional) Verify model pipeline before running the app:

```bash
pytest -q
```

---

## 🎬 Usage

1. **Start the app**  
   ```bash
   streamlit run app.py
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

## 🗄️ Data Format

- **UPLOAD_COLUMNS**:  
  `sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4`  
- **REFERENCE_GENES**: `GAPDH, GUSB, PPIB`  
- **ΔCt normalization**:  
  1. Compute mean Cq of references per sample.  
  2. ΔCt = Ct_gene – Ct_ref_mean; then take –ΔCt.

---

## ❌ Error Handling

- File-format or validation errors show a friendly `⚠️` warning in the main panel—no raw tracebacks.

---

## 🤝 Contributing

1. Fork the repo  
2. Create a branch: `git checkout -b feature/YourFeature`  
3. Commit: `git commit -m "Add awesome feature"`  
4. Push: `git push origin feature/YourFeature`  
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**—see [LICENSE](LICENSE) for details.

---

## ✉️ Contact

For questions or support, open an issue or email `ctrnacta@yahoo.com`.  

_Last updated: 2025-05-30_
