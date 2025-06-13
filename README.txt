
EDTA Blood RNA QC

A Streamlit app for quality control of EDTA-tube blood RNA profiles using a trained SVM model.
Interactive PCA plots, adjustable thresholds (FNR), ΔCt normalization, and Excel report download.

Table of Contents
1. Clone the Repository
2. Installation
   - Conda (recommended)
   - pip + venv
3. Run Tests
4. Usage
5. Data Format
6. Error Handling
7. Contributing
8. License
9. Contact

Clone the Repository:
git clone https://github.com/vlastaxiv/EDTA_QC.git
cd EDTA_QC

Installation:
Conda (recommended):
conda env create -f environment.yml
conda activate predikce_env

pip + venv:
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
pip install -r requirements.txt

Run Tests (optional):
pytest -q

Usage:
1. Start the app:
   streamlit run app.py

2. Built-in example:
   Check "Built-in example dataset" in the sidebar to load demo data.

3. Upload your own data:
   CSV or Excel (.xlsx) containing columns:
   sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4
   No missing values allowed. Decimal commas auto-converted.

4. Adjust the False Negative Rate (FNR):
   0% is strictest; increasing FNR lets more borderline samples pass; max 9% keeps 100% specificity.

5. Run Prediction:
   Click "Run Prediction"; view plot, styled table, download Excel report.

Data Format:
UPLOAD_COLUMNS: sample, BTG3, CD69, CXCR1, CXCR2, FCGR3A, GAPDH, GUSB, JUN, PPIB, STEAP4
REFERENCE_GENES: GAPDH, GUSB, PPIB
ΔCt normalization: ΔCt = Ct_gene - Ct_ref_mean; then -ΔCt.

Error Handling:
File-format or validation errors show a friendly warning in the main panel.

Contributing:
Fork the repo, create branch, commit, push, open PR.

License:
MIT License - see LICENSE for details.

Contact:
Open an issue or email youremail@example.com.
