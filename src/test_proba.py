# src/test_proba.py

from pathlib import Path
from utils import load_pipeline

# Najdi kořen projektu (o úroveň výš než src/)
ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "models" / "final_pipeline_prob.joblib"

# Načti kompletní pipeline
pipeline = load_pipeline(str(PIPELINE_PATH))
model = pipeline.named_steps['model']

# Jednoduchá kontrola, že model umí predict_proba
assert hasattr(model, "predict_proba"), "Model must implement predict_proba"
print("✔ predict_proba exists!")