from utils import load_pipeline
pipeline = load_pipeline("models/scaler.joblib", "models/pca_model.joblib", "models/final_svm_poly_model.joblib")
model = pipeline.named_steps['model']
assert hasattr(model, "predict_proba"), "Model should implement predict_proba"