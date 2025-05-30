from utils import load_pipeline
pipeline = load_pipeline("models/scaler.joblib", "models/pca_model.joblib", "models/final_svm_poly_model.joblib")
model = pipeline.named_steps['model']
print("Supports predict_proba:", hasattr(model, 'predict_proba'))
