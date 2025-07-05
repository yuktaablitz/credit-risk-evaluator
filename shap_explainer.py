# shap_explainer.py
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("model/credit_risk_model.pkl")
X_train, X_test, y_train, y_test = joblib.load("data/processed_data.pkl")

# Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Summary plot (global feature importance)
def plot_summary(output_path="shap_summary_plot.png"):
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ SHAP summary plot saved to: {output_path}")
