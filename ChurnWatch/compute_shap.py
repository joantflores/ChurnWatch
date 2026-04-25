"""Compute SHAP explanations for the best model and save summary + per-sample plots.

Requires best_model.pkl produced by preprocess_and_train.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "best_model.pkl"
OUT = ROOT / "outputs" / "shap"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    if not BEST.exists():
        print("No best_model.pkl found. Ejecuta preprocess_and_train.py primero.")
        return
    payload = joblib.load(BEST)
    model = payload.get("modelo")
    feature_cols = payload.get("feature_columns", [])

    shop = ROOT / "shopping_trends_churn.csv"
    if not shop.exists():
        print("No shopping_trends_churn.csv found.")
        return
    df = pd.read_csv(shop, encoding_errors="ignore")
    X = df[feature_cols].fillna(0)

    try:
        import shap
    except Exception:
        print("shap no instalado")
        return

    explainer = shap.Explainer(model.named_steps['clf'], model.named_steps['pre'].transform(X))
    shap_values = explainer(model.named_steps['pre'].transform(X))

    try:
        shap.summary_plot(shap_values, show=False)
        plt.savefig(OUT / "summary.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"SHAP summary guardado: {OUT / 'summary.png'}")
    except Exception as e:
        print("Error guardando SHAP summary:", e)

    for i in range(min(3, X.shape[0])):
        try:
            shap.plots.waterfall(shap_values[i], show=False)
            plt.savefig(OUT / f"waterfall_{i}.png", bbox_inches='tight', dpi=150)
            plt.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
