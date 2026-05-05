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

    # need training data sample to compute SHAP; try to use shopping_trends_churn
    shop = ROOT / "shopping_trends_churn.csv"
    if not shop.exists():
        print("No shopping_trends_churn.csv found.")
        return
    df = pd.read_csv(shop, encoding_errors="ignore")
    # Transformar el DataFrame crudo usando el pipeline de preprocesamiento
    # Preprocesar igual que en el entrenamiento: convertir categóricas a string y rellenar nulos
    num_cols = [c for c in ["previous_purchases", "purchase_amount_usd"] if c in df.columns]
    cat_cols = [c for c in df.columns if c not in num_cols + ["churn", "customer_id"]]
    X_num = df[num_cols].fillna(0) if num_cols else pd.DataFrame()
    X_cat = df[cat_cols].fillna("missing").astype(str) if cat_cols else pd.DataFrame()
    X_df = pd.concat([X_num, X_cat], axis=1) if not X_num.empty or not X_cat.empty else df
    X_proc = model.named_steps['pre'].transform(X_df)

    try:
        import shap
    except Exception:
        print("shap no instalado")
        return

    explainer = shap.Explainer(model.named_steps['clf'], X_proc)
    shap_values = explainer(X_proc)

    # summary plot
    try:
        shap.summary_plot(shap_values, show=False)
        plt.savefig(OUT / "summary.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"SHAP summary guardado: {OUT / 'summary.png'}")
    except Exception as e:
        print("Error guardando SHAP summary:", e)

    # per-sample force plot for first 3 rows
    for i in range(min(3, X_proc.shape[0])):
        try:
            shap.plots.waterfall(shap_values[i], show=False)
            plt.savefig(OUT / f"waterfall_{i}.png", bbox_inches='tight', dpi=150)
            plt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
