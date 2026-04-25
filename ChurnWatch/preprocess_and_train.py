from __future__ import annotations

from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import xgboost as xgb
except Exception:
    xgb = None

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "train"
OUT.mkdir(parents=True, exist_ok=True)

SHOP = ROOT / "shopping_trends_churn.csv"

RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()

    y = df["churn"].astype(int)

    num_cols = [c for c in ["previous_purchases", "purchase_amount_usd"] if c in df.columns]
    cat_cols = [c for c in df.columns if c not in num_cols + ["churn", "customer_id"]]


    X_num = df[num_cols].fillna(0)
    X_cat = df[cat_cols].fillna("missing").astype(str)

    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ], remainder="drop")

    X = pd.concat([X_num, X_cat], axis=1)
    return X, y, preproc, num_cols, cat_cols


def fit_and_eval(X, y, preproc, num_cols, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    if y_train.nunique() < 2:
        print("Solo una clase presente en los datos de entrenamiento. Usando DummyClassifier como fallback.")
        dummy_pipe = Pipeline([
            ("pre", preproc),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ])
        dummy_pipe.fit(X_train, y_train)
        preds = dummy_pipe.predict(X_test)
        try:
            probs = dummy_pipe.predict_proba(X_test)[:, 1]
        except Exception:
            probs = None

        results = {
            "ConstantBaseline": {
                "roc_auc": None if probs is None else float(roc_auc_score(y_test, probs)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            }
        }

        payload = {
            "modelo": dummy_pipe,
            "scaler": None,
            "feature_columns": num_cols,
            "nombre": "ConstantBaseline",
        }
        joblib.dump(payload, ROOT / "best_model.pkl")
        (OUT / "evaluation.json").write_text(json.dumps(results, indent=2))
        print(f"Fallback guardado como ConstantBaseline en {OUT}")
        return

    def make_pipeline(model):
        return Pipeline([
            ("pre", preproc),
            ("clf", model),
        ])

    models = {}
    models["LogisticRegression"] = make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    models["RandomForest"] = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    if xgb is not None:
        models["XGBoost"] = make_pipeline(xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE))

    results = {}

    for name, pipe in models.items():
        print(f"Entrenando {name}...")
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)
        results[name] = {
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    best_model = models[best_name]

    fitted_preproc = best_model.named_steps["pre"]

    num_features = num_cols
    cat_encoder = fitted_preproc.named_transformers_["cat"]
    try:
        cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_names = []
    feature_columns = list(num_features) + cat_names

    payload = {
        "modelo": best_model,
        "scaler": None,
        "feature_columns": feature_columns,
        "input_columns": list(num_cols) + list(cat_cols),
        "nombre": best_name,
    }

    joblib.dump(payload, ROOT / "best_model.pkl")
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2))
    print(f"Modelos entrenados. Mejor: {best_name}. Resultados guardados en {OUT}")


def main():
    if not SHOP.exists():
        print("Ejecuta build_churn.py primero para generar shopping_trends_churn.csv")
        return
    df = load_data(SHOP)
    X, y, preproc, num_cols, cat_cols = preprocess(df)
    fit_and_eval(X, y, preproc, num_cols, cat_cols)


if __name__ == "__main__":
    main()
