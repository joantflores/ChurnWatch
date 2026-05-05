# ChurnWatch - Setup y Flujo de Avance 2

Breve guia para reproducir el pipeline requerido por el avance 2.

1) Crear entorno y dependencias

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Colocar datasets

Coloca estos archivos en la raiz del proyecto:

- `shopping_trends.csv` (principal)
- `OnlineRetailII.csv` (contexto e-commerce)
- `TelcoCustomerChurn.csv` (benchmark con churn real)

3) Pipeline recomendado (orden)

- Construir la variable churn (Shopping Trends):

```bash
python build_churn.py
```

Genera `shopping_trends_churn.csv` con columna `churn`.

- Ejecutar EDA para los tres datasets:

```bash
python run_eda.py
```

Salidas en `outputs/eda/<dataset>/` (summary.txt, histogramas, boxplots, top categories).

- Preprocesar y entrenar modelos (LogisticRegression, RandomForest, XGBoost):

```bash
python preprocess_and_train.py
```

Genera `best_model.pkl` y `outputs/train/evaluation.json`.

- Calcular explicaciones SHAP (requiere `best_model.pkl`):

```bash
python compute_shap.py
```

Salida en `outputs/shap/` (summary + waterfall plots).

- Ejecutar la interfaz (Dash):

```bash
python app.py
```

Abre `http://localhost:8050`. La app usa `best_model.pkl` si existe, o el modo heuristico si falta.

4) Notas rapidas

- `preprocess_and_train.py` hace split estratificado 80/20.
- Los scripts usan columnas esperadas: `previous_purchases`, `frequency_of_purchases`, `purchase_amount_usd`, `subscription_status`, `discount_applied`.
- Si tus CSV usan otras mayusculas/alias, `build_churn.py` y `preprocess_and_train.py` normalizan a minusculas.

5) Problemas comunes

- Si la instalacion de `xgboost` falla puedes comentar su bloque en `preprocess_and_train.py` y usar solo RandomForest + LogisticRegression.
- Para depuracion en Visual Studio, ejecuta sin depurar (Ctrl+F5) si el depurador causa errores con `debugpy`.

Si quieres, puedo:
- Anadir export de metricas a una tabla comparativa PNG/CSV.
- Crear un demo Streamlit alternativo para presentacion en pocas horas.
