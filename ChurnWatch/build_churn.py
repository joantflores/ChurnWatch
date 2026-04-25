"""Construye la variable `churn` segun la especificacion del paper.

Regla: churn=1 si frequency es una de ['annually','every 3 months','quarterly']
y previous_purchases <= percentil25 (por defecto 13). Guarda el dataset con la columna `churn`.
"""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
INFILE = ROOT / "shopping_trends.csv"
OUTFILE = ROOT / "shopping_trends_churn.csv"


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def build_churn(df: pd.DataFrame, p25: int | None = None) -> pd.DataFrame:
    df = df.copy()
    df = normalize_cols(df)
    if "previous_purchases" in df.columns:
        prev = pd.to_numeric(df["previous_purchases"], errors="coerce").fillna(0)
    else:
        prev = pd.Series(0, index=df.index)
    if p25 is None:
        p25 = int(prev.quantile(0.25))
    low_freq = {"annually", "every 3 months", "quarterly", "cada 3 meses", "anual", "trimestral"}
    if "frequency_of_purchases" in df.columns:
        freq = df["frequency_of_purchases"].astype(str).str.lower().str.strip()
    else:
        freq = pd.Series("", index=df.index)
    churn = ((freq.isin(low_freq)) & (prev <= p25)).astype(int)
    df["churn"] = churn
    return df


def main() -> None:
    if not INFILE.exists():
        print(f"No se encontro {INFILE}. Coloca el CSV de Customer Shopping Trends como 'shopping_trends.csv'")
        return
       
    try:
        df = pd.read_csv(INFILE, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(INFILE, encoding="latin-1")
        except Exception:
    
            with open(INFILE, "rb") as f:
                raw = f.read()
            import io as _io
            text = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(_io.StringIO(text))
    out = build_churn(df)
    out.to_csv(OUTFILE, index=False)
    counts = out["churn"].value_counts()
    print(f"Guardado: {OUTFILE} balance de clases:\n{counts.to_string()}")


if __name__ == "__main__":
    main()
