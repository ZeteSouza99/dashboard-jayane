"""Inspeção inicial dos arquivos Excel."""
import pandas as pd

files = {
    "2025": "TRIMESTRES 2025 DIEGO.xlsx",
    "2026": "TRIMESTRES 2026 DIEGO.xlsx",
}

for ano, path in files.items():
    print("=" * 80)
    print(f"ARQUIVO {ano}: {path}")
    print("=" * 80)
    xl = pd.ExcelFile(path)
    print(f"Abas ({len(xl.sheet_names)}): {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        print(f"\n--- Aba: {sheet} ---")
        print(f"Shape: {df.shape}")
        print(f"Colunas: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Head:\n{df.head(8)}")
        print(f"Nulos por coluna:\n{df.isna().sum()}")
