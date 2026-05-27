"""
analytics_demo.py
-----------------
Demonstração rápida consumindo o parquet gerado por build_fact_table.py.

Mostra:
  - Top fornecedores por total no biênio
  - Curva ABC (Pareto) consolidada
  - Sazonalidade (índice mês/média) por filial
  - Maiores crescimentos e quedas 2025 -> 2026
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "data" / "staging"

fato = pd.read_parquet(STAGING / "fato_compras.parquet")
cons = pd.read_parquet(STAGING / "consolidado_trimestre.parquet")
comp = pd.read_parquet(STAGING / "comparativo_anual.parquet")
dim = pd.read_parquet(STAGING / "dim_fornecedor.parquet")

pd.options.display.float_format = "{:,.2f}".format

print("=" * 70)
print("TOP 10 FORNECEDORES - TOTAL CONSOLIDADO 2025 + 2026")
print("=" * 70)
top = (
    fato.groupby("CODFORNEC")["VALOR"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .merge(dim[["CODFORNEC", "FANTASIA"]], on="CODFORNEC")
)
print(top.to_string(index=False))

print("\n" + "=" * 70)
print("CURVA ABC (Pareto)")
print("=" * 70)
abc = (
    fato.groupby("CODFORNEC")["VALOR"].sum().sort_values(ascending=False).reset_index()
)
abc["PARTICIPACAO_PCT"] = abc["VALOR"] / abc["VALOR"].sum() * 100
abc["ACUM_PCT"] = abc["PARTICIPACAO_PCT"].cumsum()
abc["CLASSE"] = np.where(
    abc["ACUM_PCT"] <= 80, "A", np.where(abc["ACUM_PCT"] <= 95, "B", "C")
)
abc = abc.merge(dim[["CODFORNEC", "FANTASIA"]], on="CODFORNEC")
print(abc.to_string(index=False))

print("\n" + "=" * 70)
print("SAZONALIDADE (índice VALOR_MES / MEDIA_TRI) por FILIAL")
print("=" * 70)
mensal = fato.groupby(["FILIAL", "MES"])["VALOR"].sum().unstack("MES")
mensal = mensal[["JANEIRO", "FEVEREIRO", "MARCO"]]
indice = mensal.div(mensal.mean(axis=1), axis=0).round(3)
print("Soma por mês (R$):")
print(mensal.round(2).to_string())
print("\nÍndice de sazonalidade (1.0 = média do trimestre):")
print(indice.to_string())

print("\n" + "=" * 70)
print("MAIORES CRESCIMENTOS 2025 -> 2026 (top 5 absolutos +)")
print("=" * 70)
cresc = (
    comp.sort_values("VAR_ABS", ascending=False)
    .head(5)
    .merge(dim[["CODFORNEC", "FANTASIA"]], on="CODFORNEC")
)
print(cresc[["CODFORNEC", "FANTASIA", "TOTAL_2025", "TOTAL_2026", "VAR_ABS", "VAR_PCT", "STATUS"]].to_string(index=False))

print("\n" + "=" * 70)
print("MAIORES QUEDAS 2025 -> 2026 (top 5 absolutos -)")
print("=" * 70)
quedas = (
    comp.sort_values("VAR_ABS")
    .head(5)
    .merge(dim[["CODFORNEC", "FANTASIA"]], on="CODFORNEC")
)
print(quedas[["CODFORNEC", "FANTASIA", "TOTAL_2025", "TOTAL_2026", "VAR_ABS", "VAR_PCT", "STATUS"]].to_string(index=False))

print("\n" + "=" * 70)
print("MÉDIA MENSAL POR FORNECEDOR (consolidado todas filiais, ano 2026)")
print("=" * 70)
med_2026 = (
    fato.query("ANO == 2026")
    .groupby("CODFORNEC")["VALOR"]
    .agg(MEDIA_MENSAL="mean", TOTAL="sum", DESVIO="std")
    .round(2)
    .sort_values("TOTAL", ascending=False)
    .reset_index()
    .merge(dim[["CODFORNEC", "FANTASIA"]], on="CODFORNEC")
)
print(med_2026.to_string(index=False))
