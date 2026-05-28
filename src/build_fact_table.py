"""
build_fact_table.py
-------------------
Pipeline ETL: lê os Excel brutos, padroniza, transforma para formato longo,
valida e persiste como Parquet.

Saídas:
- data/staging/fato_compras.parquet   (granularidade: ano x filial x fornecedor x mês)
- data/staging/dim_fornecedor.parquet (CODFORNEC -> nomes)
- data/outputs/qa_report.txt          (relatório de qualidade)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
STAGING_DIR = ROOT / "data" / "staging"
OUTPUTS_DIR = ROOT / "data" / "outputs"

ARQUIVOS = {
    2025: RAW_DIR / "TRIMESTRES 2025 DIEGO.xlsx",
    2026: RAW_DIR / "TRIMESTRES 2026 DIEGO.xlsx",
}

MESES_COLS = [
    "JANEIRO", "FEVEREIRO", "MARCO", "ABRIL", "MAIO", "JUNHO",
    "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO",
]
MES_NUM = {m: i + 1 for i, m in enumerate(MESES_COLS)}

SCHEMA_BASE = [
    "ANO", "FILIAL", "CODFORNEC", "FORNECEDOR", "FANTASIA", "CODCOMPRADOR",
] + MESES_COLS

# --------------------------------------------------------------------------
# Camadas 1-2: Ingestão + limpeza
# --------------------------------------------------------------------------
ABAS_ESPERADAS = ["FILIAL 1", "FILIAL 2", "FILIAL 3", "FILIAL 4"]
# Identidade obrigatória; meses são opcionais (basta 1 coluna de mês).
COLUNAS_OBRIGATORIAS = {"CODFORNEC", "FILIAL", "FORNECEDOR", "CODCOMPRADOR"}


def ler_aba(path: Any, sheet: str, ano: int) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # Padroniza nomes de coluna
    df.columns = df.columns.str.strip().str.upper()

    # Descarta derivadas (vamos recalcular)
    for col in ("TOTAL", "MEDIA"):
        if col in df.columns:
            df = df.drop(columns=col)

    # Adiciona proveniência
    df["ANO"] = ano
    filial_src = int(sheet.split()[-1])
    # Valida coerência da coluna FILIAL com o nome da aba
    if "FILIAL" in df.columns and not df["FILIAL"].eq(filial_src).all():
        raise ValueError(f"{path.name}/{sheet}: coluna FILIAL diverge do nome da aba")
    df["FILIAL"] = filial_src

    # Normaliza strings
    for col in ("FORNECEDOR", "FANTASIA", "CODCOMPRADOR"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # Garante que todas as colunas de mês existem (faltantes viram NaN)
    for m in MESES_COLS:
        if m not in df.columns:
            df[m] = np.nan

    # Reindexa para schema fixo
    df = df.reindex(columns=SCHEMA_BASE)
    return df


def ingerir_arquivos(arquivos: dict[int, Any]) -> pd.DataFrame:
    """Lê todos os Excel (path ou file-like) e devolve DataFrame wide.

    `arquivos` mapeia ano -> caminho ou buffer (BytesIO).
    """
    partes: list[pd.DataFrame] = []
    for ano, path in arquivos.items():
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            partes.append(ler_aba(path, sheet, ano))
    df = pd.concat(partes, ignore_index=True)
    return df


def ingerir_tudo() -> pd.DataFrame:
    return ingerir_arquivos(ARQUIVOS)


# --------------------------------------------------------------------------
# Validação de planilha enviada via upload
# --------------------------------------------------------------------------
def detectar_ano_no_nome(nome: str) -> int | None:
    """Tenta extrair um ano (20xx) do nome do arquivo."""
    m = re.search(r"(20\d{2})", nome or "")
    return int(m.group(1)) if m else None


def validar_planilha(path_or_buf: Any, nome: str = "") -> dict:
    """Verifica se a planilha segue o padrão Jayane.

    Retorna: {ok, mensagens, abas, colunas_por_aba, ano_sugerido, linhas}.
    """
    res: dict[str, Any] = {
        "ok": False, "mensagens": [], "abas": [], "colunas_por_aba": {},
        "ano_sugerido": detectar_ano_no_nome(nome), "linhas": 0,
    }
    try:
        xl = pd.ExcelFile(path_or_buf)
    except Exception as e:  # arquivo corrompido / não é Excel
        res["mensagens"].append(f"[ERRO] Não foi possível abrir o Excel: {e}")
        return res

    res["abas"] = list(xl.sheet_names)
    abas_padronizadas = [s.strip().upper() for s in xl.sheet_names]
    faltando = [a for a in ABAS_ESPERADAS if a not in abas_padronizadas]
    if faltando:
        res["mensagens"].append(
            f"[ERRO] Abas faltando: {faltando}. Esperado: {ABAS_ESPERADAS}"
        )
        return res

    total_linhas = 0
    for sheet in xl.sheet_names:
        if sheet.strip().upper() not in ABAS_ESPERADAS:
            res["mensagens"].append(f"[AVISO] Aba ignorada: {sheet}")
            continue
        df = pd.read_excel(path_or_buf, sheet_name=sheet, nrows=0)
        cols = {c.strip().upper() for c in df.columns}
        res["colunas_por_aba"][sheet] = sorted(cols)
        faltam = COLUNAS_OBRIGATORIAS - cols
        if faltam:
            res["mensagens"].append(
                f"[ERRO] Aba '{sheet}' sem colunas: {sorted(faltam)}"
            )
            return res
        # Precisa ter pelo menos UMA coluna de mês válida
        meses_presentes = [m for m in MESES_COLS if m in cols]
        if not meses_presentes:
            res["mensagens"].append(
                f"[ERRO] Aba '{sheet}' não tem nenhuma coluna de mês reconhecida "
                f"({MESES_COLS[0]}..{MESES_COLS[-1]})."
            )
            return res
        df_full = pd.read_excel(path_or_buf, sheet_name=sheet)
        total_linhas += len(df_full)

    res["linhas"] = total_linhas
    res["ok"] = True
    res["mensagens"].append(f"[OK] Padrão Jayane validado · {total_linhas} linhas · 4 filiais")
    return res


def processar_em_memoria(arquivos: dict[int, Any]) -> dict[str, pd.DataFrame]:
    """Pipeline completo em memória: recebe {ano: buffer} e devolve os DataFrames."""
    df_wide = ingerir_arquivos(arquivos)
    df_long = para_longo(df_wide)
    return {
        "fato": df_long,
        "cons": consolidar_trimestre(df_long),
        "comp": comparativo_anual(df_long),
        "dim": dimensao_fornecedor(df_long),
        "validacoes": validar(df_wide, df_long),
    }


# --------------------------------------------------------------------------
# Camada 3: Reshape wide -> long
# --------------------------------------------------------------------------
def para_longo(df_wide: pd.DataFrame) -> pd.DataFrame:
    df_long = df_wide.melt(
        id_vars=["ANO", "FILIAL", "CODFORNEC", "FORNECEDOR", "FANTASIA", "CODCOMPRADOR"],
        value_vars=MESES_COLS,
        var_name="MES",
        value_name="VALOR",
    )
    df_long["MES_NUM"] = df_long["MES"].map(MES_NUM)
    df_long["VALOR"] = pd.to_numeric(df_long["VALOR"], errors="coerce")

    # Remove meses que não existem nos dados (NaN nos brutos)
    df_long = df_long.dropna(subset=["VALOR"]).copy()

    # Remove meses agregadamente vazios (futuro/sem movimento): por ANO+MES,
    # se o total somando todas as filiais/fornecedores for 0, considera
    # "ainda não aconteceu" e descarta — evita distorcer médias e
    # comparativos com zeros artificiais.
    soma_ano_mes = df_long.groupby(["ANO", "MES"])["VALOR"].sum()
    chaves_vazias = soma_ano_mes[soma_ano_mes <= 0].index
    if len(chaves_vazias):
        mask_vazio = df_long.set_index(["ANO", "MES"]).index.isin(chaves_vazias)
        df_long = df_long.loc[~mask_vazio].copy()

    df_long["DATA"] = pd.to_datetime(
        {"year": df_long["ANO"], "month": df_long["MES_NUM"], "day": 1}
    )
    df_long["VALOR"] = df_long["VALOR"].astype(float).round(2)
    df_long = df_long.sort_values(
        ["ANO", "FILIAL", "CODFORNEC", "MES_NUM"]
    ).reset_index(drop=True)
    return df_long


# --------------------------------------------------------------------------
# Camada 4: Métricas derivadas (consolidadas a partir do fato longo)
# --------------------------------------------------------------------------
def consolidar_trimestre(df_long: pd.DataFrame) -> pd.DataFrame:
    """Agrega por (ano, filial, fornecedor) trazendo total, média, std, pico, vale."""
    g = df_long.groupby(
        ["ANO", "FILIAL", "CODFORNEC", "FORNECEDOR", "FANTASIA", "CODCOMPRADOR"],
        as_index=False,
    )
    cons = g.agg(
        TOTAL_TRI=("VALOR", "sum"),
        MEDIA_MENSAL=("VALOR", "mean"),
        DESVIO=("VALOR", "std"),
        MIN_MES=("VALOR", "min"),
        MAX_MES=("VALOR", "max"),
    )
    cons["DESVIO"] = cons["DESVIO"].fillna(0.0)
    cons["INATIVO_TRIMESTRE"] = cons["TOTAL_TRI"] == 0
    cons[["TOTAL_TRI", "MEDIA_MENSAL", "DESVIO", "MIN_MES", "MAX_MES"]] = cons[
        ["TOTAL_TRI", "MEDIA_MENSAL", "DESVIO", "MIN_MES", "MAX_MES"]
    ].round(2)
    return cons


def dimensao_fornecedor(df_long: pd.DataFrame) -> pd.DataFrame:
    """Dim: 1 linha por CODFORNEC com nomes mais frequentes (lida com variações)."""
    dim = (
        df_long.groupby("CODFORNEC")
        .agg(
            FORNECEDOR=("FORNECEDOR", lambda s: s.mode().iat[0]),
            FANTASIA=("FANTASIA", lambda s: s.mode().iat[0]),
            N_REGISTROS=("VALOR", "size"),
        )
        .reset_index()
    )
    return dim


# --------------------------------------------------------------------------
# Camada 5: Validações
# --------------------------------------------------------------------------
def validar(df_wide: pd.DataFrame, df_long: pd.DataFrame) -> list[str]:
    achados: list[str] = []

    # 1) Não pode haver nulo nos meses
    nulos = df_long["VALOR"].isna().sum()
    if nulos:
        achados.append(f"[ERRO] {nulos} valores nulos em VALOR")

    # 2) Valores não-negativos
    neg = (df_long["VALOR"] < 0).sum()
    if neg:
        achados.append(f"[ERRO] {neg} valores negativos detectados")

    # 3) Unicidade (ANO, FILIAL, CODFORNEC, MES)
    dup = df_long.duplicated(["ANO", "FILIAL", "CODFORNEC", "MES"]).sum()
    if dup:
        achados.append(f"[ERRO] {dup} duplicatas em (ANO, FILIAL, CODFORNEC, MES)")

    # 4) CODFORNEC único por (ano, filial)
    dup2 = df_wide.duplicated(["ANO", "FILIAL", "CODFORNEC"]).sum()
    if dup2:
        achados.append(f"[ERRO] {dup2} CODFORNEC repetido na mesma (ANO, FILIAL)")

    if not achados:
        achados.append("[OK] Todas as validações passaram.")
    return achados


def comparativo_anual(df_long: pd.DataFrame) -> pd.DataFrame:
    """Por fornecedor: total 2025 vs 2026 (somando filiais)."""
    piv = (
        df_long.groupby(["CODFORNEC", "ANO"])["VALOR"].sum().unstack("ANO").fillna(0.0)
    )
    piv.columns = [f"TOTAL_{c}" for c in piv.columns]
    if {"TOTAL_2025", "TOTAL_2026"}.issubset(piv.columns):
        piv["VAR_ABS"] = piv["TOTAL_2026"] - piv["TOTAL_2025"]
        piv["VAR_PCT"] = np.where(
            piv["TOTAL_2025"] > 0,
            (piv["TOTAL_2026"] / piv["TOTAL_2025"] - 1) * 100,
            np.nan,
        )

        def status(row):
            if row["TOTAL_2025"] == 0 and row["TOTAL_2026"] > 0:
                return "NOVO"
            if row["TOTAL_2026"] == 0 and row["TOTAL_2025"] > 0:
                return "PERDIDO"
            if pd.isna(row["VAR_PCT"]):
                return "INDEFINIDO"
            if abs(row["VAR_PCT"]) < 5:
                return "ESTAVEL"
            return "CRESCENTE" if row["VAR_PCT"] > 0 else "DECRESCENTE"

        piv["STATUS"] = piv.apply(status, axis=1)
    return piv.reset_index().round(2)


# --------------------------------------------------------------------------
# Camada 6: Persistência + relatório
# --------------------------------------------------------------------------
def main() -> int:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Lendo Excel brutos...")
    df_wide = ingerir_tudo()
    print(f"      {len(df_wide)} linhas wide carregadas")

    print("[2/6] Reshape para formato longo...")
    df_long = para_longo(df_wide)
    print(f"      {len(df_long)} linhas long geradas")

    print("[3/6] Consolidando trimestre...")
    cons = consolidar_trimestre(df_long)

    print("[4/6] Comparativo anual por fornecedor...")
    comp = comparativo_anual(df_long)

    print("[5/6] Construindo dimensão fornecedor...")
    dim = dimensao_fornecedor(df_long)

    print("[6/6] Validando + persistindo...")
    achados = validar(df_wide, df_long)

    df_long.to_parquet(STAGING_DIR / "fato_compras.parquet", index=False)
    cons.to_parquet(STAGING_DIR / "consolidado_trimestre.parquet", index=False)
    comp.to_parquet(STAGING_DIR / "comparativo_anual.parquet", index=False)
    dim.to_parquet(STAGING_DIR / "dim_fornecedor.parquet", index=False)

    # Relatório QA
    linhas = []
    linhas.append("=" * 70)
    linhas.append("RELATÓRIO DE QUALIDADE - PIPELINE DASHBOARD JAYANE")
    linhas.append("=" * 70)
    linhas.append(f"Linhas fato (long): {len(df_long)}")
    linhas.append(f"Linhas consolidado: {len(cons)}")
    linhas.append(f"Fornecedores únicos: {df_long['CODFORNEC'].nunique()}")
    linhas.append(f"Filiais: {sorted(df_long['FILIAL'].unique().tolist())}")
    linhas.append(f"Anos: {sorted(df_long['ANO'].unique().tolist())}")
    linhas.append(f"Compradores: {df_long['CODCOMPRADOR'].unique().tolist()}")
    linhas.append("")
    linhas.append("VALOR TOTAL POR (ANO, FILIAL):")
    pv = (
        df_long.groupby(["ANO", "FILIAL"])["VALOR"]
        .sum()
        .unstack("FILIAL")
        .round(2)
    )
    linhas.append(pv.to_string())
    linhas.append("")
    linhas.append("VALIDAÇÕES:")
    linhas.extend(f"  {a}" for a in achados)
    linhas.append("")
    if "STATUS" in comp.columns:
        linhas.append("FORNECEDORES NOVOS (2026):")
        novos = comp[comp["STATUS"] == "NOVO"]["CODFORNEC"].tolist()
        linhas.append(f"  {novos}")
        linhas.append("FORNECEDORES PERDIDOS (sumiram em 2026):")
        perd = comp[comp["STATUS"] == "PERDIDO"]["CODFORNEC"].tolist()
        linhas.append(f"  {perd}")
    linhas.append("=" * 70)

    relatorio = "\n".join(linhas)
    (OUTPUTS_DIR / "qa_report.txt").write_text(relatorio, encoding="utf-8")
    print()
    print(relatorio)
    print()
    print("Arquivos gerados em data/staging/:")
    for p in sorted(STAGING_DIR.glob("*.parquet")):
        print(f"  - {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.1f} KB)")

    return 0 if all("[ERRO]" not in a for a in achados) else 1


if __name__ == "__main__":
    sys.exit(main())
