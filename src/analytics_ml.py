"""
analytics_ml.py
---------------
Camada de ciência de dados sobre o fato de compras.

Gera, a partir de `data/staging/fato_compras.parquet`:
  - data/staging/forecast.parquet           previsão do próximo mês (ensemble)
  - data/staging/forecast_total.parquet     previsão consolidada (filial e geral)
  - data/staging/clusters.parquet           segmentação KMeans de fornecedores
  - data/staging/anomalias.parquet          z-score + IsolationForest
  - data/staging/metricas_avancadas.parquet HHI, Gini, concentração, growth
  - data/outputs/ml_report.txt              relatório textual

Bibliotecas: pandas, numpy, scipy, scikit-learn.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "data" / "staging"
OUTPUTS = ROOT / "data" / "outputs"

MES_NUM = {"JANEIRO": 1, "FEVEREIRO": 2, "MARCO": 3,
           "ABRIL": 4, "MAIO": 5, "JUNHO": 6,
           "JULHO": 7, "AGOSTO": 8, "SETEMBRO": 9,
           "OUTUBRO": 10, "NOVEMBRO": 11, "DEZEMBRO": 12}
NUM_MES = {v: k for k, v in MES_NUM.items()}
MES_PT_CURTO = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
                7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


# ============================================================
# 1. Previsão do próximo mês
# ============================================================
def _proximo_mes(df_long: pd.DataFrame) -> tuple[int, int]:
    """Retorna (ano, mes_num) do próximo mês a ser previsto."""
    ult = df_long.sort_values(["ANO", "MES_NUM"]).iloc[-1]
    ano, mes = int(ult["ANO"]), int(ult["MES_NUM"])
    if mes == 12:
        return ano + 1, 1
    return ano, mes + 1


def _projecao_serie(valores: np.ndarray) -> dict:
    """Aplica vários modelos sobre uma série curta e retorna projeção do próximo ponto."""
    n = len(valores)
    if n == 0 or np.all(np.isnan(valores)):
        return dict(naive=0.0, media=0.0, tendencia=0.0,
                    ensemble=0.0, ic_inf=0.0, ic_sup=0.0, std=0.0)

    v = np.asarray(valores, dtype=float)
    naive = float(v[-1])
    media = float(np.mean(v))

    # Tendência linear (OLS)
    if n >= 2 and np.std(v) > 0:
        x = np.arange(n).reshape(-1, 1)
        reg = LinearRegression().fit(x, v)
        tend = float(reg.predict(np.array([[n]]))[0])
        tend = max(0.0, tend)  # compras não negativas
    else:
        tend = naive

    # Mediana ponderada (mais peso ao recente)
    pesos = np.linspace(1, 2, n)
    media_pond = float(np.average(v, weights=pesos))

    candidatos = np.array([naive, media, tend, media_pond])
    ensemble = float(np.mean(candidatos))
    std = float(np.std(candidatos))
    # IC empírico: ensemble ± 1.5 std dos candidatos OU desvio histórico
    desvio_serie = float(np.std(v)) if n >= 2 else std
    spread = max(std, desvio_serie * 0.5)
    return dict(
        naive=naive, media=media, tendencia=tend, media_pond=media_pond,
        ensemble=max(0.0, ensemble),
        ic_inf=max(0.0, ensemble - 1.5 * spread),
        ic_sup=ensemble + 1.5 * spread,
        std=spread,
    )


def forecast_por_serie(df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Previsão por (filial, fornecedor) e consolidada (filial / total)."""
    ano_alvo, mes_alvo = _proximo_mes(df_long)

    # ---- por (filial, fornecedor)
    linhas = []
    for (filial, codf, nome), g in df_long.groupby(
        ["FILIAL", "CODFORNEC", "FANTASIA"]
    ):
        # série até o último ponto (todas as observações ordenadas)
        g_ord = g.sort_values(["ANO", "MES_NUM"])
        proj = _projecao_serie(g_ord["VALOR"].to_numpy())
        linhas.append(dict(
            FILIAL=filial, CODFORNEC=codf, FANTASIA=nome,
            ANO=ano_alvo, MES_NUM=mes_alvo,
            MES=NUM_MES.get(mes_alvo, str(mes_alvo)),
            **proj,
        ))
    forecast = pd.DataFrame(linhas)

    # ---- por filial (soma agregada)
    linhas_fil = []
    serie_filial = (df_long.groupby(["FILIAL", "ANO", "MES_NUM"])["VALOR"]
                    .sum().reset_index())
    for filial, g in serie_filial.groupby("FILIAL"):
        proj = _projecao_serie(
            g.sort_values(["ANO", "MES_NUM"])["VALOR"].to_numpy()
        )
        linhas_fil.append(dict(
            ESCOPO="FILIAL", CHAVE=f"Filial {filial}", FILIAL=filial,
            ANO=ano_alvo, MES_NUM=mes_alvo, **proj,
        ))

    # ---- geral
    serie_geral = (df_long.groupby(["ANO", "MES_NUM"])["VALOR"].sum()
                   .sort_index().to_numpy())
    proj_geral = _projecao_serie(serie_geral)
    linhas_fil.append(dict(
        ESCOPO="GERAL", CHAVE="TOTAL", FILIAL=-1,
        ANO=ano_alvo, MES_NUM=mes_alvo, **proj_geral,
    ))

    forecast_total = pd.DataFrame(linhas_fil)

    meta = dict(ano_alvo=ano_alvo, mes_alvo=mes_alvo,
                mes_alvo_nome=NUM_MES.get(mes_alvo, str(mes_alvo)))
    return forecast, forecast_total, meta


# ============================================================
# 2. Modelo global supervisionado (RandomForest)
# ============================================================
def modelo_global_rf(df_long: pd.DataFrame, ano_alvo: int, mes_alvo: int):
    """
    Treina RandomForest com features (FILIAL, CODFORNEC encoded, MES_NUM, ANO,
    LAG_1, MEDIA_HIST) e prediz o próximo mês.
    Retorna df de previsão e métricas de validação.
    """
    base = df_long.copy().sort_values(["FILIAL", "CODFORNEC", "ANO", "MES_NUM"])
    # Lag e média histórica (cumulativa anterior)
    base["LAG_1"] = (base.groupby(["FILIAL", "CODFORNEC"])["VALOR"]
                     .shift(1).fillna(0))
    base["MEDIA_HIST"] = (
        base.groupby(["FILIAL", "CODFORNEC"])["VALOR"]
        .expanding().mean().reset_index(level=[0, 1], drop=True).fillna(0)
    )

    # Codifica CODFORNEC como categoria numérica
    base["FORNEC_CODE"] = base["CODFORNEC"].astype("category").cat.codes

    feats = ["FILIAL", "FORNEC_CODE", "MES_NUM", "ANO", "LAG_1", "MEDIA_HIST"]
    X = base[feats].to_numpy()
    y = base["VALOR"].to_numpy()

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )

    # Validação cruzada (5-fold) — métrica MAE e R²
    cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
    try:
        mae_scores = -cross_val_score(rf, X, y, cv=cv,
                                       scoring="neg_mean_absolute_error", n_jobs=-1)
        r2_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=-1)
        mae_cv = float(np.mean(mae_scores))
        r2_cv = float(np.mean(r2_scores))
    except Exception:
        mae_cv, r2_cv = np.nan, np.nan

    rf.fit(X, y)
    # Importância das features
    importances = dict(zip(feats, rf.feature_importances_.tolist()))

    # Predição: para cada (filial, fornecedor) usa último ponto como LAG_1
    ult = (base.sort_values(["ANO", "MES_NUM"])
           .groupby(["FILIAL", "CODFORNEC"]).tail(1))
    pred_rows = []
    for _, r in ult.iterrows():
        x_new = np.array([[r["FILIAL"], r["FORNEC_CODE"], mes_alvo, ano_alvo,
                            r["VALOR"], r["MEDIA_HIST"]]])
        y_hat = float(rf.predict(x_new)[0])
        pred_rows.append(dict(
            FILIAL=int(r["FILIAL"]), CODFORNEC=r["CODFORNEC"],
            FANTASIA=r["FANTASIA"], ANO=ano_alvo, MES_NUM=mes_alvo,
            RF_PREDICAO=max(0.0, y_hat),
        ))
    pred = pd.DataFrame(pred_rows)

    metricas = dict(mae_cv=mae_cv, r2_cv=r2_cv, n_obs=len(X),
                    importancias=importances)
    return pred, metricas


# ============================================================
# 3. Clustering de fornecedores (KMeans)
# ============================================================
def clusterizar_fornecedores(df_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Agrega features por fornecedor e aplica KMeans com k escolhido por silhouette.
    Features: TOTAL, MEDIA, CV, GROWTH_YOY, SAZ_JAN, SAZ_FEV, SAZ_MAR, N_FILIAIS.
    """
    g = df_long.groupby("CODFORNEC")
    base = g.agg(
        FANTASIA=("FANTASIA", "first"),
        TOTAL=("VALOR", "sum"),
        MEDIA=("VALOR", "mean"),
        DESVIO=("VALOR", "std"),
        N_FILIAIS=("FILIAL", "nunique"),
        N_OBS=("VALOR", "size"),
    ).reset_index()
    base["DESVIO"] = base["DESVIO"].fillna(0)
    base["CV"] = np.where(base["MEDIA"] > 0, base["DESVIO"] / base["MEDIA"], 0)

    # Growth YoY (2026 / 2025)
    piv_ano = df_long.groupby(["CODFORNEC", "ANO"])["VALOR"].sum().unstack().fillna(0)
    if 2025 in piv_ano.columns and 2026 in piv_ano.columns:
        base = base.merge(
            piv_ano.reset_index().rename(columns={2025: "T2025", 2026: "T2026"}),
            on="CODFORNEC", how="left",
        )
        base["GROWTH_YOY"] = np.where(
            base["T2025"] > 0, base["T2026"] / base["T2025"] - 1, 0
        )
    else:
        base["GROWTH_YOY"] = 0

    # Sazonalidade (média do mês / média geral do fornecedor)
    piv_mes = (df_long.groupby(["CODFORNEC", "MES_NUM"])["VALOR"]
                .mean().unstack().fillna(0))
    media_forn = piv_mes.mean(axis=1).replace(0, 1)
    saz = piv_mes.div(media_forn, axis=0).add_prefix("SAZ_M")
    base = base.merge(saz.reset_index(), on="CODFORNEC", how="left")

    feats_cols = [c for c in base.columns
                  if c.startswith("SAZ_M") or c in ("CV", "GROWTH_YOY")]
    feats_cols = list(dict.fromkeys(feats_cols + ["TOTAL", "MEDIA"]))

    X = base[feats_cols].fillna(0).to_numpy()
    if len(X) < 3:
        base["CLUSTER"] = 0
        return base, dict(k=1, silhouette=None, features=feats_cols)

    Xs = StandardScaler().fit_transform(X)

    # Escolha de k por silhueta (k de 2 a min(6, n-1))
    melhor_k, melhor_sil = 2, -1
    for k in range(2, min(6, len(X)) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
            sil = silhouette_score(Xs, km.labels_)
            if sil > melhor_sil:
                melhor_k, melhor_sil = k, sil
        except Exception:
            continue

    km = KMeans(n_clusters=melhor_k, random_state=42, n_init=10).fit(Xs)
    base["CLUSTER"] = km.labels_

    # Rotula clusters por perfil (alto valor, alta variação, crescimento, etc.)
    perfis = (base.groupby("CLUSTER")
              .agg(total=("TOTAL", "mean"),
                   cv=("CV", "mean"),
                   growth=("GROWTH_YOY", "mean"))
              .reset_index())
    perfis = perfis.sort_values("total", ascending=False).reset_index(drop=True)
    rotulos = {}
    for i, r in perfis.iterrows():
        if i == 0:
            tag = "Estratégico (alto volume)"
        elif r["growth"] > 0.15:
            tag = "Em crescimento"
        elif r["growth"] < -0.15:
            tag = "Em retração"
        elif r["cv"] > 0.5:
            tag = "Volátil"
        else:
            tag = "Cauda longa estável"
        rotulos[int(r["CLUSTER"])] = tag
    base["CLUSTER_ROTULO"] = base["CLUSTER"].map(rotulos)

    meta = dict(k=melhor_k, silhouette=float(melhor_sil), features=feats_cols,
                rotulos=rotulos)
    return base, meta


# ============================================================
# 4. Detecção de anomalias
# ============================================================
def detectar_anomalias(df_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Z-score por fornecedor (anomalia se |z|>2) +
    IsolationForest global sobre (FILIAL, VALOR, MES_NUM, ANO).
    """
    df = df_long.copy()
    df["MEDIA_FORN"] = df.groupby("CODFORNEC")["VALOR"].transform("mean")
    df["STD_FORN"] = df.groupby("CODFORNEC")["VALOR"].transform("std").fillna(0)
    df["Z_SCORE"] = np.where(df["STD_FORN"] > 0,
                              (df["VALOR"] - df["MEDIA_FORN"]) / df["STD_FORN"], 0)
    df["ANOMALIA_Z"] = df["Z_SCORE"].abs() > 2

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    feats = df[["FILIAL", "VALOR", "MES_NUM", "ANO"]].to_numpy()
    df["ISO_SCORE"] = iso.fit(feats).decision_function(feats)
    df["ANOMALIA_ISO"] = iso.predict(feats) == -1

    df["ANOMALIA"] = df["ANOMALIA_Z"] | df["ANOMALIA_ISO"]

    cols = ["ANO", "FILIAL", "MES", "MES_NUM", "DATA", "CODFORNEC", "FANTASIA",
            "VALOR", "MEDIA_FORN", "Z_SCORE", "ISO_SCORE",
            "ANOMALIA_Z", "ANOMALIA_ISO", "ANOMALIA"]
    out = df[cols].sort_values("Z_SCORE", key=abs, ascending=False)

    meta = dict(
        n_anomalias_z=int(df["ANOMALIA_Z"].sum()),
        n_anomalias_iso=int(df["ANOMALIA_ISO"].sum()),
        n_total=int(df["ANOMALIA"].sum()),
        pct_total=float(df["ANOMALIA"].mean() * 100),
    )
    return out, meta


# ============================================================
# 5. Métricas avançadas de concentração e crescimento
# ============================================================
def _hhi(shares: np.ndarray) -> float:
    """Índice de Herfindahl-Hirschman (0–10000)."""
    s = shares / shares.sum() if shares.sum() else shares
    return float(np.sum((s * 100) ** 2))


def _gini(valores: np.ndarray) -> float:
    """Coeficiente de Gini."""
    v = np.sort(np.asarray(valores, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def metricas_avancadas(df_long: pd.DataFrame) -> pd.DataFrame:
    """Concentração, growth, share top 5/10 por dimensão (geral, por filial, por ano)."""
    linhas = []

    def _bloco(escopo: str, chave: str, sub: pd.DataFrame):
        if sub.empty:
            return
        por_forn = sub.groupby("CODFORNEC")["VALOR"].sum().sort_values(ascending=False)
        total = float(por_forn.sum())
        if total == 0:
            return
        share = por_forn / total
        top5 = float(share.head(5).sum() * 100)
        top10 = float(share.head(10).sum() * 100)
        hhi = _hhi(por_forn.to_numpy())
        gini = _gini(por_forn.to_numpy())
        cr = sub.groupby(["ANO", "MES_NUM"])["VALOR"].sum()
        crescimento_mom = float(cr.pct_change().dropna().mean() * 100) if len(cr) > 1 else 0.0
        linhas.append(dict(
            ESCOPO=escopo, CHAVE=chave,
            TOTAL=total, N_FORNECEDORES=int((por_forn > 0).sum()),
            TOP5_PCT=top5, TOP10_PCT=top10,
            HHI=hhi, GINI=gini, CRESC_MEDIO_MOM_PCT=crescimento_mom,
        ))

    _bloco("GERAL", "Total", df_long)
    for ano, sub in df_long.groupby("ANO"):
        _bloco("ANO", str(ano), sub)
    for filial, sub in df_long.groupby("FILIAL"):
        _bloco("FILIAL", f"Filial {filial}", sub)
    for (ano, filial), sub in df_long.groupby(["ANO", "FILIAL"]):
        _bloco("ANO_FILIAL", f"{ano} · Filial {filial}", sub)

    return pd.DataFrame(linhas).round(2)


# ============================================================
# 6. Pipeline principal
# ============================================================
def main() -> int:
    print("[ML 1/6] Carregando fato_compras.parquet...")
    fato = pd.read_parquet(STAGING / "fato_compras.parquet")

    print("[ML 2/6] Forecast ensemble (próximo mês)...")
    fc, fc_total, meta_fc = forecast_por_serie(fato)
    print(f"        Alvo: {meta_fc['mes_alvo_nome'].title()}/{meta_fc['ano_alvo']}")

    print("[ML 3/6] RandomForest global + validação cruzada...")
    pred_rf, met_rf = modelo_global_rf(fato, meta_fc["ano_alvo"], meta_fc["mes_alvo"])
    # Junta predição RF com ensemble
    fc = fc.merge(pred_rf[["FILIAL", "CODFORNEC", "RF_PREDICAO"]],
                  on=["FILIAL", "CODFORNEC"], how="left")
    fc["PREDICAO_FINAL"] = ((fc["ensemble"] + fc["RF_PREDICAO"]) / 2).clip(lower=0)

    print(f"        MAE CV: {met_rf['mae_cv']:,.2f}  |  R² CV: {met_rf['r2_cv']:.3f}")

    print("[ML 4/6] Clusterização de fornecedores (KMeans + silhueta)...")
    clusters, meta_cl = clusterizar_fornecedores(fato)
    print(f"        k={meta_cl['k']}  silhouette={meta_cl['silhouette']:.3f}")

    print("[ML 5/6] Detecção de anomalias (z-score + IsolationForest)...")
    anom, meta_an = detectar_anomalias(fato)
    print(f"        {meta_an['n_total']} anomalias ({meta_an['pct_total']:.1f}%)")

    print("[ML 6/6] Métricas avançadas (HHI, Gini, concentração)...")
    metr = metricas_avancadas(fato)

    # Persistência
    fc.to_parquet(STAGING / "forecast.parquet", index=False)
    fc_total.to_parquet(STAGING / "forecast_total.parquet", index=False)
    clusters.to_parquet(STAGING / "clusters.parquet", index=False)
    anom.to_parquet(STAGING / "anomalias.parquet", index=False)
    metr.to_parquet(STAGING / "metricas_avancadas.parquet", index=False)

    # Relatório
    linhas = ["=" * 70, "RELATÓRIO ML — DASHBOARD JAYANE", "=" * 70]
    linhas.append(f"Observações treinadas: {met_rf['n_obs']}")
    linhas.append(f"MAE (cross-validation 5-fold): R$ {met_rf['mae_cv']:,.2f}")
    linhas.append(f"R²  (cross-validation 5-fold): {met_rf['r2_cv']:.3f}")
    linhas.append("")
    linhas.append("Importância das features (RandomForest):")
    for k, v in sorted(met_rf["importancias"].items(), key=lambda kv: -kv[1]):
        linhas.append(f"  {k:<14s} {v*100:5.1f}%")
    linhas.append("")
    linhas.append(f"Próximo mês previsto: {meta_fc['mes_alvo_nome'].title()}/{meta_fc['ano_alvo']}")
    geral = fc_total[fc_total["ESCOPO"] == "GERAL"].iloc[0]
    linhas.append(
        f"  Total previsto: R$ {geral['ensemble']:,.2f}  "
        f"(IC: R$ {geral['ic_inf']:,.2f} — R$ {geral['ic_sup']:,.2f})"
    )
    linhas.append("")
    linhas.append(f"Clusters: k={meta_cl['k']}  silhouette={meta_cl['silhouette']:.3f}")
    for cid, rot in meta_cl["rotulos"].items():
        n = int((clusters["CLUSTER"] == cid).sum())
        linhas.append(f"  Cluster {cid} ({rot}): {n} fornecedores")
    linhas.append("")
    linhas.append(f"Anomalias detectadas: {meta_an['n_total']} ({meta_an['pct_total']:.1f}%)")
    linhas.append(f"  por z-score: {meta_an['n_anomalias_z']}")
    linhas.append(f"  por IsolationForest: {meta_an['n_anomalias_iso']}")
    linhas.append("")
    linhas.append("Concentração (geral):")
    g = metr[metr["ESCOPO"] == "GERAL"].iloc[0]
    linhas.append(f"  Top 5 indústrias = {g['TOP5_PCT']:.1f}% do total")
    linhas.append(f"  Top 10 indústrias = {g['TOP10_PCT']:.1f}% do total")
    linhas.append(f"  HHI = {g['HHI']:.0f}  |  Gini = {g['GINI']:.3f}")
    linhas.append("=" * 70)
    relat = "\n".join(linhas)
    (OUTPUTS / "ml_report.txt").write_text(relat, encoding="utf-8")
    print()
    print(relat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
