"""
dashboard.py — Dashboard executivo Jayane.

Tema: corporativo, paleta teal/slate, tipografia Inter, cards customizados.
Rodar:
    .\\.venv\\Scripts\\python.exe -m streamlit run src/dashboard.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "data" / "staging"
sys.path.insert(0, str(ROOT / "src"))
from build_fact_table import (  # noqa: E402
    processar_em_memoria,
    validar_planilha,
    detectar_ano_no_nome,
)

# =====================================================================
# Configuração e tema
# =====================================================================
st.set_page_config(
    page_title="Jayane • Inteligência de Compras",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta corporativa
PRIMARY = "#0F766E"        # teal-700
PRIMARY_LIGHT = "#14B8A6"  # teal-500
ACCENT = "#F59E0B"         # amber-500
SUCCESS = "#16A34A"
DANGER = "#DC2626"
NEUTRAL = "#64748B"        # slate-500

ANO_COLORS = {2025: "#94A3B8", 2026: PRIMARY}
SEQ_TEAL = ["#CCFBF1", "#5EEAD4", "#14B8A6", "#0F766E", "#134E4A"]
CATEGORICAL = [
    "#0F766E", "#F59E0B", "#3B82F6", "#8B5CF6", "#EC4899",
    "#10B981", "#EF4444", "#06B6D4", "#A855F7", "#F97316",
    "#22C55E", "#0EA5E9", "#E11D48", "#84CC16", "#6366F1",
    "#14B8A6", "#FB923C", "#D946EF",
]

PLOTLY_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", color="#0F172A", size=13),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=CATEGORICAL,
        xaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", linecolor="#E2E8F0", zerolinecolor="#E2E8F0"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0)", borderwidth=0),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", font_size=12),
        title=dict(font=dict(size=15, color="#0F172A"), x=0, xanchor="left"),
    )
)

# =====================================================================
# CSS customizado
# =====================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stApp {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .app-header {
        background: linear-gradient(135deg, #0F766E 0%, #134E4A 100%);
        padding: 1.5rem 1.75rem;
        border-radius: 16px;
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px -8px rgba(15, 118, 110, 0.35);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .app-header h1 {
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-header .subtitle {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-top: 0.25rem;
    }
    .app-header .pill {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(8px);
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.2);
    }

    .kpi {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        height: 100%;
    }
    .kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -6px rgba(15, 23, 42, 0.08);
    }
    .kpi .label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.4rem;
    }
    .kpi .value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .kpi .delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.45rem;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.18rem 0.55rem;
        border-radius: 6px;
    }
    .kpi .delta.up { color: #15803D; background: #DCFCE7; }
    .kpi .delta.down { color: #B91C1C; background: #FEE2E2; }
    .kpi .delta.neutral { color: #475569; background: #F1F5F9; }
    .kpi .accent-bar {
        width: 32px;
        height: 3px;
        border-radius: 999px;
        background: #0F766E;
        margin-bottom: 0.7rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin: 1.5rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    .section-title::before {
        content: "";
        width: 4px;
        height: 18px;
        background: #0F766E;
        border-radius: 2px;
    }

    section[data-testid="stSidebar"] {
        background: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #F1F5F9;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 16px;
        background: transparent;
        border-radius: 7px;
        color: #475569;
        font-weight: 500;
        font-size: 0.88rem;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F766E !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }

    .stDataFrame { border-radius: 10px; overflow: hidden; }

    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Utils
# =====================================================================
def fmt_brl(x: float) -> str:
    if pd.isna(x):
        return "—"
    s = f"R$ {x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_brl_compact(x: float) -> str:
    if pd.isna(x):
        return "—"
    a = abs(x)
    if a >= 1e6:
        return f"R$ {x/1e6:,.2f} mi".replace(".", ",")
    if a >= 1e3:
        return f"R$ {x/1e3:,.1f} mil".replace(".", ",")
    return fmt_brl(x)


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_dir: str = "neutral") -> str:
    delta_html = ""
    if delta:
        arrow = {"up": "▲", "down": "▼", "neutral": "•"}[delta_dir]
        delta_html = f'<div class="delta {delta_dir}">{arrow} {delta}</div>'
    return f"""
    <div class="kpi">
        <div class="accent-bar"></div>
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


def apply_template(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE)
    if height:
        fig.update_layout(height=height)
    return fig


# =====================================================================
# Carregamento
# =====================================================================
if not (STAGING / "fato_compras.parquet").exists():
    st.error(
        "**Parquets não encontrados.** Rode primeiro: `python src/build_fact_table.py`"
    )
    st.stop()


@st.cache_data(show_spinner=False)
def load_data():
    def _opt(name):
        p = STAGING / name
        return pd.read_parquet(p) if p.exists() else None
    return {
        "fato": pd.read_parquet(STAGING / "fato_compras.parquet"),
        "cons": pd.read_parquet(STAGING / "consolidado_trimestre.parquet"),
        "comp": pd.read_parquet(STAGING / "comparativo_anual.parquet"),
        "dim": pd.read_parquet(STAGING / "dim_fornecedor.parquet"),
        "forecast": _opt("forecast.parquet"),
        "forecast_total": _opt("forecast_total.parquet"),
        "clusters": _opt("clusters.parquet"),
        "anomalias": _opt("anomalias.parquet"),
        "metricas": _opt("metricas_avancadas.parquet"),
    }


dados = st.session_state.get("dados_upload") or load_data()
fonte_label = st.session_state.get("fonte_label", "Base oficial (parquets)")
fato = dados["fato"].drop(columns=["FANTASIA", "FORNECEDOR"]).merge(
    dados["dim"][["CODFORNEC", "FANTASIA", "FORNECEDOR"]], on="CODFORNEC"
)
comp = dados["comp"]
dim = dados["dim"]
forecast_df = dados["forecast"]
forecast_total = dados["forecast_total"]
clusters_df = dados["clusters"]
anomalias_df = dados["anomalias"]
metricas_df = dados["metricas"]
HAS_ML = all(x is not None for x in (forecast_df, forecast_total, clusters_df,
                                      anomalias_df, metricas_df))

MES_ORDEM = ["JANEIRO", "FEVEREIRO", "MARCO"]
MES_PT = {"JANEIRO": "Jan", "FEVEREIRO": "Fev", "MARCO": "Mar"}

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown(
        '<div style="padding:0.5rem 0 1rem 0; border-bottom:1px solid #E2E8F0; margin-bottom:0.5rem">'
        '<div style="font-size:1.1rem; font-weight:700; color:#0F766E;">Jayane</div>'
        '<div style="font-size:0.75rem; color:#64748B;">Inteligência de Compras</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("📤 Carregar nova planilha", width="stretch"):
        st.session_state["upload_mode"] = True
        st.rerun()
    if "dados_upload" in st.session_state:
        if st.button("↩ Voltar à base oficial", width="stretch"):
            for k in ("dados_upload", "fonte_label", "filtro_ano",
                      "filtro_filial", "filtro_comprador", "filtro_mes",
                      "filtro_forn"):
                st.session_state.pop(k, None)
            st.rerun()

    st.caption(f"Fonte: **{fonte_label}**")
    st.markdown("---")

    st.markdown("## Filtros")
    anos = sorted(fato["ANO"].unique().tolist())
    filiais = sorted(fato["FILIAL"].unique().tolist())
    compradores = sorted(fato["CODCOMPRADOR"].unique().tolist())
    meses_disp = [m for m in MES_ORDEM if m in fato["MES"].unique()]

    sel_ano = st.multiselect("Ano", anos, default=anos, key="filtro_ano")
    sel_filial = st.multiselect(
        "Filial", filiais, default=filiais,
        format_func=lambda x: f"Filial {x}",
        key="filtro_filial",
    )
    sel_comprador = st.multiselect("Comprador", compradores, default=compradores,
                                     key="filtro_comprador")
    sel_mes = st.multiselect(
        "Mês", meses_disp, default=meses_disp,
        format_func=lambda x: MES_PT.get(x, x),
        key="filtro_mes",
    )

    forn_opts = (
        dim.sort_values("FANTASIA")
        .assign(label=lambda d: d["FANTASIA"] + "  ·  " + d["CODFORNEC"].astype(str))
    )
    sel_labels = st.multiselect(
        "Indústrias (vazio = todas)", forn_opts["label"].tolist(), default=[],
        key="filtro_forn",
    )
    sel_cods = forn_opts.loc[forn_opts["label"].isin(sel_labels), "CODFORNEC"].tolist()

    st.markdown("---")
    st.caption(
        f"**{fato['CODFORNEC'].nunique()}** indústrias · "
        f"**{len(filiais)}** filiais · "
        f"**{len(anos)}** anos"
    )

# =====================================================================
# Modo Upload (tela cheia)
# =====================================================================
if st.session_state.get("upload_mode"):
    st.markdown(
        """
        <div class="app-header" style="background:linear-gradient(135deg,#0F766E 0%,#134E4A 100%);">
            <div>
                <h1>📤 Carregar planilha</h1>
                <div class="subtitle">Arraste os arquivos .xlsx ou clique para selecionar</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "**Padrão aceito** — abas: `FILIAL 1`, `FILIAL 2`, `FILIAL 3`, `FILIAL 4`. "
        "Colunas obrigatórias: `CODFORNEC, FILIAL, FORNECEDOR, CODCOMPRADOR, "
        "JANEIRO, FEVEREIRO, MARCO`. O ano é detectado pelo nome do arquivo "
        "(ex.: `TRIMESTRES 2027 DIEGO.xlsx`)."
    )

    ups = st.file_uploader(
        "Solte aqui suas planilhas (.xlsx) — pode soltar várias de uma vez",
        type=["xlsx"],
        accept_multiple_files=True,
        key="uploader_xlsx_main",
        label_visibility="visible",
    )

    if ups:
        arquivos_validos: dict[int, io.BytesIO] = {}
        cols_v = st.columns(min(len(ups), 3))
        for idx, up in enumerate(ups):
            buf = io.BytesIO(up.getvalue())
            rel = validar_planilha(buf, nome=up.name)
            with cols_v[idx % len(cols_v)]:
                with st.container(border=True):
                    st.markdown(f"**📄 {up.name}**  \n*{len(up.getvalue())/1024:.1f} KB*")
                    for m in rel["mensagens"]:
                        if m.startswith("[OK]"):
                            st.success(m)
                        elif m.startswith("[AVISO]"):
                            st.warning(m)
                        else:
                            st.error(m)
                    if rel["ok"]:
                        ano = rel["ano_sugerido"]
                        ano = st.number_input(
                            "Ano destes dados", min_value=2000, max_value=2100,
                            value=int(ano) if ano else 2026, step=1,
                            key=f"ano_main_{up.name}",
                        )
                        buf.seek(0)
                        arquivos_validos[int(ano)] = buf

        st.markdown("---")
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            substituir = st.checkbox(
                "Substituir base atual completamente",
                value=True,
                help="Desmarcado: mescla os anos enviados com a base oficial.",
            )
        with col_b:
            aplicar = st.button(
                "✅ Aplicar", type="primary", width="stretch",
                disabled=not arquivos_validos,
            )
        with col_c:
            cancelar = st.button("✖ Cancelar", width="stretch")

        if cancelar:
            st.session_state.pop("upload_mode", None)
            st.rerun()

        if aplicar and arquivos_validos:
            try:
                from build_fact_table import (
                    consolidar_trimestre, comparativo_anual, dimensao_fornecedor,
                )
                if not substituir:
                    base = load_data()
                    anos_novos = set(arquivos_validos.keys())
                    fato_base = base["fato"][~base["fato"]["ANO"].isin(anos_novos)]
                    novo = processar_em_memoria(arquivos_validos)
                    fato_merge = pd.concat([fato_base, novo["fato"]], ignore_index=True)
                    dados_final = {
                        "fato": fato_merge,
                        "cons": consolidar_trimestre(fato_merge),
                        "comp": comparativo_anual(fato_merge),
                        "dim": dimensao_fornecedor(fato_merge),
                        "forecast": None, "forecast_total": None,
                        "clusters": None, "anomalias": None, "metricas": None,
                    }
                    label = f"Mesclado: oficial + upload {sorted(anos_novos)}"
                else:
                    novo = processar_em_memoria(arquivos_validos)
                    dados_final = {
                        "fato": novo["fato"], "cons": novo["cons"],
                        "comp": novo["comp"], "dim": novo["dim"],
                        "forecast": None, "forecast_total": None,
                        "clusters": None, "anomalias": None, "metricas": None,
                    }
                    label = f"Upload (anos {sorted(arquivos_validos.keys())})"
                # Limpa filtros para usarem os defaults com os novos dados
                for k in ("filtro_ano", "filtro_filial", "filtro_comprador",
                          "filtro_mes", "filtro_forn"):
                    st.session_state.pop(k, None)
                st.session_state["dados_upload"] = dados_final
                st.session_state["fonte_label"] = label
                st.session_state.pop("upload_mode", None)
                st.success("✅ Dados aplicados. Recarregando dashboard…")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Falha ao processar: {e}")
    else:
        if st.button("✖ Cancelar e voltar"):
            st.session_state.pop("upload_mode", None)
            st.rerun()
    st.stop()

# Aplica filtros
mask = (
    fato["ANO"].isin(sel_ano)
    & fato["FILIAL"].isin(sel_filial)
    & fato["CODCOMPRADOR"].isin(sel_comprador)
    & fato["MES"].isin(sel_mes)
)
if sel_cods:
    mask &= fato["CODFORNEC"].isin(sel_cods)

df = fato[mask].copy()

# =====================================================================
# Header
# =====================================================================
periodo_txt = (
    f"{', '.join(MES_PT.get(m, m) for m in sel_mes) or '—'} · "
    f"{', '.join(str(a) for a in sel_ano) or '—'}"
)
st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1>Inteligência de Compras</h1>
            <div class="subtitle">Análise consolidada por indústria, filial e período</div>
        </div>
        <div style="text-align:right">
            <div class="pill">{periodo_txt}</div>
            <div style="font-size:0.7rem; opacity:0.7; margin-top:0.5rem">
                Atualizado em {pd.Timestamp.today().strftime('%d/%m/%Y')}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("Nenhum dado para os filtros selecionados.")
    st.stop()

# =====================================================================
# KPIs
# =====================================================================
total_geral = df["VALOR"].sum()
n_fornec = df["CODFORNEC"].nunique()
media_fornec = df.groupby("CODFORNEC")["VALOR"].sum().mean()
ticket_medio = df.groupby(["ANO", "MES", "FILIAL"])["VALOR"].sum().mean()

delta_str, delta_dir = (None, "neutral")
if {2025, 2026}.issubset(sel_ano):
    t25 = df[df["ANO"] == 2025]["VALOR"].sum()
    t26 = df[df["ANO"] == 2026]["VALOR"].sum()
    if t25:
        pct = (t26 / t25 - 1) * 100
        delta_dir = "up" if pct >= 0 else "down"
        delta_str = f"{pct:+.1f}% vs 2025"

c1, c2, c3, c4 = st.columns(4)
c1.markdown(kpi_card("Total no período", fmt_brl_compact(total_geral),
                       delta=delta_str, delta_dir=delta_dir), unsafe_allow_html=True)
c2.markdown(kpi_card("Indústrias ativas", str(n_fornec)), unsafe_allow_html=True)
c3.markdown(kpi_card("Média por indústria", fmt_brl_compact(media_fornec),
                       delta="consolidado", delta_dir="neutral"), unsafe_allow_html=True)
c4.markdown(kpi_card("Ticket médio mês/filial", fmt_brl_compact(ticket_medio)),
              unsafe_allow_html=True)

# Segunda linha de KPIs avançados (ML)
if HAS_ML:
    geral_met = metricas_df[metricas_df["ESCOPO"] == "GERAL"]
    geral_fc = forecast_total[forecast_total["ESCOPO"] == "GERAL"]
    c5, c6, c7, c8 = st.columns(4)
    if not geral_fc.empty:
        g_fc = geral_fc.iloc[0]
        prox_lbl = f"{g_fc['CHAVE']} · próx. mês"
        c5.markdown(kpi_card(
            "Previsão próximo mês", fmt_brl_compact(g_fc["ensemble"]),
            delta=f"IC {fmt_brl_compact(g_fc['ic_inf'])} – {fmt_brl_compact(g_fc['ic_sup'])}",
            delta_dir="neutral",
        ), unsafe_allow_html=True)
    if not geral_met.empty:
        g_m = geral_met.iloc[0]
        c6.markdown(kpi_card(
            "Concentração top 5", f"{g_m['TOP5_PCT']:.1f}%",
            delta=f"top 10: {g_m['TOP10_PCT']:.1f}%", delta_dir="neutral",
        ), unsafe_allow_html=True)
        c7.markdown(kpi_card(
            "HHI (Herfindahl)", f"{g_m['HHI']:.0f}",
            delta="alto >2500" if g_m['HHI'] > 2500 else (
                "moderado" if g_m['HHI'] > 1500 else "baixo"),
            delta_dir="down" if g_m['HHI'] > 2500 else "neutral",
        ), unsafe_allow_html=True)
        c8.markdown(kpi_card(
            "Gini", f"{g_m['GINI']:.3f}",
            delta=f"desigualdade {'alta' if g_m['GINI']>0.6 else 'moderada'}",
            delta_dir="neutral",
        ), unsafe_allow_html=True)

# =====================================================================
# Tabs
# =====================================================================
tab_labels = [
    "Visão Geral", "Indústrias", "Filiais",
    "Evolução", "Curva ABC", "Comparativo Anual",
]
if HAS_ML:
    tab_labels += ["Previsão", "Clusters", "Anomalias", "Concentração"]
tab_labels.append("Dados")
tabs = st.tabs(tab_labels)
IDX = {name: i for i, name in enumerate(tab_labels)}

# ---------- Visão Geral ----------
with tabs[0]:
    st.markdown('<div class="section-title">Distribuição por filial e por mês</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        por_filial = df.groupby(["FILIAL", "ANO"], as_index=False)["VALOR"].sum()
        por_filial["FILIAL"] = "Filial " + por_filial["FILIAL"].astype(str)
        fig = px.bar(
            por_filial, x="FILIAL", y="VALOR", color="ANO",
            barmode="group", text_auto=".2s",
            color_discrete_map={k: v for k, v in ANO_COLORS.items()},
            title="Compras por filial",
        )
        fig.update_layout(yaxis_title="R$", xaxis_title="", legend_title="")
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(apply_template(fig, height=380), width="stretch")

    with col2:
        por_mes = df.groupby(["MES", "ANO"], as_index=False)["VALOR"].sum()
        por_mes["MES_LBL"] = por_mes["MES"].map(MES_PT)
        por_mes["MES_LBL"] = pd.Categorical(
            por_mes["MES_LBL"], [MES_PT[m] for m in MES_ORDEM], ordered=True
        )
        por_mes = por_mes.sort_values("MES_LBL")
        fig = px.bar(
            por_mes, x="MES_LBL", y="VALOR", color="ANO",
            barmode="group", text_auto=".2s",
            color_discrete_map={k: v for k, v in ANO_COLORS.items()},
            title="Compras por mês",
        )
        fig.update_layout(yaxis_title="R$", xaxis_title="", legend_title="")
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(apply_template(fig, height=380), width="stretch")

    st.markdown('<div class="section-title">Heatmap Filial × Mês × Ano</div>',
                unsafe_allow_html=True)
    heat = df.groupby(["FILIAL", "MES", "ANO"], as_index=False)["VALOR"].sum()
    heat["MES_LBL"] = heat["MES"].map(MES_PT)
    heat["MES_LBL"] = pd.Categorical(
        heat["MES_LBL"], [MES_PT[m] for m in MES_ORDEM], ordered=True
    )
    heat["FILIAL_LBL"] = "Filial " + heat["FILIAL"].astype(str)
    fig = px.density_heatmap(
        heat, x="MES_LBL", y="FILIAL_LBL", z="VALOR",
        facet_col="ANO", text_auto=".2s",
        color_continuous_scale=SEQ_TEAL,
    )
    fig.update_layout(
        xaxis_title="", yaxis_title="",
        coloraxis_colorbar=dict(title="R$", thickness=12),
    )
    fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
    st.plotly_chart(apply_template(fig, height=320), width="stretch")


# ---------- Indústrias ----------
with tabs[1]:
    rank = (
        df.groupby(["CODFORNEC", "FANTASIA"], as_index=False)
        .agg(TOTAL=("VALOR", "sum"),
             MEDIA_MENSAL=("VALOR", "mean"),
             DESVIO=("VALOR", "std"),
             N_REGISTROS=("VALOR", "size"))
        .sort_values("TOTAL", ascending=False)
    )
    rank["DESVIO"] = rank["DESVIO"].fillna(0)
    rank["CV_PCT"] = np.where(rank["MEDIA_MENSAL"] > 0,
                               rank["DESVIO"] / rank["MEDIA_MENSAL"] * 100, 0)

    c1, c2 = st.columns([1, 3])
    with c1:
        top_n = st.slider("Top N", 5, len(rank), min(15, len(rank)))
    with c2:
        st.markdown(
            f'<div style="padding-top:1.6rem; color:#64748B; font-size:0.85rem">'
            f'Exibindo <b>{top_n}</b> de <b>{len(rank)}</b> indústrias · '
            f'Total: <b>{fmt_brl(rank.head(top_n)["TOTAL"].sum())}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

    top = rank.head(top_n).sort_values("TOTAL")
    fig = px.bar(
        top, x="TOTAL", y="FANTASIA", orientation="h",
        text=top["TOTAL"].map(fmt_brl_compact),
        color="TOTAL", color_continuous_scale=SEQ_TEAL,
        title="Ranking — total no período",
    )
    fig.update_layout(yaxis_title="", xaxis_title="R$",
                       coloraxis_showscale=False)
    fig.update_traces(textposition="outside", marker_line_width=0)
    st.plotly_chart(apply_template(fig, height=max(380, top_n * 34)),
                     width="stretch")

    st.markdown('<div class="section-title">Detalhamento por indústria</div>',
                unsafe_allow_html=True)
    rank_disp = rank.copy()
    rank_disp["Total"] = rank_disp["TOTAL"].map(fmt_brl)
    rank_disp["Média mensal"] = rank_disp["MEDIA_MENSAL"].map(fmt_brl)
    rank_disp["Desvio padrão"] = rank_disp["DESVIO"].map(fmt_brl)
    rank_disp["CV"] = rank_disp["CV_PCT"].round(1).astype(str) + "%"
    rank_disp = rank_disp[["CODFORNEC", "FANTASIA", "Total", "Média mensal",
                            "Desvio padrão", "CV", "N_REGISTROS"]]
    rank_disp.columns = ["Cód.", "Indústria", "Total", "Média mensal",
                          "Desvio padrão", "CV", "Registros"]
    st.dataframe(rank_disp, width="stretch", hide_index=True)


# ---------- Filiais ----------
with tabs[2]:
    st.markdown('<div class="section-title">Composição da filial por indústria</div>',
                unsafe_allow_html=True)
    por_fil_for = df.groupby(["FILIAL", "FANTASIA"], as_index=False)["VALOR"].sum()
    por_fil_for["FILIAL_LBL"] = "Filial " + por_fil_for["FILIAL"].astype(str)
    top10 = (por_fil_for.groupby("FANTASIA")["VALOR"].sum()
             .nlargest(10).index.tolist())
    por_fil_for["GRUPO"] = np.where(
        por_fil_for["FANTASIA"].isin(top10), por_fil_for["FANTASIA"], "Outros"
    )
    agg = por_fil_for.groupby(["FILIAL_LBL", "GRUPO"], as_index=False)["VALOR"].sum()
    fig = px.bar(agg, x="FILIAL_LBL", y="VALOR", color="GRUPO",
                  text_auto=False, title="Composição (top 10 + outros)")
    fig.update_layout(yaxis_title="R$", xaxis_title="",
                       legend_title="Indústria")
    st.plotly_chart(apply_template(fig, height=460), width="stretch")

    st.markdown(
        '<div class="section-title">Sazonalidade — índice mês ÷ média do trimestre</div>',
        unsafe_allow_html=True,
    )
    saz = (df.groupby(["FILIAL", "MES"])["VALOR"].sum()
              .unstack("MES").reindex(columns=MES_ORDEM))
    indice = saz.div(saz.mean(axis=1), axis=0)
    indice.index = ["Filial " + str(i) for i in indice.index]
    indice.columns = [MES_PT[c] for c in indice.columns]
    fig = px.imshow(
        indice, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", color_continuous_midpoint=1.0,
        labels=dict(x="Mês", y="Filial", color="Índice"),
    )
    fig.update_layout(coloraxis_colorbar=dict(thickness=12))
    st.plotly_chart(apply_template(fig, height=300), width="stretch")


# ---------- Evolução ----------
with tabs[3]:
    st.markdown('<div class="section-title">Série temporal — total no período</div>',
                unsafe_allow_html=True)
    serie = df.groupby("DATA", as_index=False)["VALOR"].sum().sort_values("DATA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie["DATA"], y=serie["VALOR"], mode="lines+markers",
        line=dict(color=PRIMARY, width=3),
        marker=dict(size=9, color=PRIMARY, line=dict(color="#fff", width=2)),
        fill="tozeroy", fillcolor="rgba(15,118,110,0.08)",
        hovertemplate="<b>%{x|%b/%Y}</b><br>R$ %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title="R$", xaxis_title="", showlegend=False)
    st.plotly_chart(apply_template(fig, height=360), width="stretch")

    st.markdown('<div class="section-title">Evolução por indústria — top 8</div>',
                unsafe_allow_html=True)
    top8 = df.groupby("CODFORNEC")["VALOR"].sum().nlargest(8).index.tolist()
    sub = df[df["CODFORNEC"].isin(top8)]
    serie_for = sub.groupby(["DATA", "FANTASIA"], as_index=False)["VALOR"].sum()
    fig = px.line(serie_for, x="DATA", y="VALOR", color="FANTASIA",
                   markers=True)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
    fig.update_layout(yaxis_title="R$", xaxis_title="", legend_title="")
    st.plotly_chart(apply_template(fig, height=420), width="stretch")


# ---------- Curva ABC ----------
with tabs[4]:
    abc = (df.groupby(["CODFORNEC", "FANTASIA"], as_index=False)["VALOR"].sum()
              .sort_values("VALOR", ascending=False))
    abc["PARTICIPACAO"] = abc["VALOR"] / abc["VALOR"].sum() * 100
    abc["ACUMULADO"] = abc["PARTICIPACAO"].cumsum()
    abc["CLASSE"] = np.where(abc["ACUMULADO"] <= 80, "A",
                       np.where(abc["ACUMULADO"] <= 95, "B", "C"))

    cont = abc["CLASSE"].value_counts()
    soma = abc.groupby("CLASSE")["VALOR"].sum()
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi_card("Classe A — 80% do valor",
                          f"{int(cont.get('A',0))} indústrias",
                          delta=fmt_brl_compact(soma.get("A", 0)),
                          delta_dir="up"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Classe B — 15% do valor",
                          f"{int(cont.get('B',0))} indústrias",
                          delta=fmt_brl_compact(soma.get("B", 0)),
                          delta_dir="neutral"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Classe C — 5% do valor",
                          f"{int(cont.get('C',0))} indústrias",
                          delta=fmt_brl_compact(soma.get("C", 0)),
                          delta_dir="down"), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Curva de Pareto</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    classe_color = {"A": "#0F766E", "B": "#F59E0B", "C": "#94A3B8"}
    for classe in ["A", "B", "C"]:
        sub_abc = abc[abc["CLASSE"] == classe]
        fig.add_trace(go.Bar(
            x=sub_abc["FANTASIA"], y=sub_abc["VALOR"], name=f"Classe {classe}",
            marker_color=classe_color[classe],
            text=[fmt_brl_compact(v) for v in sub_abc["VALOR"]],
            textposition="outside",
        ))
    fig.add_trace(go.Scatter(
        x=abc["FANTASIA"], y=abc["ACUMULADO"], mode="lines+markers",
        name="% Acumulado", yaxis="y2",
        line=dict(color="#0F172A", width=2.5),
        marker=dict(size=7, color="#0F172A"),
    ))
    fig.add_hline(y=80, line=dict(color="#DC2626", dash="dash", width=1),
                   annotation_text="80%", annotation_position="right",
                   yref="y2")
    fig.update_layout(
        yaxis=dict(title="R$"),
        yaxis2=dict(title="% Acumulado", overlaying="y", side="right",
                     range=[0, 105], showgrid=False),
        xaxis_title="", legend=dict(orientation="h", y=1.1),
        barmode="stack",
    )
    st.plotly_chart(apply_template(fig, height=480), width="stretch")

    abc_disp = abc.copy()
    abc_disp["Valor"] = abc_disp["VALOR"].map(fmt_brl)
    abc_disp["Participação"] = abc_disp["PARTICIPACAO"].round(2).astype(str) + "%"
    abc_disp["Acumulado"] = abc_disp["ACUMULADO"].round(2).astype(str) + "%"
    abc_disp = abc_disp[["CODFORNEC", "FANTASIA", "Valor", "Participação",
                          "Acumulado", "CLASSE"]]
    abc_disp.columns = ["Cód.", "Indústria", "Valor", "Participação",
                         "Acumulado", "Classe"]
    st.dataframe(abc_disp, width="stretch", hide_index=True)


# ---------- Comparativo Anual ----------
with tabs[5]:
    if not {"TOTAL_2025", "TOTAL_2026"}.issubset(comp.columns):
        st.info("Comparativo requer dados de 2025 e 2026.")
    else:
        comp_view = comp.merge(dim[["CODFORNEC", "FANTASIA"]], on="CODFORNEC")
        if sel_cods:
            comp_view = comp_view[comp_view["CODFORNEC"].isin(sel_cods)]
        comp_view = comp_view.sort_values("VAR_ABS")

        status_color = {
            "CRESCENTE": SUCCESS, "DECRESCENTE": DANGER,
            "ESTAVEL": NEUTRAL, "NOVO": PRIMARY, "PERDIDO": "#9333EA",
            "INDEFINIDO": "#CA8A04",
        }
        fig = px.bar(
            comp_view, x="VAR_ABS", y="FANTASIA", color="STATUS",
            orientation="h",
            text=[fmt_brl_compact(v) for v in comp_view["VAR_ABS"]],
            color_discrete_map=status_color,
        )
        fig.update_layout(
            xaxis_title="Variação R$ (2026 − 2025)", yaxis_title="",
            legend_title="",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(apply_template(fig, height=max(400, len(comp_view)*32)),
                         width="stretch")

        comp_disp = comp_view[["CODFORNEC", "FANTASIA", "TOTAL_2025",
                                "TOTAL_2026", "VAR_ABS", "VAR_PCT", "STATUS"]].copy()
        comp_disp["2025"] = comp_disp["TOTAL_2025"].map(fmt_brl)
        comp_disp["2026"] = comp_disp["TOTAL_2026"].map(fmt_brl)
        comp_disp["Var. R$"] = comp_disp["VAR_ABS"].map(fmt_brl)
        comp_disp["Var. %"] = comp_disp["VAR_PCT"].round(2).astype(str) + "%"
        comp_disp = comp_disp[["CODFORNEC", "FANTASIA", "2025", "2026",
                                "Var. R$", "Var. %", "STATUS"]]
        comp_disp.columns = ["Cód.", "Indústria", "2025", "2026",
                              "Var. R$", "Var. %", "Status"]
        st.dataframe(comp_disp, width="stretch", hide_index=True)


# =====================================================================
# Tabs de Machine Learning (só se parquets ML existirem)
# =====================================================================
if HAS_ML:
    # ---------- Previsão ----------
    with tabs[IDX["Previsão"]]:
        geral_fc = forecast_total[forecast_total["ESCOPO"] == "GERAL"].iloc[0]
        fil_fc = forecast_total[forecast_total["ESCOPO"] == "FILIAL"].copy()
        alvo_lbl = (
            f"{['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'][int(geral_fc['MES_NUM'])]}"
            f"/{int(geral_fc['ANO'])}"
        )

        st.markdown(
            f'<div class="section-title">Previsão para {alvo_lbl} '
            f'— ensemble (naive + média + tendência + média ponderada + RandomForest)</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_card(
            "Previsão total", fmt_brl_compact(geral_fc["ensemble"]),
            delta=f"limite inf. {fmt_brl_compact(geral_fc['ic_inf'])}",
            delta_dir="neutral",
        ), unsafe_allow_html=True)
        c2.markdown(kpi_card(
            "Limite superior (IC)", fmt_brl_compact(geral_fc["ic_sup"]),
            delta="confiança 1,5σ", delta_dir="neutral",
        ), unsafe_allow_html=True)
        c3.markdown(kpi_card(
            "Incerteza (σ)", fmt_brl_compact(geral_fc["std"]),
            delta="quanto menor, melhor", delta_dir="neutral",
        ), unsafe_allow_html=True)

        # Série histórica + previsão
        st.markdown('<div class="section-title">Série histórica + projeção</div>',
                    unsafe_allow_html=True)
        serie = (fato.groupby(["DATA", "ANO", "MES_NUM"], as_index=False)["VALOR"]
                  .sum().sort_values("DATA"))
        next_date = pd.Timestamp(year=int(geral_fc["ANO"]),
                                  month=int(geral_fc["MES_NUM"]), day=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=serie["DATA"], y=serie["VALOR"], mode="lines+markers",
            name="Histórico", line=dict(color=PRIMARY, width=3),
            marker=dict(size=10),
            fill="tozeroy", fillcolor="rgba(15,118,110,0.07)",
        ))
        ult_data = serie["DATA"].iloc[-1]
        ult_val = serie["VALOR"].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[ult_data, next_date],
            y=[ult_val, geral_fc["ensemble"]],
            mode="lines+markers", name="Previsão",
            line=dict(color=ACCENT, width=3, dash="dash"),
            marker=dict(size=12, symbol="diamond"),
        ))
        # Banda de IC
        fig.add_trace(go.Scatter(
            x=[next_date, next_date], y=[geral_fc["ic_inf"], geral_fc["ic_sup"]],
            mode="lines", name="IC", line=dict(color=ACCENT, width=10),
            opacity=0.25, showlegend=True,
        ))
        fig.update_layout(yaxis_title="R$", xaxis_title="",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(apply_template(fig, height=380), width="stretch")

        # Por filial
        st.markdown('<div class="section-title">Previsão por filial</div>',
                    unsafe_allow_html=True)
        fil_fc["FIL_LBL"] = fil_fc["CHAVE"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fil_fc["FIL_LBL"], y=fil_fc["ensemble"],
            text=[fmt_brl_compact(v) for v in fil_fc["ensemble"]],
            textposition="outside",
            marker_color=PRIMARY, name="Previsão",
            error_y=dict(
                type="data", symmetric=False,
                array=fil_fc["ic_sup"] - fil_fc["ensemble"],
                arrayminus=fil_fc["ensemble"] - fil_fc["ic_inf"],
                color="#0F172A", thickness=1.5, width=8,
            ),
        ))
        fig.update_layout(yaxis_title="R$", xaxis_title="", showlegend=False)
        st.plotly_chart(apply_template(fig, height=380), width="stretch")

        # Top fornecedores previstos
        st.markdown(
            f'<div class="section-title">Top 20 indústrias — previsão {alvo_lbl}</div>',
            unsafe_allow_html=True,
        )
        fc_view = forecast_df.copy()
        if sel_cods:
            fc_view = fc_view[fc_view["CODFORNEC"].isin(sel_cods)]
        if sel_filial:
            fc_view = fc_view[fc_view["FILIAL"].isin(sel_filial)]
        fc_agg = (fc_view.groupby(["CODFORNEC", "FANTASIA"], as_index=False)
                  .agg(ensemble=("ensemble", "sum"),
                       ic_inf=("ic_inf", "sum"),
                       ic_sup=("ic_sup", "sum"),
                       PREDICAO_FINAL=("PREDICAO_FINAL", "sum"))
                  .sort_values("ensemble", ascending=False)
                  .head(20).sort_values("ensemble"))
        fig = px.bar(
            fc_agg, x="ensemble", y="FANTASIA", orientation="h",
            text=fc_agg["ensemble"].map(fmt_brl_compact),
            color="ensemble", color_continuous_scale=SEQ_TEAL,
            title="Previsão por indústria (ensemble)",
        )
        fig.update_layout(yaxis_title="", xaxis_title="R$",
                           coloraxis_showscale=False)
        fig.update_traces(textposition="outside", marker_line_width=0)
        st.plotly_chart(apply_template(fig, height=max(420, len(fc_agg)*32)),
                         width="stretch")

        st.markdown('<div class="section-title">Tabela completa de previsões</div>',
                    unsafe_allow_html=True)
        fc_tab = forecast_df.copy()
        fc_tab["FIL_LBL"] = "Filial " + fc_tab["FILIAL"].astype(str)
        for col_money in ["naive", "media", "tendencia", "ensemble",
                           "ic_inf", "ic_sup", "RF_PREDICAO", "PREDICAO_FINAL"]:
            if col_money in fc_tab.columns:
                fc_tab[col_money] = fc_tab[col_money].map(fmt_brl)
        fc_tab = fc_tab[["FIL_LBL", "CODFORNEC", "FANTASIA",
                          "naive", "media", "tendencia", "ensemble",
                          "ic_inf", "ic_sup", "RF_PREDICAO", "PREDICAO_FINAL"]]
        fc_tab.columns = ["Filial", "Cód.", "Indústria",
                           "Naive", "Média", "Tendência", "Ensemble",
                           "IC inf", "IC sup", "RF", "Final (ens+RF)"]
        st.dataframe(fc_tab, width="stretch", hide_index=True, height=420)

    # ---------- Clusters ----------
    with tabs[IDX["Clusters"]]:
        st.markdown(
            '<div class="section-title">Segmentação de indústrias (KMeans + silhueta)</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Cada indústria é representada por features de volume, dispersão (CV), "
            "crescimento YoY e sazonalidade mensal. KMeans agrupa perfis similares."
        )

        cl = clusters_df.copy()
        # KPI: nº clusters e tamanho
        cnt = cl.groupby(["CLUSTER", "CLUSTER_ROTULO"]).size().reset_index(name="N")
        cnt = cnt.sort_values("N", ascending=False)

        cols = st.columns(min(4, len(cnt)))
        for i, row in cnt.head(4).reset_index(drop=True).iterrows():
            cols[i].markdown(kpi_card(
                f"Cluster {int(row['CLUSTER'])}",
                f"{int(row['N'])} indústrias",
                delta=row["CLUSTER_ROTULO"], delta_dir="neutral",
            ), unsafe_allow_html=True)

        # Scatter 2D: TOTAL vs CV colorido por cluster
        st.markdown('<div class="section-title">Mapa Total × Variação (CV) × Crescimento</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(
            cl, x="TOTAL", y="CV", color="CLUSTER_ROTULO",
            size=np.clip(cl["TOTAL"] / cl["TOTAL"].max() * 60 + 8, 8, 70),
            hover_data={"FANTASIA": True, "GROWTH_YOY": ":.1%",
                         "TOTAL": ":,.0f", "CV": ":.2f"},
            color_discrete_sequence=CATEGORICAL,
        )
        fig.update_layout(xaxis_title="Total comprado (R$)",
                           yaxis_title="Coef. variação (dispersão)",
                           legend_title="Perfil")
        fig.update_xaxes(tickformat=",.0f")
        st.plotly_chart(apply_template(fig, height=440), width="stretch")

        # Detalhamento
        st.markdown('<div class="section-title">Indústrias por cluster</div>',
                    unsafe_allow_html=True)
        cl_disp = cl.copy()
        cl_disp["Total"] = cl_disp["TOTAL"].map(fmt_brl)
        cl_disp["Média mensal"] = cl_disp["MEDIA"].map(fmt_brl)
        cl_disp["CV"] = cl_disp["CV"].round(2)
        cl_disp["Crescim. YoY"] = (cl_disp["GROWTH_YOY"] * 100).round(1).astype(str) + "%"
        cl_disp = cl_disp[["CLUSTER", "CLUSTER_ROTULO", "CODFORNEC", "FANTASIA",
                            "Total", "Média mensal", "CV", "Crescim. YoY"]]
        cl_disp.columns = ["#", "Perfil", "Cód.", "Indústria",
                            "Total", "Média mensal", "CV", "Crescim. YoY"]
        st.dataframe(cl_disp.sort_values(["#", "Total"], ascending=[True, False]),
                      width="stretch", hide_index=True, height=420)

    # ---------- Anomalias ----------
    with tabs[IDX["Anomalias"]]:
        an = anomalias_df.copy()
        n_total = int(an["ANOMALIA"].sum())
        n_z = int(an["ANOMALIA_Z"].sum())
        n_iso = int(an["ANOMALIA_ISO"].sum())

        st.markdown(
            '<div class="section-title">Detecção de outliers (z-score + IsolationForest)</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(kpi_card("Total de anomalias", str(n_total),
                              delta=f"{n_total/len(an)*100:.1f}% das observações",
                              delta_dir="down" if n_total > 30 else "neutral"),
                     unsafe_allow_html=True)
        c2.markdown(kpi_card("Por z-score |z|>2", str(n_z),
                              delta="atípico vs. histórico do fornecedor",
                              delta_dir="neutral"), unsafe_allow_html=True)
        c3.markdown(kpi_card("Por IsolationForest", str(n_iso),
                              delta="atípico no padrão multivariado",
                              delta_dir="neutral"), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Top 30 observações mais atípicas</div>',
                    unsafe_allow_html=True)
        top_an = an.head(30).copy()
        top_an["FIL"] = "Filial " + top_an["FILIAL"].astype(str)
        top_an["Período"] = top_an["DATA"].dt.strftime("%b/%Y")
        top_an["Valor"] = top_an["VALOR"].map(fmt_brl)
        top_an["Média histórica"] = top_an["MEDIA_FORN"].map(fmt_brl)
        top_an["Z-score"] = top_an["Z_SCORE"].round(2)
        flag = top_an.apply(
            lambda r: "Z+ISO" if r["ANOMALIA_Z"] and r["ANOMALIA_ISO"]
            else ("Z" if r["ANOMALIA_Z"] else ("ISO" if r["ANOMALIA_ISO"] else "")),
            axis=1,
        )
        top_an["Flag"] = flag
        top_an = top_an[["FIL", "Período", "CODFORNEC", "FANTASIA",
                          "Valor", "Média histórica", "Z-score", "Flag"]]
        top_an.columns = ["Filial", "Período", "Cód.", "Indústria",
                           "Valor", "Média histórica", "Z-score", "Detectado por"]
        st.dataframe(top_an, width="stretch", hide_index=True, height=520)

        # Distribuição dos z-scores
        st.markdown('<div class="section-title">Distribuição dos z-scores</div>',
                    unsafe_allow_html=True)
        fig = px.histogram(an, x="Z_SCORE", nbins=40,
                            color_discrete_sequence=[PRIMARY])
        fig.add_vline(x=2, line=dict(color=DANGER, dash="dash"))
        fig.add_vline(x=-2, line=dict(color=DANGER, dash="dash"))
        fig.update_layout(xaxis_title="Z-score", yaxis_title="Observações")
        st.plotly_chart(apply_template(fig, height=320), width="stretch")

    # ---------- Concentração ----------
    with tabs[IDX["Concentração"]]:
        st.markdown(
            '<div class="section-title">Métricas de concentração de mercado</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "**HHI** (Herfindahl-Hirschman): 0–10000. >2500 = concentração alta. "
            "**Gini**: 0 = perfeitamente distribuído, 1 = um único fornecedor. "
            "**Top 5 / Top 10 %**: parcela das maiores indústrias no total."
        )

        m_disp = metricas_df.copy()
        m_disp["Total"] = m_disp["TOTAL"].map(fmt_brl)
        m_disp["Top 5"] = m_disp["TOP5_PCT"].round(1).astype(str) + "%"
        m_disp["Top 10"] = m_disp["TOP10_PCT"].round(1).astype(str) + "%"
        m_disp["HHI"] = m_disp["HHI"].round(0).astype(int)
        m_disp["Gini"] = m_disp["GINI"].round(3)
        m_disp["Cresc. m/m"] = m_disp["CRESC_MEDIO_MOM_PCT"].round(2).astype(str) + "%"
        m_disp = m_disp[["ESCOPO", "CHAVE", "Total", "N_FORNECEDORES",
                          "Top 5", "Top 10", "HHI", "Gini", "Cresc. m/m"]]
        m_disp.columns = ["Escopo", "Recorte", "Total", "Fornec.",
                           "Top 5", "Top 10", "HHI", "Gini", "Cresc. m/m"]
        st.dataframe(m_disp, width="stretch", hide_index=True, height=420)

        # Barras: HHI por filial
        st.markdown('<div class="section-title">HHI por filial</div>',
                    unsafe_allow_html=True)
        hhi_fil = metricas_df[metricas_df["ESCOPO"] == "FILIAL"].copy()
        if not hhi_fil.empty:
            fig = px.bar(hhi_fil, x="CHAVE", y="HHI",
                          color="HHI", color_continuous_scale=SEQ_TEAL,
                          text=hhi_fil["HHI"].round(0).astype(int))
            fig.add_hline(y=1500, line=dict(color=ACCENT, dash="dash"),
                           annotation_text="moderado", annotation_position="right")
            fig.add_hline(y=2500, line=dict(color=DANGER, dash="dash"),
                           annotation_text="alto", annotation_position="right")
            fig.update_traces(textposition="outside")
            fig.update_layout(xaxis_title="", yaxis_title="HHI",
                               coloraxis_showscale=False)
            st.plotly_chart(apply_template(fig, height=360), width="stretch")

        # Lorenz curve
        st.markdown('<div class="section-title">Curva de Lorenz (desigualdade)</div>',
                    unsafe_allow_html=True)
        valores = (fato.groupby("CODFORNEC")["VALOR"].sum()
                   .sort_values().to_numpy())
        cum_pop = np.arange(1, len(valores) + 1) / len(valores)
        cum_val = np.cumsum(valores) / valores.sum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.insert(cum_pop, 0, 0),
                                  y=np.insert(cum_val, 0, 0),
                                  mode="lines", name="Lorenz",
                                  line=dict(color=PRIMARY, width=3),
                                  fill="tozeroy",
                                  fillcolor="rgba(15,118,110,0.12)"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Igualdade perfeita",
                                  line=dict(color=NEUTRAL, dash="dash")))
        fig.update_layout(xaxis_title="% acumulado de fornecedores",
                           yaxis_title="% acumulado de compras",
                           xaxis=dict(tickformat=".0%"),
                           yaxis=dict(tickformat=".0%"))
        st.plotly_chart(apply_template(fig, height=400), width="stretch")


# ---------- Dados ----------
with tabs[IDX["Dados"]]:
    st.markdown('<div class="section-title">Base filtrada (formato longo)</div>',
                unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar CSV", data=csv,
        file_name="compras_filtrado.csv", mime="text/csv",
        type="primary",
    )

    if HAS_ML:
        st.markdown('<div class="section-title">Downloads dos artefatos ML</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Previsão (CSV)",
            data=forecast_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast.csv", mime="text/csv",
        )
        col2.download_button(
            "Clusters (CSV)",
            data=clusters_df.to_csv(index=False).encode("utf-8"),
            file_name="clusters.csv", mime="text/csv",
        )
        col3.download_button(
            "Anomalias (CSV)",
            data=anomalias_df.to_csv(index=False).encode("utf-8"),
            file_name="anomalias.csv", mime="text/csv",
        )
