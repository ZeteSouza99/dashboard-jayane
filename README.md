# Dashboard Jayane — Inteligência de Compras

Pipeline de dados + dashboard interativo para análise de compras por filial,
indústria, período e previsão do próximo mês com machine learning.

**Trabalho de pós-graduação** | Python · pandas · scikit-learn · Streamlit · Plotly

---

## Stack

| Camada | Tecnologia |
|---|---|
| ETL | pandas, numpy, openpyxl, pyarrow (Parquet) |
| ML  | scikit-learn (RandomForest, KMeans, IsolationForest), scipy |
| Visualização | Streamlit + Plotly |
| Deploy | Streamlit Community Cloud (grátis) |

---

## Estrutura

```
.
├── data/
│   ├── raw/                 # planilhas Excel originais
│   ├── staging/             # parquets gerados (fato + ML)
│   └── outputs/             # relatórios texto (QA, ML)
├── src/
│   ├── build_fact_table.py  # ETL: Excel → Parquet
│   ├── analytics_ml.py      # ML: previsão, clusters, anomalias, concentração
│   ├── analytics_demo.py    # relatório CLI rápido
│   └── dashboard.py         # Streamlit app
├── .streamlit/config.toml   # tema corporativo
├── requirements.txt
└── README.md
```

---

## Rodar localmente

```powershell
# 1. ambiente
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. pipeline (gera parquets em data/staging/)
python src\build_fact_table.py
python src\analytics_ml.py

# 3. dashboard
streamlit run src\dashboard.py
```

Acesse `http://localhost:8501`.

---

## Modelos e métricas

- **Previsão do próximo mês** (`forecast.parquet`): ensemble de naive, média,
  tendência OLS, média ponderada e RandomForest treinado em (FILIAL, FORNEC,
  MES, ANO, LAG_1, MEDIA_HIST). Validação 5-fold cross-validation.
- **Clusters** (`clusters.parquet`): KMeans em features de volume, dispersão
  (CV), crescimento YoY e sazonalidade mensal. `k` escolhido por silhueta.
- **Anomalias** (`anomalias.parquet`): z-score por fornecedor (|z|>2) +
  IsolationForest com contamination=5%.
- **Concentração** (`metricas_avancadas.parquet`): HHI (Herfindahl), Gini,
  top 5/10 share, crescimento médio mês-a-mês — por geral/ano/filial.

---

## Deploy em nuvem (Streamlit Community Cloud — grátis)

1. **Crie um repositório no GitHub** com este projeto (público ou privado).
2. Suba via:
   ```powershell
   git init
   git add .
   git commit -m "Dashboard Jayane v1.0"
   git branch -M main
   git remote add origin https://github.com/<seu-usuario>/dashboard-jayane.git
   git push -u origin main
   ```
3. Acesse https://share.streamlit.io e clique em **New app**.
4. Selecione o repositório, branch `main`, arquivo principal `src/dashboard.py`.
5. Em **Advanced settings**, deixe Python 3.11+.
6. Deploy automático. URL pública gerada em ~2 min.

> Atenção LGPD: as planilhas em `data/raw/` contêm nomes de fornecedores
> reais. Se o repositório for público, ou habilite `data/raw/*.xlsx` no
> `.gitignore` ou use Streamlit secrets / anonimize os nomes.

### Alternativas de deploy

| Plataforma | Vantagem | Como |
|---|---|---|
| **Streamlit Cloud** | grátis, 1-click | acima |
| Hugging Face Spaces | grátis, GPU opcional | tipo "Streamlit" |
| Render / Railway | controle total | Dockerfile + `streamlit run` |
| Azure App Service | corporativo | `azure-pipelines.yml` |

---

## Atualizando dados

Quando chegarem novos meses (abril, maio…):

1. Adicione/atualize as colunas `ABRIL`, `MAIO`, ... nas planilhas em
   `data/raw/` (ou edite `MESES_COLS` em `src/build_fact_table.py`).
2. Reexecute os pipelines:
   ```powershell
   python src\build_fact_table.py
   python src\analytics_ml.py
   ```
3. O dashboard recarrega automaticamente (cache invalidado).
