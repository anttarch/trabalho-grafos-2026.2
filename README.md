# Análise do Grafo Reddit Hyperlink Network (SNAP)

Repositório com os scripts, dados tratados e resultados gerados para o trabalho da disciplina de Teoria dos Grafos, utilizando o conjunto de dados [Reddit Hyperlink Network (SNAP)](https://snap.stanford.edu/data/soc-RedditHyperlinks.html).

## Estrutura do Projeto

```text
trabalho-grafos/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # Arquivos .tsv baixados do SNAP (não versionados no Git)
│   └── processed/    # Grafos tratados (.csv.gz)
├── notebooks/
│   └── analise_completa.ipynb  # Notebook Jupyter interativo (Etapa 0 + Partes I, II e III)
├── src/
│   ├── download_data.py     # Download automático dos arquivos do SNAP
│   ├── preprocess.py        # Etapa 0: limpeza, agregação e extração da GCC/subgrafos
│   ├── parte1_estrutural.py # Parte I: análise estrutural obrigatória e gráficos
│   ├── parte2_algoritmos.py # Parte II: 8 algoritmos em Python puro com média, DP e IC 95%
│   └── parte3_avancada.py   # Parte III: Small-World próprio, Power Law, Robustez, Descoberta e Extra
└── results/
    ├── tables/       # Tabelas (.csv e .json)
    └── figures/      # Figuras (.png e .pdf)
```

## Requisitos

- Python 3.10 ou superior
- `pip`

## Como Executar

### 1. Criar ambiente virtual e instalar dependências

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Executar via terminal (sem Jupyter)

Com o ambiente virtual ativado, execute os scripts na ordem abaixo:

1. **Baixar os dados do SNAP:**
   ```bash
   python src/download_data.py
   ```

2. **Executar o pré-processamento (Etapa 0):**
   ```bash
   python src/preprocess.py
   ```

3. **Executar a Análise Estrutural (Parte I):**
   ```bash
   python src/parte1_estrutural.py
   ```

4. **Executar os Algoritmos da Disciplina (Parte II):**
   ```bash
   python src/parte2_algoritmos.py
   ```

5. **Executar a Análise Estrutural Avançada (Parte III):**
   ```bash
   python src/parte3_avancada.py
   ```

### 3. Executar com Jupyter (JupyterLab)

O projeto inclui o notebook interativo [`notebooks/analise_completa.ipynb`](notebooks/analise_completa.ipynb), que permite explorar todas as tabelas, gráficos e reexecutar todos os módulos de `src/` célula a célula.

Com o ambiente virtual ativado:
```bash
jupyter lab notebooks/analise_completa.ipynb
```

> **Nota:** No notebook, a variável `RECOMPUTE` vem definida como `False` por padrão para carregar rapidamente os resultados já gerados em `results/`. Para executar todo o pipeline do zero pelo Jupyter (incluindo download e pré-processamento), altere `RECOMPUTE = True` na primeira célula de código e execute todas as células.
