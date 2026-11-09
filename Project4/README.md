# Projeto 4 — Classificação de Chuva em Série Temporal (LSTM / GRU)

Classificação binária de séries temporais meteorológicas: dada uma janela de
observações horárias recentes, o modelo recorrente prevê se **vai chover nas
próximas K horas**. Os dados são da estação automática A701 (São Paulo –
Mirante) do INMET, enriquecidos com o índice **ONI** (El Niño / La Niña) como
variável exógena. O eixo central do projeto é a comparação entre duas
estratégias de partição — **cronológica** (honesta) e **aleatória** (com
vazamento temporal) — para medir o quanto o vazamento infla o desempenho
aparente.

## Estrutura

```
Project4/
├── src/
│   ├── data.py          # download INMET + ONI, features e janelas rotuladas
│   ├── split_data.py    # partição cronológica (com embargo) e aleatória
│   ├── model.py         # RainClassifier (LSTM ou GRU selecionável)
│   ├── train.py         # laço de treino, scaler só-treino, early stopping
│   ├── search.py        # busca de hiperparâmetros por loss de validação
│   ├── main.py          # treino final + curvas de aprendizagem
│   ├── evaluate.py      # métricas no teste + matriz de confusão + comparação
│   ├── explore.py       # figuras de contexto climático (ONI) e vazamento
│   └── report_macros.py # gera os números do relatório a partir dos CSVs
├── data/            # dados brutos (cache) e windows.npz (não versionados)
├── results/         # splits, buscas, checkpoints (.pt) e tabelas (.csv)
├── figures/         # curvas, matrizes de confusão e figuras de clima
├── report/          # relatório LaTeX (main.tex)
└── requirements.txt # dependências Python
```

## Como executar

A partir da pasta `Project4`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) dados: baixa INMET (2022-2024) + ONI e monta as janelas rotuladas
python -m src.data

# 2) partições cronológica e aleatória
python -m src.split_data

# 3) busca de hiperparâmetros (LSTM/GRU, tamanho oculto, camadas)
python -m src.search            # ou: python -m src.search chronological

# 4) treino final dos modelos selecionados + curvas
python -m src.main

# 5) avaliação no teste e comparação entre partições
python -m src.evaluate

# 6) figuras de exploração (clima/El Niño e vazamento temporal)
python -m src.explore

# 7) números do relatório e compilação
python -m src.report_macros
cd report && latexmk -pdf main.tex
```

Os dados do INMET e o índice ONI são baixados automaticamente na primeira
execução e cacheados em `data/raw/`.

## Decisões de projeto

- **Tarefa:** classificação binária "chove ≥ 0,2 mm nas próximas 6 h?", a
  partir de uma janela de 24 h.
- **Variáveis (13):** 8 meteorológicas + 4 cíclicas (hora/dia do ano) + ONI.
- **Desbalanceamento:** ~17 % de classe positiva, tratado com `pos_weight` na
  `BCEWithLogitsLoss`; avaliação por F1 e AUC além da acurácia.
- **Partição cronológica:** blocos contíguos 70/15/15 com *embargo* entre eles
  para evitar sobreposição de janelas na fronteira.
- **Partição aleatória:** mantida apenas como experimento didático, para
  quantificar o vazamento temporal.
- **Reprodutibilidade:** `SEED = 42`; padronização ajustada só no treino.

## Contexto climático

O período 2022–2024 inclui o El Niño forte de 2023–2024. O índice ONI entra
como variável de contexto e embasa a discussão ecológico-social do relatório:
a não estacionariedade da série (regimes de chuva que se deslocam com o clima)
é justamente o que torna a avaliação cronológica mais difícil — e mais honesta
— que a aleatória.
