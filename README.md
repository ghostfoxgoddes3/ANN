# ANN — Redes Neurais Artificiais

Repositório dos projetos da disciplina **Introdução às Redes Neurais
Artificiais** (UNIFESP). Cada projeto é autocontido, com seu próprio código em
`src/`, dependências em `requirements.txt` e um relatório LaTeX em `report/`.

**Autor:** Gabriely Di Folco Rocha (RA: 188892)

## Projetos

| # | Tema | Modelos | Pasta |
|---|------|---------|-------|
| 1 | Regressão e estudo de ablação | MLP (PyTorch) | [`Project1/`](Project1/README.md) |
| 2 | Quantização de cores | SOM e GNG | [`Project2/`](Project2/README.md) |
| 3 | Modelo generativo | VAE (Fashion-MNIST) | [`Project3/`](Project3/README.md) |
| 4 | Séries temporais | LSTM / GRU | [`Project4/`](Project4/instructions.md) |

- **Projeto 1 — Regressão com MLP:** ajuste de topologia e taxas de uma MLP para
  uma função não linear, seguido de ablação (momentum, L1, L2, Dropout) sob duas
  estratégias de partição. Seleção pelo MSE de validação.
- **Projeto 2 — Quantização de cores:** SOM e GNG para reduzir a paleta de cores
  de imagens, avaliados por erro de quantização e erro topológico.
- **Projeto 3 — VAE generativo:** Autoencoders Variacionais no Fashion-MNIST com
  espaços latentes de 2D e 8D, e exploração do espaço latente (clusters, rótulos,
  PCA).
- **Projeto 4 — Séries temporais:** LSTM/GRU para previsão ou classificação,
  seguindo a mesma estrutura de relatório do Projeto 1 *(em desenvolvimento)*.

## Convenções comuns

Os projetos compartilham um mesmo padrão de organização e reprodutibilidade:

```
ProjectN/
├── src/              # código-fonte, executável via python -m src.<modulo>
├── data/             # dados de entrada (brutos e processados)
├── results/          # artefatos gerados (.pt, .csv)
├── figures/          # figuras geradas
├── report/           # relatório LaTeX (main.tex)
└── requirements.txt  # dependências Python
```

- Semente aleatória fixa (`SEED = 42`) para reprodutibilidade.
- Seleção de modelos sempre pelo conjunto de validação, com o teste reservado
  para a avaliação final.

## Como começar

Cada projeto possui seu próprio ambiente e dependências. A partir da pasta do
projeto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Consulte o `README.md` (ou `instructions.md`) de cada projeto para os comandos
de execução específicos.
