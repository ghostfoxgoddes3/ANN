# Projeto 1 — Regressão com MLP e Estudo de Ablação

Regressão de uma função não linear ($x \mapsto y$, 300 observações) com um
Perceptron Multicamadas (MLP) em PyTorch. O modelo *baseline* é treinado com
SGD puro e, em seguida, submetido a um estudo de ablação (momentum, L2, L1,
Dropout e combinação), sempre selecionando configurações pelo MSE de validação.
Todo o experimento é repetido sob duas estratégias de partição (aleatória e
sistemática) para separar o efeito do método do efeito da composição dos dados.

## Estrutura

```
Project1/
├── src/
│   ├── inspect_data.py     # estatísticas descritivas e figura do dataset
│   ├── split_data.py       # partições aleatória e sistemática (10/10/80)
│   ├── preprocess_data.py  # StandardScaler ajustado só no treino -> .npz
│   ├── baseline_search.py  # busca de topologia, taxa e batch (val MSE)
│   ├── baseline.py         # treino do baseline (SGD puro) + curvas
│   ├── momentum.py / l2.py / l1.py / dropout.py / combined.py  # ablações
│   ├── ablation.py         # consolidação comparativa das ablações
│   ├── final_models.py     # treino das configurações selecionadas
│   ├── final_test.py       # avaliação final no conjunto de teste
│   └── analysis.py         # paridade e análise de resíduos
├── data/            # dataset_p1.csv e cópias por partição
├── results/         # checkpoints (.pt), históricos e tabelas (.csv)
├── figures/         # curvas de aprendizagem, paridade e resíduos
├── report/          # relatório LaTeX (main.tex)
└── requirements.txt # dependências Python
```

## Como executar

A partir da pasta `Project1`:

```bash
pip install -r requirements.txt

# 1) inspeção e partição dos dados
python -m src.inspect_data
python -m src.split_data
python -m src.preprocess_data

# 2) busca de hiperparâmetros e treino do baseline
python -m src.baseline_search
python -m src.baseline

# 3) estudo de ablação
python -m src.momentum
python -m src.l2
python -m src.l1
python -m src.dropout
python -m src.combined
python -m src.ablation

# 4) modelos finais e avaliação no teste
python -m src.final_models
python -m src.final_test
python -m src.analysis

# 5) compilação do relatório
cd report && latexmk -pdf main.tex
```

## Decisões experimentais

- **Partições:** aleatória (seed 42) e sistemática (por resto da divisão do
  índice por 30), ambas com 30 treino / 30 validação / 240 teste.
- **Pré-processamento:** `StandardScaler` ajustado apenas no treino e aplicado
  a validação e teste; as métricas finais são reportadas na escala original.
- **Seleção:** sempre pelo menor MSE de validação, com o conjunto de teste
  mantido intocado até a avaliação final.
- **Treino:** `SEED = 42`, até 1000 épocas, *early stopping* com `patience = 20`.

## Resumo dos resultados

Topologia selecionada em ambas as partições: `(128, 64)` com ReLU, $\eta = 0{,}01$.

| Partição    | Modelo final | MSE teste | RMSE   | MAE    | $R^2$   |
|-------------|--------------|----------:|-------:|-------:|--------:|
| Aleatória   | Momentum     | 0,4471    | 0,6687 | 0,5561 | 0,1127  |
| Sistemática | Baseline     | 0,4195    | 0,6477 | 0,5254 | 0,1909  |

(Métricas na escala original da variável resposta.)

- No split aleatório, adicionar *momentum* reduziu o MSE de validação (~14 %);
  no sistemático, não houve ganho.
- L1 e L2 ($\lambda = 10^{-4}$) tiveram efeito discreto; Dropout ($p = 0{,}2$)
  piorou o desempenho, compatível com regularização excessiva para o número
  reduzido de amostras de treino.
