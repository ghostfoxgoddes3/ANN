# Projeto 3 — Modelo Generativo (VAE) para Fashion-MNIST

Treinamento de Autoencoders Variacionais (VAE) sobre o Fashion-MNIST, com
ajuste de topologia por loss de validação para dois espaços latentes (2D e 8D)
e exploração do espaço latente (clusters, separação de rótulos e PCA).

## Estrutura

```
Project3/
├── src/
│   ├── data.py      # download/leitura do Fashion-MNIST (IDX) e splits
│   ├── vae.py       # modelo VAE (MLP) e função de custo (ELBO)
│   ├── train.py     # laço de treino com early stopping
│   ├── main.py      # busca de topologia + treino final (gera results/)
│   └── explore.py   # visualizações do espaço latente (gera figures/)
├── data/            # arquivos brutos do Fashion-MNIST (baixados)
├── results/         # checkpoints (.pt), históricos e tabelas (.csv)
├── figures/         # figuras geradas
└── report/          # relatório LaTeX (main.tex)
```

## Como executar

```bash
pip install -r requirements.txt

# 1) busca de topologia + treino dos modelos finais (2D e 8D)
python -m src.main

# 2) geração das figuras de exploração do espaço latente
python -m src.explore

# 3) compilação do relatório
cd report && latexmk -pdf main.tex
```

Os dados do Fashion-MNIST são baixados automaticamente na primeira execução
e cacheados em `data/`.

## Resumo dos resultados

| Latente | Topologia      | Loss teste | Recon. | KL    | Var. PCA 2D |
|---------|----------------|-----------:|-------:|------:|------------:|
| 2D      | (512,256,128)  | 258,2      | 251,5  | 6,7   | — (direta)  |
| 8D      | (512,256)      | 240,7      | 227,9  | 12,7  | 34,5 %      |

- O VAE 2D gera clusters legíveis; o VAE 8D reconstrói melhor.
- No espaço 2D há clusters claros para calçados, calças e bolsas; peças de
  parte superior do corpo se sobrepõem.
- A projeção PCA do espaço 8D ilustra apenas 34,5 % da variância (PC1 18,0 % +
  PC2 16,6 %).
