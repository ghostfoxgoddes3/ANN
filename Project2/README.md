# Projeto 2 — Quantização de Cores com SOM e GNG

Quantização de cores de imagens usando dois modelos neurais não supervisionados:
um **Mapa Auto-Organizável (SOM)** e um **Growing Neural Gas (GNG)**. Cada pixel
é representado por seus componentes RGB normalizados, e os protótipos aprendidos
pela rede são usados para reconstruir a imagem com uma paleta reduzida. A
avaliação combina erro de quantização, erro topológico e comparação visual das
reconstruções.

## Estrutura

```
Project2/
├── src/
│   ├── data.py                   # leitura de imagens e reconstrução
│   ├── som.py                    # implementação do Self-Organizing Map
│   ├── gng.py                    # implementação do Growing Neural Gas
│   ├── metrics.py                # erro de quantização e erro topológico
│   ├── resize_images.py          # pré-processamento (redimensionamento)
│   ├── main.py                   # execução dos experimentos
│   ├── generate_mosaics.py       # mosaicos comparativos
│   ├── palette_visualization.py  # visualização das paletas
│   └── plot_som_palettes.py      # figuras das paletas do SOM
├── figures/
│   ├── resized/     # imagens redimensionadas (geradas, não versionadas)
│   └── ...          # imagens de entrada originais
├── results/         # reconstruções e tabelas (geradas)
├── report/          # relatório LaTeX
└── requirements.txt # dependências Python
```

## Como executar

A partir da pasta `Project2`:

```bash
pip install -r requirements.txt

# 1) pré-processamento: gera cópias redimensionadas em figures/resized/
python -m src.resize_images

# 2) execução dos experimentos (reconstruções salvas em results/)
python -m src.main

# 3) figuras complementares
python -m src.generate_mosaics
python -m src.palette_visualization
python -m src.plot_som_palettes

# 4) compilação do relatório
cd report && latexmk -pdf main.tex
```

As imagens originais são preservadas; apenas as cópias redimensionadas são
geradas em `figures/resized/`.

## Configurações experimentais

- **SOM:** grades bidimensionais de $4 \times 4$, $8 \times 8$ e $16 \times 16$.
- **GNG:** redes com no máximo 16, 64 e 256 neurônios.
- Amostragem de 5000 pixels para treino, repetida com seeds 42, 43 e 44.

A comparação final considera o erro de quantização e o erro topológico, além da
análise visual das imagens reconstruídas com a paleta reduzida.
