import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_SIZE = 30
VAL_SIZE = 30
TEST_SIZE = 240


# carrega o dataset
df = pd.read_csv("data/dataset_p1.csv")

n = len(df)
indices = np.arange(n)

assert n == TRAIN_SIZE + VAL_SIZE + TEST_SIZE


# ============================================================
# split aleatório
# ============================================================

train_random, temp_random = train_test_split(
    indices,
    train_size=TRAIN_SIZE,
    random_state=SEED
)

val_random, test_random = train_test_split(
    temp_random,
    train_size=VAL_SIZE,
    random_state=SEED
)

random_split = np.full(n, "test", dtype=object)
random_split[train_random] = "train"
random_split[val_random] = "val"


# ============================================================
# split sistemático
#
# cada sequência possui a forma:
# r, r + 30, r + 60, ..., r + 270
#
# treino:     r = 0, 10, 20
# validação:  r = 5, 15, 25
# teste:      demais posições
# ============================================================

train_remainders = [0, 10, 20]
val_remainders = [5, 15, 25]

systematic_split = np.full(n, "test", dtype=object)

systematic_split[
    np.isin(indices % 30, train_remainders)
] = "train"

systematic_split[
    np.isin(indices % 30, val_remainders)
] = "val"


# verifica os tamanhos
assert np.sum(random_split == "train") == TRAIN_SIZE
assert np.sum(random_split == "val") == VAL_SIZE
assert np.sum(random_split == "test") == TEST_SIZE

assert np.sum(systematic_split == "train") == TRAIN_SIZE
assert np.sum(systematic_split == "val") == VAL_SIZE
assert np.sum(systematic_split == "test") == TEST_SIZE


# ============================================================
# salva as atribuições
# ============================================================

split_df = df.copy()

split_df.insert(0, "index", indices)
split_df["random_split"] = random_split
split_df["systematic_split"] = systematic_split

split_df.to_csv(
    "results/split_assignments.csv",
    index=False
)


# ============================================================
# salva os datasets separados
# ============================================================

for split_name in ["random", "systematic"]:
    for subset in ["train", "val", "test"]:
        subset_df = split_df[
            split_df[f"{split_name}_split"] == subset
        ][["index", "x", "y"]]

        subset_df.to_csv(
            f"results/{split_name}_{subset}.csv",
            index=False
        )


# ============================================================
# resumo
# ============================================================

with open("results/split_summary.txt", "w") as f:
    f.write("DATASET SPLIT SUMMARY\n")
    f.write("=====================\n\n")

    f.write(f"Dataset size: {n}\n")
    f.write(f"Seed: {SEED}\n\n")

    f.write("Random split:\n")
    f.write(
        f"  Train: {np.sum(random_split == 'train')}\n"
    )
    f.write(
        f"  Validation: {np.sum(random_split == 'val')}\n"
    )
    f.write(
        f"  Test: {np.sum(random_split == 'test')}\n\n"
    )

    f.write("Systematic split:\n")
    f.write(
        f"  Train: {np.sum(systematic_split == 'train')}\n"
    )
    f.write(
        f"  Validation: {np.sum(systematic_split == 'val')}\n"
    )
    f.write(
        f"  Test: {np.sum(systematic_split == 'test')}\n\n"
    )

    f.write("Systematic train remainders: ")
    f.write(f"{train_remainders}\n")

    f.write("Systematic validation remainders: ")
    f.write(f"{val_remainders}\n")


# ============================================================
# visualização
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, split_name, title in zip(
    axes,
    ["random_split", "systematic_split"],
    ["Random split", "Systematic split"]
):
    for subset in ["train", "val", "test"]:
        mask = split_df[split_name] == subset

        ax.scatter(
            split_df.loc[mask, "x"],
            split_df.loc[mask, "y"],
            label=subset,
            alpha=0.7
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig(
    "figures/split_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()