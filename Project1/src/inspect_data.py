import pandas as pd
import matplotlib.pyplot as plt

#load dataset
df = pd.read_csv("data/dataset_p1.csv")

#save dataset summary
with open("results/dataset_summary.txt", "w") as f:
    f.write(f"Dataset shape: {df.shape}\n\n")

    f.write("Columns:\n")
    f.write(f"{df.columns.tolist()}\n\n")

    f.write("Data types:\n")
    f.write(f"{df.dtypes}\n\n")

    f.write("Missing values:\n")
    f.write(f"{df.isna().sum()}\n\n")

    f.write("Descriptive statistics:\n")
    f.write(f"{df.describe()}\n\n")

    f.write("First rows:\n")
    f.write(f"{df.head(20)}\n\n")

    f.write("Last rows:\n")
    f.write(f"{df.tail(20)}\n\n")

    f.write("Unique x values:\n")
    f.write(f"{df['x'].nunique()}\n\n")

    f.write("Value counts of x:\n")
    f.write(f"{df['x'].value_counts().sort_index()}\n")

#plot data
plt.figure(figsize=(8, 5))
plt.scatter(df["x"], df["y"])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset")
plt.grid(True)

plt.savefig("figures/dataset.png", dpi=300, bbox_inches="tight")
plt.show()