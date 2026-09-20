import pandas as pd
import matplotlib.pyplot as plt


#load dataset
df = pd.read_csv("data/dataset.csv")


#basic information
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDescriptive statistics:")
print(df.describe())


#plot data
plt.figure(figsize=(8, 5))

plt.scatter(
    df["x"],
    df["y"]
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Dataset")

plt.grid(True)
plt.show()