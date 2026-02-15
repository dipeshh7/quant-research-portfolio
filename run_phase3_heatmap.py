import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("phase3_grid_results.csv")

pivot = df.pivot(index="short_window", columns="long_window", values="sharpe")

plt.figure(figsize=(10, 5))
plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title("Sharpe Heatmap (Portfolio MA Crossover, With Costs)")
plt.xlabel("Long Window")
plt.ylabel("Short Window")
plt.colorbar(label="Sharpe")
plt.tight_layout()
plt.show()
