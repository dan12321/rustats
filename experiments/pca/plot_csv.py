#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

iris = pd.read_csv('examples/pca/iris_pca.csv')
cols = iris.columns[:-1]

color_map = {
    "setosa": "green",
    "versicolor": "blue",
    "virginica": "red",
}
colors = [color_map[spec] for spec in iris["species"]]
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, layout="constrained")
pos = 0
for i in range(3):
    for j in range(i+1, 4):
        x_col = cols[i]
        y_col = cols[j]
        x = iris[x_col]
        y = iris[y_col]
        axs_i = pos // 3
        axs_j = pos % 3
        axs[axs_i, axs_j].scatter(x, y, c=colors)
        axs[axs_i, axs_j].set_xlabel(x_col)
        axs[axs_i, axs_j].set_ylabel(y_col)
        pos += 1

plt.show()