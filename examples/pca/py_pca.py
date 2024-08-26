#!/usr/bin/python3
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

iris = pd.read_csv('datasets/iris.csv')
iris_cols = iris.columns[:-1]

pca = PCA(n_components=len(iris_cols))
iris_numerics = iris.loc[:, iris_cols].values
components = pca.fit_transform(iris_numerics)

cols = [f"pca{i+1}" for i in range(len(iris_cols))]
df = pd.DataFrame(data = components, columns = cols)

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
        x = df[x_col]
        y = df[y_col]
        axs_i = pos // 3
        axs_j = pos % 3
        axs[axs_i, axs_j].scatter(x, y, c=colors)
        axs[axs_i, axs_j].set_xlabel(x_col)
        axs[axs_i, axs_j].set_ylabel(y_col)
        pos += 1

plt.show()