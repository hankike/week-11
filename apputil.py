import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# Exercise one


def kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    return model.cluster_centers_, model.labels_


# Exercise two
# Picking diamonds
diamonds = sns.load_dataset('diamonds')

# Only numeric
diamonds_numeric = diamonds.select_dtypes(include=[np.number])


def kmeans_diamonds(n, k):
    X = diamonds_numeric.iloc[:n].to_numpy()

    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    times = []

    for i in range(n_iter):
        start = time()

        kmeans_diamonds(n, k)

        t = time() - start
        times.append(t)

    return np.mean(times)
