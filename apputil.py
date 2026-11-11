import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return (centroids, labels)
