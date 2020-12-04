import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def evaluate_clustering(labels_true: list, labels_pred: list) -> dict:
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    return {'ari': ari, 'ami': ami}


def get_kmean_clusters(X: np.array, n_clusters: int) -> np.array:
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters).fit(X)
    return model.labels_


def get_hira_clusters(X: np.array, n_clusters: int) -> np.array:
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',  # "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed". If linkage is "ward", only "euclidean" is accepted
        compute_full_tree=False,
        linkage='ward',  # “complete”, “average”, “single”}, default=”ward”
    ).fit(X)
    return model.labels_


def get_dbscan_clusters(X: np.array) -> np.array:
    from sklearn.cluster import DBSCAN
    model = DBSCAN(n_jobs=-1).fit(X)
    return model.labels_


def get_optics_clusters(X: np.array) -> np.array:
    from sklearn.cluster import OPTICS
    model = OPTICS(n_jobs=-1).fit(X)
    return model.labels_


def get_corex_clusters(X: np.array, n_clusters: int) -> np.array:
    from corex_linearcorex import Corex
    corex = Corex(n_hidden=n_clusters, gaussianize=None, verbose=True)
    corex.fit(X)
    return corex.clusters()
