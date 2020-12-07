import numpy as np
import pandas as pd
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster


def cluster_metrics(lables: pd.Series, dist_mat: pd.DataFrame, sym_meta: pd.DataFrame) -> pd.DataFrame:
    results = []
    for k in np.unique(lables):
        clust_idx = lables[lables == k]

        clust_dist_mat = dist_mat.iloc[clust_idx.index, clust_idx.index]
        clust_dist_mat[clust_dist_mat==0] = None  # remove identity cor/dist
        clust_avg_similarity = 1 - clust_dist_mat.mean().mean()

        clust_meta = sym_meta[sym_meta.index.isin(clust_dist_mat.columns)]
        clust_meta = clust_meta.drop(columns=['sic','cik','listdate','exchangeSymbol','type'])
        
        unq_sector = len(clust_meta.sector.unique())
        sector_purity = (clust_meta.shape[0] / unq_sector) / clust_meta.shape[0]
        unq_industry = len(clust_meta.industry.unique())
        industry_purity = (clust_meta.shape[0] / unq_industry) / clust_meta.shape[0]
        
        clust = {
            'label': k, 
            'size': clust_meta.shape[0],
            'avg_similartiy': clust_avg_similarity,
            'sector_purity': sector_purity,
            'industry_purity': industry_purity,
            'avg_range_value_pct': clust_meta.range_value_pct.mean(),
            'med_daily_dollar_volume': clust_meta.dollar_total.median(),
            'dist_mat': clust_dist_mat, 
            'symbol_meta': clust_meta, 
        }
        results.append(clust)
        
    return pd.DataFrame(results).set_index('label')


def cluster_ground_truth_eval(get_cluster_fn, n_clusters: int, X: np.array, ground_truth: np.array) -> dict:
    model = get_cluster_fn(X, n_clusters)
    if get_cluster_fn.__name__ != 'get_corex_clusters':
        cluster_labels = model.labels_
    else:
        cluster_labels = model.clusters()
    eval_metrics = gt_cluster_eval(labels_true=ground_truth, labels_pred=cluster_labels)
    eval_metrics.update({'cluster_labels': cluster_labels, 'model': model})
    return eval_metrics


def gt_cluster_eval(labels_true: list, labels_pred: list) -> dict:
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    return {'adj. rand index': ari, 'adj. multual information': ami}


def get_kmean_clusters(X: np.array, n_clusters: int):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters).fit(X)


def get_hira_clusters(X: np.array, n_clusters: int):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',  # "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed". If linkage is "ward", only "euclidean" is accepted
        compute_full_tree=False,
        linkage='ward',  # “complete”, “average”, “single”}, default=”ward”
    ).fit(X)
    return model


def get_dbscan_clusters(X: np.array, n_clusters: int=None):
    from sklearn.cluster import DBSCAN
    return DBSCAN(n_jobs=-1).fit(X)


def get_optics_clusters(X: np.array, n_clusters: int=None):
    from sklearn.cluster import OPTICS
    return OPTICS(n_jobs=-1).fit(X)


def get_corex_clusters(X: np.array, n_clusters: int):
    from corex_linearcorex import Corex
    corex = Corex(n_hidden=n_clusters, gaussianize=None, verbose=True)
    return corex.fit(X)


def get_spectral_clusters(X: np.array, n_clusters: int):
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(n_clusters, n_jobs=-1).fit(X)


def get_meanshift_clusters(X: np.array, n_clusters: int=None):
    from sklearn.cluster import MeanShift
    return MeanShift(n_jobs=-1).fit(X)
    

def get_affinityprop_clusters(X: np.array, n_clusters: int=None):
    from sklearn.cluster import AffinityPropagation
    return AffinityPropagation().fit(X)


def get_birch_clusters(X: np.array, n_clusters: int):
    from sklearn.cluster import Birch
    return Birch().fit(X)


def get_sparse_pca_model(X: np.array, n_clusters: int):
    from sklearn.decomposition import SparsePCA
    return SparsePCA(n_componentsint=n_clusters, alpha=1.0).fit(X)
