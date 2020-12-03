from datetime import datetime
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from corex_linearcorex import Corex


def evaluate_clustering(labels_true: list, labels_pred: list) -> dict:
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    return {'ari': ari, 'ami': ami}


def corex_fit(X: pd.DataFrame, n_hidden: int) -> tuple:
    corex = Corex(n_hidden=n_hidden, gaussianize='outliers', verbose=True)
    start_dt = datetime.now()  # measure fit time
    corex.fit(X)
    fit_td = datetime.now() - start_dt
    print(fit_td) # print fit time
    sym_clust = pd.DataFrame(list(zip(X.columns, corex.clusters()))).rename(columns={0: 'symbol', 1: 'cluster'})
    full_df = pd.merge(sym_clust, pd.Series(data=corex.tcs, name='tcs'), left_on='cluster', right_index=True)
    return corex, full_df


def corex_results(fit_corex: Corex):
    # number of latent factors fit
    n_latent_factors = len(pd.Series(fit_corex.clusters()).unique())
    # total correlation explained
    total_cor = fit_corex.tc
    # cluster size distrabution
    cluster_size_dist = fit_corex.cluster.value_counts().describe()
    # median pairwise correlation between latent factors/clusters
    median_pairwise_corr = pd.DataFrame(fit_corex.transform(X)).corr().median().median()
    # factor/cluser TC distrabution
    cluster_tcs_dist = pd.Series(fit_corex.tcs).describe(percentiles=[.25,.75,.9,.99,.999])
    # factor/cluster ranking by TC
    pct_cluseter_tcs = pd.Series(fit_corex.tcs / sum(fit_corex.tcs))
    # pct_cluseter_tcs = pd.Series(fit_corex.tcs / fit_corex.tc)  # alt
    pd.DataFrame(fit_corex.tcs).describe(percentiles=[.7,.8,.9,.99])


def full_df_results(full_df: pd.DataFrame):
    
    tops_clust = full_df[full_df.symbol=='TOPS'].cluster.values[0]  # get cluster for 'TOPS'
    other_clust_symbols = full_df.loc[full_df.cluster == tops_clust]

    symbols_from_top_factor = full_df.sort_values(['tcs', 'symbol'], ascending=False)[0:20]


def fit_hclust_model(similarity: pd.DataFrame, n_clusters: int) -> tuple:
    # from sklearn.neighbors import kneighbors_graph
    #  connectivity = kneighbors_graph(par_cor_mat, n_neighbors=5, include_self=False)
    hclust_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',  # "euclidean", "l1", "l2", "manhattan", "cosine", or "precomputed". If linkage is "ward", only "euclidean" is accepted
        memory=None,
        connectivity=None,
        compute_full_tree=True,
        linkage='ward',  # “complete”, “average”, “single”}, default=”ward”
        distance_threshold=None,
    ).fit(similarity)
    cluster_labels = pd.DataFrame({'symbol': similarity.columns, f"cluster_n{n_clusters}": hclust_model.labels_})
    return hclust_model, cluster_labels


# def fit_kmeans():
