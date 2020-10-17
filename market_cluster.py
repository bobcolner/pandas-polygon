import numpy as np
import pandas as pd
from tqdm import tqdm


def read_market_daily(result_path:str) -> pd.DataFrame:    
    df = read_matching_files(glob_string=result_path+'/market_daily/*.feather', reader=pd.read_feather)
    df = find_compleat_symbols(df, compleat_only=True)
    df = df.sort_index()
    return df


def plot_daily_symbols(df:pd.DataFrame, symbols=['SPY', 'QQQ'], metric='vwap') -> pd.DataFrame:
    fdf = df[['symbol', metric]][df.symbol.isin(symbols)]
    pdf = fdf.pivot(columns='symbol', values=metric)
    pdf.plot_bokeh(kind='line', sizing_mode="scale_height", rangetool=True, title=str(symbols), ylabel=metric+' [$]', number_format="1.00 $")
    return pdf


def normalize_market_df(market_df:pd.DataFrame) -> pd.DataFrame:
    # pivot symbols to columns
    close_prices = market_df.pivot(columns='symbol', values='close')
    # get returns from price ts
    returns = close_prices.diff().drop(close_prices.index[0])  # drop NA first rowa
    # get sharpe ratio
    sharpe_ratios = returns.mean() / returns.std(ddof=0)
    sharpe_ratios.name = 'sharpe_ratio'
    # get z-score of returns
    zscore_returns = (returns - returns.mean()) / returns.std(ddof=0)
    return zscore_returns, sharpe_ratios

    
def linreg_residuals(x:pd.Series, y:pd.Series, summary=False):
    from sklearn.linear_model import LinearRegression  # BayesianRidge
    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)
    ols = LinearRegression()
    ols.fit(x_df, y_df)
    beta = ols.coef_[0][0]
    intercept = ols.intercept_[0]
    y_hat = ols.predict(x_df).squeeze()
    residual = y - y_hat
    df = pd.DataFrame(
        {'y': y, 
         'y_hat': y_hat, 
         'residual': residual
        }
    )
    if summary is True:
        import statsmodels.api as sm
        mod = sm.OLS(y, X)
        fit_mod = mod.fit()
        print(fit_mod.summary())
    return df, beta, intercept


def colwise_linreg(df, beta_symbol='SPY') -> pd.DataFrame:
    results = []
    betas = []
    for col in tqdm(df.columns):
        # linreg out market'beta' and return residuals
        output, beta, intercept = linreg_residuals(x=df[beta_symbol], y=df[col])
        results.append(output['residual'])
        betas.append(beta)
    # convert list of pd.Series to df
    resid_df = pd.DataFrame(results).transpose()
    resid_df.columns = df.columns    
    return resid_df


def colwise_partial_distcorr(df, col1:str, partial:str):
    import dcor
    pdc_list = []
    dc_list = []
    ipdc_list = []
    idc_list = []
#     for col2 in tqdm(df.columns):
    for col2 in df.columns:
        dc = dcor.distance_correlation(x=df[col1], y=df[col2])
        dc_list.append(dc)
        if partial is not None:
            pdc = dcor.partial_distance_correlation(x=df[col1], y=df[col2], z=df[partial])
            pdc_list.append(pdc)
    
    result_df = pd.DataFrame()
    result_df['distance_corr'] = dc_list
    result_df['partial_distance_corr'] = pdc_list
    result_df['col1'] = col1
    result_df['col2'] = df.columns
    result_df['partial'] = partial
    return result_df


def plot_pca(X:pd.DataFrame, n_components:int):
    from sklearn.decomposition import PCA
    pca = PCA(n_components)
    X_r = pca.fit(X).transform(X)
    pd.Series(pca.explained_variance_ratio_).cumsum().plot_bokeh(sizing_mode="scale_height")


def get_linkage_matrix(model) -> np.array:
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    return linkage_matrix


def plot_dendrogram(hclust_mod, p, truncate_mode='level', figsize=(20, 14)):
    from scipy.cluster.hierarchy import dendrogram
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(
        Z=get_linkage_matrix(hclust_mod),
        p=p,
        truncate_mode=truncate_mode,
        color_threshold=30,
        get_leaves=True,
        orientation='left',  # left, right, bottom
        labels=None,
        count_sort=False,
        distance_sort=True,
        show_leaf_counts=True,
        no_plot=False,
        no_labels=False,
        leaf_font_size=None,
        leaf_rotation=None,
        leaf_label_func=None,
        show_contracted=False,
        link_color_func=None,
        ax=None,
        above_threshold_color='b',
    )
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


def fit_hclust_model(similarity:pd.DataFrame, n_clusters:int):
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import AgglomerativeClustering
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
    cluster_labels = pd.DataFrame({'symbol':similarity.columns, f"cluster_n{n_clusters}":hclust_model.labels_})
    return hclust_model, cluster_labels


def cluster_sim_matrix(similarity:pd.DataFrame) -> pd.DataFrame:
    # cluster similarity matrix
    hclust_model_n10, cluster_labels_n10 = fit_hclust_model(similarity, n_clusters=10)
    hclust_model_n50, cluster_labels_n50 = fit_hclust_model(similarity, n_clusters=50)
    hclust_model_n100, cluster_labels_n100 = fit_hclust_model(similarity, n_clusters=100)
    hclust_model_n200, cluster_labels_n200 = fit_hclust_model(similarity, n_clusters=200)
    # join all cluster lables
    cluster_lables = cluster_labels_n10.join(cluster_labels_n50, rsuffix='_').join(cluster_labels_n100, rsuffix='__').join(cluster_labels_n200, rsuffix='___').drop(columns=['symbol_', 'symbol__', 'symbol___'])
    return cluster_lables


def join_symbol_data(
    details_df:pd.DataFrame, 
    cluster_lables:pd.DataFrame, 
    sharpe_ratios:pd.DataFrame, 
    mdf:pd.DataFrame
) -> pd.DataFrame:
    # join clust labs w details
    clusters_dets = pd.merge(cluster_lables, details_df[['symbol', 'name', 'sector', 'industry', 'tags', 'similar', 'type']], on='symbol', how='left')
    # add sharpe ratio
    clusters_sr = pd.merge(clusters_dets, sharpe_ratios, left_on='symbol', right_index=True)
    daily_avg_vol = mdf.groupby('symbol')['dollar_total'].mean()
    daily_avg_vol.name = 'daily_avg_dollar_volume'
    symbol_meta = pd.merge(clusters_sr, daily_avg_vol, left_on='symbol', right_index=True)
    return symbol_meta


def get_cluster_coheasion(sim_df:pd.DataFrame, symbol_meta:pd.DataFrame, cluster_col:str) -> pd.DataFrame:
    
    data = pd.merge(sim_df, symbol_meta, left_index=True, right_on='symbol')
    out = []
    for clust in data[cluster_col].unique(): 
        avg_sim = data[data[cluster_col]==clust].iloc[:,0].mean()
        out.append({'cluster':clust, 'avg_similarity':avg_sim})

    cluster_coheasion = pd.DataFrame(out)
    cluster_size = symbol_meta.groupby(cluster_col).count()['symbol']
    cluster_size.name = 'size'
    cluster_meta = pd.merge(cluster_coheasion.avg_similarity, cluster_size, left_index=True, right_index=True).sort_index()
    return cluster_meta
