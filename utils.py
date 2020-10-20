import scipy.stats as stats


def compound_interest(principle:float, rate:float, peroids:int): 
    # Calculates compound interest  
    total_return = principle * (pow((1 + rate / 100), peroids)) 
    print("Total Interest $:", round(total_return, 2))
    print("Anualized Peroid %", round(total_return / principle, 1) * 100)

# compount daily for 1 year (market days)
compound_interest(principle=100000, rate=.5, peroids=250)


from scipy.cluster.hierarchy import linkage, is_valid_linkage, fcluster
from scipy.spatial.distance import pdist

## Load dataset
X = np.load("dataset.npy")

## Hierarchical clustering
dists = pdist(X)
Z = linkage(dists, method='centroid', metric='euclidean')

print(is_valid_linkage(Z))

## Now let's say we want the flat cluster assignement with 10 clusters.
#  If cut_tree() was working we would do
from scipy.cluster.hierarchy import cut_tree
cut = cut_tree(Z, 10)

clust = fcluster(Z, k, criterion='maxclust')


### from scipy.cluster.hierarchy import cut_tree
from scipy import cluster
np.random.seed(23)
X = np.random.randn(50, 4)
Z = cluster.hierarchy.ward(X, )
cutree = cluster.hierarchy.cut_tree(Z, n_clusters=[5, 10])
cutree[:10]


# good = [0, 1, 3, 4, 8, 9, 11, 14, 23, 25, 27, 28, 30, 34, 36]
# listed_bad = [2, 5, 7, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 29, 33, 37, 52, 53]
# confirmed_bad = [2, 7, 10, 12, 13, 15, 16, 17, 20, 22, 38, 52, 53]
# neverseen_bad = [5, 18, 19, 21, 29, 33]
# listed_blank = [6, 17, 18, 19, 24, 26, 32, 35, 39-51, 54, 55, 56, 59]
# after_hours = [12]
# odd_lot = [37]
# neutral = [41]


def apply_fft(series, components=[3, 6, 9, 100]):
    
    close_fft = np.fft.fft(series)
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    plt.figure(figsize=(14, 7), dpi=100)
    fft_list = np.asarray(fft_df['fft'].tolist())

    for num_ in components:
        fft_list_m10 = np.copy(fft_list) 
        fft_list_m10[num_:-num_] = 0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))

    plt.plot(series, label='Real')
    plt.xlabel('Time')
    plt.ylabel('USD')
    plt.title('Stock trades & Fourier transforms')
    plt.legend()
    plt.show()

def apply_condtion_filter(df, keep_afterhours):
    keep_afterhours=True
    conditions_idx=[]
    afterhours_idx=[]
    for row in df.itertuples():
        filter_conditions = [2, 5, 6, 7, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 29, 
            32, 33, 35, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
        if keep_afterhours is False:
            filter_conditions.remove(12)
        if row.condition is not None:
            condition_bool = any(np.isin(row.condition, np.array(filter_conditions)))
            afterhours_bool = any(np.isin(row.condition, 12))
        else: 
            condition_bool = False
            afterhours_bool = False
        conditions_idx.append(condition_bool)
        afterhours_idx.append(afterhours_bool)
    return pd.Series(conditions_idx), pd.Series(afterhours_idx)


def weighted_kernel_density_1d(values, weights, bw='silverman', plot=False):
    from statsmodels.nonparametric.kde import KDEUnivariate
    kden= KDEUnivariate(values)
    kden.fit(weights=weights, bw=bw, fft=False)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(kden.support, [kden.evaluate(xi) for xi in kden.support], 'o-')
    return kden


def quantile_from_kdensity(kden, quantile=0.5):
    return kden.support[kde1.cdf >= quantile][0]


def wkde1d(state, new_bar):
    
    try:
        wkde = weighted_kernel_density_1d(
                values=np.array(state['trades']['price']), 
                weights=np.array(state['trades']['volume'])
                )
        # new_bar['wkde'] = wkde
        new_bar['kd_10'] = quantile_from_kdensity(new_bar['kden'], quantile=0.1)
        new_bar['kd_50'] = quantile_from_kdensity(new_bar['kden'], quantile=0.5)
        new_bar['kd_90'] = quantile_from_kdensity(new_bar['kden'], quantile=0.9)
    except:
        new_bar['wkde'] = None
        new_bar['kd_10'] = None
        new_bar['kd_50'] = None
        new_bar['kd_90'] = None

    return new_bar



def rolling_decay_vwap(ts, window_len=7, decay='none'):
    nrows = list(range(len(ts)))
    vwap = []
    for nrow in nrows:
        if nrow >= window_len:
            price = ts['price'][nrow - window_len:nrow].values
            volume = ts['volume'][nrow - window_len:nrow].values
            if decay == 'exp':
                exp_decay = stats.expon.pdf(x=list(range(0, window_len)))
                weight = volume * exp_decay
            if decay == 'linear':
                linear_decay = np.array(range(1, window_len+1)) / window_len
                weight = volume * linear_decay
            if decay == 'none':
                weight = volume
            tmp = (price * weight).sum() / weight.sum()
            vwap.append(tmp)
        else:
            vwap.append(None)
    return vwap


def trunc_timestamp(ts, trunc_list=None, add_date_time=False):
    if add_date_time:
        ts['date_time'] = pd.to_datetime(ts['epoch'], utc=True, unit='ns')
    if 'micro' in trunc_list:
        ts['epoch_micro'] = ts['epoch'].floordiv(10 ** 3)
    if 'ms' in trunc_list:
        ts['epoch_ms'] = ts['epoch'].floordiv(10 ** 6)
        if  add_date_time:
            ts['date_time_ms'] = ts['date_time'].values.astype('<M8[ms]')
    if 'cs' in trunc_list:
        ts['epoch_cs'] = ts['epoch'].floordiv(10 ** 7)
    if 'ds' in trunc_list:
        ts['epoch_ds'] = ts['epoch'].floordiv(10 ** 8)
    if 'sec' in trunc_list:
        ts['epoch_sec'] = ts['epoch'].floordiv(10 ** 9)
        if add_date_time:
            ts['date_time_sec'] = ts['date_time'].values.astype('<M8[s]')
    if 'min' in trunc_list:
        ts['epoch_min'] = ts['epoch'].floordiv((10 ** 9) * 60)
        if add_date_time:
            ts['date_time_min'] = ts['date_time'].values.astype('<M8[m]')
    if 'min5' in trunc_list:
        ts['epoch_min5'] = ts['epoch'].floordiv((10 ** 9) * 60 * 5)
    if 'min10' in trunc_list:
        ts['epoch_min10'] = ts['epoch'].floordiv((10 ** 9) * 60 * 10)
    if 'min15' in trunc_list:
        ts['epoch_min15'] = ts['epoch'].floordiv((10 ** 9) * 60 * 15)
    if 'min30' in trunc_list:
        ts['epoch_min30'] = ts['epoch'].floordiv((10 ** 9) * 60 * 30)
    if 'hour' in trunc_list:
        ts['epoch_hour'] = ts['epoch'].floordiv((10 ** 9) * 60 * 60)
        if add_date_time:
            ts['date_time_hour'] = ts['date_time'].values.astype('<M8[h]')
    return ts


def ts_groupby(df, column='epoch_cs'):
    groups = df.groupby(column, as_index=False, squeeze=True).agg({'price': ['count', 'mean'], 'volume':'sum'})
    groups.columns = ['_'.join(tup).rstrip('_') for tup in groups.columns.values]
    return groups


def ticks_df_tofile(df:pd.DataFrame, symbol:str, date:str, result_path:str, 
    date_partition:str, formats=['parquet', 'feather']):

    if date_partition == 'file_dates':
        partion_path = f"{symbol}/{date}"
    elif date_partition == 'dir_dates':
        partion_path = f"{symbol}/{date}/"
    elif date_partition == 'hive':
        partion_path = f"symbol={symbol}/date={date}/"
    
    if 'csv' in formats:
        path = result_path + '/csv/' + partion_path
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_csv(
            path_or_buf=path+'data.csv',
            index=False,
        )
    if 'parquet' in formats:
        path = result_path + '/parquet/' + partion_path
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            path=path+'data.parquet',
            engine='auto',
            index=False,
            partition_cols=None,
        )
    if 'feather' in formats:
        path = result_path + '/feather/' + partion_path
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_feather(path+'data.feather', version=2)


def dates_from_path(dates_path:str, date_partition:str) -> list:
    if os.path.exists(dates_path):
        file_list = os.listdir(dates_path)
        if '.DS_Store' in file_list:
            file_list.remove('.DS_Store')

        if date_partition == 'file_symbol_date':
            # assumes {symbol}_{yyyy-mm-dd}.{format} filename template
            existing_dates = [i.split('_')[1].split('.')[0] for i in file_list]
        
        elif date_partition == 'file_dates':
            # assumes {yyyy-mm-dd}.{format} filename template
            existing_dates = [i.split('.')[0] for i in file_list]
        
        elif date_partition == 'dir_dates':
            # assumes {yyyy-mm-dd}/data.{format} filename template
            existing_dates = file_list

        elif date_partition == 'hive':
            # assumes {date}={yyyy-mm-dd}/data.{format} filename template
            existing_dates = [i.split('=')[1] for i in file_list]

        return existing_dates




last_tick_time = pd.to_datetime(last_tick, utc=True, unit='ns').tz_convert('America/New_York')
print('Downloaded:', len(ticks_batch), symbol, 'ticks; latest tick timestamp(NYC):', last_tick_time)

from dask.distributed import Client, progress, fire_and_forget
client = Client(threads_per_worker=4, n_workers=4)

request_dates = get_open_market_dates(start_date, end_date)
futures = []
for symbol in symbols: 
    existing_dates = dates_from_s3(symbol, tick_type)
    remaining_dates = find_remaining_dates(request_dates, existing_dates)
    for date in remaining_dates:
        result = client.submit(
            backfill_date,
            symbol=symbol, 
            date=date, 
            tick_type=tick_type, 
            result_path=result_path,
            save_local=True,
            upload_to_s3=True,
        )
        futures.append(result)
        
client.gather(futures)