def compound_interest(principle: float, rate: float, peroids: int): 
    # Calculates compound interest  
    total_return = principle * (pow((1 + rate / 100), peroids)) 
    print("Total Interest $:", round(total_return, 2))
    print("Anualized Peroid %", round(total_return / principle, 1) * 100)


def read_matching_files(glob_string: str, reader=pd.read_csv) -> pd.DataFrame:
    from glob import glob
    return pd.concat(map(reader, glob(path.join('', glob_string))), ignore_index=True)



def add_filter_features(df: pd.DataFrame, col: str, length: int=10, power: float=1.0) -> pd.DataFrame:
    # compute jma filter
    df.loc[:, col+'_jma_'+str(length)] = jma_expanding_filter(df[col], length=length, power=power)
    # jma diff(n)
    df.loc[:, col+'_jma_'+str(length)+'_diff1'] = df.loc[:, col+'_jma_'+str(length)].diff(1)
    df.loc[:, col+'_jma_'+str(length)+'_diff3'] = df.loc[:, col+'_jma_'+str(length)].diff(3)
    df.loc[:, col+'_jma_'+str(length)+'_diff6'] = df.loc[:, col+'_jma_'+str(length)].diff(6)
    df.loc[:, col+'_jma_'+str(length)+'_diff9'] = df.loc[:, col+'_jma_'+str(length)].diff(9)
    # compute value - jma filter residual
    df.loc[:, col+'_jma_'+str(length)+'_resid'] = df[col] - df[col+'_jma_'+str(length)]
    # compute resid abs mean
    df.loc[:, col+'_jma_'+str(length)+'_resid_mean'] = abs(df[col+'_jma_'+str(length)+'_resid']).rolling(window=length, min_periods=0).mean()
    # compute resid abs median
    df.loc[:, col+'_jma_'+str(length)+'_resid_median'] = abs(df[col+'_jma_'+str(length)+'_resid']).rolling(window=length, min_periods=0).median()
    # compute resid abs jma
    df.loc[:, col+'_jma_'+str(length)+'_resid_jma'] = jma_expanding_filter(abs(df[col+'_jma_'+str(length)+'_resid']), length=length, phase=50, power=2)
    return df


def add_bands(df: pd.DataFrame, base_col: str, vol_col: str, multipler: int=2):
    df.loc[:, base_col+'_upper'] = df[base_col] + df[vol_col] * multipler
    df.loc[:, base_col+'_lower'] = df[base_col] - df[vol_col] * multipler
    return df


def add_volitiliy_features(df, col: str, length: int) -> pd.DataFrame:   
    df.loc[:, col+'_'+str(length)+'_std'] = df[col].rolling(window=length, min_periods=0).std()
    return df


from io import BytesIO
import pandas as pd
import pyarrow.feather as pf
from prefect.engine.serializers import Serializer


class FeatherSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes        
        bytes_buffer = BytesIO()
        pf.write_feather(
            df=value,
            dest=bytes_buffer,
            version=2,
        )
        return bytes_buffer.getvalue()

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes
        df_bytes_io = BytesIO(value)
        return pd.read_feather(df_bytes_io)


class ParquetSerializer(Serializer):

    def serialize(self, value: pd.DataFrame) -> bytes:
        # transform a Python object into bytes
        bytes_buffer = BytesIO()
        value.to_parquet(
            path=bytes_buffer,
            index=False
        )
        return bytes_buffer.getvalue()

    def deserialize(self, value:bytes) -> pd.DataFrame:
        # recover a Python object from bytes        
        df_bytes_io = BytesIO(value)
        return pd.read_parquet(df_bytes_io)


import ipdb; ipdb.set_trace(context=10)

bars[['open_time', 'price_vwap', 'tick_count', 'volume_sum', 'tick_run_max', 'tick_imbalance']]

def bar_stats(ticks):
    bar = {}
    bar['wkde'] = weighted_kernel_density_1d(values=ticks['price'], weights=ticks['volume'])
    bar['price_min'] = ticks['price'].min()
    bar['kd_10'] = quantile_from_kdensity(bar['kden'], quantile=0.1)
    bar['kd_50'] = quantile_from_kdensity(bar['kden'], quantile=0.5)
    bar['vwap'] = (ticks['price'] * ticks['volume']).sum() / ticks['volume'].sum()
    bar['kd_90'] = quantile_from_kdensity(bar['kden'], quantile=0.9)
    bar['price_max'] = ticks['price'].max()
    bar['price_std'] = ticks['price'].std()
    bar['price_range'] = bar['price_max'] - bar['price_min']
    bar['price_open'] = ticks['price'][0]
    bar['price_close'] = ticks['price'][-1]
    bar['bar_return'] = bar['price_close'] - bar['price_open']
    bar['volume'] = ticks['volume'].sum()
    bar['dollars'] = ticks['price'].sum() * ticks['volume'].sum()
    return bar


def time_bars(ts, freq='5min'):
    dr = pd.date_range(start='2019-05-09', end='2019-05-10', freq='5min', tz='utc')
    bars = []
    for i in list(range(len(dr))):
        ticks = ts[(ts.date_time >= dr[i]) & (ts.date_time < dr[i+1])]
        
        bar = bar_stats(ticks)


def find_bar_params(ts, num_bars=100):
    d = {}
    d['volume_thresh'] = round(sum(ts['volume']) / num_bars)
    d['tick_thresh'] = round(len(ts) / num_bars)
    d['dollar_thresh'] = round((np.mean(ts['price']) * sum(ts['volume'])) / num_bars)
    d['minute_thresh'] = round((6.5 / num_bars) * 60)
    return d

# state['thresh_duration_ns'] = params['thresh_duration_ns']
    # state['thresh_ticks'] = 250
    # state['thresh_volume'] = 50000
    # state['thresh_dollar'] = 6000000
    # state['thresh_tick_imbalance'] = 10 ** 3
    # state['thresh_volume_imbalance'] = 10 ** 4
    # state['thresh_dollar_imbalance']  = 10 ** 5 
    # state['thresh_price_range'] = 0.3
    # state['thresh_return'] = 0.2
    # state['thresh_renko'] = 0.1
    # state['thresh_tick_run'] = 10
    # state['thresh_volume_run'] = 10 ** 4
    # state['thresh_dollar_run'] = 10 ** 6
    
if len(output_bars) > 0:
        last_bar_side = output_bars[-1]['bar_return']
    else:
        last_bar_side = 0 

import ipdb; ipdb.set_trace(context=10)

def timeit(func):
    from functools import wraps
    from time import time
    @wraps(func)
    def newfunc(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        diff = time() - start
        print('function [{}] finished in {} ms'.format(func.__name__, int(diff * 1000)))
    return newfunc


import scipy.stats as stats


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

# sort symbol-date tuples
sort symbol-date pairs by date
sd_df = pd.DataFrame(symbol_dates).sort_values(1)
sd_d = sd_df.to_dict(orient='records')
symbol_dates = []
for row in sd_d:
    symbol_dates.append((row[0], row[1]))


# open_dates = pb.get_open_market_dates(start_date, end_date)
daily_vol_df = ps3.get_symbol_vol_filter(result_path, symbol, start_date)
bar_dates = []
for row in daily_vol_df.itertuples():
    # load ticks
    ticks_df = ps3.load_ticks(result_path, symbol, row.date, tick_type)
    # sample bars
    thresh.update({'renko_size': row.range_jma / 15})
    bars, state = bs.build_bars(ticks_df, thresh)
    d = {'date:': date,
         'bars': bars
        }
    bar_dates.append(d)    


from os import environ
from time import time_ns
import websocket as ws


if 'POLYGON_API_KEY' in environ:
    API_KEY = environ['POLYGON_API_KEY']
else:
    raise ValueError('missing poloyon api key')


def on_message(wsc: ws._app.WebSocketApp, message: str):
    print(message)
    with open('data.txt', 'a') as out_file:
        out_file.write(str(time_ns()) + ' || ' + message + '\n')


def on_error(wsc: ws._app.WebSocketApp, error: str):
    print(error)


def on_close(wsc: ws._app.WebSocketApp):
    print("### closed ###")


def on_open(wsc: ws._app.WebSocketApp, symbols: str=None):
    wsc.send(data='{"action":"auth", "params":"' + API_KEY + '"}')
    wsc.send(data='{"action":"subscribe","params":"T.SPY, T.GLD"}')
    # wsc.send(data='{"action":"subscribe","params":"' + symbols + '"}')


def run(symbols: str=None):
    wsc = ws.WebSocketApp(
        url="wss://alpaca.socket.polygon.io/stocks",
        on_message = on_message,
        on_error = on_error,
        on_close = on_close
        )
    wsc.on_open = on_open
    wsc.run_forever()
    
def backfill_dates(symbol: str, start_date: str, end_date: str, result_path: str, tick_type: str, 
    save_local=True, upload_to_s3=True):
    
    request_dates = get_open_market_dates(start_date, end_date)
    print('requested', len(request_dates), 'dates')
    
    if upload_to_s3:
        existing_dates = list_symbol_dates(symbol, tick_type)
    else:
        existing_dates = list_dates_from_path(symbol, tick_type, result_path)

    if existing_dates is not None:
        request_dates = find_remaining_dates(request_dates, existing_dates)
    
    print(len(request_dates), 'remaining dates')
    
    for date in request_dates:
        print('fetching:', date)
        backfill_date(symbol, date, tick_type, result_path, save_local, upload_to_s3)

    


def jma_filter_update(series_last: float, e0_last: float, e1_last: float, e2_last: float,
    jma_last: float, length: int=7, phase: int=50, power: int=2) -> tuple:

    if phase < -100:
        phase_ratio = 0.5
    elif phase > 100:
        phase_ratio = 2.5
    else:
        phase_ratio = phase / (100 + 1.5)
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2)
    alpha = pow(beta, power)
    e0_next = (1 - alpha) * series_last + alpha * e0_last
    e1_next = (series_last - e0_next) * (1 - beta) + beta * e1_last
    e2_next = (e0_next + phase_ratio * e1_next - jma_last) * pow(1 - alpha, 2) + pow(alpha, 2) * e2_last
    jma_next = e2_next + jma_last
    return jma_next, e0_next, e1_next, e2_next


def jma_rolling_filter(series: pd.Series, length: int=7, phase: int=50, power: int=2) -> list:

    e0_next = 0
    e1_next = 0
    e2_next = 0
    jma_next = series.values[0]
    jma = []
    for value in series:
        jma_next, e0_next, e1_next, e2_next  = jma_filter_update(
            series_last=value, e0_last=e0_next, e1_last=e1_next,
            e2_last=e2_next, jma_last=jma_next, length=length,
            phase=phase, power=power
        )
        jma.append(jma_next)

    jma[0:(length-1)] = [None] * (length-1)
    return jma    

def clean_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    # get origional number of ticks
    og_tick_count = df.shape[0]
    # drop irrgular trade conditions
    df = df.loc[df.irregular==False]
    # drop trades with >1sec timestamp diff
    dt_diff = (df.sip_dt - df.exchange_dt)
    df = df.loc[dt_diff < pd.to_timedelta(1, unit='S')]
    # add median filter and remove outlier trades
    df = median_outlier_filter(df)
    # remove duplicate trades
    num_dups = sum(df.duplicated(subset=['sip_dt', 'exchange_dt', 'sequence', 'trade_id', 'price', 'size']))
    if num_dups > 0: 
        print(num_dups, 'duplicated trade removed')
        df = df.drop_duplicates(subset=['sip_dt', 'exchange_dt', 'sequence', 'trade_id', 'price', 'size'])
    # drop trades with zero size/volume
    df = df.loc[df['size'] > 0]
    droped_rows = og_tick_count - df.shape[0]
    print('dropped', droped_rows, 'ticks (', round((droped_rows / og_tick_count) * 100, 2), '%)')
    # sort df
    df = df.sort_values(['sip_dt', 'exchange_dt', 'sequence'])
    # small cols subset
    df = df[['sip_dt', 'price', 'size']]
    return df.rename(columns={'sip_dt': 'date_time', 'size': 'volume'}).reset_index(drop=True)    


def time_bars(ticks_df: pd.DataFrame, date: str, freq: str='15min') -> list:
    import datetime as dt
    from tqdm import tqdm

    start_date = dt.datetime.strptime(date, '%Y-%m-%d')
    end_date = start_date + dt.timedelta(days=1)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq, tz='utc', closed=None)
    bars = []
    for i in tqdm(list(range(len(dates)-2))):
        ticks = ticks_df.loc[(ticks_df['date_time'] >= dates[i]) & (ticks_df['date_time'] < dates[i+1])]
        _, state = build_bars(ticks_df=ticks, thresh={})
        bar = output_new_bar(state)
        bar['open_at'] = dates[i]
        bar['close_at'] = dates[i+1]
        bars.append(bar)

    return bars
