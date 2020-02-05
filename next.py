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
