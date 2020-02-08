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