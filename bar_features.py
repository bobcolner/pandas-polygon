from statsmodels.stats.weightstats import DescrStatsW
    

def output_new_bar(state: dict) -> dict:
    from statsmodels.stats.weightstats import DescrStatsW
    
    new_bar = {}
    if state['stat']['tick_count'] == 0:
        return new_bar

    new_bar['bar_trigger'] = state['trigger_yet?!']
    # time
    new_bar['open_at'] = state['trades']['date_time'][0]
    new_bar['close_at'] = state['trades']['date_time'][-1]
    new_bar['duration_td'] = new_bar['close_at'] - new_bar['open_at']
    new_bar['duration_sec'] = state['stat']['duration_sec']
    new_bar['duration_min'] = new_bar['duration_sec'] / 60
    # price
    new_bar['price_open'] = state['trades']['price'][0]
    new_bar['price_close'] = state['trades']['price'][-1]
    new_bar['price_low'] = state['stat']['price_min']
    new_bar['price_high'] = state['stat']['price_max']
    new_bar['price_range'] = state['stat']['price_range']
    new_bar['price_return'] = state['stat']['price_return']
    # volume weighted price
    dsw = DescrStatsW(data=state['trades']['price'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    new_bar['price_wq10'] = qtiles[0]
    new_bar['price_wq50'] = qtiles[1]
    new_bar['price_wq90'] = qtiles[2]
    new_bar['price_wq_range'] = new_bar['price_wq90'] - new_bar['price_wq10']
    new_bar['price_wmean'] = dsw.mean
    new_bar['price_wstd'] = dsw.std
    # jma
    new_bar['jma_open'] = state['trades']['jma'][0]
    new_bar['jma_close'] = state['trades']['jma'][-1]
    new_bar['jma_low'] = state['stat']['jma_min']
    new_bar['jma_high'] = state['stat']['jma_max']
    new_bar['jma_range'] = state['stat']['jma_range']
    new_bar['jma_return'] = state['stat']['jma_return']
    # volume weighted jma
    dsw = DescrStatsW(data=state['trades']['jma'], weights=state['trades']['volume'])
    qtiles = dsw.quantile(probs=[0.1, 0.5, 0.9]).values
    new_bar['jma_wq10'] = qtiles[0]
    new_bar['jma_wq50'] = qtiles[1]
    new_bar['jma_wq90'] = qtiles[2]
    new_bar['jma_wq_range'] = new_bar['jma_wq90'] - new_bar['jma_wq10']
    new_bar['jma_wmean'] = dsw.mean
    new_bar['jma_wstd'] = dsw.std
    # tick/vol/dollar/imbalance
    new_bar['tick_count'] = state['stat']['tick_count']
    new_bar['volume'] = state['stat']['volume']
    new_bar['dollars'] = state['stat']['dollars']
    new_bar['tick_imbalance'] = state['stat']['tick_imbalance']
    new_bar['volume_imbalance'] = state['stat']['volume_imbalance']
    new_bar['dollar_imbalance'] = state['stat']['dollar_imbalance']
    # new_bar['tick_imbalance_run'] = state['stat']['tick_run']
    # new_bar['volume_imbalance_run'] = state['stat']['volume_run']
    # new_bar['dollar_imbalance_run'] = state['stat']['dollar_run']
    return new_bar
