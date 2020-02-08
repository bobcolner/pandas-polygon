import api
import bars
# %load_ext autoreload
# %autoreload 2

symbol='GLD'
fmat = 'parquet'
date = '2019-05-09'

ts = api.read_matching_files(f"../data/ticks/{fmat}/{symbol}/{symbol}_{date}.{fmat}", format=fmat)

bars_15m = bars.time_bars(ts, freq='15min', date=date)
