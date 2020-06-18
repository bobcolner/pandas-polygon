# %load_ext autoreload
# %autoreload 2
import rest_api
import bars
import trio
import streaming_api


# start ws streaming event-loop
trio.run(streaming_api.stream_ticks, "T.SPY, T.GLD", 'data.txt')

symbol='GLD'
fmat = 'parquet'
# date = '2019-05-09'
start_date = '2020-06-04'
end_date = '2020-06-04'

trades_df = rest_api.get_trades_date(symbol, date=start_date)

# get_trades_dates(symbol, start_date, end_date, save='both')

# ts = read_matching_files(f"../data/ticks/{fmat}/{symbol}/{symbol}_{date}.{fmat}", format=fmat)

# bars_15m = bars.time_bars(ts, freq='15min', date=date)

