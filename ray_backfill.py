from psutil import cpu_count
import ray
from tenacity import retry, stop_after_attempt
from polygon_backfill import get_open_market_dates, dates_from_s3, find_remaining_dates, backfill_date

ray.init(num_cpus=5, ignore_reinit_error=True)

@ray.remote
@retry(stop=stop_after_attempt(2))
def backfill_date_ray(symbol, date, tick_type, result_path):
    result = backfill_date(
        symbol=symbol, 
        date=date, 
        tick_type=tick_type, 
        result_path=result_path,
        save_local=True,
        upload_to_s3=True,
    )
    return symbol + '|' + date


def backfill_dates_ray(symbols: list, start_date: str, end_date: str, tick_type: str, result_path: str) ->list:
    request_dates = get_open_market_dates(start_date, end_date)
    futures = []
    for symbol in symbols:
        existing_dates = dates_from_s3(symbol, tick_type)
        remaining_dates = find_remaining_dates(request_dates, existing_dates)
        for date in remaining_dates:
            result = backfill_date_ray.remote(
                symbol=symbol,
                date=date,
                tick_type=tick_type,
                result_path=result_path
            )
            futures.append(result)
    results = ray.get(futures)
    return results

symbols = ['GLD', 'GDX', 'DUST', 'SPY']
start_date = '2020-01-01'
end_date = '2020-10-20'
tick_type = 'trades'
result_path = '/Users/bobcolner/QuantClarity/pandas-polygon/data'

request_dates = get_open_market_dates(start_date, end_date)
futures = []
for symbol in symbols:
    existing_dates = dates_from_s3(symbol, tick_type)
    remaining_dates = find_remaining_dates(request_dates, existing_dates)
    for date in remaining_dates:
        result = backfill_date_ray.remote(
            symbol=symbol, 
            date=date, 
            tick_type=tick_type, 
            result_path=result_path
        )
        futures.append(result)
results = ray.get(futures)