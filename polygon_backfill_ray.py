import ray
from tenacity import retry, stop_after_attempt
from dates import get_open_market_dates, find_remaining_dates
from polygon_s3 import list_symbol_dates, get_and_save_date_df


@ray.remote
@retry(stop=stop_after_attempt(2))
def backfill_ray_task(symbol: str, date: str, tick_type: str):
    df = get_and_save_date_df(symbol, date, tick_type)
    return True


def backfill(start_date: str, end_date: str, symbols: list, tick_type: str) -> list:

    request_dates = get_open_market_dates(start_date, end_date)
    futures = []
    for symbol in symbols:
        existing_dates = list_symbol_dates(symbol, tick_type)
        remaining_dates = find_remaining_dates(request_dates, existing_dates)
        for date in remaining_dates:
            result = backfill_ray_task.remote(symbol, date, tick_type)
            futures.append(result)

    return ray.get(futures)
