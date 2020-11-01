import ray
from tenacity import retry, stop_after_attempt
from polygon_backfill import get_open_market_dates, find_remaining_dates, backfill_date
from polygon_s3 import get_symbol_dates


@ray.remote
@retry(stop=stop_after_attempt(2))
class Backfill(object):
    def backfill_date(symbol: str, date: str):
        df = backfill_date(
            symbol=symbol,
            date=date,
            tick_type='trades',
            result_path='/Users/bobcolner/QuantClarity/pandas-polygon/data',
            save_local=True,
            upload_to_s3=True,
        )
        return True


def get_remaining_symbol_dates(start_date: str, end_date: str, symbols: list):
    request_dates = get_open_market_dates(start_date, end_date)
    futures = []
    backfill_actor = Backfill.remote()
    for symbol in symbols:
        existing_dates = get_symbol_dates(symbol, tick_type='trades')
        remaining_dates = find_remaining_dates(request_dates, existing_dates)
        for date in remaining_dates:
            done = backfill_actor.backfill_date(symbol, date)
            futures.append(done)

    ray.get(futures)
