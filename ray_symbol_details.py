import ray
# from tenacity import retry, stop_after_attempt, wait_exponential
from polygon_rest_api import get_ticker_details


@ray.remote
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_symbol_details_ray(symbol: str) -> dict:
    print(symbol)
    return get_ticker_details(symbol)


def symbol_details_ray(symbols: list) -> list:
    futures = []
    for symbol in symbols:
        result = get_symbol_details_ray.remote(symbol)
        if result:
            futures.append(result)

    return ray.get(futures)


def dets_to_df(symbols):
	results = symbol_details_ray(symbols)
	results = [i for i in results if i]  # remove empty/null items from list
	return pd.DataFrame(results)


def get_remaining_symbols(requested_symbols: list, current_symbol_dets: list) -> list:
    return list(set(requested_symbols) - set(current_symbol_dets))