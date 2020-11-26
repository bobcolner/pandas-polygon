import ray
from polygon_rest_api import get_ticker_details


@ray.remote
def get_symbol_details_ray(symbol: str) -> dict:
    return get_ticker_details(symbol)


def symbol_details_ray(symbols: list) -> list:
    futures = []
    for symbol in symbols:
        result = get_symbol_details_ray.remote(symbol)
        futures.append(result)

    return ray.get(futures)
