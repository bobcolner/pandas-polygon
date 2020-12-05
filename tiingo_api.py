from requests import get
from utils_globals import TIINGO_API_KEY

# https://api.tiingo.com/documentation/
BASE_URL = 'https://api.tiingo.com'


def validate_response(response: str):
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def get_stock_symbol_daily(symbol: str, start_date: str, end_date: str, resample_freq: str='daily'):
    url = BASE_URL + f"/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&resampleFreq={resample_freq}&token={TIINGO_API_KEY}"
    response = get(url)
    return validate_response(response)


def get_crypto_symbol_intaday(symbol: str, start_date: str, end_date: str, resample_freq: str='daily'):
    url = BASE_URL + f"/tiingo/crypto/prices?tickers={symbols}?startDate={start_date}&endDate={end_date}&resampleFreq={resample_freq}&token={TIINGO_API_KEY}"
    response = get(url)
    return validate_response(response)


def get_fx_symbol_intaday(symbol: str, start_date: str, end_date: str, resample_freq: str='1hour'):
    url = BASE_URL + f"/tiingo/fx/{symbol}/prices?startDate={start_date}&endDate={end_date}&resampleFreq={resample_freq}&token={TIINGO_API_KEY}"
    response = get(url)
    return validate_response(response)


def get_iex_symbol_intaday(symbol: str, start_date: str, end_date: str, resample_freq: str='1hour'):
    url = BASE_URL + f"/iex/{symbol}/prices?startDate={start_date}&endDate={end_date}&resampleFreq={resample_freq}&token={TIINGO_API_KEY}"
    response = get(url)
    return validate_response(response)
