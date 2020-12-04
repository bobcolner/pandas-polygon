from requests import get
import pandas as pd
from utils_globals import ALPHAVANTAGE_API_KEY

# https://www.alphavantage.co/documentation/
BASE_URL = 'https://www.alphavantage.co'


def validate_response(response: str):
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def get_symbol_details(symbol: str):
    url = BASE_URL + f"/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    response = get(url)
    return validate_response(response)


def get_fx_intaday(from_symbol: str, to_symbol: str, interval: str='15min'):
    url = BASE_URL + f"/query?function=FX_INTRADAY&from_symbol={from_symbol}&to_symbol={to_symbol}&interval={interval}&outputsize=full&apikey={ALPHAVANTAGE_API_KEY}"
    response = get(url)
    data = validate_response(response)
    return pd.DataFrame(data[f"Time Series FX ({interval})"], dtype='float').transpose()


def get_crypto_daily(symbol: str='BTC', market: str='USD'):
    url = BASE_URL + f"/query?function=DIGITAL_CURRENCY_DAILY&symbol={symbol}&market={market}&apikey={ALPHAVANTAGE_API_KEY}"
    response = get(url)
    data = validate_response(response)
    return pd.DataFrame(data['Time Series (Digital Currency Daily)'], dtype='float').transpose()
