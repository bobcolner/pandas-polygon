import os
from requests import get


BASE_URL = 'https://api.polygon.io'

if 'POLYGON_API_KEY' in os.environ:
    API_KEY = os.environ['POLYGON_API_KEY']
else:
    raise ValueError('missing poloyon api key')


def validate_response(response):
    if response.status_code == 200:
        return response.json()['results']
    else:
        response.raise_for_status()


def get_grouped_daily(locale='us', market='stocks', date='2020-01-02'):
    url = BASE_URL + f"/v2/aggs/grouped/locale/{locale}/market/{market}/{date}?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_markets():
    url = BASE_URL + f"/v2/reference/markets?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_locales():
    url = BASE_URL + f"/v2/reference/locales?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_types():
    url = BASE_URL + f"/v2/reference/types?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_tickers_page(page=1, stock_type='etp'):
    path = BASE_URL + f"/v2/reference/tickers"
    params = {}
    params['apiKey'] = API_KEY
    params['page'] = page
    params['active'] = 'true'
    params['perpage'] = 50
    params['market'] = 'stocks'
    params['locale'] = 'us'
    params['type'] = stock_type
    response = get(path, params)
    if response.status_code != 200:
        response.raise_for_status()
    tickers_list = response.json()['tickers']
    return tickers_list


def get_all_tickers(start_page=1, end_page=None, stock_type='etp'):
    # stock_type: [cs, etp]
    run = True
    page_num = start_page
    all_tickers = []
    while run == True:
        print('getting page #: ', page_num)
        tickers = get_tickers_page(page=page_num, stock_type=stock_type)
        all_tickers = all_tickers + tickers
        page_num = page_num + 1
        if len(tickers) < 50:
            run = False
        if end_page and page_num >= end_page:
            run = False
    
    return all_tickers


def get_ticker_details(symbol):
    url = BASE_URL + f"/v1/meta/symbols/{symbol}/company?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_market_status():
    url = BASE_URL + f"/v1/marketstatus/now?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_stock_ticks(symbol: str, date: str, tick_type: str, timestamp_first=None, timestamp_limit=None, reverse=False, limit=50000):
    if tick_type == 'quotes':
        path = BASE_URL + f"/v2/ticks/stocks/nbbo/{symbol}/{date}"
    elif tick_type == 'trades':
        path = BASE_URL + f"/v2/ticks/stocks/trades/{symbol}/{date}"
    params = {}
    params['apiKey'] = API_KEY
    if timestamp_first is not None:
        params['timestamp'] = timestamp_first
    if timestamp_limit is not None:
        params['timestampLimit'] = timestamp_limit
    if reverse is not None:
        params['reverse'] = reverse
    if limit is not None:
        params['limit'] = limit
    response = get(path, params)
    return validate_response(response)
