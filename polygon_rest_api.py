from os import environ
from numpy import isin
from requests import get


BASE_URL = 'https://api.polygon.io'

if 'POLYGON_API_KEY' in environ:
    API_KEY = environ['POLYGON_API_KEY']
else:
    raise ValueError('missing poloyon api key')


def validate_response(response):
    if response.status_code == 200:
        return response.json()['results']
    else:
        response.raise_for_status()


def get_market_date(date:str, locale='us', market='stocks'):
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


def get_ticker_details(symbol:str):
    url = BASE_URL + f"/v1/meta/symbols/{symbol}/company?apiKey={API_KEY}"
    response = get(url)
    if response.status_code == 200:
        return response.json()


def get_market_status():
    url = BASE_URL + f"/v1/marketstatus/now?apiKey={API_KEY}"
    response = get(url)
    return validate_response(response)


def get_stock_ticks_batch(symbol: str, date: str, tick_type: str, timestamp_first=None, 
    timestamp_limit=None, reverse=False, limit=50000) -> list:

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


def add_condition_groups(ticks: list) -> list:
    green_conditions = [0, 1, 3, 4, 8, 9, 11, 14, 23, 25, 27, 28, 30, 36, 41]
    irregular_conditions = [2, 5, 7, 10, 13, 15, 16, 20, 21, 22, 29, 33, 38, 52, 53]
    blank_conditions = [6, 17, 18, 19, 24, 26, 32, 35, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57]
    for idx, tick in enumerate(ticks):
        if 'c' in tick:
            ticks[idx]['green'] = any(isin(tick['c'], green_conditions))
            ticks[idx]['irregular'] = any(isin(tick['c'], irregular_conditions))
            ticks[idx]['afterhours'] = any(isin(tick['c'], 12))
            ticks[idx]['odd_lot'] = any(isin(tick['c'], 37))
            ticks[idx]['blank'] = any(isin(tick['c'], blank_conditions))
        else:
            ticks[idx]['green'] = False
            ticks[idx]['irregular'] = False
            ticks[idx]['afterhours'] = False
            ticks[idx]['odd_lot'] = False
            ticks[idx]['blank'] = True
    return ticks


def get_stocks_ticks_date(symbol: str, date: str, tick_type: str) -> list:
    last_tick = None
    limit = 50000
    ticks = []
    batch = 0
    run = True
    while run == True:
        # get batch of ticks
        batch += 1
        print('Batch#:', batch, symbol, date)
        ticks_batch = get_stock_ticks_batch(symbol, date, tick_type, timestamp_first=last_tick, limit=limit)
        if len(ticks_batch) < 1: # empty tick batch
            print('Empty Batch!', batch, symbol, date)
            run = False
            continue
        # filter ticks
        ticks_batch = add_condition_groups(ticks_batch)
        # update last_tick
        last_tick = ticks_batch[-1]['t'] # sip ts
        # logging
        print('Downloaded:', len(ticks_batch), 'ticks', symbol, date)
        # append batch to ticks list
        ticks = ticks + ticks_batch
        # check if we are done pulling ticks
        if len(ticks_batch) < limit:
            run = False
        elif len(ticks_batch) == limit:
            del ticks[-1] # drop last row to avoid dups

    return ticks
