import os
import glob
import datetime
import pathlib
import requests
import scipy.stats as stats
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


BASE_URL = 'https://api.polygon.io'

if 'polygon_api_key' in os.environ:
    API_KEY = os.environ['polygon_api_key']
else:
    raise ValueError('missing poloyon api key')


def validate_response(response):
    if response.status_code == 200:
        return response.json()['results']
    else:
        response.raise_for_status()


def get_grouped_daily(locale='us', market='stocks', date='2020-01-02'):
    url = BASE_URL + f"/v2/aggs/grouped/locale/{locale}/market/{market}/{date}?apiKey={API_KEY}"
    response = requests.get(url)
    return validate_response(response)


def get_market_candels_date(date='2020-01-02'):
    daily = get_grouped_daily(locale='us', market='stocks', date=date)
    daily_df = pd.DataFrame(daily)
    daily_df = daily_df.rename(columns={'T': 'symbol',
                                        'v': 'volume',
                                        'o': 'open',
                                        'c': 'close',
                                        'h': 'high',
                                        'l': 'low',
                                        't': 'epoch'})
    daily_df['date_time'] = pd.to_datetime(daily_df.epoch, utc=True, unit='ms') #.tz_convert('America/New_York')
    return daily_df


def get_market_candels_dates(start_date, end_date, save='both'):

    req_dates = get_market_dates(start_date=start_date, end_date=end_date)
    
    if save == 'both':
        existing_dates = get_file_dates(f"data/daily_bars/parquet")
    else:
        existing_dates = get_file_dates(f"data/daily_bars/{save}")

    dates = get_remaining_dates(req_dates=req_dates, existing_dates=existing_dates)
    set_df = pd.DataFrame()

    for date in dates: 
        print('downloading: ', date)
        daily_df = get_market_candels_date(date=date)
        
        if save in ['csv', 'both']:
            pathlib.Path(f"/Users/bobcolner/QuantClarity/data/daily_bars/csv").mkdir(parents=True, exist_ok=True)
            daily_df.to_csv(f"data/daily_bars/csv/market_{date}.csv", index=False)

        if save in ['parquet', 'both']:
            pathlib.Path(f"/Users/bobcolner/QuantClarity/data/daily_bars/parquet").mkdir(parents=True, exist_ok=True)
            daily_df.to_parquet(f"data/daily_bars/parquet/market_{date}.parquet", index=False, engine='fastparquet')
            pathlib.Path(f"/Users/bobcolner/QuantClarity/data/daily_bars/feather").mkdir(parents=True, exist_ok=True)
            daily_df.to_feather(f"data/daily_bars/feather/market_{date}.feather")
    


def get_markets():
    url = BASE_URL + f"/v2/reference/markets?apiKey={API_KEY}"
    response = requests.get(url)
    return validate_response(response)


def get_locales():
    url = BASE_URL + f"/v2/reference/locales?apiKey={API_KEY}"
    response = requests.get(url)
    return validate_response(response)


def get_types():
    url = BASE_URL + f"/v2/reference/types?apiKey={API_KEY}"
    response = requests.get(url)
    return validate_response(response)


def get_all_tickers(start_page=1, end_page=None, stock_type='etp', return_type='df'):
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
            
    if return_type == 'df':
        all_tickers_df = pd.DataFrame(all_tickers)
        all_tickers_df = all_tickers_df.drop(columns=['codes', 'url', 'updated'])
        return all_tickers_df
    else:
        return all_tickers


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
    response = requests.get(path, params)
    if response.status_code != 200:
        response.raise_for_status()
    tickers_list = response.json()['tickers']
    return tickers_list


def get_ticker_details(symbol):
    url = BASE_URL + f"/v1/meta/symbols/{symbol}/company?apiKey={API_KEY}"
    response = requests.get(url)
    return validate_response(response)


def get_market_status():
    url = BASE_URL + f"/v1/marketstatus/now?apiKey={API_KEY}"
    response = requests.get(url)
    return validate_response(response)


def get_market_dates(start_date, end_date):
    market = mcal.get_calendar('NYSE')
    schedule = market.schedule(start_date=start_date, end_date=end_date)
    dates = [i.date().isoformat() for i in schedule.index]
    return dates


def get_file_dates(dates_path):
    if os.path.exists(dates_path):
        file_list = os.listdir(dates_path)
        existing_dates = [i.split('_')[1].split('.')[0] for i in file_list]
        existing_dates.sort()
        return existing_dates
    else:
        return []


def get_remaining_dates(req_dates, existing_dates):
    existing_dates_set = set(existing_dates)
    remaining_dates = [x for x in req_dates if x not in existing_dates_set]
    next_dates = [i for i in remaining_dates if i <= datetime.date.today().isoformat()]
    print('pull new dates: ', next_dates)
    return next_dates


def get_stock_trades(symbol: str, date: str, timestamp_first=None, timestamp_limit=None, reverse=False, limit=50000):
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
    response = requests.get(path, params)
    return validate_response(response)


def trades_to_df(trades):
    df = pd.DataFrame(trades, columns=['t', 'y', 'q', 'i', 'x', 'p', 's'])
    df = df.rename(columns={'t': 'epoch',
                            'q': 'sequence',
                            'p': 'price',
                            's': 'volume',
                            'y': 'exchange_epoch',
                            'x': 'exchange_id',
                            'i': 'trade_id'
                            })
    df = df.drop(columns=['exchange_id', 'exchange_epoch', 'trade_id'])
    return df


def get_trades_date(symbol: str, date: str):
    last_trade = None
    limit = 50000
    trade_set = pd.DataFrame()
    run = True
    while run == True:
        trades = get_stock_trades(symbol, date, timestamp_first=last_trade, limit=limit)
        trade_batch = trades_to_df(trades)
        print('Trades count: ', len(trades), '; Time (NYC): ', pd.to_datetime(trade_batch.epoch.iloc[1], utc=True, unit='ns').tz_convert('America/New_York'))
        trade_set = trade_set.append(trade_batch, ignore_index=True, verify_integrity=False)
        last_trade = trade_batch.epoch.iloc[-1]    
        if len(trades) < limit:
            run = False
        else:
            trade_set = trade_set.drop(trade_set.tail(1).index) # drop last row to avoid dups
    trade_set = trade_set.sort_values(['epoch', 'sequence'])
    return trade_set


def get_trades_dates(symbol, start_date, end_date, save='both'):

    def validate_trades(trades_df):
        if len(trades_df) < 1:
            raise ValueError('0 length trades df')
        elif any(trades_df.count() == 0):
            raise ValueError('trades df missing fields. Recent historic data may not be ready for consumption')
        else:
            return(trades_df)

    req_dates = get_market_dates(start_date=start_date, end_date=end_date)
    
    if save == 'both':
        existing_dates = get_file_dates(f"data/ticks/parquet/{symbol}")
    else:
        existing_dates = get_file_dates(f"data/ticks/{save}/{symbol}")

    dates = get_remaining_dates(req_dates=req_dates, existing_dates=existing_dates)

    for date in dates:
        
        ticks_df = get_trades_date(symbol=symbol, date=date)
        ticks_df = validate_trades(trades_df=ticks_df)

        if save in ['csv', 'both']:
            pathlib.Path(f"/Users/bobcolner/QuantClarity/data/ticks/csv/{symbol}").mkdir(parents=True, exist_ok=True)
            ticks_df.to_csv(f"data/ticks/csv/{symbol}/{symbol}_{date}.csv", index=False)

        if save in ['parquet', 'both']:
            pathlib.Path(f"/Users/bobcolner/QuantClarity/data/ticks/parquet/{symbol}").mkdir(parents=True, exist_ok=True)
            ticks_df.to_parquet(f"data/ticks/parquet/{symbol}/{symbol}_{date}.parquet", index=False, engine='fastparquet')
