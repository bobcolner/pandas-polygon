from os.path import join
from glob import glob
from pathlib import Path
from polygon_backfil import *

def get_trades_dates(symbol, start_date, end_date):

    def validate_trades(trades_df):
        if len(trades_df) < 1:
            raise ValueError('0 length trades df')
        elif any(trades_df.count() == 0):
            raise ValueError('trades df missing fields. Recent historic data may not be ready for consumption')
        else:
            return(trades_df)

    req_dates = get_market_dates(start_date=start_date, end_date=end_date)
    
    existing_dates = get_file_dates(f"/Users/bobcolner/QuantClarity/data/ticks/parquet/{symbol}")

    dates = get_remaining_dates(req_dates=req_dates, existing_dates=existing_dates)

    for date in dates:
        
        trades = get_trades_date(symbol=symbol, date=date)
        ticks_df = trades_to_df(trades)
        ticks_df = validate_trades(trades_df=ticks_df)
        
        Path(f"/Users/bobcolner/QuantClarity/data/ticks/csv/{symbol}").mkdir(parents=True, exist_ok=True)
        ticks_df.to_csv(f"/Users/bobcolner/QuantClarity/data/ticks/csv/{symbol}/{symbol}_{date}.csv", index=False)
        Path(f"/Users/bobcolner/QuantClarity/data/ticks/parquet/{symbol}").mkdir(parents=True, exist_ok=True)
        ticks_df.to_parquet(f"/Users/bobcolner/QuantClarity/data/ticks/parquet/{symbol}/{symbol}_{date}.parquet", index=False, engine='fastparquet')
        Path("/Users/bobcolner/QuantClarity/data/ticks/feather").mkdir(parents=True, exist_ok=True)
        ticks_df.to_feather(f"/Users/bobcolner/QuantClarity/data/ticks/feather/{symbol}/{symbol}_{date}.feather", index=False)


def get_market_candels_dates(start_date, end_date, path):

    req_dates = get_market_dates(start_date=start_date, end_date=end_date)
    
    existing_dates = get_file_dates(str(path))
    
    existing_dates = get_file_dates(f"/Users/bobcolner/QuantClarity/data/daily_bars/{save}")

    dates = get_remaining_dates(req_dates=req_dates, existing_dates=existing_dates)
    set_df = pd.DataFrame()

    for date in dates: 
        print('downloading: ', date)
        daily_df = get_market_candels_date(date=date)

        # (path_obj / 'daily_bars/csv').mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(f"/Users/bobcolner/QuantClarity/data/daily_bars/csv/market_{date}.csv", index=False)
        Path("/Users/bobcolner/QuantClarity/data/daily_bars/parquet").mkdir(parents=True, exist_ok=True)
        daily_df.to_parquet(f"/Users/bobcolner/QuantClarity/data/daily_bars/parquet/market_{date}.parquet", index=False, engine='fastparquet')
        Path("/Users/bobcolner/QuantClarity/data/daily_bars/feather").mkdir(parents=True, exist_ok=True)
        daily_df.to_feather(f"/Users/bobcolner/QuantClarity/data/daily_bars/feather/market_{date}.feather")


def read_matching_files(glob_string, format='csv'):
    if format == 'csv':
        return pd.concat(map(pd.read_csv, glob(join('', glob_string))), ignore_index=True)
    elif format == 'parquet':
        return pd.concat(map(pd.read_parquet, glob(join('', glob_string))), ignore_index=True)
    elif format == 'feather':
        return pd.concat(map(pd.read_feather, glob(join('', glob_string))), ignore_index=True)
