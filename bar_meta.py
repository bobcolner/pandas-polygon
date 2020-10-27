import pandas as pd
import ray
from polygon_s3 import load_ticks
from bar_samples import build_bars
from filters import jma_filter_df

@ray.remote
def build_bars_ray(result_path: str, symbol: str, date: str, thresh: dict) -> dict:
    # get ticks for current date
    ticks_df = load_ticks(result_path, symbol, date, 'trades')
    # sample bars
    bars, state = build_bars(ticks_df, thresh)
    return {'date': date, 'bars': bars}

    

def build_bars_dates_ray(daily_stats_df: pd.DataFrame, thresh: dict, result_path: str, symbol: str) -> list:	
	ray.init(ignore_reinit_error=True)
	futures = []
	for row in daily_stats_df.itertuples():
	 
	    thresh.update({'renko_size': row.range_jma_lag / 15})
	    
	    if 'tick_imbalance_thresh_jma_lag' in daily_stats_df.columns:
	    	thresh.update({'tick_imbalance': row.tick_imbalance_thresh_jma_lag})

	    bars = build_bars_ray.remote(
	        result_path=result_path,
	        symbol=symbol, 
	        date=row.date,
	        thresh=thresh
	    )
	    futures.append(bars)
	   
	bar_dates = ray.get(futures)
	ray.shutdown()
	return bar_dates 


def process_bar_dates(bar_dates: list) -> pd.DataFrame:
	results = []
	for date_d in bar_dates:
	    
	    imbalances = []    
	    for bar in date_d['bars']:
	        imbalances.append(bar['tick_imbalance'])
	   
	    imbal_thresh = pd.Series(imbalances).quantile(q=.95)
	    results.append({'date': date_d['date'], 'bar_count': len(date_d['bars']), 'tick_imbalance_thresh': imbal_thresh})

	daily_bar_stats_df = jma_filter_df(pd.DataFrame(results), 'tick_imbalance_thresh', length=5, power=1)

	daily_bar_stats_df.loc[:,'tick_imbalance_thresh_jma_lag'] = daily_bar_stats_df['tick_imbalance_thresh_jma'].shift(1)

	daily_bar_stats_df = daily_bar_stats_df.dropna()

	return daily_bar_stats_df
