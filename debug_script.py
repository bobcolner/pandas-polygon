# import pudb; pu.db

import api
import bars

symbol='GLD'
fmat = 'parquet'
date = '2019-05-09'

ts = api.read_matching_files(f"../data/ticks/{fmat}/{symbol}/{symbol}_{date}.{fmat}", format=fmat)

bars.build_bars(ts)
