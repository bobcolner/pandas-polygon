# docker run -p 8888:8888 jupyter/datascience-notebook:latest
import os
import datetime
import pickle
import numpy as np
import pandas as pd
import polygon_rest_api as api
# import pudb; pu.db

symbol='GLD'
fmat = 'parquet'
date = '2019-05-09'
ts = api.read_matching_files(f"data/ticks/{fmat}/{symbol}/{symbol}_{date}.{fmat}", format=fmat)
bars = []
state = api.reset_state()
nrow = -1
while nrow < 999:
    nrow += 1
    epoch = ts['epoch'][nrow]
    price = ts['price'][nrow]
    volume = ts['volume'][nrow]
    bars, state = api.update_bar(epoch, price, volume, bars, s=state)

