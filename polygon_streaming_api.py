from time import time_ns, time
import json
import pandas as pd
from trio_websocket import open_websocket_url


async def stream_ticks(tickers="T.GLD, Q.GLD", output_fname='data.txt'):
    # connect to ws
    async with open_websocket_url('wss://alpaca.socket.polygon.io/stocks') as ws:
        msg1 = await ws.get_message()
        print(msg1)
        # authenticate
        await ws.send_message('{"action":"auth", "params":"AKW7AP27P3ZWEYEP02BZ"}')
        msg2 = await ws.get_message()
        print(msg2)
        # subscribe to symbols
        await ws.send_message(f'{{"action":"subscribe", "params":"{tickers}"}}')
        msg3 = await ws.get_message()
        print(msg3)
        # listen for events
        while True:
            with open(output_fname, 'a') as out_file:
                message = await ws.get_message()
                # out_file.write(message + '\n')
                ticks = json.loads(message)
                for tick in ticks:
                    if tick['ev'] not in ['T', 'Q']:
                        continue
                    tick['a'] = time_ns()
                    tick_str = json.dumps(tick)
                    out_file.write(tick_str + '\n')


def read_ticks(file_name):
    df = pd.read_json(file_name, lines=True, convert_dates=False)
    df = df.rename(columns={'ev': 'event',
                            'sym': 'symbol',
                            'c': 'condition',
                            'x': 'exchange_id',
                            'i': 'trade_id',
                            'z': 'tape',
                            'p': 'price',
                            's': 'size',
                            'bx': 'bid_exchange_id',
                            'ax': 'ask_exchange_id',
                            'bp': 'bid_price',
                            'ap': 'ask_price',
                            'bs': 'bid_size',
                            'as': 'ask_size',
                            't': 'tick_epoch',
                            'a': 'arrival_epoch'}
    )
    df = df.drop(columns=['tape'])
    # fix category types
    df['symbol'] = df['symbol'].astype('category')
    df['event'] = df['event'].astype('category')
    # parse datetimes
    df['tick_dt'] = pd.to_datetime(df.tick_epoch, utc=True, unit='ms') #.tz_convert('America/New_York')
    df['arrival_dt'] = pd.to_datetime(df.arrival_epoch, utc=True, unit='ns') #.tz_convert('America/New_York')
    # df['latency'] = df.arrival_dt - df.tick_dt
    return df


# def parse_streaming_data(file_name):
#     ticks = []
#     quotes = []
#     with open(file_name) as out_file:
#         for line in out_file:
#             line_data = json.loads(line)
#             for item in line_data:
#                 if item['ev'] == 'T':
#                     ticks.append(item)
#                 elif item['ev'] == 'Q':
#                     quotes.append(item)
#     return ticks, quotes


# def quotes_to_dataframe(quotes):
#     df = pd.DataFrame(quotes)
#     df = df.rename(columns={'ev': 'event',
#                             'sym': 'symbol',
#                             'c': 'quote_condition',
#                             'bx': 'bid_exchange_id',
#                             'ax': 'ask_exchange_id',
#                             'bp': 'bid_price',
#                             'ap': 'ask_price',
#                             'bs': 'bid_size',
#                             'as': 'ask_size',
#                             't': 'tick_epoch'}
#     )
#     # df = df.drop(columns=[])
#     df['datetime'] = pd.to_datetime(df.tick_epoch, utc=True, unit='ms') #.tz_convert('America/New_York')
#     return df


# def ticks_to_dataframe(ticks):
#     # df = pd.DataFrame(ticks)
#     df = df.rename(columns={'ev': 'event',
#                             'sym': 'symbol',
#                             'c': 'trade_condition',
#                             'x': 'exchange_id',
#                             'i': 'trade_id',
#                             'z': 'tape',
#                             'p': 'price',
#                             's': 'size',
#                             't': 'tick_epoch'}
#     )   
#     # df = df.drop(columns=[])
#     df['tick_dt'] = pd.to_datetime(df.tick_epoch, utc=True, unit='ms') #.tz_convert('America/New_York')
#     return df
