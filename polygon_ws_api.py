from os import environ
from time import time_ns
from json import loads
import pandas as pd
from trio import run
from trio_websocket import open_websocket_url
from utils_globals import POLYGON_API_KEY


def run_stream_ticks(tickers: str='T.SPY, Q.SPY', file_name: str='data.txt'):
    run(stream_ticks, tickers, file_name)


# https://www.willmcgugan.com/blog/tech/post/speeding-up-websockets-60x/
async def stream_ticks(tickers: str, output_fname: str, print_msg: bool=False):
    # connect to ws
    async with open_websocket_url('wss://alpaca.socket.polygon.io/stocks') as ws:
        # authenticate
        await ws.send_message('{"action":"auth", "params":"' + POLYGON_API_KEY + '"}')
        # subscribe to symbols
        await ws.send_message(f'{{"action": "subscribe", "params": "{tickers}"}}')
        # listen for events
        while True:
            with open(output_fname, 'a') as out_file:
                message = await ws.get_message()
                if print_msg:
                    print(message)
                out_file.write(str(time_ns()) + '||' + message + '\n')
                # ticks = json.loads(message)
                # for tick in ticks:
                #     if tick['ev'] not in ['T', 'Q']:
                #         continue
                #     tick['a'] = time_ns()
                #     out_file.write(json.dumps(tick) + '\n')


def trades_to_df(trades: list) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    df = df.rename(columns={'ev': 'event',
                            'sym': 'symbol',
                            'c': 'condition',
                            'x': 'exchange_id',
                            'i': 'trade_id',
                            'z': 'tape',
                            'p': 'price',
                            's': 'size',
                            't': 'trade_epoch'}
    )
    # add datetimes
    df['trade_dt'] = pd.to_datetime(df['trade_epoch'], unit='ms') #.tz_convert('America/New_York')
    df.loc[:, 'arrive_dt'] = pd.to_datetime(df['arrived_epoch'])
    df.loc[:, 'latency'] = df['arrive_dt'] - df['trade_dt']
    # convert types
    df['trade_id'] = df['trade_id'].astype('string')
    df['symbol'] = df['symbol'].astype('string')
    df['price'] = df['price'].astype('float32')
    df['size'].fillna(value=0, inplace=True)
    df['size'] = df['size'].astype('uint32')
    df['exchange_id'] = df['exchange_id'].astype('uint8')
    # drop columns
    df = df.drop(columns=['arrived_epoch', 'trade_epoch', 'event', 'tape'])
    return df


def process_ws_file(file_name: str='data.txt') -> pd.DataFrame:
    with open(file=file_name, mode='r') as fio:
        trades = []
        for line in fio:
            line_items = line.split(sep='||')
            arrived_epoch = int(line_items[0])
            ticks = loads(line_items[1])
            for tick in ticks:
                if tick['ev'] != 'T':
                    continue
                tick.update({'arrived_epoch': arrived_epoch})
                trades.append(tick)

    return trades_to_df(trades)


if __name__ == '__main__':
    from datetime import datetime
    start_at = datetime.today().isoformat()
    run_stream_ticks(tickers='T.SPY, T.GLD', file_name=f'data_{start_at}.txt')
