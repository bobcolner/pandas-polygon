from os import environ
from time import time_ns
import websocket as ws


if 'POLYGON_API_KEY' in environ:
    API_KEY = environ['POLYGON_API_KEY']
else:
    raise ValueError('missing poloyon api key')


def on_message(wsc: ws._app.WebSocketApp, message: str):
    print(message)
    with open('data.txt', 'a') as out_file:
        out_file.write(str(time_ns()) + ' || ' + message + '\n')


def on_error(wsc: ws._app.WebSocketApp, error: str):
    print(error)


def on_close(wsc: ws._app.WebSocketApp):
    print("### closed ###")


def on_open(wsc: ws._app.WebSocketApp, symbols: str=None):
    wsc.send(data='{"action":"auth", "params":"' + API_KEY + '"}')
    wsc.send(data='{"action":"subscribe","params":"T.SPY, T.GLD"}')
    # wsc.send(data='{"action":"subscribe","params":"' + symbols + '"}')


def run(symbols: str=None):
	wsc = ws.WebSocketApp(
		url="wss://alpaca.socket.polygon.io/stocks",
		on_message = on_message,
	    on_error = on_error,
	    on_close = on_close
	    )
	wsc.on_open = on_open
	wsc.run_forever()
