# pip3 install websocket-client
# pip install git+https://github.com/tangentlabs/django-oscar-paypal.git@issue/34/oscar-0.6

import websocket

def on_message(ws, message):
    print(message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ws.send('{"action":"auth","params":"A2l7oJ6FxwU0nO0_m_mB3SZSOI3cBghopWfbV6"}')
    ws.send('{"action":"subscribe","params":"C.AUD/USD,C.USD/EUR,C.USD/JPY"}')

def run():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://socket.polygon.io/forex",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()

if __name__ == "__main__":
    run()