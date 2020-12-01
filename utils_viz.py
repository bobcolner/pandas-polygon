import finplot as fplt
import pandas as pd


def plot_daily_symbols(df: pd.DataFrame, symbols: list, metric: str='close') -> pd.DataFrame:
    fdf = df[['symbol', metric]][df.symbol.isin(symbols)]
    pdf = fdf.pivot(columns='symbol', values=metric)
    pdf.plot_bokeh(kind='line', sizing_mode="scale_height", rangetool=True, title=str(symbols), ylabel=metric+' [$]', number_format="1.00 $")
    return pdf


bars_df = pd.read_feather('bars.feather')
bars_df = bars_df.set_index('close_at')

fplt.candlestick_ochl(bars_df[['price_open','price_close','price_high','price_low']])
fplt.show()



def plot_bars(bars:pd.DataFrame):
    ax1, ax2 = fplt.create_plot('Prices+Vol', rows=2)
    fplt.candlestick_ochl(bars[['price_open','price_close','price_high','price_low']], ax=ax1)
    fplt.volume_ocv(bars[['price_open','price_close','volume_sum']], ax=ax2)
    fplt.show()


# candle stick example
from math import pi
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.stocks import MSFT

df = pd.DataFrame(MSFT)[:50]
df["date"] = pd.to_datetime(df["date"])

inc = df.close > df.open
dec = df.open > df.close
w = 12*60*60*1000 # half day in ms

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1000, title = "MSFT Candlestick")
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.3

p.segment(df.date, df.high, df.date, df.low, color="black")
p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")

output_file("candlestick.html", title="candlestick.py example")

show(p)  # open a browser