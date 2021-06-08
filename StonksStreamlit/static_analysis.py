import sys
from from_root import from_root
from matplotlib import ticker
import matplotlib.pyplot as plt
import yfinance as yf


sys.path.append(str(from_root(".")))
from utils.ticker_symbols import get_ticker_symbols
from utils.dict_print import dict_print


def retrieve_ticker_data(tickerSymbol):
    tickerData = yf.Ticker(tickerSymbol)
    tickerDF = tickerData.history(period="1y")
    return tickerDF


ticker_dfs = [{ticker: retrieve_ticker_data(ticker)} for ticker in get_ticker_symbols()]

for dict_ in ticker_dfs:
    [(ticker, ticker_data)] = dict_.items()
    # print(ticker_data.columns)
    series = ticker_data["Open"]

    series.plot()
    plt.show()
