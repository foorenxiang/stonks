from ticker_symbols import get_ticker_symbols
from show_tickers import ShowTickers

if __name__ == "__main__":
    ShowTickers.show_tickers(get_ticker_symbols())
