import sys
from from_root import from_root

sys.path.append(str(from_root(".")))

from utils.ticker_symbols import get_ticker_symbols
from show_tickers import StreamlitShowTickers

if __name__ == "__main__":
    StreamlitShowTickers.show_tickers(get_ticker_symbols())
