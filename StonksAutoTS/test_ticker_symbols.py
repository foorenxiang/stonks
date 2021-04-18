from StonksStreamlit.ticker_symbols import get_ticker_symbols


def test_get_symbols_type():
    assert type(get_ticker_symbols()) == list


def test_get_symbols_are_unique():
    tickerSymbols = get_ticker_symbols()
    testSymbolsUnique = list({*tickerSymbols})
    testSymbolsUnique.sort()
    assert testSymbolsUnique == tickerSymbols
