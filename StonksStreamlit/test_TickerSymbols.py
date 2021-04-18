from StonksStreamlit.TickerSymbols import getTickerSymbols


def test_get_symbols_type():
    assert type(getTickerSymbols()) == list


def test_get_symbols_are_unique():
    tickerSymbols = getTickerSymbols()
    testSymbolsUnique = list({*tickerSymbols})
    testSymbolsUnique.sort()
    assert testSymbolsUnique == tickerSymbols
