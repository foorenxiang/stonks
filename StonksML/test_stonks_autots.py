from .stonks_autots import get_stocks


def test_get_stocks():
    stocks = get_stocks()
    assert type(stocks) == list
    assert not not stocks
