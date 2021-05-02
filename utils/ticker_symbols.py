def get_ticker_symbols():
    ticker_symbols = list(
        {
            *{
                "AAPL",
                "^GSPC",
                "GOOG",
                # "PLTR",
                # "TIGR",
                # "TBLT",
                # "GME",
                # "SQ",
                # "WMT",
                "V",
                # "DIS",
                # "OCGN",
                # "NVAX",
                "MZH.SI",
                # "D01.SI",
                "^IXIC",
                "TSLA",
            },
            *{
                # "PLTR",
                # "TIGR",
                # "TBLT",
                # "GME",
                # "SQ",
                # "WMT",
                # "V",
                # "AAPL",
                # "DIS",
                # "OCGN",
                # "NVAX",
                # "MZH.SI",
                # "D01.SI",
            },
        }
    )
    ticker_symbols.sort()
    return ticker_symbols
