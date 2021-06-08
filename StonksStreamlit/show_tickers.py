import yfinance as yf
import streamlit as st

import sys
from from_root import from_root

sys.path.append(str(from_root(".")))
from utils import paths_catalog
from StonksML.timeseries.generate_streamlit_results import generate_streamlit_results


class StreamlitShowTickers:
    tickerSymbols = set()

    @classmethod
    def _ml_predictions(cls):
        # stocks_predictions = generate_streamlit_results()
        stocks_predictions = None
        if not stocks_predictions:
            st.write(
                "##### Please wait while we train/load the ML results, check back again later"
            )

        return stocks_predictions

    @classmethod
    def show_tickers(cls, tickerSymbols={}):
        if not tickerSymbols:
            st.write("### Please add ticker symbols to tickerSymbols.py to continue...")
            return

        stocks_predictions = cls._ml_predictions()

        cls.tickerSymbols = set(tickerSymbols)

        cls._display_tickers(stocks_predictions)

    @staticmethod
    def _retrieve_ticker_data(tickerSymbol):
        tickerData = yf.Ticker(tickerSymbol)

        tickerDF = tickerData.history(period="1y")
        return tickerDF

    @classmethod
    def _display_tickers(cls, stocks_predictions):
        for tickerSymbol in cls.tickerSymbols:
            tickerDF = cls._retrieve_ticker_data(tickerSymbol)

            if not tickerDF.Close.empty | tickerDF.Volume.empty:
                try:
                    st.write(f"## {tickerSymbol}")
                    st.line_chart(tickerDF.Close)
                    st.write(f"{tickerSymbol} Closing Prices")

                    st.line_chart(tickerDF.Volume)
                    st.write(f"{tickerSymbol} Volume\n\n\n")
                except TypeError:
                    st.write(
                        f"Error encountered with writing full results for {tickerSymbol}"
                    )

            else:
                st.write(f"{tickerSymbol} is not a valid symbol on Yahoo Finance!!")

            for prediction in stocks_predictions:
                if prediction["name"] == tickerSymbol:
                    st.write(f"## Forecast for {tickerSymbol}")
                    for key in prediction.keys():
                        if key != "name":
                            st.line_chart(prediction[key])
                    break
