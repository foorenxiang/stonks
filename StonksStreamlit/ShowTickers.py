import yfinance as yf
import streamlit as st


class ShowTickers:
    tickerSymbols = set()

    @classmethod
    def showTickers(cls, tickerSymbols={}):
        if not tickerSymbols:
            st.write("### Please add ticker symbols to tickerSymbols.py to continue...")
        cls.tickerSymbols = tickerSymbols
        for tickerSymbol in cls.tickerSymbols:
            tickerData = yf.Ticker(tickerSymbol)

            tickerDF = tickerData.history(period="1y")

            if not tickerDF.Close.empty | tickerDF.Volume.empty:
                st.write(f"## {tickerSymbol}")
                st.line_chart(tickerDF.Close)
                st.write(f"{tickerSymbol} Closing Prices")

                st.line_chart(tickerDF.Volume)
                st.write(f"{tickerSymbol} Volume\n\n\n")

            else:
                st.write(f"{tickerSymbol} is not a valid symbol on Yahoo Finance!!")
