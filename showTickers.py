import yfinance as yf
import streamlit as st


def showTickers(tickerSymbols={}):
    if not tickerSymbols:
        st.write("### Please add ticker symbols to tickerSymbols.py to continue...")
    for tickerSymbol in tickerSymbols:
        tickerData = yf.Ticker(tickerSymbol)

        tickerDF = tickerData.history(period="1d", start="2020-1-1", end="2021-2-28")

        if not tickerDF.Close.empty | tickerDF.Volume.empty:
            st.write(f"## {tickerSymbol}")
            st.line_chart(tickerDF.Close)
            st.write(f"{tickerSymbol} Closing Prices")

            st.line_chart(tickerDF.Volume)
            st.write(f"{tickerSymbol} Volume\n\n\n")

        else:
            st.write(f"{tickerSymbol} is not a valid symbol on Yahoo Finance!!")