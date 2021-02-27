import yfinance as yf
import streamlit as st
import pandas as pd

# st.write(
#     """
# # Stonks

# Stock closing prices & volume of various stonks

# """
# )

st.write("# Stonks")

tickerSymbol = "GME"

tickerData = yf.Ticker(tickerSymbol)

tickerDF = tickerData.history(period="1d", start="2020-1-1", end="2021-2-28")

st.line_chart(tickerDF.Close)
st.write("GME Closing Prices")

st.line_chart(tickerDF.Volume)
st.write("GME Volume")