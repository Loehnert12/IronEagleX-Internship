import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App
Shown are the stock **closing price** and ***volume*** of Apple!
""")

tickerSymbol = 'AAPL'

tickerData = yf.Ticker(tickerSymbol)

tickerDF = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

st.write("""
 ## Closing Price
 """)
st.line_chart(tickerDF.Close)
st.write("""
 ## Volume Price
 """)
st.line_chart(tickerDF.Volume)