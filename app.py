import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# RSI
def calculate_rsi(series, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()
    # Separate positive and negative gains/losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate rolling means of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

#MACD
def calculate_macd(series, span_short=12, span_long=26, span_signal=9):
    """
    Calculate the MACD (Moving Average Convergence Divergence) and Signal line.
    Returns MACD, Signal, and the MACD Histogram.
    """
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

#Streamlit App
st.set_page_config(page_title="Stock Data Analysis", layout="wide")

st.title("Stock Data Analysis App")

# Create centered searchbar and filter options using columns
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    ticker_input = st.text_input("Enter Stock Ticker", value="OKLO", key="ticker")
    
# Additional filter options (for example, to choose data period and interval)
st.markdown("### Data Filter Options")
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    period = st.selectbox(
        "Select Data Period", 
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=5,
        key="period"
    )
with filter_col2:
    interval = st.selectbox(
        "Select Interval", 
        options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        index=8,
        key="interval"
    )

# A filter button to trigger data retrieval and processing
if st.button("Apply Filter"):
    # Retrieve the ticker data from yfinance
    ticker_obj = yf.Ticker(ticker_input)
    history_data = ticker_obj.history(period=period, interval=interval)
    
    if history_data.empty:
        st.error(f"No data found for ticker: {ticker_input}")
    else:
        st.subheader(f"Historical Data for {ticker_input}")
        st.write(history_data.head())
        
        # Calculate RSI and add as a new column
        history_data['RSI'] = calculate_rsi(history_data['Close'])
        
        # Calculate MACD, Signal, and Histogram and add as new columns
        macd, signal_line, histogram = calculate_macd(history_data['Close'])
        history_data['MACD'] = macd
        history_data['Signal'] = signal_line
        history_data['Histogram'] = histogram
        
        # ---------------------------
        # Plotting the RSI and MACD indicators
        # ---------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot RSI
        ax1.plot(history_data.index, history_data['RSI'], label='RSI', color='blue')
        ax1.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax1.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        ax1.set_ylabel("RSI")
        ax1.set_title("Relative Strength Index (RSI)")
        ax1.legend()
        
        # Plot MACD and Signal Line, with Histogram as bar chart
        ax2.plot(history_data.index, history_data['MACD'], label='MACD', color='purple')
        ax2.plot(history_data.index, history_data['Signal'], label='Signal Line', color='orange')
        ax2.bar(history_data.index, history_data['Histogram'], label='Histogram', color='gray', alpha=0.3)
        ax2.set_ylabel("MACD")
        ax2.set_title("MACD")
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)