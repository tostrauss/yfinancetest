import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import time
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

###############################################
# SECTION 1: TECHNICAL INDICATOR CALCULATIONS
###############################################

def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI) of a price series.
    Uses a rolling average of gains and losses.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, span_short=12, span_long=26, span_signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) indicator.
    Returns the MACD line, Signal line, and Histogram.
    """
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculate Bollinger Bands for given series.
    Returns the middle band (SMA), upper band, and lower band.
    """
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def calculate_sma(series, window):
    """
    Calculate Simple Moving Average (SMA) of a series.
    """
    return series.rolling(window=window).mean()

###############################################
# SECTION 2: DATA FETCHING & PROCESSING FOR STOCKS
###############################################

def fetch_stock_data(ticker, period="1d", interval="1m"):
    """
    Fetch historical stock data using yfinance.
    Designed for real‑time data (e.g., period="1d", interval="1m")
    but supports multiple periods and intervals.
    
    Returns:
      A DataFrame with added columns for RSI, MACD, Bollinger Bands,
      SMA20, SMA50, SMA200, and daily high/low.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        raise

    try:
        data["RSI"] = calculate_rsi(data["Close"])
        macd, signal, _ = calculate_macd(data["Close"])
        data["MACD"] = macd
        data["Signal"] = signal
        data["SMA20"] = calculate_sma(data["Close"], 20)
        data["SMA50"] = calculate_sma(data["Close"], 50)
        data["SMA200"] = calculate_sma(data["Close"], 200)
        bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(data["Close"])
        data["BB_Mid"] = bb_mid
        data["BB_Upper"] = bb_upper
        data["BB_Lower"] = bb_lower
        data["Day_High"] = data["High"].cummax()
        data["Day_Low"] = data["Low"].cummin()
    except Exception as e:
        st.error(f"Error processing data for {ticker}: {e}")
        raise
    return data

###############################################
# SECTION 3: BLACK-SCHOLES & GREEKS CALCULATIONS
###############################################

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Compute Black-Scholes Greeks for a European option.
    
    Parameters:
      S : float - Underlying asset price.
      K : float - Strike price.
      T : float - Time to expiration in years.
      r : float - Risk-free interest rate (annualized, decimal).
      sigma : float - Implied volatility (annualized, decimal).
      option_type : str - 'call' or 'put'.
    
    Returns:
      delta, gamma, theta (per day), vega, rho, bs_price
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (np.nan,)*6
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
    except Exception as e:
        st.error(f"Error computing d1, d2: {e}")
        return (np.nan,)*6

    try:
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
            bs_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
        else:
            delta = -norm.cdf(-d1)
            theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
            bs_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0

        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0
    except Exception as e:
        st.error(f"Error computing Greeks: {e}")
        return (np.nan,)*6

    return delta, gamma, theta, vega, rho, bs_price

def add_greeks(options_df, S, T, r=0.01, option_type='call'):
    """
    Add Black-Scholes Greeks columns to an options DataFrame.
    The DataFrame must have columns "strike" and "impliedVolatility".
    
    Parameters:
      options_df : DataFrame with options chain data.
      S : float - Current underlying stock price.
      T : float - Time to expiration in years.
      r : float - Risk-free interest rate.
      option_type : 'call' or 'put'.
    
    Returns:
      The DataFrame with additional columns: Delta, Gamma, Theta, Vega, Rho, BS_Price.
    """
    def compute_row(row):
        if pd.notna(row.get("impliedVolatility", np.nan)):
            return pd.Series(
                black_scholes_greeks(S, row["strike"], T, r, row["impliedVolatility"], option_type)
            )
        else:
            return pd.Series([np.nan] * 6)
    
    greeks = options_df.apply(compute_row, axis=1)
    greeks.columns = ["Delta", "Gamma", "Theta", "Vega", "Rho", "BS_Price"]
    return pd.concat([options_df, greeks], axis=1)

###############################################
# SECTION 4: OPTIONS CHAIN DATA FETCHING
###############################################

def get_option_chain(ticker, expiration=None):
    """
    Retrieve the options chain for the given ticker.
    
    Parameters:
      ticker : str - Stock symbol.
      expiration : str or None - Expiration date in YYYY-MM-DD. If None, the app returns the list.
    
    Returns:
      calls, puts DataFrames and the expiration date (str) if successful;
      otherwise, an error message in the third return.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if not expirations:
            return None, None, "No options data available."
        # If an expiration date is provided and valid, use it; otherwise, return the list.
        if expiration is None or expiration not in expirations:
            # Here we choose the first available expiration if the user did not choose one.
            expiration = expirations[0]
        chain = ticker_obj.option_chain(expiration)
        return chain.calls, chain.puts, expiration
    except Exception as e:
        return None, None, f"Error retrieving options chain: {e}"

###############################################
# SECTION 5: ENHANCED EMAIL NOTIFICATIONS
###############################################

def send_email_notification(to_email, subject, body):
    """
    Send an email using SMTP.
    Update SMTP_SERVER, SMTP_PORT, SMTP_USER, and SMTP_PASSWORD with your actual settings.
    """
    SMTP_SERVER = ""  
    SMTP_PORT = 1
    SMTP_USER = ""
    SMTP_PASSWORD = ""
    FROM_EMAIL = SMTP_USER

    msg = MIMEMultipart()
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def enhanced_notification(ticker, email, period="1d", interval="1m"):
    """
    Check the latest stock data for the ticker and send an email if RSI is critical.
    The email includes current price, volume, SMAs, and RSI.
    
    Critical thresholds: RSI < 35 (Oversold) or RSI > 65 (Overbought).
    """
    try:
        data = fetch_stock_data(ticker, period, interval)
        latest = data.iloc[-1]
        current_rsi = latest["RSI"]
        current_price = latest["Close"]
        current_volume = int(latest["Volume"])
        sma20 = latest["SMA20"]
        sma50 = latest["SMA50"]
        sma200 = latest["SMA200"]
        
        alert = False
        message = (f"Ticker: {ticker}\n"
                   f"Current Price: ${current_price:.2f}\n"
                   f"Volume: {current_volume}\n"
                   f"SMA20: {sma20:.2f}, SMA50: {sma50:.2f}, SMA200: {sma200:.2f}\n"
                   f"RSI: {current_rsi:.2f}\n\n")
        
        if current_rsi < 35:
            message += "Alert: RSI is below 35 (Oversold)!\n"
            alert = True
        elif current_rsi > 65:
            message += "Alert: RSI is above 65 (Overbought)!\n"
            alert = True
        
        if alert:
            subject = f"Stock Alert for {ticker} – RSI Alert"
            if send_email_notification(email, subject, message):
                st.success("Notification sent successfully!")
            else:
                st.error("Failed to send notification.")
        else:
            st.info("RSI is within normal range. No notification sent.")
    except Exception as e:
        st.error(f"Error during notification: {e}")

#########################################################
# SECTION 6: STREAMLIT APP LAYOUT & MULTI-PAGE NAVIGATION
#########################################################

st.set_page_config(page_title="ToFu´s Stock Analysis & Options Trading", layout="wide")
st.title("ToFu´s Stock Analysis & Options Trading")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Stock Analysis", "Options Trading", "Notification Subscription"])
st_autorefresh(interval=60 * 1000, key="real_time_refresh")

###############################################
# PAGE 1: REAL‑TIME STOCK ANALYSIS
###############################################
if page == "Stock Analysis":
    st.header("Real‑Time Stock Analysis")
    st.markdown(
        """
        **Overview:**  
        This page provides real‑time data and key technical indicators:
        - **RSI (Relative Strength Index)**
        - **MACD (Moving Average Convergence Divergence)**
        - **Bollinger Bands**
        - **Simple Moving Averages (SMA20, SMA50, SMA200)**
        - **Daily High/Low Levels**
        """
    )
    
    ticker_input = st.text_input("Enter Stock Ticker", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Select Data Period", options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=0)
    with col2:
        interval = st.selectbox("Select Data Interval", options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"], index=0)
    
    if st.button("Analyze Stock"):
        try:
            data = fetch_stock_data(ticker_input, period, interval)
            st.subheader(f"Intraday Data for {ticker_input} ({period}, {interval} interval)")
            st.dataframe(data.tail(10))
            
            # Price chart with SMAs and Bollinger Bands.
            fig_price, ax_price = plt.subplots(figsize=(12, 6))
            ax_price.plot(data.index, data["Close"], label="Close Price", color="blue")
            ax_price.plot(data.index, data["SMA20"], label="SMA20", linestyle="--", color="orange")
            ax_price.plot(data.index, data["SMA50"], label="SMA50", linestyle="--", color="green")
            ax_price.plot(data.index, data["SMA200"], label="SMA200", linestyle="--", color="red")
            ax_price.fill_between(data.index, data["BB_Lower"], data["BB_Upper"], color="gray", alpha=0.2, label="Bollinger Bands")
            ax_price.set_title(f"{ticker_input} - Price Chart (Intraday)")
            ax_price.legend()
            st.pyplot(fig_price)
            
            # Subplots for RSI and MACD.
            fig_indicators, (ax_rsi, ax_macd) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax_rsi.plot(data.index, data["RSI"], label="RSI", color="purple")
            ax_rsi.axhline(70, color="red", linestyle="--", label="Overbought (70)")
            ax_rsi.axhline(30, color="green", linestyle="--", label="Oversold (30)")
            ax_rsi.set_ylabel("RSI")
            ax_rsi.legend()
            
            ax_macd.plot(data.index, data["MACD"], label="MACD", color="brown")
            ax_macd.plot(data.index, data["Signal"], label="Signal", color="magenta")
            ax_macd.set_ylabel("MACD")
            ax_macd.legend()
            st.pyplot(fig_indicators)
            
        except Exception as e:
            st.error(f"Error analyzing {ticker_input}: {e}")

###############################################
# PAGE 2: OPTIONS TRADING & GREEKS (with Graph)
###############################################
elif page == "Options Trading":
    st.header("Options Trading Analysis & Greeks")
    st.markdown(
        """
        Retrieve and analyze options chain data including Black–Scholes Greeks.
        
        **Instructions:**
        - Enter the ticker below.
        - The app will fetch the list of available expiration dates.  
          Select the expiration date from the list.
        - Data will include Delta, Gamma, Theta (per day), Vega, Rho, and the estimated option price.
        - The graph below displays the Black–Scholes estimated option price as a function of strike price,
          with separate line plots for calls and puts, and the underlying price indicated.
        """
    )
    
    ticker_option = st.text_input("Enter Stock Ticker for Options", value="AAPL", key="option_ticker")
    
    # Fetch available expirations for the entered ticker.
    try:
        ticker_obj = yf.Ticker(ticker_option)
        expirations = ticker_obj.options
        if not expirations:
            st.error("No options expirations available for this ticker.")
        else:
            expiration_selected = st.selectbox("Select Expiration Date", expirations)
    except Exception as e:
        st.error(f"Error retrieving expiration dates: {e}")
        expirations = []
        expiration_selected = None
    
    if st.button("Get Option Chain") and expiration_selected:
        with st.spinner("Fetching options data..."):
            calls, puts, expiration_info = get_option_chain(ticker_option, expiration_selected)
        if (calls is None) or (puts is None):
            st.error(f"Error retrieving options: {expiration_info}")
        else:
            st.success(f"Options data for expiration: {expiration_info}")
            # Calculate time to expiration in years.
            try:
                exp_date = datetime.datetime.strptime(expiration_info, "%Y-%m-%d")
                today = datetime.datetime.today()
                T = max((exp_date - today).days / 365.0, 0.001)
            except Exception as e:
                st.error(f"Error parsing expiration date: {e}")
                T = 0.001

            # Get the current underlying price.
            try:
                current_data = yf.Ticker(ticker_option).history(period="1d", interval="1m")
                S = current_data["Close"].iloc[-1]
            except Exception as e:
                st.error(f"Error retrieving current price for {ticker_option}: {e}")
                S = np.nan
            
            st.subheader("Call Options")
            if not calls.empty:
                calls = add_greeks(calls, S, T, r=0.01, option_type='call')
                st.dataframe(calls)
            else:
                st.info("No call options data available.")
            
            st.subheader("Put Options")
            if not puts.empty:
                puts = add_greeks(puts, S, T, r=0.01, option_type='put')
                st.dataframe(puts)
            else:
                st.info("No put options data available.")
            
            # ---------------------------
            # NEW: OPTION GRAPH (Line Plot)
            # ---------------------------
            st.markdown("### Black–Scholes Option Price vs. Strike Price")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Plot calls as a line chart (sorted by strike)
            if not calls.empty:
                calls_sorted = calls.sort_values("strike")
                ax.plot(calls_sorted["strike"], calls_sorted["BS_Price"], label="Calls", color="blue", marker="o")
            # Plot puts as a line chart (sorted by strike)
            if not puts.empty:
                puts_sorted = puts.sort_values("strike")
                ax.plot(puts_sorted["strike"], puts_sorted["BS_Price"], label="Puts", color="red", marker="o")
            # Mark the underlying price as a vertical line.
            ax.axvline(x=S, color="black", linestyle="--", label="Underlying Price")
            ax.set_xlabel("Strike Price")
            ax.set_ylabel("Option Price (Black–Scholes Estimate)")
            ax.set_title(f"Option Price vs. Strike Price for Expiration: {expiration_info}")
            ax.legend()
            st.pyplot(fig)

###############################################
# PAGE 3: NOTIFICATION SUBSCRIPTION & TESTING
###############################################
elif page == "Notification Subscription":
    st.header("RSI Notification Subscription")
    st.markdown(
        """
        Subscribe to receive email notifications when RSI crosses critical thresholds:
        
        - **RSI < 35:** Oversold condition.
        - **RSI > 65:** Overbought condition.
        
        The notification email will include current price, volume, SMAs, and RSI.
        """
    )
    subscription_email = st.text_input("Enter Your Email Address", value="", key="notify_email")
    ticker_notify = st.text_input("Enter Stock Ticker to Monitor", value="AAPL", key="notify_ticker")
    col_n1, col_n2 = st.columns(2)
    with col_n1:
        period_notify = st.selectbox("Select Data Period for Monitoring", options=["1d", "5d", "1mo"], index=0, key="notify_period")
    with col_n2:
        interval_notify = st.selectbox("Select Data Interval", options=["1m", "5m", "15m", "30m"], index=0, key="notify_interval")
    
    if st.button("Subscribe for RSI Alerts"):
        if not subscription_email or "@" not in subscription_email:
            st.error("Please enter a valid email address.")
        else:
            # In production, store the subscription (e.g., in a database).
            st.success("Subscription successful! (For demo, click 'Test Notification' to simulate an alert.)")
    
    if st.button("Test Notification Now"):
        if subscription_email and ticker_notify:
            enhanced_notification(ticker_notify, subscription_email, period_notify, interval_notify)
        else:
            st.error("Please provide both an email and a ticker to monitor.")
st_autorefresh(interval=60 * 1000, key="real_time_refresh")
