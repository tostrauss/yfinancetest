import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import concurrent.futures
import time
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# Helper Functions for Technical Indicators
# =============================================================================

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, span_short=12, span_long=26, span_signal=9):
    """Calculate MACD, signal line and histogram."""
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def calculate_sma(series, window):
    """Calculate simple moving average (SMA)."""
    return series.rolling(window=window).mean()

# =============================================================================
# Functions for Data Fetching & Processing
# =============================================================================

def fetch_stock_data(ticker, period="1d", interval="1d"):
    """Fetch stock data from yfinance."""
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period=period, interval=interval)
    if data.empty:
        raise ValueError("No data found")
    # Technical indicators
    data["RSI"] = calculate_rsi(data["Close"])
    macd, signal, _ = calculate_macd(data["Close"])
    data["MACD"] = macd
    data["Signal"] = signal
    sma20 = calculate_sma(data["Close"], 20)
    sma50 = calculate_sma(data["Close"], 50)
    sma200 = calculate_sma(data["Close"], 200)
    data["SMA20"] = sma20
    data["SMA50"] = sma50
    data["SMA200"] = sma200
    sma, upper_band, lower_band = calculate_bollinger_bands(data["Close"])
    data["BB_Mid"] = sma
    data["BB_Upper"] = upper_band
    data["BB_Lower"] = lower_band
    data["Day_High"] = data["High"].cummax()
    data["Day_Low"] = data["Low"].cummin()
    return data

def process_ticker(ticker, period, interval, rsi_threshold=40):
    """
    Process a single ticker and return key indicators if the current RSI is below the threshold which is 40.
    """
    try:
        data = fetch_stock_data(ticker, period, interval)
        if len(data) < 15:
            return {"Ticker": ticker, "Error": "Insufficient data"}
        current_rsi = data["RSI"].iloc[-1]
        if current_rsi < rsi_threshold:
            return {
                "Ticker": ticker,
                "Current RSI": round(current_rsi, 2),
                "Current MACD": round(data["MACD"].iloc[-1], 2),
                "Current Signal": round(data["Signal"].iloc[-1], 2),
            }
        else:
            return None
    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

def screen_tickers_parallel(ticker_list, period, interval, rsi_threshold=40, max_workers=10):
    """
    Process multiple tickers in parallel.
    Returns a DataFrame of tickers that meet criteria and a list of errors.
    """
    results = []
    errors = []
    total = len(ticker_list)
    progress_text = st.empty()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_ticker, ticker, period, interval, rsi_threshold): ticker
            for ticker in ticker_list
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            result = future.result()
            completed += 1
            progress_text.text(f"Processed {completed}/{total} tickers...")
            if result:
                if "Error" in result:
                    errors.append(result)
                else:
                    results.append(result)
    progress_text.empty()
    results_df = pd.DataFrame(results)
    return results_df, errors

# =============================================================================
# Functions for Option Chain Data
# =============================================================================

def get_option_chain(ticker, expiration=None):
    """
    Retrieve the options chain data for a given ticker.
    If expiration is not provided, return the first available expiration.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if not expirations:
            return None, None, "No options data available."
        # If no expiration is specified, choose the nearest expiration.
        if expiration is None or expiration not in expirations:
            expiration = expirations[0]
        chain = ticker_obj.option_chain(expiration)
        return chain.calls, chain.puts, expiration
    except Exception as e:
        return None, None, str(e)

# =============================================================================
# Functions for Notifications (Subscription and Sending Emails)
# =============================================================================

def send_email_notification(to_email, subject, body):
    """
    Send an email notification using SMTP.
    """
    # --- SMTP server !!! ---
    SMTP_SERVER = ""   
    SMTP_PORT = 587
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

def check_and_notify(ticker, email, period="1d", interval="1m"):
    """
    Check the latest RSI for a ticker and send an email if RSI crosses key levels.
    In this demo, thresholds are RSI < 35 (oversold) or RSI > 65 (overbought).
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
        message = f"Ticker: {ticker}\nCurrent Price: ${current_price:.2f}\nVolume: {current_volume}\n"
        message += f"SMA20: {sma20:.2f}, SMA50: {sma50:.2f}, SMA200: {sma200:.2f}\n"
        message += f"RSI: {current_rsi:.2f}\n\n"
        
        if current_rsi < 35:
            message += "Alert: RSI has fallen below 35 (Oversold)!\n"
            alert = True
        elif current_rsi > 65:
            message += "Alert: RSI has risen above 65 (Overbought)!\n"
            alert = True
        
        if alert:
            subject = f"Stock Alert: {ticker} - RSI Alert"
            if send_email_notification(email, subject, message):
                st.success("Notification sent successfully!")
            else:
                st.error("Notification failed to send.")
        else:
            st.info("RSI is within normal limits. No alert triggered.")
    except Exception as e:
        st.error(f"Error during notification check: {e}")

# =============================================================================
# Streamlit App: Multi-Page Layout via Sidebar Navigation
# =============================================================================

st.set_page_config(page_title="ToFu´s Stock Analysis & Options Trading", layout="wide")
st.title("ToFu´s Stock Analysis & Options Trading App")

# Use radio buttons to switch pages
page = st.sidebar.radio("Navigation", ["Stock Analysis", "Options Trading", "Notification Subscription"])

# ---------------------------
# Page 1: Stock Analysis
# ---------------------------
if page == "Stock Analysis":
    st.header("Real-Time Stock Analysis")
    st.markdown(
        """
        This page displays a range of technical indicators that have been shown to be important:
        - **RSI (Relative Strength Index)**
        - **MACD (Moving Average Convergence Divergence)**
        - **Bollinger Bands**
        - **Simple Moving Averages (20, 50, 200)**
        - **Daily High/Low Levels"
        
        **Timeframes:** You can select daily, intraday (e.g., 5m, 15m, 30m) or weekly data.
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
            st.write(f"Intraday Data for {ticker_input}({period}), ({period}), {interval} interval")
            st.dataframe(data.tail(10))
            
            # Plot the Close price with moving averages and Bollinger Bands
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data["Close"], label="Close Price", color="blue")
            ax.plot(data.index, data["SMA20"], label="SMA 20", linestyle="--")
            ax.plot(data.index, data["SMA50"], label="SMA 50", linestyle="--")
            ax.plot(data.index, data["SMA200"], label="SMA 200", linestyle="--")
            ax.fill_between(data.index, data["BB_Lower"], data["BB_Upper"], color="gray", alpha=0.2, label="Bollinger Bands")
            ax.set_title(f"{ticker_input} Price Chart (Intra)")
            ax.legend()
            st.pyplot(fig)
            
            # Plot RSI & MACD in separate charts
            fig2, (ax_rsi, ax_macd) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax_rsi.plot(data.index, data["RSI"], label="RSI", color="purple")
            ax_rsi.axhline(70, color="red", linestyle="--", label="Overbought (70)")
            ax_rsi.axhline(30, color="green", linestyle="--", label="Oversold (30)")
            ax_rsi.set_ylabel("RSI")
            ax_rsi.legend()
            ax_macd.plot(data.index, data["MACD"], label="MACD", color="orange")
            ax_macd.plot(data.index, data["Signal"], label="Signal", color="magenta")
            ax_macd.set_ylabel("MACD")
            ax_macd.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error fetching data for {ticker_input}: {e}")

    st.markdown("----")
    st.info("You can also perform multi-ticker screening from the sidebar (future enhancement).")

# ---------------------------
# Page 2: Options Trading
# ---------------------------
elif page == "Options Trading":
    st.header("Options Trading Analysis")
    st.markdown(
        """
        On this page, you can explore options data (calls and puts) for any stock.  
        In addition to viewing the option chain, the app displays relevant technical analysis to aid option trading decisions.
        """
    )
    ticker_option = st.text_input("Enter Stock Ticker for Options", value="AAPL", key="option_ticker")
    expiration_input = st.text_input("Enter Expiration Date (YYYY-MM-DD) or leave blank for nearest", value="", key="option_exp")
    
    if st.button("Get Option Chain"):
        with st.spinner("Fetching options data..."):
            calls, puts, expiration_info = get_option_chain(ticker_option, expiration_input.strip() or None)
        if isinstance(expiration_info, str) and ("No options" in expiration_info or "error" in expiration_info.lower()):
            st.error(expiration_info)
        else:
            st.success(f"Displaying options for expiration: {expiration_info}")
            st.subheader("Calls")
            st.dataframe(calls)
            st.subheader("Puts")
            st.dataframe(puts)
            
            # You can also display the underlying stock chart with its technical indicators
            try:
                data_option = fetch_stock_data(ticker_option, period="6mo", interval="1d")
                fig_option, ax_option = plt.subplots(figsize=(12, 6))
                ax_option.plot(data_option.index, data_option["Close"], label="Close Price")
                ax_option.plot(data_option.index, data_option["SMA20"], label="SMA20", linestyle="--")
                ax_option.fill_between(data_option.index, data_option["BB_Lower"], data_option["BB_Upper"], color="gray", alpha=0.2, label="Bollinger Bands")
                ax_option.set_title(f"{ticker_option} Underlying Price")
                ax_option.legend()
                st.pyplot(fig_option)
            except Exception as e:
                st.error(f"Error displaying underlying chart: {e}")

# ---------------------------
# Page 3: Notification Subscription
# ---------------------------
elif page == "Notification Subscription":
    st.header("Subscribe for RSI Alerts")
    st.markdown(
        """
        Sign up to receive email notifications when a stock's RSI crosses critical thresholds:
        
        - **RSI < 35:** Indicates an oversold condition.
        - **RSI > 65:** Indicates an overbought condition.
        
        The alert will include the current price, volume, key moving averages, and RSI.
        """
    )
    subscription_email = st.text_input("Enter your email address to subscribe", value="", key="subscribe_email")
    ticker_notify = st.text_input("Enter Stock Ticker to Monitor", value="AAPL", key="notify_ticker")
    col_notify, col_check = st.columns(2)
    with col_notify:
        period_notify = st.selectbox("Select Data Period for Monitoring", options=["1d", "5d", "1mo"], index=0, key="notify_period")
    with col_check:
        interval_notify = st.selectbox("Select Data Interval", options=["1m", "5m", "15m", "30m"], index=0, key="notify_interval")
    
    if st.button("Subscribe for Notifications"):
        if not subscription_email or "@" not in subscription_email:
            st.error("Please enter a valid email address.")
        else:
            # For demonstration, we write the subscription to a CSV file.
            subscription_df = pd.DataFrame({
                "Email": [subscription_email],
                "Ticker": [ticker_notify],
                "Period": [period_notify],
                "Interval": [interval_notify],
                "Subscribed At": [datetime.datetime.now()]
            })
            try:
                # Append to a subscriptions CSV (in production, use a database)
                subscription_file = "subscriptions.csv"
                try:
                    existing = pd.read_csv(subscription_file)
                    subscription_df = pd.concat([existing, subscription_df], ignore_index=True)
                except FileNotFoundError:
                    pass
                subscription_df.to_csv(subscription_file, index=False)
                st.success("Subscription successful! You will be notified when the RSI crosses under 35 or over 65.")
            except Exception as e:
                st.error(f"Error saving subscription: {e}")
    
    # For demonstration purposes, add a "Test Notification" button.
    if st.button("Test Notification Now"):
        if subscription_email and ticker_notify:
            check_and_notify(ticker_notify, subscription_email, period_notify, interval_notify)
            st.info("If the RSI is in a critical zone, a notification email was sent (check your inbox).")
        else:
            st.error("Please fill in both the email and ticker fields.")

# =============================================================================
# OPTIONAL: Auto-Refresh Mechanism for Notifications (Demo Purpose)
# =============================================================================
# Note: In a production system, you would run a separate background job that
# checks all subscriptions at scheduled intervals. The following auto-refresh
# simply re-runs the app every 5 minutes.
if page == "Notification Subscription":
    st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")
