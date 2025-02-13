##########################################
# Happy Trading & Coding. It´s ToFu-Time #
##########################################

import yfinance as yf
import pandas as pd
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
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
import pandas_ta as ta

st.set_page_config(page_title="ToFu´s Stock Analysis & Options Trading", layout="wide")
st.title("ToFu´s Stock Analysis & Options Trading")

###############################################
# CUSTOM CSS
###############################################
st.markdown(
    """
    <style>
    /* Main container with a bright gradient background */
    .reportview-container {
        background: linear-gradient(135deg, #fdfbfb, #ebedee);
    }
    /* Sidebar with a bright teal background */
    .sidebar .sidebar-content {
        background-color: #e0f7fa;
    }
    /* Headings with a deep blue color */
    h1, h2, h3, h4, h5, h6 {
        color: #003366;
    }
    /* Buttons with a vibrant orange style */
    .stButton>button {
        background-color: #ff5722;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    color: #333;
    text-align: center;
    padding: 10px 0;
    box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    </style>
    
    """,
    unsafe_allow_html=True,
)

# Refresh the entire page every 15 seconds
st_autorefresh(interval=15 * 1000, key="real_time_refresh")

###############################################
# SECTION 1: TECHNICAL INDICATOR CALCULATIONS USING PANDAS_TA
###############################################
def add_technical_indicators(data):
    """
    Uses pandas_ta to compute:
    - RSI (14)
    - MACD (fast=12, slow=26, signal=9)
    - Bollinger Bands (length=20, std=2)
    - SMAs (20, 50, 200)
    - VWAP
    - ADX (14)
    Also computes pivot points (PP, R1, S1) manually,
    and adds daily high and low values.
    """
    try:
        data["RSI"] = ta.rsi(data["Close"], length=14)
        macd = ta.macd(data["Close"], fast=12, slow=26, signal=9)
        data["MACD"] = macd["MACD_12_26_9"]
        data["Signal"] = macd["MACDs_12_26_9"]
        data["MACD_Hist"] = macd["MACDh_12_26_9"]
        bb = ta.bbands(data["Close"], length=20, std=2)
        data["BBL"] = bb["BBL_20_2.0"]
        data["BBM"] = bb["BBM_20_2.0"]
        data["BBU"] = bb["BBU_20_2.0"]
        data["SMA20"] = ta.sma(data["Close"], length=20)
        data["SMA50"] = ta.sma(data["Close"], length=50)
        data["SMA200"] = ta.sma(data["Close"], length=200)
        data["VWAP"] = ta.vwap(data["High"], data["Low"], data["Close"], data["Volume"])
        adx = ta.adx(data["High"], data["Low"], data["Close"], length=14)
        data["ADX"] = adx["ADX_14"]
        data["PP"] = (data["High"] + data["Low"] + data["Close"]) / 3
        data["R1"] = 2 * data["PP"] - data["Low"]
        data["S1"] = 2 * data["PP"] - data["High"]
        data["Day_High"] = data["High"].cummax()
        data["Day_Low"] = data["Low"].cummin()
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
    return data

###############################################
# SECTION 2: DATA FETCHING & PROCESSING FOR STOCKS
###############################################
def fetch_stock_data(ticker, period="1d", interval="1m"):
    """
    Fetch historical stock data using yfinance and enrich it with technical indicators.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        data.index = data.index.tz_localize(None)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        raise

    try:
        data = add_technical_indicators(data)
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
    Add Black-Scholes Greeks to an options DataFrame.
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
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        expirations = ticker_obj.options
        if not expirations:
            return None, None, "No options data available."
        if expiration is None or expiration not in expirations:
            expiration = expirations[0]
        chain = ticker_obj.option_chain(expiration)
        return chain.calls, chain.puts, expiration
    except Exception as e:
        return None, None, f"Error retrieving options chain: {e}"

###############################################
# SECTION 5: EMAIL NOTIFICATIONS & SMTP SERVER
###############################################
def send_email_notification(to_email, subject, body):
    """
    Send an email notification using SMTP.
    SMTP settings are taken from st.session_state if available.
    """
    # Use configured SMTP settings if available, otherwise use placeholders.
    SMTP_SERVER = st.session_state.get("SMTP_SERVER", "")
    SMTP_PORT = st.session_state.get("SMTP_PORT", 587)
    SMTP_USER = st.session_state.get("SMTP_USER", "")
    SMTP_PASSWORD = st.session_state.get("SMTP_PASSWORD", "")
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
    Check the latest stock data for the given ticker and send an email alert if RSI is critical.
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

###############################################
# SECTION 6: STREAMLIT APP LAYOUT & MULTI-PAGE NAVIGATION
###############################################
pages = [
    "Stock Analysis", 
    "Watchlist", 
    "Options Trading", 
    "SMTP Server", 
    "Notification Subscription", 
    "Investment Information", 
    "Set Option Calls"
]
page = st.sidebar.radio("Navigation", pages)

###############################################
# PAGE 1: REAL‑TIME STOCK ANALYSIS
###############################################
if page == "Stock Analysis":
    st.header("Real‑Time Stock Analysis")
    st.markdown(
        """
        **Overview:**  
        This page provides real‑time data along with a comprehensive set of technical indicators:
        - **RSI (Relative Strength Index)**
        - **MACD (Moving Average Convergence Divergence)**
        - **Bollinger Bands**
        - **Simple Moving Averages (SMA20, SMA50, SMA200)**
        - **VWAP (Volume Weighted Average Price)**
        - **ADX (Average Directional Index)**
        - **Pivot Points (PP, R1, S1)**
        - **Daily High/Low Levels**
        """
    )
    if "auto_update" not in st.session_state:
        st.session_state.auto_update = False

    col_update1, col_update2 = st.columns(2)
    with col_update1:
        if st.button("Start Auto‑Update"):
            st.session_state.auto_update = True
    with col_update2:
        if st.button("Stop Auto‑Update"):
            st.session_state.auto_update = False

    ticker_input = st.text_input("Enter Stock Ticker", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Select Data Period", 
                              options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=0)
    with col2:
        interval = st.selectbox("Select Data Interval", 
                                options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"], index=0)
    
    if st.session_state.auto_update:
        try:
            data = fetch_stock_data(ticker_input, period, interval)
            st.subheader(f"Intraday Data for {ticker_input} ({period}, {interval} interval)")
            st.dataframe(data.tail(10))
            
            # Price chart with SMAs, Bollinger Bands, VWAP, and Pivot Points.
            fig_price, ax_price = plt.subplots(figsize=(12, 6))
            ax_price.plot(data.index, data["Close"], label="Close Price", color="blue")
            ax_price.plot(data.index, data["SMA20"], label="SMA20", linestyle="--", color="orange")
            ax_price.plot(data.index, data["SMA50"], label="SMA50", linestyle="--", color="green")
            ax_price.plot(data.index, data["SMA200"], label="SMA200", linestyle="--", color="red")
            ax_price.plot(data.index, data["VWAP"], label="VWAP", linestyle="-.", color="magenta")
            ax_price.fill_between(data.index, data["BBL"], data["BBU"], color="gray", alpha=0.2, label="Bollinger Bands")
            ax_price.axhline(y=data["PP"].iloc[-1], label="Pivot Point (PP)", color="grey", linestyle="--")
            ax_price.axhline(y=data["R1"].iloc[-1], label="Resistance 1 (R1)", color="red", linestyle="--")
            ax_price.axhline(y=data["S1"].iloc[-1], label="Support 1 (S1)", color="green", linestyle="--")
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
            
            # ADX Chart.
            fig_adx, ax_adx = plt.subplots(figsize=(12, 4))
            ax_adx.plot(data.index, data["ADX"], label="ADX", color="brown")
            ax_adx.axhline(25, color="red", linestyle="--", label="Trend Threshold (25)")
            ax_adx.set_title("Average Directional Index (ADX)")
            ax_adx.legend()
            st.pyplot(fig_adx)
            
            # Fundamental Metrics Display.
            st.subheader("Fundamental Analysis")
            try:
                ticker_obj = yf.Ticker(ticker_input)
                info = ticker_obj.info
                st.write("**Current Ratio:**", info.get("currentRatio", "N/A"))
                st.write("**Debt to Equity Ratio:**", info.get("debtToEquity", "N/A"))
                st.write("**Return on Equity (ROE):**", info.get("returnOnEquity", "N/A"))
                st.write("**Gross Profit Margin:**", info.get("grossMargins", "N/A"))
                st.write("**Net Profit Margin:**", info.get("profitMargins", "N/A"))
                st.write("**Return on Assets (ROA):**", info.get("returnOnAssets", "N/A"))
            except Exception as e:
                st.error(f"Error fetching fundamental metrics: {e}")
            
        except Exception as e:
            st.error(f"Error analyzing {ticker_input}: {e}")
    else:
        st.info("Click **Start Auto‑Update** to analyze the stock data automatically every 15 seconds.")

    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 2: WATCHLIST
###############################################
elif page == "Watchlist":
    st.header("Watchlist")
    st.markdown(
        """
        **Overview:**  
        Add tickers to your watchlist and see their latest RSI values along with industry information.
        You can also compare the RSI of each stock to the average RSI of its industry.
        """
    )
    # Initialize watchlist in session state if not exists.
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    
    with st.form("add_watchlist_form"):
        new_ticker = st.text_input("Enter Ticker to Add", value="AAPL")
        add_button = st.form_submit_button("Add to Watchlist")
    if add_button:
        ticker_upper = new_ticker.strip().upper()
        if ticker_upper and ticker_upper not in st.session_state.watchlist:
            st.session_state.watchlist.append(ticker_upper)
            st.success(f"{ticker_upper} added to watchlist!")
        else:
            st.info("Ticker is already in your watchlist or invalid.")
    
    # Option to clear the watchlist.
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []
        st.info("Watchlist cleared!")
    
    if st.session_state.watchlist:
        watchlist_data = []
        for ticker in st.session_state.watchlist:
            try:
                # Use 1d and 1m interval for a quick refresh of current RSI.
                data = fetch_stock_data(ticker, period="1d", interval="1m")
                latest = data.iloc[-1]
                current_rsi = latest["RSI"]
                # Try to get industry info from ticker.info
                info = yf.Ticker(ticker).info
                industry = info.get("industry", "N/A")
                watchlist_data.append({"Ticker": ticker, "Industry": industry, "RSI": current_rsi})
            except Exception as e:
                watchlist_data.append({"Ticker": ticker, "Industry": "Error", "RSI": np.nan})
        df_watchlist = pd.DataFrame(watchlist_data)
        
        # Compute average RSI per industry if available.
        if not df_watchlist.empty:
            df_avg = df_watchlist.groupby("Industry")["RSI"].mean().reset_index().rename(columns={"RSI": "Industry Avg RSI"})
            df_watchlist = pd.merge(df_watchlist, df_avg, on="Industry", how="left")
        
        st.dataframe(df_watchlist)
    else:
        st.info("Your watchlist is empty. Please add tickers.")
    
    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 3: OPTIONS TRADING & GREEKS (with Graph)
###############################################
elif page == "Options Trading":
    st.header("Options Trading Analysis & Greeks")
    st.markdown(
        """
        Retrieve and analyze options chain data including Black–Scholes Greeks.
        
        **Instructions:**
        - Enter the ticker below.
        - The app will fetch the list of available expiration dates.  
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
            try:
                exp_date = datetime.datetime.strptime(expiration_info, "%Y-%m-%d")
                today = datetime.datetime.today()
                T = max((exp_date - today).days / 365.0, 0.001)
            except Exception as e:
                st.error(f"Error parsing expiration date: {e}")
                T = 0.001

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
            
            # OPTION GRAPH (Dual-panel Layout)
            st.markdown("### Black–Scholes Option Price vs. Strike Price")
            fig, (ax_calls, ax_puts) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
            
            # Plot for Call Options
            if not calls.empty:
                calls_sorted = calls.sort_values("strike")
                ax_calls.plot(calls_sorted["strike"], calls_sorted["BS_Price"], label="Calls", color="blue", marker="o", linestyle="-")
                ax_calls.axvline(x=S, color="black", linestyle="--", label="Underlying Price")
                ax_calls.set_title("Call Options")
                ax_calls.set_ylabel("BS Price")
                ax_calls.grid(True)
                ax_calls.legend()
            else:
                ax_calls.text(0.5, 0.5, "No call options data available", transform=ax_calls.transAxes, ha="center", va="center")
                ax_calls.set_title("Call Options")
            
            # Plot for Put Options
            if not puts.empty:
                puts_sorted = puts.sort_values("strike")
                ax_puts.plot(puts_sorted["strike"], puts_sorted["BS_Price"], label="Puts", color="red", marker="o", linestyle="-")
                ax_puts.axvline(x=S, color="black", linestyle="--", label="Underlying Price")
                ax_puts.set_title("Put Options")
                ax_puts.set_xlabel("Strike Price")
                ax_puts.set_ylabel("BS Price")
                ax_puts.grid(True)
                ax_puts.legend()
            else:
                ax_puts.text(0.5, 0.5, "No put options data available", transform=ax_puts.transAxes, ha="center", va="center")
                ax_puts.set_title("Put Options")
            
            fig.suptitle(f"Option Prices vs. Strike Price for Expiration: {expiration_info}", fontsize=16)
            st.pyplot(fig)
    
    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 4: SMTP SERVER SETTINGS
###############################################
elif page == "SMTP Server":
    st.header("SMTP Server Settings")
    st.markdown(
        """
        **Configure your SMTP server settings for email notifications.**  
        (Ensure you update these settings so that email notifications can be sent successfully.)
        """
    )
    with st.form("smtp_form"):
        smtp_server = st.text_input("SMTP Server", value=st.session_state.get("SMTP_SERVER", "smtp.example.com"))
        smtp_port = st.number_input("SMTP Port", value=st.session_state.get("SMTP_PORT", 587), step=1)
        smtp_user = st.text_input("SMTP Username", value=st.session_state.get("SMTP_USER", "your_email@example.com"))
        smtp_password = st.text_input("SMTP Password", type="password", value=st.session_state.get("SMTP_PASSWORD", ""))
        submit_smtp = st.form_submit_button("Save SMTP Settings")
    if submit_smtp:
        st.session_state.SMTP_SERVER = smtp_server
        st.session_state.SMTP_PORT = smtp_port
        st.session_state.SMTP_USER = smtp_user
        st.session_state.SMTP_PASSWORD = smtp_password
        st.success("SMTP settings saved!")
    
    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 5: NOTIFICATION SUBSCRIPTION & TESTING
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
            st.success("Subscription successful! (For demo, click 'Test Notification' to simulate an alert.)")
    
    if st.button("Test Notification Now"):
        if subscription_email and ticker_notify:
            enhanced_notification(ticker_notify, subscription_email, period_notify, interval_notify)
        else:
            st.error("Please provide both an email and a ticker to monitor.")
    
    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 6: INVESTMENT INFORMATION
###############################################
elif page == "Investment Information":
    investment_info_content = r"""
    # Investment Analysis & Fundamentals
    ---
    
    ## 1. Options Fundamentals
    
    ### Call Options
    - **Definition:** A call option gives the buyer the **right, but not the obligation**, to purchase the underlying asset at a predetermined **strike price** on or before the expiration date.
    - **Example:** If you buy a call option for stock XYZ with a strike price of \$100 and a premium of \$5, and the stock rises to \$120, the intrinsic value is \$20 per share, netting you a profit of \$15 per share (ignoring transaction costs).
    
    ### Put Options
    - **Definition:** A put option gives the buyer the **right, but not the obligation**, to sell the underlying asset at a predetermined **strike price** on or before the expiration date.
    - **Example:** If you buy a put option for stock XYZ with a strike price of \$100 and a premium of \$4, and the stock falls to \$80, the intrinsic value is \$20 per share, netting you a profit of \$16 per share.
    
    ### Option Pricing Considerations
    - **Intrinsic Value and Time Value:** Options are priced based on the difference between the underlying asset's price and the strike price, as well as the time left until expiration.
    - **Volatility:** Higher volatility increases the premium due to a greater likelihood of favorable price movements.
    - **Pricing Models:** Black-Scholes and binomial models are commonly used to estimate option prices.
    
    ---
    
    ## 2. Hedging with Options
    
    ### Protective Put
    - **Strategy:** Buy put options while holding the underlying asset to limit downside risk.
    - **Example:** Owning 100 shares of Company ABC at \$50 per share, you buy a put option at a \$50 strike for a \$2 premium. If the stock falls to \$40, the put option gains value, offsetting losses.
    
    ### Covered Call
    - **Strategy:** Hold the underlying asset and sell call options to generate additional income.
    - **Example:** Owning shares at \$50, you sell a call option with a strike of \$55. If the stock remains below \$55, you keep both the shares and the premium.
    
    ### Collar Strategy
    - **Strategy:** Combine buying a protective put and selling a covered call to create a range of acceptable prices.
    - **Example:** Buy a put at \$50 and sell a call at \$60 to limit both downside risk and upside potential.
    
    ---
    
    ## 3. Options Strategies
    
    ### Butterfly Spread
    - **Overview:** A limited-risk, limited-reward strategy using three strike prices.
    - **Example:** Buy one call at \$90, sell two calls at \$100, and buy one call at \$110. Maximum profit is achieved if the underlying asset is at \$100 at expiration.
    
    ### Condor Spread
    - **Overview:** Similar to the butterfly spread but with four strike prices, providing a wider profit zone.
    - **Example:** Buy calls at \$90 and \$120, sell calls at \$100 and \$110.
    
    ### Bull and Bear Spreads
    - **Bull Spread (Call Spread):**  
      - Buy a call at a lower strike and sell a call at a higher strike.  
      - **Example:** Buy a call at \$100 and sell a call at \$110 if expecting a moderate rise.
    - **Bear Spread (Put Spread):**  
      - Buy a put at a higher strike and sell a put at a lower strike.  
      - **Example:** Buy a put at \$100 and sell a put at \$90 if expecting a moderate decline.
    
    ### “Free Lunch” Strategies
    - **Overview:** Strategies like risk reversals that aim to create positions with minimal net premium.
    - **Example:** Sell a put while buying a call to create a synthetic long position with low upfront cost.
    
    ---
    
    ## 4. Financial Ratios and Metrics
    
    ### Current Ratio
    - **Formula:** `Current Ratio = Current Assets / Current Liabilities`
    - **Interpretation:** A ratio above 1 indicates adequate short-term liquidity.
    
    ### Debt to Equity Ratio
    - **Formula:** `Debt to Equity Ratio = Total Liabilities / Shareholders’ Equity`
    - **Interpretation:** A high ratio may indicate potential financial risk due to excessive borrowing.
    
    ### Return on Equity (ROE)
    - **Formula:** `ROE = Net Income / Shareholders’ Equity`
    - **Interpretation:** Measures how effectively a company uses equity to generate profits.
    
    ### Gross Profit Margin
    - **Formula:** `Gross Profit Margin = (Revenue - COGS) / Revenue`
    - **Interpretation:** Higher margins indicate better production efficiency or pricing power.
    
    ### Net Profit Margin
    - **Formula:** `Net Profit Margin = Net Income / Revenue`
    - **Interpretation:** Reflects overall profitability after all expenses.
    
    ### Return on Assets (ROA)
    - **Formula:** `ROA = Net Income / Average Total Assets`
    - **Interpretation:** Indicates how efficiently a company uses its assets to generate profit.
    
    ### Cash Flow Ratio
    - **Formula:** `Cash Flow Ratio = Operating Cash Flow / Current Liabilities`
    - **Interpretation:** A ratio above 1 suggests strong liquidity from operating activities.
    """
    st.markdown(investment_info_content, unsafe_allow_html=True)
    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 7: SET OPTION CALLS
###############################################
elif page == "Set Option Calls":
    st.header("Set Option Calls")
    st.markdown("### Place Your Option Call Order")
    st.markdown("Fill out the details below to simulate your call option order.")
    
    # Use a form for inputting the option call parameters.
    with st.form("option_call_form"):
        ticker_call = st.text_input("Ticker", value="AAPL")
        # Attempt to fetch the current price from yfinance; if unavailable, default to 100.
        try:
            ticker_obj_call = yf.Ticker(ticker_call)
            current_data_call = ticker_obj_call.history(period="1d", interval="1m")
            S_default = current_data_call["Close"].iloc[-1]
        except Exception as e:
            S_default = 100.0
        S_call = st.number_input("Underlying Current Price (S)", value=float(S_default), step=0.1)
        strike_call = st.number_input("Strike Price (K)", value=float(S_default * 1.05), step=0.1)
        premium_call = st.number_input("Option Premium Paid", value=5.0, step=0.1)
        days_to_expiration = st.number_input("Days to Expiration", value=30, min_value=1, step=1)
        risk_free_rate = st.number_input("Risk-Free Interest Rate (annual, decimal)", value=0.01, step=0.001, format="%.3f")
        implied_vol = st.number_input("Implied Volatility (annual, decimal)", value=0.20, step=0.01, format="%.2f")
        num_contracts = st.number_input("Number of Contracts", value=1, min_value=1, step=1)
        submit_option_call = st.form_submit_button("Simulate Option Call")
    
    if submit_option_call:
        T_call = days_to_expiration / 365.0
        # Calculate the Black-Scholes theoretical price and Greeks for a call option.
        delta, gamma, theta, vega, rho, bs_price = black_scholes_greeks(S_call, strike_call, T_call, risk_free_rate, implied_vol, option_type='call')
        st.subheader("Option Call Details and Greeks")
        st.write(f"**Ticker:** {ticker_call}")
        st.write(f"**Underlying Price (S):** {S_call}")
        st.write(f"**Strike Price (K):** {strike_call}")
        st.write(f"**Premium Paid:** {premium_call}")
        st.write(f"**Days to Expiration:** {days_to_expiration}")
        st.write(f"**Risk-Free Rate:** {risk_free_rate}")
        st.write(f"**Implied Volatility:** {implied_vol}")
        st.write(f"**Black-Scholes Theoretical Price:** {bs_price:.2f}")
        st.write(f"**Delta:** {delta:.2f}")
        st.write(f"**Gamma:** {gamma:.4f}")
        st.write(f"**Theta (per day):** {theta:.4f}")
        st.write(f"**Vega:** {vega:.2f}")
        st.write(f"**Rho:** {rho:.2f}")
        st.write(f"**Number of Contracts:** {num_contracts}")
        
        # Simulate the option payoff at expiration.
        st.subheader("Simulated Payoff at Expiration")
        contract_size = 100  # Standard option contract size (100 shares)
        price_range = np.linspace(0.5 * S_call, 1.5 * S_call, 100)
        # For a call option, payoff per share = max(price - strike, 0) - premium paid.
        payoff = np.maximum(price_range - strike_call, 0) - premium_call  
        total_payoff = payoff * contract_size * num_contracts
        
        # Plot the payoff profile.
        fig_payoff, ax_payoff = plt.subplots(figsize=(10, 6))
        ax_payoff.plot(price_range, total_payoff, label="Profit / Loss")
        ax_payoff.axhline(0, color="black", linestyle="--")
        ax_payoff.set_xlabel("Underlying Price at Expiration")
        ax_payoff.set_ylabel("Profit / Loss ($)")
        ax_payoff.set_title("Option Call Payoff at Expiration")
        ax_payoff.legend()
        st.pyplot(fig_payoff)
    
    footer_html = """
    <div class="footer">
    <p>© 2025 Tobias Strauss</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
