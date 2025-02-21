import os
import alpaca_trade_api as tradeapi
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
import requests
from transformers import pipeline
import yfinance as yf

###############################################
# SETUP & CUSTOM CSS
###############################################
st.set_page_config(page_title="ToFu´s Stock Analysis & Options Trading", layout="wide")
st.title("ToFu´s Stock Analysis & Options Trading")

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

###############################################
# LOAD FOOTER FROM EXTERNAL FILE
###############################################
def load_footer():
    if os.path.exists("footer.html"):
        with open("footer.html", "r") as f:
            return f.read()
    else:
        # Fallback footer if file not found.
        return '<div class="footer"><p>© 2025 Tobias Strauss</p></div>'

footer_html = load_footer()

###############################################
# ALPACA API SETTINGS & HELPER FUNCTION
###############################################
if "ALPACA_API_KEY" not in st.session_state:
    st.session_state.ALPACA_API_KEY = "AKAZIT5RT3SU1KNI8OQN"
if "ALPACA_API_SECRET" not in st.session_state:
    st.session_state.ALPACA_API_SECRET = "xJAPz5u5gmod424tQbes1PqfzvkPymCmiWHnC8Ex"
if "ALPACA_BASE_URL" not in st.session_state:
    st.session_state.ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper trading endpoint

def get_alpaca_api():
    return tradeapi.REST(
        st.session_state.ALPACA_API_KEY,
        st.session_state.ALPACA_API_SECRET,
        st.session_state.ALPACA_BASE_URL,
        api_version="v2"
    )

###############################################
# SECTION 1: TECHNICAL INDICATOR CALCULATIONS USING PANDAS_TA
###############################################
def add_technical_indicators(data):
    try:
        data["RSI"] = ta.rsi(data["close"], length=14)
        macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
        data["MACD"] = macd["MACD_12_26_9"]
        data["Signal"] = macd["MACDs_12_26_9"]
        data["MACD_Hist"] = macd["MACDh_12_26_9"]
        bb = ta.bbands(data["close"], length=20, std=2)
        data["BBL"] = bb["BBL_20_2.0"]
        data["BBM"] = bb["BBM_20_2.0"]
        data["BBU"] = bb["BBU_20_2.0"]
        data["SMA20"] = ta.sma(data["close"], length=20)
        data["SMA50"] = ta.sma(data["close"], length=50)
        data["SMA200"] = ta.sma(data["close"], length=200)
        if {"high", "low", "close", "volume"}.issubset(data.columns):
            data["VWAP"] = ta.vwap(data["high"], data["low"], data["close"], data["volume"])
        adx = ta.adx(data["high"], data["low"], data["close"], length=14)
        data["ADX"] = adx["ADX_14"]
        data["PP"] = (data["high"] + data["low"] + data["close"]) / 3
        data["R1"] = 2 * data["PP"] - data["low"]
        data["S1"] = 2 * data["PP"] - data["high"]
        data["Day_High"] = data["high"].cummax()
        data["Day_Low"] = data["low"].cummin()
    except Exception as e:
        st.error(f"Error adding technical indicators: {e}")
    return data

###############################################
# SECTION 2: DATA FETCHING & PROCESSING FOR STOCKS (and crypto/bond data via yfinance)
###############################################
def fetch_stock_data(ticker, period="1d", interval="1m"):
    try:
        api = get_alpaca_api()
        now = datetime.datetime.now()
        if period == "1d":
            start = now - datetime.timedelta(days=1)
        elif period == "5d":
            start = now - datetime.timedelta(days=5)
        elif period == "1mo":
            start = now - datetime.timedelta(days=30)
        elif period == "3mo":
            start = now - datetime.timedelta(days=90)
        elif period == "6mo":
            start = now - datetime.timedelta(days=180)
        elif period == "1y":
            start = now - datetime.timedelta(days=365)
        elif period == "2y":
            start = now - datetime.timedelta(days=730)
        elif period == "5y":
            start = now - datetime.timedelta(days=1825)
        elif period == "10y":
            start = now - datetime.timedelta(days=3650)
        elif period == "max":
            start = now - datetime.timedelta(days=3650)
        else:
            start = now - datetime.timedelta(days=1)
        end = now

        start_iso = start.isoformat()
        end_iso = end.isoformat()

        interval_mapping = {
            "1m": "1Min",
            "2m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "15Min",
            "60m": "1Hour",
            "90m": "1Hour",
            "1h": "1Hour",
            "1d": "1Day"
        }
        alpaca_interval = interval_mapping.get(interval, "1Min")
        bars = api.get_bars(ticker, alpaca_interval, start_iso, end_iso).df
        if bars.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        bars.index = pd.to_datetime(bars.index).tz_localize(None)
        data = bars.copy()
    except Exception as e:
        st.warning(f"Alpaca API error: {e}. Falling back to yfinance for {ticker}.")
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data returned for ticker: {ticker}")
            data.columns = [col.lower() for col in data.columns]
            data.index = data.index.tz_localize(None)
        except Exception as e2:
            st.error(f"Error fetching data for {ticker} via yfinance: {e2}")
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
# SECTION 4: OPTIONS CHAIN DATA FETCHING (ALWAYS USING YFINANCE)
###############################################
def get_option_chain(ticker, expiration=None):
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
    try:
        data = fetch_stock_data(ticker, period, interval)
        latest = data.iloc[-1]
        current_rsi = latest["RSI"]
        current_price = latest["close"]
        current_volume = int(latest["volume"]) if "volume" in latest else 0
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
# SECTION 6: METRICS & SENTIMENT ANALYSIS FUNCTIONS
###############################################
@st.cache_resource(show_spinner=False)
def load_finbert():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
    except Exception as e:
        st.error(f"Error loading FinBERT: {e}")
        return None

finbert = load_finbert()

def fetch_news(ticker, api_key, from_date, to_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": api_key,
        "pageSize": 20
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data.get("status") != "ok":
            st.error("Error fetching news data.")
            return []
        articles = data.get("articles", [])
        headlines = [article["title"] for article in articles if article.get("title")]
        return headlines
    except Exception as e:
        st.error(f"Error in NewsAPI request: {e}")
        return []

def compute_average_sentiment(headlines):
    if not headlines or not finbert:
        return 0
    scores = []
    for headline in headlines:
        try:
            result = finbert(headline)
            label = result[0]["label"]
            if label.upper() == "POSITIVE":
                scores.append(1)
            elif label.upper() == "NEGATIVE":
                scores.append(-1)
            else:
                scores.append(0)
        except Exception as e:
            st.error(f"Error analyzing sentiment for a headline: {e}")
            scores.append(0)
    if scores:
        return np.mean(scores)
    return 0

def generate_trading_signal(rsi, avg_sentiment, rsi_buy=30, rsi_sell=70, sentiment_threshold=0.2):
    if rsi < rsi_buy:
        tech_signal = 1
    elif rsi > rsi_sell:
        tech_signal = -1
    else:
        tech_signal = 0

    if avg_sentiment > sentiment_threshold:
        sentiment_signal = 1
    elif avg_sentiment < -sentiment_threshold:
        sentiment_signal = -1
    else:
        sentiment_signal = 0

    combined = tech_signal + sentiment_signal
    if combined >= 1:
        return "STRONG BUY"
    elif combined <= -1:
        return "STRONG SELL"
    else:
        return "HOLD"

###############################################
# SECTION 7: BROKER ORDER FUNCTION FOR STOCKS USING ALPACA
###############################################
def place_order(symbol, qty, side, order_type, time_in_force):
    api = get_alpaca_api()
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )
        st.success(f"Order submitted: {order}")
    except Exception as e:
        st.error(f"Error placing order: {e}")

###############################################
# SECTION 8: STREAMLIT APP LAYOUT & MULTI-PAGE NAVIGATION
###############################################
pages = [
    "Stock Analysis", 
    "Watchlist", 
    "Options Trading", 
    "SMTP Server", 
    "Notification Subscription", 
    "Investment Information", 
    "Set Option Calls",
    "Risk/Reward Calculator",
    "Metrics & Sentiment Tracker",
    "Bond Analysis",
    "Crypto Analysis"
]
    
page = st.sidebar.selectbox("Select Page", pages)   
if page != "Metrics & Sentiment Tracker":
    st_autorefresh(interval=15 * 1000, key="real_time_refresh_unique")

###############################################
# PAGE 1: REAL‑TIME STOCK ANALYSIS
###############################################
if page == "Stock Analysis":
    st.header("Real‑Time Stock Analysis")
    st.markdown(""" 
        **Overview:**  
        This page provides real‑time data along with a comprehensive set of technical indicators:
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        - Simple Moving Averages (SMA20, SMA50, SMA200)
        - VWAP (Volume Weighted Average Price)
        - ADX (Average Directional Index)
        - Pivot Points (PP, R1, S1)
        - Daily High/Low Levels
    """)
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
        period = st.selectbox("Select Data Period", options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=0)
    with col2:
        interval = st.selectbox("Select Data Interval", options=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"], index=0)
    
    if st.session_state.auto_update:
        try:
            data = fetch_stock_data(ticker_input, period, interval)
            st.subheader(f"Intraday Data for {ticker_input} ({period}, {interval} interval)")
            st.dataframe(data.tail(10))
            fig_price, ax_price = plt.subplots(figsize=(12, 6))
            ax_price.plot(data.index, data["close"], label="Close Price", color="blue")
            ax_price.plot(data.index, data["SMA20"], label="SMA20", linestyle="--", color="orange")
            ax_price.plot(data.index, data["SMA50"], label="SMA50", linestyle="--", color="green")
            ax_price.plot(data.index, data["SMA200"], label="SMA200", linestyle="--", color="red")
            if "VWAP" in data.columns:
                ax_price.plot(data.index, data["VWAP"], label="VWAP", linestyle="-.", color="magenta")
            if "BBL" in data.columns and "BBU" in data.columns:
                ax_price.fill_between(data.index, data["BBL"], data["BBU"], color="gray", alpha=0.2, label="Bollinger Bands")
            ax_price.axhline(y=data["PP"].iloc[-1], label="Pivot Point (PP)", color="grey", linestyle="--")
            ax_price.axhline(y=data["R1"].iloc[-1], label="Resistance 1 (R1)", color="red", linestyle="--")
            ax_price.axhline(y=data["S1"].iloc[-1], label="Support 1 (S1)", color="green", linestyle="--")
            ax_price.set_title(f"{ticker_input} - Price Chart (Intraday)")
            ax_price.legend()
            st.pyplot(fig_price)
            
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
            
            fig_adx, ax_adx = plt.subplots(figsize=(12, 4))
            ax_adx.plot(data.index, data["ADX"], label="ADX", color="brown")
            ax_adx.axhline(25, color="red", linestyle="--", label="Trend Threshold (25)")
            ax_adx.set_title("Average Directional Index (ADX)")
            ax_adx.legend()
            st.pyplot(fig_adx)
            
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

    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 2: WATCHLIST
###############################################
elif page == "Watchlist":
    st.header("Watchlist")
    st.markdown("""
        **Overview:**  
        Add tickers to your watchlist and see their latest RSI values along with industry information.
    """)
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
    
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []
        st.info("Watchlist cleared!")
    
    if st.session_state.watchlist:
        watchlist_data = []
        for ticker in st.session_state.watchlist:
            try:
                data = fetch_stock_data(ticker, period="1d", interval="1m")
                latest = data.iloc[-1]
                current_rsi = latest["RSI"]
                info = yf.Ticker(ticker).info
                industry = info.get("industry", "N/A")
                watchlist_data.append({"Ticker": ticker, "Industry": industry, "RSI": current_rsi})
            except Exception as e:
                watchlist_data.append({"Ticker": ticker, "Industry": "Error", "RSI": np.nan})
        df_watchlist = pd.DataFrame(watchlist_data)
        if not df_watchlist.empty:
            df_avg = df_watchlist.groupby("Industry")["RSI"].mean().reset_index().rename(columns={"RSI": "Industry Avg RSI"})
            df_watchlist = pd.merge(df_watchlist, df_avg, on="Industry", how="left")
        st.dataframe(df_watchlist)
    else:
        st.info("Your watchlist is empty. Please add tickers.")

    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 3: OPTIONS TRADING & GREEKS
###############################################
elif page == "Options Trading":
    st.header("Options Trading Analysis & Greeks")
    st.markdown("""
        Retrieve and analyze options chain data including Black–Scholes Greeks.
    """)
    
    ticker_option = st.text_input("Enter Stock Ticker for Options", value="AAPL", key="option_ticker")
    try:
        calls, puts, expiration_info = get_option_chain(ticker_option)
        if isinstance(expiration_info, str) and "Error" in expiration_info:
            st.error(expiration_info)
        else:
            expiration_selected = st.selectbox("Select Expiration Date", [expiration_info])
    except Exception as e:
        st.error(f"Error retrieving expiration dates: {e}")
        calls, puts, expiration_selected = None, None, None
    
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
                current_data = fetch_stock_data(ticker_option, period="1d", interval="1m")
                S = current_data["close"].iloc[-1]
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
            
            st.markdown("### Black–Scholes Option Price vs. Strike Price")
            fig, (ax_calls, ax_puts) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
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

    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 4: SMTP SERVER SETTINGS
###############################################
elif page == "SMTP Server":
    st.header("SMTP Server Settings")
    st.markdown("""
        **Configure your SMTP server settings for email notifications.**
    """)
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

    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 5: NOTIFICATION SUBSCRIPTION & TESTING
###############################################
elif page == "Notification Subscription":
    st.header("RSI Notification Subscription")
    st.markdown("""
        Subscribe to receive email notifications when RSI crosses critical thresholds.
    """)
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

    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 6: INVESTMENT INFORMATION
###############################################
elif page == "Investment Information":
    investment_info_content = r"""
    # Investment Analysis & Fundamentals
    ---
    
    ## 1. Options Fundamentals
    - Gives the right to buy or sell an asset at a predetermined price.
    
    ## 2. Hedging with Options
    - Protective Put, Covered Call, Collar Strategy.
    
    ## 3. Options Strategies
    - Butterfly Spread, Condor Spread.
    
    ## 4. Financial Ratios and Metrics
    - Current Ratio, Debt to Equity, ROE, Gross/Net Profit Margins, ROA.
    """
    st.markdown(investment_info_content, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 7: SET OPTION CALLS
###############################################
elif page == "Set Option Calls":
    st.header("Set Option Calls")
    st.markdown("### Place Your Option Call Order")
    st.markdown("Fill out the details below to simulate your call option order.")
    
    with st.form("option_call_form"):
        ticker_call = st.text_input("Ticker", value="AAPL")
        try:
            current_data_call = fetch_stock_data(ticker_call, period="1d", interval="1m")
            S_default = current_data_call["close"].iloc[-1]
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
        
        st.subheader("Simulated Payoff at Expiration")
        contract_size = 100  # Standard option contract size (100 shares)
        price_range = np.linspace(0.5 * S_call, 1.5 * S_call, 100)
        payoff = np.maximum(price_range - strike_call, 0) - premium_call  
        total_payoff = payoff * contract_size * num_contracts
        
        fig_payoff, ax_payoff = plt.subplots(figsize=(10, 6))
        ax_payoff.plot(price_range, total_payoff, label="Profit / Loss")
        ax_payoff.axhline(0, color="black", linestyle="--")
        ax_payoff.set_xlabel("Underlying Price at Expiration")
        ax_payoff.set_ylabel("Profit / Loss ($)")
        ax_payoff.set_title("Option Call Payoff at Expiration")
        ax_payoff.legend()
        st.pyplot(fig_payoff)
        
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 8: RISK/REWARD CALCULATOR
###############################################
elif page == "Risk/Reward Calculator":
    st.header("Risk/Reward Calculator")
    st.markdown("Enter your trade parameters to calculate your risk/reward ratio and visualize the payoff.")
    trade_type = st.selectbox("Select Trade Type", ["Long", "Short"])
    entry_price = st.number_input("Entry Price", value=100.0, step=0.1)
    stop_loss = st.number_input("Stop Loss Price", value=95.0, step=0.1)
    target_price = st.number_input("Target Price", value=110.0, step=0.1)
    position_size = st.number_input("Position Size", value=1, step=1)
    if trade_type == "Long":
        risk = entry_price - stop_loss
        reward = target_price - entry_price
    else:
        risk = stop_loss - entry_price
        reward = entry_price - target_price
    if risk <= 0:
        st.error("Invalid parameters: Risk must be positive. Please check your entry, stop loss, and target prices.")
    else:
        risk_reward_ratio = reward / risk
        st.markdown(f"**Risk per unit:** ${risk:.2f}")
        st.markdown(f"**Reward per unit:** ${reward:.2f}")
        st.markdown(f"**Risk/Reward Ratio:** {risk_reward_ratio:.2f}")
        low_bound = min(stop_loss, target_price) * 0.95
        high_bound = max(stop_loss, target_price) * 1.05
        price_range = np.linspace(low_bound, high_bound, 100)
        if trade_type == "Long":
            profit_loss = (price_range - entry_price) * position_size
        else:
            profit_loss = (entry_price - price_range) * position_size
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(price_range, profit_loss, label="Profit / Loss", color="blue")
        ax.axvline(entry_price, color="black", linestyle="--", label="Entry Price")
        ax.axvline(stop_loss, color="red", linestyle="--", label="Stop Loss")
        ax.axvline(target_price, color="green", linestyle="--", label="Target Price")
        ax.axhline(0, color="gray", linestyle="-")
        ax.set_xlabel("Price")
        ax.set_ylabel("Profit / Loss ($)")
        ax.set_title("Risk/Reward Diagram")
        ax.legend()
        st.pyplot(fig)

    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 9: METRICS & SENTIMENT TRACKER
###############################################
elif page == "Metrics & Sentiment Tracker":
    st.header("Metrics & Sentiment Stock Tracker")
    st.markdown("""
    This page combines real time technical metrics with live news sentiment analysis to deliver a clear recommendation:
    
    **STRONG BUY | HOLD | STRONG SELL**
    
    **How it works:**
    - **News Flow:** Fetches news articles from the past 24 hours using NewsAPI.
    - **Sentiment Analysis:** Analyzes headlines using FinBERT.
    - **Technical Metrics:** Uses key metrics (e.g., RSI).
    - **Combined Signal:** Outputs a recommendation based on aggregated metrics.
    """)
    
    ticker_ms = st.text_input("Enter Stock Ticker", value="AAPL", key="ms_ticker")
    api_key_news = "c5c9ea5b981f4c6ab85badcf610fba78"
    st.write("Using NEWSAPI key:", api_key_news)
    
    if st.button("Analyze Stock", key="ms_analyze"):
        try:
            data = fetch_stock_data(ticker_ms, period="1d", interval="1m")
            if data is not None:
                latest = data.iloc[-1]
                rsi = latest.get("RSI", 50)
                st.write("Latest Technical Data:")
                st.write(f"Close: {latest['close']:.2f} | RSI: {rsi:.2f}")
            else:
                rsi = 50
        except Exception as e:
            st.error(f"Error fetching technical data: {e}")
            rsi = 50
        
        today = datetime.datetime.now()
        yesterday = today - datetime.timedelta(days=1)
        from_date = yesterday.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        headlines = fetch_news(ticker_ms, api_key_news, from_date, to_date)
        if headlines:
            st.write("Fetched News Headlines:")
            for head in headlines:
                st.write(f"- {head}")
        else:
            st.write("No recent news found.")
        
        avg_sentiment = compute_average_sentiment(headlines)
        st.write(f"Average Sentiment Score: {avg_sentiment:.2f} (scale: -1 negative, +1 positive)")
        
        recommendation = generate_trading_signal(rsi, avg_sentiment)
        st.markdown(f"## Recommendation: **{recommendation}**")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["RSI"], [rsi], color="skyblue")
        ax.axhline(30, color="green", linestyle="--", label="Oversold (30)")
        ax.axhline(70, color="red", linestyle="--", label="Overbought (70)")
        ax.set_ylabel("RSI")
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("""
    **How Often?**  
    - This analysis runs on demand when you click **Analyze Stock**.
    """)
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 10: BOND ANALYSIS
###############################################
elif page == "Bond Analysis":
    st.header("Bond Analysis")
    st.markdown("### Bond Price Calculator")
    with st.form("bond_form"):
        face_value = st.number_input("Face Value ($)", value=1000.0, step=1.0)
        coupon_rate = st.number_input("Coupon Rate (annual, %)", value=5.0, step=0.1) / 100.0
        years_to_maturity = st.number_input("Years to Maturity", value=10, step=1)
        yield_to_maturity = st.number_input("Yield to Maturity (annual, %)", value=4.0, step=0.1) / 100.0
        coupon_frequency = st.selectbox("Coupon Frequency", options=[1, 2, 4], index=1)
        submit_bond = st.form_submit_button("Calculate Bond Price")
    if submit_bond:
        periods = years_to_maturity * coupon_frequency
        coupon_payment = face_value * coupon_rate / coupon_frequency
        price = sum([coupon_payment / ((1 + yield_to_maturity / coupon_frequency) ** (i + 1)) for i in range(int(periods))])
        price += face_value / ((1 + yield_to_maturity / coupon_frequency) ** periods)
        st.write(f"Calculated Bond Price: ${price:.2f}")
        yields = np.linspace(0.01, 0.10, 100)
        prices = []
        for y in yields:
            p = sum([coupon_payment / ((1 + y / coupon_frequency) ** (i + 1)) for i in range(int(periods))])
            p += face_value / ((1 + y / coupon_frequency) ** periods)
            prices.append(p)
        fig_bond, ax_bond = plt.subplots(figsize=(10, 6))
        ax_bond.plot(yields * 100, prices, label="Bond Price Curve", color="blue")
        ax_bond.axvline(yield_to_maturity * 100, color="red", linestyle="--", label="Selected Yield")
        ax_bond.set_xlabel("Yield to Maturity (%)")
        ax_bond.set_ylabel("Bond Price ($)")
        ax_bond.set_title("Bond Price vs. Yield to Maturity")
        ax_bond.legend()
        st.pyplot(fig_bond)
    st.markdown(footer_html, unsafe_allow_html=True)

###############################################
# PAGE 11: CRYPTO ANALYSIS
###############################################
elif page == "Crypto Analysis":
    st.header("Crypto Analysis")
    st.markdown("### Real-Time Cryptocurrency Analysis")
    crypto_ticker = st.text_input("Enter Crypto Ticker (e.g., BTC-USD, ETH-USD)", value="BTC-USD")
    crypto_period = st.selectbox("Select Data Period", options=["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=0)
    crypto_interval = st.selectbox("Select Data Interval", options=["1m", "5m", "15m", "30m", "1h", "1d"], index=0)
    if st.button("Analyze Crypto"):
        try:
            crypto_data = fetch_stock_data(crypto_ticker, period=crypto_period, interval=crypto_interval)
            st.subheader(f"Crypto Data for {crypto_ticker}")
            st.dataframe(crypto_data.tail(10))
            fig_crypto, ax_crypto = plt.subplots(figsize=(10, 6))
            ax_crypto.plot(crypto_data.index, crypto_data["close"], label="Close Price", color="blue")
            ax_crypto.set_title(f"{crypto_ticker} Price Chart")
            ax_crypto.set_xlabel("Time")
            ax_crypto.set_ylabel("Price ($)")
            ax_crypto.legend()
            st.pyplot(fig_crypto)
        except Exception as e:
            st.error(f"Error analyzing {crypto_ticker}: {e}")
    st.markdown(footer_html, unsafe_allow_html=True)
