import os
import io
import base64
import math
import datetime
import smtplib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from flask import Flask, render_template_string, request, redirect, url_for, flash, session, has_request_context
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from scipy.stats import norm
import pandas_ta as ta
import yfinance as yf
import alpaca_trade_api as tradeapi
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# ---------------------------
# SET DEFAULT ALPACA SETTINGS IN app.config
# ---------------------------
app.config['ALPACA_API_KEY'] = "AKAZIT5RT3SU1KNI8OQN"
app.config['ALPACA_API_SECRET'] = "xJAPz5u5gmod424tQbes1PqfzvkPymCmiWHnC8Ex"
app.config['ALPACA_BASE_URL'] = "https://paper-api.alpaca.markets"

# ---------------------------
# BASE TEMPLATE (using a content placeholder)
# ---------------------------
base_template = """
<!DOCTYPE html>
<html>
<head>
  <title>{{ title }}</title>
  <style>
    body {
      background-color: #013220;
      color: #fff;
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
    }
    .navbar {
      background-color: #004d00;
      padding: 10px;
    }
    .navbar a {
      color: #fff;
      margin-right: 15px;
      text-decoration: none;
      font-weight: bold;
    }
    .container {
      padding: 20px;
    }
    .button {
      background-color: red;
      border: none;
      color: white;
      padding: 10px 20px;
      text-align: center;
      text-decoration: none;
      display: inline-block;
      font-size: 16px;
      margin: 4px 2px;
      cursor: pointer;
      border-radius: 8px;
    }
    .footer {
      text-align: center;
      padding: 10px;
      background-color: #004d00;
      position: fixed;
      bottom: 0;
      width: 100%;
    }
    input, select, textarea {
      padding: 8px;
      border-radius: 4px;
      border: 1px solid #ccc;
    }
    label {
      display: block;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="navbar">
    <a href="{{ url_for('stock_analysis') }}">Stock Analysis</a>
    <a href="{{ url_for('watchlist') }}">Watchlist</a>
    <a href="{{ url_for('options_trading') }}">Options Trading</a>
    <a href="{{ url_for('smtp_server') }}">SMTP Server</a>
    <a href="{{ url_for('notification_subscription') }}">Notification Subscription</a>
    <a href="{{ url_for('investment_information') }}">Investment Information</a>
    <a href="{{ url_for('set_option_calls') }}">Set Option Calls</a>
    <a href="{{ url_for('risk_reward_calculator') }}">Risk/Reward Calculator</a>
    <a href="{{ url_for('metrics_sentiment_tracker') }}">Metrics & Sentiment Tracker</a>
    <a href="{{ url_for('bond_analysis') }}">Bond Analysis</a>
    <a href="{{ url_for('crypto_analysis') }}">Crypto Analysis</a>
  </div>
  <div class="container">
    {% for category, message in get_flashed_messages(with_categories=True) %}
      <div>{{ message }}</div>
    {% endfor %}
    {{ content|safe }}
  </div>
  <div class="footer">
    © 2025 Tobias Strauss
  </div>
</body>
</html>
"""

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def get_alpaca_api():
    return tradeapi.REST(
        app.config.get('ALPACA_API_KEY'),
        app.config.get('ALPACA_API_SECRET'),
        app.config.get('ALPACA_BASE_URL'),
        api_version="v2"
    )

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
        flash(f"Error adding technical indicators: {e}", "error")
    return data

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
        flash(f"Alpaca API error: {e}. Falling back to yfinance for {ticker}.", "warning")
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data returned for ticker: {ticker}")
            data.columns = [col.lower() for col in data.columns]
            data.index = data.index.tz_localize(None)
        except Exception as e2:
            flash(f"Error fetching data for {ticker} via yfinance: {e2}", "error")
            raise
    try:
        data = add_technical_indicators(data)
    except Exception as e:
        flash(f"Error processing data for {ticker}: {e}", "error")
        raise
    return data

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (np.nan,)*6
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
    except Exception as e:
        flash(f"Error computing d1, d2: {e}", "error")
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
        flash(f"Error computing Greeks: {e}", "error")
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

def send_email_notification(to_email, subject, body):
    SMTP_SERVER = session.get("SMTP_SERVER", "")
    SMTP_PORT = session.get("SMTP_PORT", 587)
    SMTP_USER = session.get("SMTP_USER", "")
    SMTP_PASSWORD = session.get("SMTP_PASSWORD", "")
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
        flash(f"Failed to send email: {e}", "error")
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
                flash("Notification sent successfully!", "success")
            else:
                flash("Failed to send notification.", "error")
        else:
            flash("RSI is within normal range. No notification sent.", "info")
    except Exception as e:
        flash(f"Error during notification: {e}", "error")

# ---------------------------
# SENTIMENT ANALYSIS
# ---------------------------
finbert = None
def load_finbert():
    global finbert
    try:
        finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
    except Exception as e:
        if has_request_context():
            flash(f"Error loading FinBERT: {e}", "error")
        else:
            print(f"Error loading FinBERT: {e}")
        finbert = None

load_finbert()

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
            flash("Error fetching news data.", "error")
            return []
        articles = data.get("articles", [])
        headlines = [article["title"] for article in articles if article.get("title")]
        return headlines
    except Exception as e:
        flash(f"Error in NewsAPI request: {e}", "error")
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
            flash(f"Error analyzing sentiment for a headline: {e}", "error")
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
        flash(f"Order submitted: {order}", "success")
    except Exception as e:
        flash(f"Error placing order: {e}", "error")

# ---------------------------
# HELPER: CONVERT FIGURE TO BASE64 IMAGE
# ---------------------------
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return img_data

# ---------------------------
# ROUTES
# ---------------------------

@app.route('/')
def index():
    return redirect(url_for('stock_analysis'))

# Page 1: Real‑Time Stock Analysis
@app.route('/stock_analysis', methods=['GET', 'POST'])
def stock_analysis():
    chart_images = {}
    table_html = ""
    if request.method == 'POST':
        ticker = request.form.get("ticker", "AAPL").upper()
        period = request.form.get("period", "1d")
        interval = request.form.get("interval", "1m")
        try:
            data = fetch_stock_data(ticker, period, interval)
            # Price chart
            fig_price, ax_price = plt.subplots(figsize=(8,4))
            ax_price.plot(data.index, data["close"], label="Close Price", color="blue")
            if "SMA20" in data.columns:
                ax_price.plot(data.index, data["SMA20"], label="SMA20", linestyle="--", color="orange")
            if "SMA50" in data.columns:
                ax_price.plot(data.index, data["SMA50"], label="SMA50", linestyle="--", color="green")
            if "SMA200" in data.columns:
                ax_price.plot(data.index, data["SMA200"], label="SMA200", linestyle="--", color="red")
            if "VWAP" in data.columns:
                ax_price.plot(data.index, data["VWAP"], label="VWAP", linestyle="-.", color="magenta")
            if "BBL" in data.columns and "BBU" in data.columns:
                ax_price.fill_between(data.index, data["BBL"], data["BBU"], color="gray", alpha=0.2, label="Bollinger Bands")
            if "PP" in data.columns:
                ax_price.axhline(y=data["PP"].iloc[-1], label="Pivot Point (PP)", color="grey", linestyle="--")
            ax_price.set_title(f"{ticker} - Price Chart")
            ax_price.legend()
            chart_images["price_chart"] = plot_to_img(fig_price)
            
            # RSI and MACD
            fig_indicators, (ax_rsi, ax_macd) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
            if "RSI" in data.columns:
                ax_rsi.plot(data.index, data["RSI"], label="RSI", color="purple")
                ax_rsi.axhline(70, color="red", linestyle="--", label="Overbought (70)")
                ax_rsi.axhline(30, color="green", linestyle="--", label="Oversold (30)")
                ax_rsi.set_ylabel("RSI")
                ax_rsi.legend()
            if "MACD" in data.columns and "Signal" in data.columns:
                ax_macd.plot(data.index, data["MACD"], label="MACD", color="brown")
                ax_macd.plot(data.index, data["Signal"], label="Signal", color="magenta")
                ax_macd.set_ylabel("MACD")
                ax_macd.legend()
            chart_images["indicators_chart"] = plot_to_img(fig_indicators)
            
            # ADX chart
            fig_adx, ax_adx = plt.subplots(figsize=(8,4))
            if "ADX" in data.columns:
                ax_adx.plot(data.index, data["ADX"], label="ADX", color="brown")
                ax_adx.axhline(25, color="red", linestyle="--", label="Trend Threshold (25)")
                ax_adx.set_title("Average Directional Index (ADX)")
                ax_adx.legend()
            chart_images["adx_chart"] = plot_to_img(fig_adx)
            
            table_html = data.tail(10).to_html(classes="table", border=1)
        except Exception as e:
            flash(f"Error analyzing {ticker}: {e}", "error")
    
    page_content = """
    <h2>Real‑Time Stock Analysis</h2>
    <form method="POST">
      <label for="ticker">Enter Stock Ticker:</label>
      <input type="text" name="ticker" value="AAPL" required>
      <label for="period">Select Data Period:</label>
      <select name="period">
        <option value="1d">1d</option>
        <option value="5d">5d</option>
        <option value="1mo">1mo</option>
        <option value="3mo">3mo</option>
        <option value="6mo">6mo</option>
        <option value="1y">1y</option>
        <option value="2y">2y</option>
        <option value="5y">5y</option>
        <option value="10y">10y</option>
        <option value="max">max</option>
      </select>
      <label for="interval">Select Data Interval:</label>
      <select name="interval">
        <option value="1m">1m</option>
        <option value="2m">2m</option>
        <option value="5m">5m</option>
        <option value="15m">15m</option>
        <option value="30m">30m</option>
        <option value="60m">60m</option>
        <option value="90m">90m</option>
        <option value="1h">1h</option>
        <option value="1d">1d</option>
      </select>
      <button class="button" type="submit">Analyze Stock</button>
    </form>
    {% if table_html %}
      <h3>Latest Data</h3>
      <div>{{ table_html|safe }}</div>
    {% endif %}
    {% for key, img in chart_images.items() %}
      <h3>{{ key.replace('_', ' ').title() }}</h3>
      <img src="data:image/png;base64,{{ img }}" alt="{{ key }}">
    {% endfor %}
    """
    return render_template_string(base_template, title="Stock Analysis", content=page_content, table_html=table_html, chart_images=chart_images)

# Page 2: Watchlist
@app.route('/watchlist', methods=['GET', 'POST'])
def watchlist():
    if "watchlist" not in session:
        session["watchlist"] = []
    if request.method == 'POST':
        if 'add_ticker' in request.form:
            new_ticker = request.form.get("new_ticker", "").strip().upper()
            if new_ticker and new_ticker not in session["watchlist"]:
                watchlist = session["watchlist"]
                watchlist.append(new_ticker)
                session["watchlist"] = watchlist
                flash(f"{new_ticker} added to watchlist!", "success")
            else:
                flash("Ticker is already in your watchlist or invalid.", "info")
        elif 'clear_watchlist' in request.form:
            session["watchlist"] = []
            flash("Watchlist cleared!", "info")
    watchlist_data = []
    for ticker in session.get("watchlist", []):
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
        table_html = df_watchlist.to_html(classes="table", border=1, index=False)
    else:
        table_html = ""
    page_content = """
    <h2>Watchlist</h2>
    <form method="POST">
      <label for="new_ticker">Enter Ticker to Add:</label>
      <input type="text" name="new_ticker" value="AAPL" required>
      <button class="button" type="submit" name="add_ticker">Add to Watchlist</button>
      <button class="button" type="submit" name="clear_watchlist">Clear Watchlist</button>
    </form>
    {% if table_html %}
      <h3>Your Watchlist</h3>
      <div>{{ table_html|safe }}</div>
    {% else %}
      <p>Your watchlist is empty. Please add tickers.</p>
    {% endif %}
    """
    return render_template_string(base_template, title="Watchlist", content=page_content, table_html=table_html)

# Page 3: Options Trading & Greeks
@app.route('/options_trading', methods=['GET', 'POST'])
def options_trading():
    option_data = {}
    if request.method == 'POST':
        ticker = request.form.get("ticker_option", "AAPL").upper()
        expiration_selected = request.form.get("expiration")
        if not expiration_selected:
            calls, puts, expiration_selected = get_option_chain(ticker)
        else:
            calls, puts, expiration_selected = get_option_chain(ticker, expiration_selected)
        if isinstance(expiration_selected, str) and "Error" in expiration_selected:
            flash(expiration_selected, "error")
        else:
            try:
                exp_date = datetime.datetime.strptime(expiration_selected, "%Y-%m-%d")
                today = datetime.datetime.today()
                T = max((exp_date - today).days / 365.0, 0.001)
            except Exception as e:
                flash(f"Error parsing expiration date: {e}", "error")
                T = 0.001
            try:
                current_data = fetch_stock_data(ticker, period="1d", interval="1m")
                S = current_data["close"].iloc[-1]
            except Exception as e:
                flash(f"Error retrieving current price for {ticker}: {e}", "error")
                S = np.nan
            if not calls.empty:
                calls = add_greeks(calls, S, T, r=0.01, option_type='call')
                calls_html = calls.to_html(classes="table", border=1)
            else:
                calls_html = "No call options data available."
            if not puts.empty:
                puts = add_greeks(puts, S, T, r=0.01, option_type='put')
                puts_html = puts.to_html(classes="table", border=1)
            else:
                puts_html = "No put options data available."
            fig, (ax_calls, ax_puts) = plt.subplots(2, 1, figsize=(8,8), sharex=True)
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
            option_chart = plot_to_img(fig)
            option_data = {
                "ticker": ticker,
                "expiration": expiration_selected,
                "calls_html": calls_html,
                "puts_html": puts_html,
                "option_chart": option_chart
            }
    page_content = """
    <h2>Options Trading Analysis & Greeks</h2>
    <form method="POST">
      <label for="ticker_option">Enter Stock Ticker for Options:</label>
      <input type="text" name="ticker_option" value="AAPL" required>
      <label for="expiration">Expiration Date (optional, YYYY-MM-DD):</label>
      <input type="text" name="expiration" placeholder="YYYY-MM-DD">
      <button class="button" type="submit">Get Option Chain</button>
    </form>
    {% if option_data %}
      <h3>Options Data for Expiration: {{ option_data.expiration }}</h3>
      <h4>Call Options</h4>
      <div>{{ option_data.calls_html|safe }}</div>
      <h4>Put Options</h4>
      <div>{{ option_data.puts_html|safe }}</div>
      <h4>Option Price vs. Strike Price</h4>
      <img src="data:image/png;base64,{{ option_data.option_chart }}" alt="Option Chart">
    {% endif %}
    """
    return render_template_string(base_template, title="Options Trading", content=page_content, option_data=option_data)

# Page 4: SMTP Server Settings
@app.route('/smtp_server', methods=['GET', 'POST'])
def smtp_server():
    if request.method == 'POST':
        session["SMTP_SERVER"] = request.form.get("smtp_server", "smtp.example.com")
        session["SMTP_PORT"] = int(request.form.get("smtp_port", 587))
        session["SMTP_USER"] = request.form.get("smtp_user", "your_email@example.com")
        session["SMTP_PASSWORD"] = request.form.get("smtp_password", "")
        flash("SMTP settings saved!", "success")
    page_content = """
    <h2>SMTP Server Settings</h2>
    <form method="POST">
      <label for="smtp_server">SMTP Server:</label>
      <input type="text" name="smtp_server" value="{{ session.get('SMTP_SERVER', 'smtp.example.com') }}" required>
      <label for="smtp_port">SMTP Port:</label>
      <input type="number" name="smtp_port" value="{{ session.get('SMTP_PORT', 587) }}" required>
      <label for="smtp_user">SMTP Username:</label>
      <input type="text" name="smtp_user" value="{{ session.get('SMTP_USER', 'your_email@example.com') }}" required>
      <label for="smtp_password">SMTP Password:</label>
      <input type="password" name="smtp_password" value="{{ session.get('SMTP_PASSWORD', '') }}" required>
      <button class="button" type="submit">Save SMTP Settings</button>
    </form>
    """
    return render_template_string(base_template, title="SMTP Server", content=page_content)

# Page 5: Notification Subscription & Testing
@app.route('/notification_subscription', methods=['GET', 'POST'])
def notification_subscription():
    if request.method == 'POST':
        subscription_email = request.form.get("subscription_email", "")
        ticker_notify = request.form.get("ticker_notify", "AAPL").upper()
        period_notify = request.form.get("period_notify", "1d")
        interval_notify = request.form.get("interval_notify", "1m")
        if "subscribe" in request.form:
            if not subscription_email or "@" not in subscription_email:
                flash("Please enter a valid email address.", "error")
            else:
                flash("Subscription successful! (For demo, click 'Test Notification' to simulate an alert.)", "success")
        elif "test_notification" in request.form:
            if subscription_email and ticker_notify:
                enhanced_notification(ticker_notify, subscription_email, period_notify, interval_notify)
            else:
                flash("Please provide both an email and a ticker to monitor.", "error")
    page_content = """
    <h2>RSI Notification Subscription</h2>
    <form method="POST">
      <label for="subscription_email">Enter Your Email Address:</label>
      <input type="email" name="subscription_email" required>
      <label for="ticker_notify">Enter Stock Ticker to Monitor:</label>
      <input type="text" name="ticker_notify" value="AAPL" required>
      <label for="period_notify">Select Data Period for Monitoring:</label>
      <select name="period_notify">
        <option value="1d">1d</option>
        <option value="5d">5d</option>
        <option value="1mo">1mo</option>
      </select>
      <label for="interval_notify">Select Data Interval:</label>
      <select name="interval_notify">
        <option value="1m">1m</option>
        <option value="5m">5m</option>
        <option value="15m">15m</option>
        <option value="30m">30m</option>
      </select>
      <button class="button" type="submit" name="subscribe">Subscribe for RSI Alerts</button>
      <button class="button" type="submit" name="test_notification">Test Notification Now</button>
    </form>
    """
    return render_template_string(base_template, title="Notification Subscription", content=page_content)

# Page 6: Investment Information (Static Content)
@app.route('/investment_information')
def investment_information():
    content = """
    <h1>Investment Analysis & Fundamentals</h1>
    <hr>
    <h2>1. Options Fundamentals</h2>
    <p>- Gives the right to buy or sell an asset at a predetermined price.</p>
    <h2>2. Hedging with Options</h2>
    <p>- Protective Put, Covered Call, Collar Strategy.</p>
    <h2>3. Options Strategies</h2>
    <p>- Butterfly Spread, Condor Spread.</p>
    <h2>4. Financial Ratios and Metrics</h2>
    <p>- Current Ratio, Debt to Equity, ROE, Gross/Net Profit Margins, ROA.</p>
    """
    return render_template_string(base_template, title="Investment Information", content=content)

# Page 7: Set Option Calls
@app.route('/set_option_calls', methods=['GET', 'POST'])
def set_option_calls():
    result = {}
    if request.method == 'POST':
        ticker_call = request.form.get("ticker_call", "AAPL").upper()
        try:
            current_data_call = fetch_stock_data(ticker_call, period="1d", interval="1m")
            S_default = current_data_call["close"].iloc[-1]
        except Exception as e:
            S_default = 100.0
        S_call = float(request.form.get("S_call", S_default))
        strike_call = float(request.form.get("strike_call", S_default * 1.05))
        premium_call = float(request.form.get("premium_call", 5.0))
        days_to_expiration = int(request.form.get("days_to_expiration", 30))
        risk_free_rate = float(request.form.get("risk_free_rate", 0.01))
        implied_vol = float(request.form.get("implied_vol", 0.20))
        num_contracts = int(request.form.get("num_contracts", 1))
        T_call = days_to_expiration / 365.0
        delta, gamma, theta, vega, rho, bs_price = black_scholes_greeks(S_call, strike_call, T_call, risk_free_rate, implied_vol, option_type='call')
        result = {
            "ticker_call": ticker_call,
            "S_call": S_call,
            "strike_call": strike_call,
            "premium_call": premium_call,
            "days_to_expiration": days_to_expiration,
            "risk_free_rate": risk_free_rate,
            "implied_vol": implied_vol,
            "bs_price": bs_price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho,
            "num_contracts": num_contracts
        }
        contract_size = 100
        price_range = np.linspace(0.5 * S_call, 1.5 * S_call, 100)
        payoff = np.maximum(price_range - strike_call, 0) - premium_call
        total_payoff = payoff * contract_size * num_contracts
        fig_payoff, ax_payoff = plt.subplots(figsize=(8,4))
        ax_payoff.plot(price_range, total_payoff, label="Profit / Loss")
        ax_payoff.axhline(0, color="black", linestyle="--")
        ax_payoff.set_xlabel("Underlying Price at Expiration")
        ax_payoff.set_ylabel("Profit / Loss ($)")
        ax_payoff.set_title("Option Call Payoff at Expiration")
        ax_payoff.legend()
        result["payoff_chart"] = plot_to_img(fig_payoff)
    page_content = """
    <h2>Set Option Calls</h2>
    <form method="POST">
      <label for="ticker_call">Ticker:</label>
      <input type="text" name="ticker_call" value="AAPL" required>
      <label for="S_call">Underlying Current Price (S):</label>
      <input type="number" step="0.1" name="S_call" value="{{ request.form.get('S_call', '') }}">
      <label for="strike_call">Strike Price (K):</label>
      <input type="number" step="0.1" name="strike_call" value="{{ request.form.get('strike_call', '') }}">
      <label for="premium_call">Option Premium Paid:</label>
      <input type="number" step="0.1" name="premium_call" value="5.0">
      <label for="days_to_expiration">Days to Expiration:</label>
      <input type="number" name="days_to_expiration" value="30">
      <label for="risk_free_rate">Risk-Free Interest Rate:</label>
      <input type="number" step="0.001" name="risk_free_rate" value="0.01">
      <label for="implied_vol">Implied Volatility:</label>
      <input type="number" step="0.01" name="implied_vol" value="0.20">
      <label for="num_contracts">Number of Contracts:</label>
      <input type="number" name="num_contracts" value="1">
      <button class="button" type="submit">Simulate Option Call</button>
    </form>
    {% if result %}
      <h3>Option Call Details and Greeks</h3>
      <p><strong>Ticker:</strong> {{ result.ticker_call }}</p>
      <p><strong>Underlying Price (S):</strong> {{ result.S_call }}</p>
      <p><strong>Strike Price (K):</strong> {{ result.strike_call }}</p>
      <p><strong>Premium Paid:</strong> {{ result.premium_call }}</p>
      <p><strong>Days to Expiration:</strong> {{ result.days_to_expiration }}</p>
      <p><strong>Risk-Free Rate:</strong> {{ result.risk_free_rate }}</p>
      <p><strong>Implied Volatility:</strong> {{ result.implied_vol }}</p>
      <p><strong>Black-Scholes Theoretical Price:</strong> {{ result.bs_price|round(2) }}</p>
      <p><strong>Delta:</strong> {{ result.delta|round(2) }}</p>
      <p><strong>Gamma:</strong> {{ result.gamma|round(4) }}</p>
      <p><strong>Theta (per day):</strong> {{ result.theta|round(4) }}</p>
      <p><strong>Vega:</strong> {{ result.vega|round(2) }}</p>
      <p><strong>Rho:</strong> {{ result.rho|round(2) }}</p>
      <p><strong>Number of Contracts:</strong> {{ result.num_contracts }}</p>
      <h3>Simulated Payoff at Expiration</h3>
      <img src="data:image/png;base64,{{ result.payoff_chart }}" alt="Payoff Chart">
    {% endif %}
    """
    return render_template_string(base_template, title="Set Option Calls", content=page_content, result=result)

# Page 8: Risk/Reward Calculator
@app.route('/risk_reward_calculator', methods=['GET', 'POST'])
def risk_reward_calculator():
    result = {}
    if request.method == 'POST':
        trade_type = request.form.get("trade_type", "Long")
        entry_price = float(request.form.get("entry_price", 100.0))
        stop_loss = float(request.form.get("stop_loss", 95.0))
        target_price = float(request.form.get("target_price", 110.0))
        position_size = int(request.form.get("position_size", 1))
        if trade_type == "Long":
            risk = entry_price - stop_loss
            reward = target_price - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - target_price
        if risk <= 0:
            flash("Invalid parameters: Risk must be positive.", "error")
        else:
            risk_reward_ratio = reward / risk
            result["risk"] = risk
            result["reward"] = reward
            result["ratio"] = risk_reward_ratio
            low_bound = min(stop_loss, target_price) * 0.95
            high_bound = max(stop_loss, target_price) * 1.05
            price_range = np.linspace(low_bound, high_bound, 100)
            if trade_type == "Long":
                profit_loss = (price_range - entry_price) * position_size
            else:
                profit_loss = (entry_price - price_range) * position_size
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(price_range, profit_loss, label="Profit / Loss", color="blue")
            ax.axvline(entry_price, color="black", linestyle="--", label="Entry Price")
            ax.axvline(stop_loss, color="red", linestyle="--", label="Stop Loss")
            ax.axvline(target_price, color="green", linestyle="--", label="Target Price")
            ax.axhline(0, color="gray", linestyle="-")
            ax.set_xlabel("Price")
            ax.set_ylabel("Profit / Loss ($)")
            ax.set_title("Risk/Reward Diagram")
            ax.legend()
            result["chart"] = plot_to_img(fig)
            result["entry_price"] = entry_price
            result["stop_loss"] = stop_loss
            result["target_price"] = target_price
            result["position_size"] = position_size
    page_content = """
    <h2>Risk/Reward Calculator</h2>
    <form method="POST">
      <label for="trade_type">Select Trade Type:</label>
      <select name="trade_type">
        <option value="Long">Long</option>
        <option value="Short">Short</option>
      </select>
      <label for="entry_price">Entry Price:</label>
      <input type="number" step="0.1" name="entry_price" value="100.0" required>
      <label for="stop_loss">Stop Loss Price:</label>
      <input type="number" step="0.1" name="stop_loss" value="95.0" required>
      <label for="target_price">Target Price:</label>
      <input type="number" step="0.1" name="target_price" value="110.0" required>
      <label for="position_size">Position Size:</label>
      <input type="number" name="position_size" value="1" required>
      <button class="button" type="submit">Calculate</button>
    </form>
    {% if result %}
      <p><strong>Risk per unit:</strong> ${{ result.risk|round(2) }}</p>
      <p><strong>Reward per unit:</strong> ${{ result.reward|round(2) }}</p>
      <p><strong>Risk/Reward Ratio:</strong> {{ result.ratio|round(2) }}</p>
      <h3>Risk/Reward Diagram</h3>
      <img src="data:image/png;base64,{{ result.chart }}" alt="Risk Reward Chart">
    {% endif %}
    """
    return render_template_string(base_template, title="Risk/Reward Calculator", content=page_content, result=result)

# Page 9: Metrics & Sentiment Tracker
@app.route('/metrics_sentiment_tracker', methods=['GET', 'POST'])
def metrics_sentiment_tracker():
    analysis = {}
    if request.method == 'POST':
        ticker_ms = request.form.get("ticker_ms", "AAPL").upper()
        api_key_news = "c5c9ea5b981f4c6ab85badcf610fba78"
        try:
            data = fetch_stock_data(ticker_ms, period="1d", interval="1m")
            if data is not None:
                latest = data.iloc[-1]
                rsi = latest.get("RSI", 50)
                analysis["technical"] = f"Close: {latest['close']:.2f} | RSI: {rsi:.2f}"
            else:
                rsi = 50
        except Exception as e:
            flash(f"Error fetching technical data: {e}", "error")
            rsi = 50
        today = datetime.datetime.now()
        yesterday = today - datetime.timedelta(days=1)
        from_date = yesterday.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        headlines = fetch_news(ticker_ms, api_key_news, from_date, to_date)
        analysis["headlines"] = headlines
        avg_sentiment = compute_average_sentiment(headlines)
        analysis["avg_sentiment"] = avg_sentiment
        recommendation = generate_trading_signal(rsi, avg_sentiment)
        analysis["recommendation"] = recommendation
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["RSI"], [rsi], color="skyblue")
        ax.axhline(30, color="green", linestyle="--", label="Oversold (30)")
        ax.axhline(70, color="red", linestyle="--", label="Overbought (70)")
        ax.set_ylabel("RSI")
        ax.legend()
        analysis["chart"] = plot_to_img(fig)
        analysis["rsi"] = rsi
    page_content = """
    <h2>Metrics & Sentiment Stock Tracker</h2>
    <form method="POST">
      <label for="ticker_ms">Enter Stock Ticker:</label>
      <input type="text" name="ticker_ms" value="AAPL" required>
      <button class="button" type="submit">Analyze Stock</button>
    </form>
    {% if analysis %}
      <h3>Latest Technical Data</h3>
      <p>{{ analysis.technical }}</p>
      <h3>Fetched News Headlines:</h3>
      {% if analysis.headlines %}
        <ul>
          {% for head in analysis.headlines %}
            <li>{{ head }}</li>
          {% endfor %}
        </ul>
      {% else %}
        <p>No recent news found.</p>
      {% endif %}
      <p>Average Sentiment Score: {{ analysis.avg_sentiment|round(2) }} (scale: -1 negative, +1 positive)</p>
      <h2>Recommendation: <strong>{{ analysis.recommendation }}</strong></h2>
      <h3>RSI Chart</h3>
      <img src="data:image/png;base64,{{ analysis.chart }}" alt="RSI Chart">
    {% endif %}
    """
    return render_template_string(base_template, title="Metrics & Sentiment Tracker", content=page_content, analysis=analysis)

# Page 10: Bond Analysis
@app.route('/bond_analysis', methods=['GET', 'POST'])
def bond_analysis():
    result = {}
    if request.method == 'POST':
        face_value = float(request.form.get("face_value", 1000.0))
        coupon_rate = float(request.form.get("coupon_rate", 5.0)) / 100.0
        years_to_maturity = int(request.form.get("years_to_maturity", 10))
        yield_to_maturity = float(request.form.get("yield_to_maturity", 4.0)) / 100.0
        coupon_frequency = int(request.form.get("coupon_frequency", 2))
        periods = years_to_maturity * coupon_frequency
        coupon_payment = face_value * coupon_rate / coupon_frequency
        price = sum([coupon_payment / ((1 + yield_to_maturity / coupon_frequency) ** (i + 1)) for i in range(int(periods))])
        price += face_value / ((1 + yield_to_maturity / coupon_frequency) ** periods)
        result["price"] = price
        yields = np.linspace(0.01, 0.10, 100)
        prices = []
        for y in yields:
            p = sum([coupon_payment / ((1 + y / coupon_frequency) ** (i + 1)) for i in range(int(periods))])
            p += face_value / ((1 + y / coupon_frequency) ** periods)
            prices.append(p)
        fig_bond, ax_bond = plt.subplots(figsize=(8,4))
        ax_bond.plot(yields * 100, prices, label="Bond Price Curve", color="blue")
        ax_bond.axvline(yield_to_maturity * 100, color="red", linestyle="--", label="Selected Yield")
        ax_bond.set_xlabel("Yield to Maturity (%)")
        ax_bond.set_ylabel("Bond Price ($)")
        ax_bond.set_title("Bond Price vs. Yield to Maturity")
        ax_bond.legend()
        result["chart"] = plot_to_img(fig_bond)
    page_content = """
    <h2>Bond Analysis</h2>
    <form method="POST">
      <label for="face_value">Face Value ($):</label>
      <input type="number" step="0.1" name="face_value" value="1000.0" required>
      <label for="coupon_rate">Coupon Rate (annual, %):</label>
      <input type="number" step="0.1" name="coupon_rate" value="5.0" required>
      <label for="years_to_maturity">Years to Maturity:</label>
      <input type="number" name="years_to_maturity" value="10" required>
      <label for="yield_to_maturity">Yield to Maturity (annual, %):</label>
      <input type="number" step="0.1" name="yield_to_maturity" value="4.0" required>
      <label for="coupon_frequency">Coupon Frequency:</label>
      <select name="coupon_frequency">
        <option value="1">1</option>
        <option value="2" selected>2</option>
        <option value="4">4</option>
      </select>
      <button class="button" type="submit">Calculate Bond Price</button>
    </form>
    {% if result.price %}
      <p>Calculated Bond Price: ${{ result.price|round(2) }}</p>
      <h3>Bond Price vs. Yield</h3>
      <img src="data:image/png;base64,{{ result.chart }}" alt="Bond Chart">
    {% endif %}
    """
    return render_template_string(base_template, title="Bond Analysis", content=page_content, result=result)

# Page 11: Crypto Analysis
@app.route('/crypto_analysis', methods=['GET', 'POST'])
def crypto_analysis():
    chart_img = ""
    table_html = ""
    if request.method == 'POST':
        crypto_ticker = request.form.get("crypto_ticker", "BTC-USD").upper()
        crypto_period = request.form.get("crypto_period", "1d")
        crypto_interval = request.form.get("crypto_interval", "1m")
        try:
            crypto_data = fetch_stock_data(crypto_ticker, period=crypto_period, interval=crypto_interval)
            table_html = crypto_data.tail(10).to_html(classes="table", border=1)
            fig_crypto, ax_crypto = plt.subplots(figsize=(8,4))
            ax_crypto.plot(crypto_data.index, crypto_data["close"], label="Close Price", color="blue")
            ax_crypto.set_title(f"{crypto_ticker} Price Chart")
            ax_crypto.set_xlabel("Time")
            ax_crypto.set_ylabel("Price ($)")
            ax_crypto.legend()
            chart_img = plot_to_img(fig_crypto)
        except Exception as e:
            flash(f"Error analyzing {crypto_ticker}: {e}", "error")
    page_content = """
    <h2>Crypto Analysis</h2>
    <form method="POST">
      <label for="crypto_ticker">Enter Crypto Ticker (e.g., BTC-USD):</label>
      <input type="text" name="crypto_ticker" value="BTC-USD" required>
      <label for="crypto_period">Select Data Period:</label>
      <select name="crypto_period">
        <option value="1d">1d</option>
        <option value="5d">5d</option>
        <option value="1mo">1mo</option>
        <option value="3mo">3mo</option>
        <option value="6mo">6mo</option>
        <option value="1y">1y</option>
      </select>
      <label for="crypto_interval">Select Data Interval:</label>
      <select name="crypto_interval">
        <option value="1m">1m</option>
        <option value="5m">5m</option>
        <option value="15m">15m</option>
        <option value="30m">30m</option>
        <option value="1h">1h</option>
        <option value="1d">1d</option>
      </select>
      <button class="button" type="submit">Analyze Crypto</button>
    </form>
    {% if table_html %}
      <h3>Crypto Data</h3>
      <div>{{ table_html|safe }}</div>
      <h3>Price Chart</h3>
      <img src="data:image/png;base64,{{ chart_img }}" alt="Crypto Chart">
    {% endif %}
    """
    return render_template_string(base_template, title="Crypto Analysis", content=page_content, chart_img=chart_img, table_html=table_html)

# ---------------------------
# RUN THE APP
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
