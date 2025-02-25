##########################################
# Happy Trading & Coding. It´s ToFu-Time #
##########################################

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import datetime
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from scipy.stats import norm

# -------------------------------
# HELPER FUNCTIONS & SETUP
# -------------------------------


def add_technical_indicators(data):
    if data is None or data.empty:
        return data
    # Ensure required columns are filled to avoid None values.
    for col in ["close", "high", "low"]:
        if col in data.columns:
            data[col] = data[col].ffill()
    
    try:
        data["RSI"] = ta.rsi(data["close"], length=14)
        
        macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            data["MACD"] = macd.get("MACD_12_26_9", np.nan)
            data["Signal"] = macd.get("MACDs_12_26_9", np.nan)
            data["MACD_Hist"] = macd.get("MACDh_12_26_9", np.nan)
        else:
            data["MACD"] = np.nan
            data["Signal"] = np.nan
            data["MACD_Hist"] = np.nan

        bb = ta.bbands(data["close"], length=20, std=2)
        if bb is not None:
            data["BBL"] = bb.get("BBL_20_2.0", np.nan)
            data["BBM"] = bb.get("BBM_20_2.0", np.nan)
            data["BBU"] = bb.get("BBU_20_2.0", np.nan)
        else:
            data["BBL"] = np.nan
            data["BBM"] = np.nan
            data["BBU"] = np.nan

        data["SMA20"] = ta.sma(data["close"], length=20)
        data["SMA50"] = ta.sma(data["close"], length=50)
        data["SMA200"] = ta.sma(data["close"], length=200)

        if {"high", "low", "close", "volume"}.issubset(data.columns):
            data["VWAP"] = ta.vwap(data["high"], data["low"], data["close"], data["volume"])

        adx = ta.adx(data["high"], data["low"], data["close"], length=14)
        if adx is not None:
            data["ADX"] = adx.get("ADX_14", np.nan)
        else:
            data["ADX"] = np.nan

        data["PP"] = (data["high"] + data["low"] + data["close"]) / 3
        data["R1"] = 2 * data["PP"] - data["low"]
        data["S1"] = 2 * data["PP"] - data["high"]

        data["Day_High"] = data["high"].cummax()
        data["Day_Low"] = data["low"].cummin()
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
    return data




def fetch_stock_data(ticker, period="1d", interval="1m"):
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        data.columns = [col.lower() for col in data.columns]
        data.index = data.index.tz_localize(None)
        data = add_technical_indicators(data)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return (np.nan,)*6
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
    except Exception as e:
        print(f"Error computing d1, d2: {e}")
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
        print(f"Error computing Greeks: {e}")
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
            return None, None, "No options data available.", []
        if expiration is None or expiration not in expirations:
            expiration = expirations[0]
        chain = ticker_obj.option_chain(expiration)
        return chain.calls, chain.puts, expiration, expirations
    except Exception as e:
        return None, None, f"Error retrieving options chain: {e}", []

def send_email_notification(to_email, subject, body, smtp_settings):
    SMTP_SERVER = smtp_settings.get("server", "")
    SMTP_PORT = smtp_settings.get("port", 587)
    SMTP_USER = smtp_settings.get("user", "")
    SMTP_PASSWORD = smtp_settings.get("password", "")
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
        print(f"Failed to send email: {e}")
        return False

def generate_tech_signal(rsi, rsi_buy=30, rsi_sell=70):
    if rsi < rsi_buy:
        return "STRONG BUY"
    elif rsi > rsi_sell:
        return "STRONG SELL"
    else:
        return "HOLD"

# -------------------------------
# BACKTESTING UTILS
# -------------------------------

def calculate_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    return drawdown, max_dd

def backtest_ma_rsi(data, short_window=20, long_window=50, rsi_buy=30, rsi_sell=70):
    """
    Backtest a moving average crossover strategy filtered by RSI.
    Buy when MA_short > MA_long and RSI < rsi_buy; sell when MA_short < MA_long and RSI > rsi_sell.
    """
    df = data.copy()
    if df.empty or "close" not in df.columns:
        print("Insufficient data for backtesting.")
        return df

    df["MA_short"] = df["close"].rolling(window=short_window, min_periods=1).mean()
    df["MA_long"] = df["close"].rolling(window=long_window, min_periods=1).mean()

    if "RSI" not in df.columns:
        try:
            df["RSI"] = ta.rsi(df["close"], length=14)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            df["RSI"] = np.nan

    df["Signal"] = 0
    df.loc[(df["MA_short"] > df["MA_long"]) & (df["RSI"] < rsi_buy), "Signal"] = 1
    df.loc[(df["MA_short"] < df["MA_long"]) & (df["RSI"] > rsi_sell), "Signal"] = -1

    # Forward-fill positions
    df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)

    df["MarketReturn"] = df["close"].pct_change()
    df["StrategyReturn"] = df["MarketReturn"] * df["Position"].shift(1)
    df["CumulativeMarket"] = (1 + df["MarketReturn"]).cumprod()
    df["CumulativeStrategy"] = (1 + df["StrategyReturn"].fillna(0)).cumprod()

    df["BuyPrice"] = np.where(df["Signal"] == 1, df["close"], np.nan)
    df["SellPrice"] = np.where(df["Signal"] == -1, df["close"], np.nan)
    return df

# -------------------------------
# DASH APP & LAYOUT SETUP
# -------------------------------

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "ToFu´s Stock Analysis & Options Trading"

store_components = [
    dcc.Store(id="watchlist-store", data=[]),
    dcc.Store(id="smtp-store", data={}),
    dcc.Store(id="options-store", data={}),
]

header = dbc.Navbar(
    dbc.Container([dbc.NavbarBrand("ToFu´s Stock Analysis & Options Trading", className="ms-2")]),
    color="primary", dark=True, sticky="top"
)

footer = html.Footer(
    dbc.Container(html.P("© 2025 Tobias Strauss", className="text-center text-muted my-2")),
    style={"position": "fixed", "left": "0", "bottom": "0", "width": "100%",
           "backgroundColor": "#f8f9fa", "padding": "10px 0", "boxShadow": "0 -2px 5px rgba(0,0,0,0.1)"}
)

# -------------------------------
# TAB LAYOUTS
# -------------------------------

def stock_analysis_layout():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H2("Real‑Time Stock Analysis", className="text-primary mb-3"))),
        dbc.Row([
            dbc.Col([
                dbc.Label("Enter Stock Ticker"),
                dcc.Input(id="stock-ticker-input", type="text", value="AAPL", className="form-control")
            ], md=4),
            dbc.Col([
                dbc.Label("Select Data Period"),
                dcc.Dropdown(id="stock-period",
                             options=[{"label": p, "value": p} for p in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]],
                             value="1d", className="form-select")
            ], md=3),
            dbc.Col([
                dbc.Label("Select Data Interval"),
                dcc.Dropdown(id="stock-interval",
                             options=[{"label": i, "value": i} for i in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"]],
                             value="1m", className="form-select")
            ], md=3),
            dbc.Col(dbc.Button("Analyze Stock", id="analyze-stock-button", color="primary", className="mt-4"), md=2)
        ], className="mb-3"),
        dbc.Row(dbc.Col(html.Div(id="stock-error", className="text-danger"))),
        dbc.Row(dbc.Col(dash_table.DataTable(id="stock-data-table", page_size=10, style_table={"overflowX": "auto"}))),
        dbc.Row(dbc.Col(dcc.Graph(id="price-chart"))),
        dbc.Row(dbc.Col(dcc.Graph(id="indicators-chart"))),
        dbc.Row(dbc.Col(dcc.Graph(id="adx-chart"))),
        dbc.Row(dbc.Col(html.Div(id="fundamentals-div", className="mt-3")))
    ], fluid=True)

def watchlist_layout():
    return dbc.Container([
        html.H2("Watchlist", className="text-primary mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Enter Ticker to Add"),
                dcc.Input(id="watchlist-input", type="text", value="AAPL", className="form-control")
            ], md=4),
            dbc.Col(dbc.Button("Add to Watchlist", id="add-watchlist", color="success", className="mt-4"), md=2)
        ], className="mb-3"),
        dbc.Row(dbc.Col(dbc.Button("Clear Watchlist", id="clear-watchlist", color="danger"), className="mb-3")),
        dbc.Row(dbc.Col(html.Div(id="watchlist-div")))
    ], fluid=True)

def options_trading_layout():
    return dbc.Container([
        html.H2("Options Trading Analysis & Greeks", className="text-primary mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Enter Stock Ticker for Options"),
            dcc.Input(id="option-ticker-input", type="text", value="AAPL", className="form-control")
        ], md=4)),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="expiration-div"))),
        dbc.Row(dbc.Col(dbc.Button("Get Option Chain", id="get-option-chain", color="primary"))),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="options-output")))
    ], fluid=True)

def smtp_server_layout():
    return dbc.Container([
        html.H2("SMTP Server Settings", className="text-primary mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("SMTP Server"),
                dcc.Input(id="smtp-server", type="text", value="smtp.example.com", className="form-control")
            ], md=4),
            dbc.Col([
                dbc.Label("SMTP Port"),
                dcc.Input(id="smtp-port", type="number", value=587, className="form-control")
            ], md=2)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("SMTP Username"),
                dcc.Input(id="smtp-user", type="text", value="your_email@example.com", className="form-control")
            ], md=4),
            dbc.Col([
                dbc.Label("SMTP Password"),
                dcc.Input(id="smtp-password", type="password", value="", className="form-control")
            ], md=4)
        ], className="mb-3"),
        dbc.Row(dbc.Col(dbc.Button("Save SMTP Settings", id="save-smtp", color="primary"))),
        dbc.Row(dbc.Col(html.Div(id="smtp-status", className="mt-2 text-success")))
    ], fluid=True)

def notification_subscription_layout():
    return dbc.Container([
        html.H2("RSI Notification Subscription", className="text-primary mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Enter Your Email Address"),
                dcc.Input(id="notify-email", type="email", value="", className="form-control")
            ], md=4),
            dbc.Col([
                dbc.Label("Enter Stock Ticker to Monitor"),
                dcc.Input(id="notify-ticker", type="text", value="AAPL", className="form-control")
            ], md=4)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Data Period for Monitoring"),
                dcc.Dropdown(id="notify-period",
                             options=[{"label": p, "value": p} for p in ["1d", "5d", "1mo"]],
                             value="1d", className="form-select")
            ], md=3),
            dbc.Col([
                dbc.Label("Select Data Interval"),
                dcc.Dropdown(id="notify-interval",
                             options=[{"label": i, "value": i} for i in ["1m", "5m", "15m", "30m"]],
                             value="1m", className="form-select")
            ], md=3)
        ], className="mb-3"),
        dbc.Row(dbc.Col(dbc.Button("Subscribe for RSI Alerts", id="subscribe-button", color="success"))),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Button("Test Notification Now", id="test-notify-button", color="primary"))),
        dbc.Row(dbc.Col(html.Div(id="notification-status", className="mt-2")))
    ], fluid=True)

def investment_information_layout():
    return dbc.Container([
        dcc.Markdown(r"""
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
                     
    ### -             
    """)
    ], fluid=True)

def set_option_calls_layout():
    return dbc.Container([
        html.H2("Set Option Calls", className="text-primary mb-3"),
        html.H4("Place Your Option Call Order"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Ticker"),
                dcc.Input(id="option-call-ticker", type="text", value="AAPL", className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Underlying Current Price (S)"),
                dcc.Input(id="option-call-S", type="number", value=100.0, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Strike Price (K)"),
                dcc.Input(id="option-call-K", type="number", value=105.0, className="form-control")
            ], md=3)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Option Premium Paid"),
                dcc.Input(id="option-premium", type="number", value=5.0, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Days to Expiration"),
                dcc.Input(id="days-to-expiration", type="number", value=30, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Risk-Free Interest Rate (annual, decimal)"),
                dcc.Input(id="risk-free-rate", type="number", value=0.01, step=0.001, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Implied Volatility (annual, decimal)"),
                dcc.Input(id="implied-vol", type="number", value=0.20, step=0.01, className="form-control")
            ], md=3)
        ], className="mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Number of Contracts"),
            dcc.Input(id="num-contracts", type="number", value=1, min=1, className="form-control")
        ], md=3)),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Button("Simulate Option Call", id="simulate-option-call", color="primary"))),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="option-call-output")))
    ], fluid=True)

def risk_reward_layout():
    return dbc.Container([
        html.H2("Risk/Reward Calculator", className="text-primary mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Select Trade Type"),
            dcc.Dropdown(id="trade-type",
                         options=[{"label": "Long", "value": "Long"}, {"label": "Short", "value": "Short"}],
                         value="Long", className="form-select")
        ], md=4)),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Entry Price"),
                dcc.Input(id="entry-price", type="number", value=100.0, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Stop Loss Price"),
                dcc.Input(id="stop-loss", type="number", value=95.0, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Target Price"),
                dcc.Input(id="target-price", type="number", value=110.0, className="form-control")
            ], md=3)
        ], className="mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Position Size"),
            dcc.Input(id="position-size", type="number", value=1, className="form-control")
        ], md=3)),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Button("Calculate Risk/Reward", id="calc-risk-reward", color="primary"))),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="risk-reward-output", className="mb-3"))),
        dbc.Row(dbc.Col(dcc.Graph(id="risk-reward-chart")))
    ], fluid=True)

def metrics_tracker_layout():
    return dbc.Container([
        html.H2("Metrics Tracker", className="text-primary mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Enter Stock Ticker"),
            dcc.Input(id="ms-ticker", type="text", value="AAPL", className="form-control")
        ], md=4)),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Button("Analyze Stock", id="analyze-ms", color="primary"))),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="ms-output"))),
        dbc.Row(dbc.Col(dcc.Graph(id="ms-rsi-chart")))
    ], fluid=True)

def bond_analysis_layout():
    return dbc.Container([
        html.H2("Bond Analysis", className="text-primary mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Face Value ($)"),
                dcc.Input(id="face-value", type="number", value=1000.0, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Coupon Rate (annual, %)"),
                dcc.Input(id="coupon-rate", type="number", value=5.0, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Years to Maturity"),
                dcc.Input(id="years-maturity", type="number", value=10, className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Yield to Maturity (annual, %)"),
                dcc.Input(id="ytm", type="number", value=4.0, className="form-control")
            ], md=3)
        ], className="mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Coupon Frequency"),
            dcc.Dropdown(id="coupon-frequency",
                         options=[
                             {"label": "Annual (1)", "value": 1},
                             {"label": "Semi-Annual (2)", "value": 2},
                             {"label": "Quarterly (4)", "value": 4}
                         ],
                         value=2, className="form-select")
        ], md=3)),
        html.Br(),
        dbc.Row(dbc.Col(dbc.Button("Calculate Bond Price", id="calc-bond", color="primary"))),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="bond-price-output"), className="mb-3")),
        dbc.Row(dbc.Col(dcc.Graph(id="bond-chart")))
    ], fluid=True)

def crypto_analysis_layout():
    return dbc.Container([
        html.H2("Crypto Analysis", className="text-primary mb-3"),
        dbc.Row(dbc.Col([
            dbc.Label("Enter Crypto Ticker (e.g., BTC-USD, ETH-USD)"),
            dcc.Input(id="crypto-ticker", type="text", value="BTC-USD", className="form-control")
        ], md=4)),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Label("Select Data Period"),
                dcc.Dropdown(id="crypto-period",
                             options=[{"label": p, "value": p} for p in ["1d", "5d", "1mo", "3mo", "6mo", "1y"]],
                             value="1d", className="form-select")
            ], md=3),
            dbc.Col([
                dbc.Label("Select Data Interval"),
                dcc.Dropdown(id="crypto-interval",
                             options=[{"label": i, "value": i} for i in ["1m", "5m", "15m", "30m", "1h", "1d"]],
                             value="1m", className="form-select")
            ], md=3)
        ], className="mb-3"),
        dbc.Row(dbc.Col(dbc.Button("Analyze Crypto", id="analyze-crypto", color="primary"))),
        html.Br(),
        dbc.Row(dbc.Col(html.Div(id="crypto-output"))),
        dbc.Row(dbc.Col(dcc.Graph(id="crypto-chart")))
    ], fluid=True)

def backtesting_layout():
    return dbc.Container([
        html.H2("Back Testing", className="text-primary mb-3"),
        html.P("Test a MA + RSI strategy on historical data. Buy when the short-term MA exceeds the long-term MA and RSI is below the oversold threshold; sell when the short-term MA falls below the long-term MA and RSI is above the overbought threshold."),
        dbc.Row([
            dbc.Col([
                dbc.Label("Enter Stock Ticker"),
                dcc.Input(id="backtest-ticker", type="text", value="AAPL", className="form-control")
            ], md=3),
            dbc.Col([
                dbc.Label("Select Data Period"),
                dcc.Dropdown(id="backtest-period",
                             options=[{"label": p, "value": p} for p in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]],
                             value="1y", className="form-select")
            ], md=2),
            dbc.Col([
                dbc.Label("Select Data Interval"),
                dcc.Dropdown(id="backtest-interval",
                             options=[{"label": i, "value": i} for i in ["1d", "1wk", "1mo"]],
                             value="1d", className="form-select")
            ], md=2),
            dbc.Col([
                dbc.Label("RSI Buy Level"),
                dcc.Input(id="rsi-buy-level", type="number", value=30, className="form-control")
            ], md=2),
            dbc.Col([
                dbc.Label("RSI Sell Level"),
                dcc.Input(id="rsi-sell-level", type="number", value=70, className="form-control")
            ], md=2),
            dbc.Col(dbc.Button("Run Backtest", id="run-backtest-button", color="primary", className="mt-4"), md=1)
        ], className="mb-3"),
        dbc.Row(dbc.Col(html.Div(id="backtest-error", className="text-danger"))),
        dbc.Row(dbc.Col(dcc.Graph(id="backtest-chart"))),
        dbc.Row(dbc.Col(dcc.Graph(id="equity-curve-chart"))),
        dbc.Row(dbc.Col(dcc.Graph(id="trade-profit-chart"))),
        dbc.Row(dbc.Col(html.Div(id="backtest-results", className="mt-3")))
    ], fluid=True)

# -------------------------------
# APP LAYOUT & TABS
# -------------------------------

app.layout = dbc.Container(
    store_components + [
        header,
        dbc.Tabs([
            dbc.Tab(label="Stock Analysis", tab_id="tab-stock"),
            dbc.Tab(label="Watchlist", tab_id="tab-watchlist"),
            dbc.Tab(label="Options Trading", tab_id="tab-options"),
            dbc.Tab(label="SMTP Server", tab_id="tab-smtp"),
            dbc.Tab(label="Notification Subscription", tab_id="tab-notify"),
            dbc.Tab(label="Investment Information", tab_id="tab-investment"),
            dbc.Tab(label="Set Option Calls", tab_id="tab-option-calls"),
            dbc.Tab(label="Risk/Reward Calculator", tab_id="tab-risk-reward"),
            dbc.Tab(label="Metrics Tracker", tab_id="tab-metrics"),
            dbc.Tab(label="Bond Analysis", tab_id="tab-bond"),
            dbc.Tab(label="Crypto Analysis", tab_id="tab-crypto"),
            dbc.Tab(label="Back Testing", tab_id="tab-backtest")
        ], id="tabs", active_tab="tab-stock", className="my-3"),
        html.Div(id="tabs-content", className="mb-5"),
        footer
    ],
    fluid=True
)

@app.callback(Output("tabs-content", "children"), [Input("tabs", "active_tab")])
def render_content(active_tab):
    if active_tab == "tab-stock":
        return stock_analysis_layout()
    elif active_tab == "tab-watchlist":
        return watchlist_layout()
    elif active_tab == "tab-options":
        return options_trading_layout()
    elif active_tab == "tab-smtp":
        return smtp_server_layout()
    elif active_tab == "tab-notify":
        return notification_subscription_layout()
    elif active_tab == "tab-investment":
        return investment_information_layout()
    elif active_tab == "tab-option-calls":
        return set_option_calls_layout()
    elif active_tab == "tab-risk-reward":
        return risk_reward_layout()
    elif active_tab == "tab-metrics":
        return metrics_tracker_layout()
    elif active_tab == "tab-bond":
        return bond_analysis_layout()
    elif active_tab == "tab-crypto":
        return crypto_analysis_layout()
    elif active_tab == "tab-backtest":
        return backtesting_layout()
    else:
        return html.Div("Page not found")

# -------------------------------
# CALLBACKS FOR STOCK ANALYSIS
# -------------------------------
@app.callback(
    [Output("stock-data-table", "data"),
     Output("stock-data-table", "columns"),
     Output("price-chart", "figure"),
     Output("indicators-chart", "figure"),
     Output("adx-chart", "figure"),
     Output("fundamentals-div", "children"),
     Output("stock-error", "children")],
    Input("analyze-stock-button", "n_clicks"),
    [State("stock-ticker-input", "value"),
     State("stock-period", "value"),
     State("stock-interval", "value")]
)
def update_stock_analysis(n_clicks, ticker, period, interval):
    if not n_clicks:
        return [[], [], {}, {}, {}, "", ""]
    try:
        data = fetch_stock_data(ticker, period, interval)
        if data.empty:
            return [[], [], {}, {}, {}, "", f"No data returned for ticker: {ticker}"]
        table_data = data.tail(10).reset_index().to_dict("records")
        table_cols = [{"name": i, "id": i} for i in data.tail(10).reset_index().columns]

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data["close"], mode="lines", name="Close Price"))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA20", line=dict(dash="dash", color="orange")))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA50", line=dict(dash="dash", color="green")))
        fig_price.add_trace(go.Scatter(x=data.index, y=data["SMA200"], mode="lines", name="SMA200", line=dict(dash="dash", color="red")))
        if "VWAP" in data.columns:
            fig_price.add_trace(go.Scatter(x=data.index, y=data["VWAP"], mode="lines", name="VWAP", line=dict(dash="dot", color="magenta")))
        if "BBL" in data.columns and "BBU" in data.columns:
            fig_price.add_trace(go.Scatter(x=data.index, y=data["BBL"], mode="lines", name="Bollinger Lower", line=dict(color="gray"), opacity=0.5))
            fig_price.add_trace(go.Scatter(x=data.index, y=data["BBU"], mode="lines", name="Bollinger Upper", line=dict(color="gray"), opacity=0.5, fill="tonexty"))
        last = data.iloc[-1]
        if "PP" in data.columns:
            fig_price.add_hline(y=last["PP"], line=dict(dash="dash", color="grey"), annotation_text="PP")
        if "R1" in data.columns:
            fig_price.add_hline(y=last["R1"], line=dict(dash="dash", color="red"), annotation_text="R1")
        if "S1" in data.columns:
            fig_price.add_hline(y=last["S1"], line=dict(dash="dash", color="green"), annotation_text="S1")
        fig_price.update_layout(title=f"{ticker} Price Chart ({period}, {interval})")

        fig_ind = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("RSI", "MACD"))
        fig_ind.add_trace(go.Scatter(x=data.index, y=data["RSI"], mode="lines", name="RSI"), row=1, col=1)
        fig_ind.add_hline(y=70, line=dict(dash="dash", color="red"), row=1, col=1)
        fig_ind.add_hline(y=30, line=dict(dash="dash", color="green"), row=1, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD"), row=2, col=1)
        fig_ind.add_trace(go.Scatter(x=data.index, y=data["Signal"], mode="lines", name="Signal"), row=2, col=1)
        fig_ind.update_layout(height=600, title_text="RSI and MACD")

        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(x=data.index, y=data["ADX"], mode="lines", name="ADX"))
        fig_adx.add_hline(y=25, line=dict(dash="dash", color="red"), annotation_text="Trend Threshold")
        fig_adx.update_layout(title="Average Directional Index (ADX)")

        try:
            info = yf.Ticker(ticker).info
            fundamentals = html.Div([
                html.P(f"Current Ratio: {info.get('currentRatio', 'N/A')}"),
                html.P(f"Debt to Equity Ratio: {info.get('debtToEquity', 'N/A')}"),
                html.P(f"Return on Equity (ROE): {info.get('returnOnEquity', 'N/A')}"),
                html.P(f"Gross Profit Margin: {info.get('grossMargins', 'N/A')}"),
                html.P(f"Net Profit Margin: {info.get('profitMargins', 'N/A')}"),
                html.P(f"Return on Assets (ROA): {info.get('returnOnAssets', 'N/A')}")
            ], className="mt-3")
        except Exception:
            fundamentals = html.P("Error fetching fundamental metrics.")

        return table_data, table_cols, fig_price, fig_ind, fig_adx, fundamentals, ""
    except Exception as e:
        return [[], [], {}, {}, {}, "", f"Error: {e}"]

# -------------------------------
# CALLBACKS FOR WATCHLIST
# -------------------------------
@app.callback(
    Output("watchlist-store", "data"),
    [Input("add-watchlist", "n_clicks"), Input("clear-watchlist", "n_clicks")],
    [State("watchlist-input", "value"), State("watchlist-store", "data")]
)
def update_watchlist(add_clicks, clear_clicks, ticker_input, stored_list):
    ctx = callback_context
    if not ctx.triggered:
        return stored_list
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "add-watchlist":
        ticker = ticker_input.strip().upper()
        if ticker and ticker not in stored_list:
            stored_list.append(ticker)
    elif button_id == "clear-watchlist":
        stored_list = []
    return stored_list

@app.callback(
    Output("watchlist-div", "children"),
    Input("watchlist-store", "data")
)
def display_watchlist(ticker_list):
    if not ticker_list:
        return html.P("No tickers in your watchlist.")
    rows = []
    for ticker in ticker_list:
        try:
            data = fetch_stock_data(ticker, period="1d", interval="1m")
            latest = data.iloc[-1] if not data.empty else {}
            rsi = latest.get("rsi", "N/A")
            info = yf.Ticker(ticker).info
            industry = info.get("industry", "N/A")
        except Exception:
            rsi = "Error"
            industry = "Error"
        rows.append({"Ticker": ticker, "Industry": industry, "RSI": rsi})
    df = pd.DataFrame(rows)
    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={"overflowX": "auto"}
    )
    return table

# -------------------------------
# CALLBACKS FOR OPTIONS TRADING
# -------------------------------
@app.callback(
    [Output("expiration-div", "children"), Output("options-store", "data")],
    Input("get-option-chain", "n_clicks"),
    State("option-ticker-input", "value")
)
def get_options(n_clicks, ticker):
    if not n_clicks:
        return "", {}
    calls, puts, exp_selected, expirations = get_option_chain(ticker)
    if not expirations:
        return html.P("No expiration dates available."), {}
    exp_options = [{"label": exp, "value": exp} for exp in expirations]
    store_data = {"ticker": ticker, "expirations": expirations}
    dropdown = dcc.Dropdown(id="expiration-dropdown", options=exp_options, value=expirations[0], className="form-select")
    return dropdown, store_data

@app.callback(
    Output("options-output", "children"),
    [Input("expiration-dropdown", "value")],
    State("option-ticker-input", "value")
)
def update_option_chain(expiration, ticker):
    if not expiration:
        return html.P("Please select an expiration date.")
    try:
        calls, puts, exp_selected, _ = get_option_chain(ticker, expiration)
    except Exception as e:
        return html.P(f"Error retrieving options: {e}")
    if calls is None or puts is None:
        return html.P(f"No options data available for {ticker}.")
    try:
        exp_date = datetime.datetime.strptime(expiration, "%Y-%m-%d")
        today = datetime.datetime.today()
        T = max((exp_date - today).days / 365.0, 0.001)
    except Exception:
        T = 0.001
    try:
        current_data = fetch_stock_data(ticker, period="1d", interval="1m")
        S = current_data["close"].iloc[-1]
    except Exception:
        S = np.nan
    try:
        calls = add_greeks(calls, S, T, r=0.01, option_type="call")
        puts = add_greeks(puts, S, T, r=0.01, option_type="put")
    except Exception:
        pass

    calls_table = dash_table.DataTable(
        data=calls.to_dict("records"),
        columns=[{"name": i, "id": i} for i in calls.columns],
        style_table={"overflowX": "auto"}
    )
    puts_table = dash_table.DataTable(
        data=puts.to_dict("records"),
        columns=[{"name": i, "id": i} for i in puts.columns],
        style_table={"overflowX": "auto"}
    )
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Call Options", "Put Options"))
    if not calls.empty:
        calls_sorted = calls.sort_values("strike")
        fig.add_trace(go.Scatter(x=calls_sorted["strike"], y=calls_sorted["BS_Price"],
                                 mode="markers+lines", name="Calls", marker=dict(color="blue")), row=1, col=1)
        fig.add_hline(y=S, line=dict(dash="dash", color="black"), row=1, col=1, annotation_text="Underlying Price")
    else:
        fig.add_annotation(text="No call options data", row=1, col=1)
    if not puts.empty:
        puts_sorted = puts.sort_values("strike")
        fig.add_trace(go.Scatter(x=puts_sorted["strike"], y=puts_sorted["BS_Price"],
                                 mode="markers+lines", name="Puts", marker=dict(color="red")), row=2, col=1)
        fig.add_hline(y=S, line=dict(dash="dash", color="black"), row=2, col=1, annotation_text="Underlying Price")
    else:
        fig.add_annotation(text="No put options data", row=2, col=1)
    fig.update_layout(title=f"Option Prices vs. Strike Price for Expiration: {expiration}")
    return html.Div([
        html.H5("Call Options"),
        calls_table,
        html.Hr(),
        html.H5("Put Options"),
        puts_table,
        dcc.Graph(figure=fig)
    ])

# -------------------------------
# CALLBACKS FOR SMTP SERVER
# -------------------------------
@app.callback(
    Output("smtp-status", "children"),
    Input("save-smtp", "n_clicks"),
    [State("smtp-server", "value"), State("smtp-port", "value"),
     State("smtp-user", "value"), State("smtp-password", "value")]
)
def save_smtp(n_clicks, server, port, user, password):
    if not n_clicks:
        return ""
    return "SMTP settings saved successfully!"

@app.callback(
    Output("smtp-store", "data"),
    Input("save-smtp", "n_clicks"),
    [State("smtp-server", "value"), State("smtp-port", "value"),
     State("smtp-user", "value"), State("smtp-password", "value")]
)
def update_smtp_store(n_clicks, server, port, user, password):
    if not n_clicks:
        return {}
    return {"server": server, "port": port, "user": user, "password": password}

# -------------------------------
# CALLBACK FOR NOTIFICATION SUBSCRIPTION
# -------------------------------
@app.callback(
    Output("notification-status", "children"),
    [Input("subscribe-button", "n_clicks"), Input("test-notify-button", "n_clicks")],
    [State("notify-email", "value"), State("notify-ticker", "value"),
     State("notify-period", "value"), State("notify-interval", "value"),
     State("smtp-store", "data")]
)
def handle_notification(subscribe_clicks, test_clicks, email, ticker, period, interval, smtp_settings):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "subscribe-button":
        if "@" not in email:
            return html.P("Please enter a valid email address.", className="text-danger")
        return html.P("Subscription successful! (For demo purposes.)", className="text-success")
    elif button_id == "test-notify-button":
        if not email or "@" not in email:
            return html.P("Please provide a valid email address.", className="text-danger")
        try:
            data = fetch_stock_data(ticker, period, interval)
            if data.empty:
                return html.P(f"No data for {ticker}", className="text-danger")
            latest = data.iloc[-1]
            rsi = latest.get("rsi", 50)
            message = f"Ticker: {ticker}\nCurrent Price: ${latest['close']:.2f}\nRSI: {rsi:.2f}"
            subject = f"Stock Alert for {ticker}"
            if rsi < 35 or rsi > 65:
                if smtp_settings:
                    success = send_email_notification(email, subject, message, smtp_settings)
                    if success:
                        return html.P("Notification sent successfully!", className="text-success")
                    else:
                        return html.P("Failed to send notification.", className="text-danger")
                else:
                    return html.P("SMTP settings not configured.", className="text-danger")
            else:
                return html.P("RSI is within normal range. No notification sent.", className="text-info")
        except Exception as e:
            return html.P(f"Error during notification: {e}", className="text-danger")
    return ""

# -------------------------------
# CALLBACKS FOR SET OPTION CALLS
# -------------------------------
@app.callback(
    Output("option-call-output", "children"),
    Input("simulate-option-call", "n_clicks"),
    [State("option-call-ticker", "value"), State("option-call-S", "value"),
     State("option-call-K", "value"), State("option-premium", "value"),
     State("days-to-expiration", "value"), State("risk-free-rate", "value"),
     State("implied-vol", "value"), State("num-contracts", "value")]
)
def simulate_option_call(n_clicks, ticker, S, K, premium, days, r, vol, contracts):
    if not n_clicks:
        return ""
    T = days / 365.0
    delta, gamma, theta, vega, rho, bs_price = black_scholes_greeks(S, K, T, r, vol, option_type="call")
    details = html.Div([
        html.P(f"Ticker: {ticker}"),
        html.P(f"Underlying Price (S): {S}"),
        html.P(f"Strike Price (K): {K}"),
        html.P(f"Option Premium Paid: {premium}"),
        html.P(f"Days to Expiration: {days}"),
        html.P(f"Risk-Free Rate: {r}"),
        html.P(f"Implied Volatility: {vol}"),
        html.P(f"Black-Scholes Theoretical Price: {bs_price:.2f}"),
        html.P(f"Delta: {delta:.2f}"),
        html.P(f"Gamma: {gamma:.4f}"),
        html.P(f"Theta (per day): {theta:.4f}"),
        html.P(f"Vega: {vega:.2f}"),
        html.P(f"Rho: {rho:.2f}"),
        html.P(f"Number of Contracts: {contracts}")
    ])
    contract_size = 100
    price_range = np.linspace(0.5 * S, 1.5 * S, 100)
    payoff = np.maximum(price_range - K, 0) - premium
    total_payoff = payoff * contract_size * contracts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=total_payoff, mode="lines", name="Profit / Loss"))
    fig.add_hline(y=0, line=dict(color="black", dash="dash"))
    fig.update_layout(title="Option Call Payoff at Expiration", xaxis_title="Underlying Price", yaxis_title="Profit / Loss ($)")
    return html.Div([details, dcc.Graph(figure=fig)])

# -------------------------------
# CALLBACKS FOR RISK/REWARD CALCULATOR
# -------------------------------
@app.callback(
    [Output("risk-reward-output", "children"), Output("risk-reward-chart", "figure")],
    Input("calc-risk-reward", "n_clicks"),
    [State("trade-type", "value"), State("entry-price", "value"),
     State("stop-loss", "value"), State("target-price", "value"),
     State("position-size", "value")]
)
def update_risk_reward(n_clicks, trade_type, entry, stop, target, pos_size):
    if not n_clicks:
        return "", {}
    if trade_type == "Long":
        risk = entry - stop
        reward = target - entry
    else:
        risk = stop - entry
        reward = entry - target
    if risk <= 0:
        output = "Invalid parameters: Risk must be positive."
        fig = {}
    else:
        ratio = reward / risk
        output = f"Risk per unit: ${risk:.2f} | Reward per unit: ${reward:.2f} | Risk/Reward Ratio: {ratio:.2f}"
        low_bound = min(stop, target) * 0.95
        high_bound = max(stop, target) * 1.05
        price_range = np.linspace(low_bound, high_bound, 100)
        if trade_type == "Long":
            profit_loss = (price_range - entry) * pos_size
        else:
            profit_loss = (entry - price_range) * pos_size
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_range, y=profit_loss, mode="lines", name="Profit / Loss"))
        fig.add_vline(x=entry, line=dict(dash="dash", color="black"), annotation_text="Entry")
        fig.add_vline(x=stop, line=dict(dash="dash", color="red"), annotation_text="Stop Loss")
        fig.add_vline(x=target, line=dict(dash="dash", color="green"), annotation_text="Target")
        fig.add_hline(y=0, line=dict(color="gray"))
        fig.update_layout(title="Risk/Reward Diagram", xaxis_title="Price", yaxis_title="Profit / Loss ($)")
    return output, fig

# -------------------------------
# CALLBACKS FOR METRICS TRACKER
# -------------------------------
@app.callback(
    [Output("ms-output", "children"), Output("ms-rsi-chart", "figure")],
    Input("analyze-ms", "n_clicks"),
    State("ms-ticker", "value")
)
def update_metrics_tracker(n_clicks, ticker):
    if not n_clicks:
        return "", {}
    try:
        data = fetch_stock_data(ticker, period="1d", interval="1m")
        if data.empty:
            return f"No data returned for ticker: {ticker}", {}
        latest = data.iloc[-1]
        rsi = latest.get("rsi", 50)
        recommendation = generate_tech_signal(rsi)
        output_text = f"Latest Close: ${latest['close']:.2f} | RSI: {rsi:.2f} | Recommendation: {recommendation}"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["RSI"], y=[rsi], marker_color="skyblue"))
        fig.add_hline(y=30, line=dict(dash="dash", color="green"), annotation_text="Oversold")
        fig.add_hline(y=70, line=dict(dash="dash", color="red"), annotation_text="Overbought")
        fig.update_layout(title="RSI Indicator", yaxis_title="RSI")
        return output_text, fig
    except Exception as e:
        return f"Error: {e}", {}

# -------------------------------
# CALLBACKS FOR BOND ANALYSIS
# -------------------------------
@app.callback(
    [Output("bond-price-output", "children"), Output("bond-chart", "figure")],
    Input("calc-bond", "n_clicks"),
    [State("face-value", "value"), State("coupon-rate", "value"),
     State("years-maturity", "value"), State("ytm", "value"), State("coupon-frequency", "value")]
)
def calculate_bond(n_clicks, face_value, coupon_rate, years, ytm, freq):
    if not n_clicks:
        return "", {}
    try:
        periods = int(years * freq)
        coupon_payment = face_value * (coupon_rate/100) / freq
        price = sum([coupon_payment / ((1 + (ytm/100) / freq) ** (i + 1)) for i in range(periods)])
        price += face_value / ((1 + (ytm/100) / freq) ** periods)
        text = f"Calculated Bond Price: ${price:.2f}"
        yields = np.linspace(0.01, 0.10, 100)
        prices = []
        for y in yields:
            p = sum([coupon_payment / ((1 + y / freq) ** (i + 1)) for i in range(periods)])
            p += face_value / ((1 + y / freq) ** periods)
            prices.append(p)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yields*100, y=prices, mode="lines", name="Bond Price Curve", line=dict(color="blue")))
        fig.add_vline(x=ytm, line=dict(dash="dash", color="red"), annotation_text="Selected Yield")
        fig.update_layout(title="Bond Price vs. Yield to Maturity", xaxis_title="Yield to Maturity (%)", yaxis_title="Bond Price ($)")
        return text, fig
    except Exception as e:
        return f"Error calculating bond: {e}", {}

# -------------------------------
# CALLBACKS FOR CRYPTO ANALYSIS
# -------------------------------
@app.callback(
    [Output("crypto-output", "children"), Output("crypto-chart", "figure")],
    Input("analyze-crypto", "n_clicks"),
    [State("crypto-ticker", "value"), State("crypto-period", "value"), State("crypto-interval", "value")]
)
def analyze_crypto(n_clicks, ticker, period, interval):
    if not n_clicks:
        return "", {}
    try:
        data = fetch_stock_data(ticker, period=period, interval=interval)
        if data.empty:
            return f"No data returned for {ticker}", {}
        table_data = data.tail(10).reset_index().to_dict("records")
        table_cols = [{"name": i, "id": i} for i in data.tail(10).reset_index().columns]
        table = dash_table.DataTable(data=table_data, columns=table_cols, style_table={"overflowX": "auto"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["close"], mode="lines", name="Close Price", line=dict(color="blue")))
        fig.update_layout(title=f"{ticker} Price Chart", xaxis_title="Time", yaxis_title="Price ($)")
        return table, fig
    except Exception as e:
        return f"Error analyzing {ticker}: {e}", {}

# -------------------------------
# CALLBACK FOR BACKTESTING (MA + RSI Strategy with Trade Profit Graph)
# -------------------------------
@app.callback(
    [Output("backtest-chart", "figure"),
     Output("equity-curve-chart", "figure"),
     Output("trade-profit-chart", "figure"),
     Output("backtest-results", "children"),
     Output("backtest-error", "children")],
    Input("run-backtest-button", "n_clicks"),
    [State("backtest-ticker", "value"), State("backtest-period", "value"),
     State("backtest-interval", "value"), State("rsi-buy-level", "value"),
     State("rsi-sell-level", "value")]
)
def run_backtest(n_clicks, ticker, period, interval, rsi_buy, rsi_sell):
    if not n_clicks:
        return {}, {}, {}, "", ""
    try:
        data = fetch_stock_data(ticker, period, interval)
        if data.empty:
            return {}, {}, {}, "", f"No data returned for {ticker}."
        result_df = backtest_ma_rsi(data, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
        
        # Price chart with buy and sell signals
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=result_df.index, y=result_df["close"], mode="lines", name="Close Price"))
        buy_points = result_df[result_df["Signal"] == 1]
        sell_points = result_df[result_df["Signal"] == -1]
        fig_price.add_trace(go.Scatter(x=buy_points.index, y=buy_points["BuyPrice"], mode="markers",
                                        name="Buy Signal", marker_symbol="triangle-up", marker_color="green", marker_size=10))
        fig_price.add_trace(go.Scatter(x=sell_points.index, y=sell_points["SellPrice"], mode="markers",
                                        name="Sell Signal", marker_symbol="triangle-down", marker_color="red", marker_size=10))
        fig_price.update_layout(title=f"Backtest Price Chart: {ticker}")
        
        # Equity curve chart
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=result_df.index, y=result_df["CumulativeMarket"], mode="lines", name="Buy & Hold"))
        fig_equity.add_trace(go.Scatter(x=result_df.index, y=result_df["CumulativeStrategy"], mode="lines", name="Strategy"))
        fig_equity.update_layout(title="Equity Curve Comparison")
        
        # Compute per-trade profits and create a bar chart
        trades = []
        trade_open = False
        buy_price = None
        buy_time = None
        for idx, row in result_df.iterrows():
            if not trade_open and row["Signal"] == 1:
                trade_open = True
                buy_price = row["close"]
                buy_time = idx
            elif trade_open and row["Signal"] == -1:
                sell_price = row["close"]
                sell_time = idx
                profit = sell_price - buy_price
                trades.append({"BuyTime": buy_time, "SellTime": sell_time, "Profit": profit})
                trade_open = False
        
        if trades:
            trade_df = pd.DataFrame(trades)
            fig_profit = go.Figure()
            fig_profit.add_trace(go.Bar(
                x=trade_df["BuyTime"],
                y=trade_df["Profit"],
                marker_color=np.where(trade_df["Profit"] >= 0, "green", "red"),
                name="Trade Profit"
            ))
            fig_profit.update_layout(title="Per-Trade Profit (Buy to Sell)", xaxis_title="Buy Time", yaxis_title="Profit ($)")
        else:
            fig_profit = go.Figure()
            fig_profit.update_layout(title="No complete trades to display profit.")
        
        final_strategy = result_df["CumulativeStrategy"].iloc[-1] - 1.0
        final_market = result_df["CumulativeMarket"].iloc[-1] - 1.0
        total_signals = int(result_df["Signal"].abs().sum())
        _, max_dd = calculate_drawdown(result_df["CumulativeStrategy"])
        results_text = [
            html.P(f"Final Strategy Return: {final_strategy*100:.2f}%"),
            html.P(f"Final Buy & Hold Return: {final_market*100:.2f}%"),
            html.P(f"Number of Signals (Trades): {total_signals}"),
            html.P(f"Max Drawdown (Strategy): {max_dd*100:.2f}%")
        ]
        return fig_price, fig_equity, fig_profit, results_text, ""
    except Exception as e:
        return {}, {}, {}, "", f"Error during backtest: {e}"

# -------------------------------
# RUN THE APP
# -------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
