ToFu’s Stock Analysis & Options Trading - README.md

🚀 Key Features Overview

📊 Real-Time Stock Analysis

Displays live stock prices with technical indicators from pandas-ta:

RSI: Detects overbought/oversold signals.

MACD: Highlights trend momentum.

Bollinger Bands: Shows price volatility.

SMA (20, 50, 200-day): Reveals short- and long-term trends.

VWAP: Indicates fair market value.

ADX: Measures trend strength.

Pivot Points: Identifies support and resistance.

✅ Interactive Watchlist

Add favorite stocks to your personal watchlist.

View live RSI values and compare against industry averages.

💹 Options Trading & Black-Scholes Analysis

Retrieves options chains using yfinance.

Calculates Black-Scholes option pricing and Greeks:

Delta, Gamma, Theta, Vega, Rho with estimated option price.

Displays option pricing with interactive charts.

📧 Automated Email Alerts for RSI

Sends alerts if RSI > 65 (Overbought) or RSI < 35 (Oversold).

Configurable SMTP settings for customized notifications.

📖 Investment Learning Hub

Explains options strategies: Covered Calls, Protective Puts, Spreads.

Breaks down essential financial metrics: ROE, Profit Margins, Debt-to-Equity.

🧮 Options Call Simulations

Simulates call option payoffs with adjustable parameters.

Displays profit/loss charts for easy strategy analysis.

🗂️ Intuitive Multi-Page Navigation

Navigate easily between pages:

Stock Analysis, Watchlist, Options Trading

SMTP Setup, RSI Alerts, Learning Hub, Call Simulations

🛠️ Setup & Installation

1️⃣ Install Dependencies:

git clone https://github.com/tostrauss/streamlit-stock-analysis.git
cd streamlit-stock-analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2️⃣ Run the Application:

streamlit run streamlit_app.py

3️⃣ Docker Option:

docker-compose up --build

✉️ Sample Email Alert:

Subject: Stock Alert for AAPL - RSI Alert

Ticker: AAPL
Current Price: $179.25
Volume: 2,150,000
SMA20: $175.43 | SMA50: $172.80 | SMA200: $165.20
RSI: 68.45 (Overbought)

📜 License

This project is licensed under the MIT License.

Created with ❤️ by Tobias Strauss 🚀

