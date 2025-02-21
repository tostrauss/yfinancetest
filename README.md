# ToFu’s Stock Analysis & Options Trading

Welcome to ToFu’s Stock Analysis & Options Trading – a comprehensive, interactive Dash application designed to help you analyze stocks in real time, trade options using Black‑Scholes and technical indicators, and backtest your trading strategies using a robust MA+RSI framework.

---

## 🚀 Key Features Overview

### 📊 **Real-Time Stock Analysis**
- **Live Stock Data:** Fetches historical and real-time stock data via `yfinance`.
- **Technical Indicators:** Uses `pandas-ta` to compute:
  - **RSI:** Detects overbought/oversold conditions.
  - **MACD:** Shows trend momentum and signal line.
  - **Bollinger Bands:** Displays price volatility.
  - **SMA (20, 50, 200-day):** Highlights short-, medium-, and long-term trends.
  - **VWAP:** Provides a volume-weighted average price (fair value).
  - **ADX:** Measures the strength of the trend.
  - **Pivot Points:** Identifies key support and resistance levels.

### ✅ **Interactive Watchlist**
- **Personalized Watchlist:** Add or clear your favorite stocks.
- **Real-Time Metrics:** View live RSI values and industry information.

### 💹 **Options Trading & Black-Scholes Analysis**
- **Options Chain Retrieval:** Pulls options data for a given stock.
- **Greeks Calculation:** Computes Black-Scholes theoretical prices and Greeks (Delta, Gamma, Theta, Vega, Rho) for call and put options.
- **Interactive Charts:** Visualize option pricing and the relationship between strike prices and theoretical values.

### 📧 **Automated Email Alerts for RSI**
- **Custom Alerts:** Receive email notifications when RSI crosses defined thresholds (overbought/oversold).
- **SMTP Configuration:** Easily configure your SMTP settings to enable alert delivery.

### 📖 **Investment Learning Hub**
- **Educational Content:** Includes explanations of options strategies (e.g., Covered Calls, Protective Puts, Spreads) and essential financial metrics (ROE, profit margins, debt-to-equity).
  
### 🧮 **Options Call Simulations**
- **Simulated Trades:** Simulate call option orders with parameters such as strike price, premium, days to expiration, risk-free rate, and implied volatility.
- **Profit/Loss Analysis:** Generate payoff charts to visualize potential profit or loss at expiration.

### ⚖️ **Risk/Reward Calculator**
- **Trade Analysis:** Calculate risk per unit, reward per unit, and risk/reward ratio.
- **Visual Diagram:** Generate interactive charts to display trade profit/loss scenarios and support effective decision-making.

### 📊 **Metrics Tracker**
- **RSI Analysis:** Analyze live RSI data and generate recommendations (e.g., STRONG BUY, STRONG SELL, HOLD).

### 🔗 **Bond & Crypto Analysis**
- **Bond Analysis:** Calculate bond prices based on face value, coupon rate, yield to maturity, and more.
- **Crypto Analysis:** Analyze cryptocurrencies (e.g., BTC-USD, ETH-USD) with similar technical tools.

### 📈 **Robust Backtesting Module**
- **MA + RSI Strategy:** Backtest a moving average crossover strategy filtered by RSI.
  - **Buy Signal:** When the short-term moving average exceeds the long-term moving average **and** RSI is below the buy threshold.
  - **Sell Signal:** When the short-term moving average falls below the long-term moving average **and** RSI is above the sell threshold.
- **Cumulative Returns:** Compare the strategy’s equity curve against a simple buy-and-hold approach.
- **Per-Trade Profit Analysis:** Calculate and visualize individual trade profits (from buy to sell) using an interactive bar chart.
- **Performance Metrics:** Displays key metrics including final strategy return, buy-and-hold return, number of signals (trades), and maximum drawdown.

### 🗂️ **Intuitive Multi-Page Navigation**
- Easily switch between pages:
  - **Stock Analysis**, **Watchlist**, **Options Trading**, **SMTP Setup**, **RSI Alerts**, **Investment Information**, **Call Simulations**, **Risk/Reward Calculator**, **Metrics Tracker**, **Bond Analysis**, **Crypto Analysis**, and **Back Testing**.

---

## 🛠️ Setup & Installation

### 1️⃣ **Clone and Install Dependencies:**

```bash
git clone https://github.com/tostrauss/your-dash-stock-analysis.git
cd your-dash-stock-analysis
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ **Run the Application:**
```bash
python dashTry.py
```

### 3️⃣ (Optional) Docker Option:
```bash
docker-compose up --build
```

### ✉️ Sample Email Alert
```bash
Subject: Stock Alert for AAPL - RSI Alert

Ticker: AAPL
Current Price: $179.25
Volume: 2,150,000
SMA20: $175.43 | SMA50: $172.80 | SMA200: $165.20
RSI: 68.45 (Overbought)
```

### 📈 Backtesting Module Metrics
- **Cumulative Strategy Return:** Overall percentage return of the MA+RSI strategy.
- **Cumulative Market Return:** Return from a simple buy-and-hold approach.
- **Number of Signals (Trades):** Total buy and sell events generated.
- **Max Drawdown:** Maximum observed loss from a peak in the strategy’s equity curve.
- **Per-Trade Profit Analysis:** Bar chart visualization of profit (or loss) per trade based on -paired buy and sell signals.

### 📜 License
This project is licensed under the MIT License.

### Created with ❤️ by Tobias Strauss 🚀



