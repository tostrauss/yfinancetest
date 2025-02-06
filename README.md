# ToFu’s Stock Analysis & Options Trading App

This Streamlit application provides real-time stock analysis and options trading insights by fetching data from Yahoo Finance. It computes technical indicators such as RSI, MACD, Bollinger Bands, and various SMAs, and offers email notifications for critical RSI levels.

## Features

- **Stock Analysis:**  
  - View real-time stock data with technical indicators.
  - Display charts for price, RSI, MACD, and Bollinger Bands.
  - Select timeframes and intervals for customized analysis.

- **Options Trading:**  
  - Retrieve and display options chains (calls and puts) for a given ticker.
  - Overlay underlying stock charts with technical analysis to support option trading decisions.

- **Notification Subscription:**  
  - Subscribe to receive email alerts when a stock’s RSI crosses critical thresholds (RSI < 35 for oversold or RSI > 65 for overbought conditions).
  - Test notifications directly from the app.
  - (Note: In this demo, subscriptions are stored in a CSV file; a production system may use a database.)

- **Auto-Refresh:**  
  - Automatically refreshes the Notification Subscription page at regular intervals (every 5 minutes) for demo purposes.

## How It Works

1. **Data Fetching and Technical Analysis:**  
   The app leverages the [yfinance](https://pypi.org/project/yfinance/) library to fetch historical stock data. Several helper functions calculate technical indicators:
   - **RSI:** Uses rolling averages of gains and losses.
   - **MACD:** Computes exponential moving averages (EMAs) for short and long periods, then derives the signal line.
   - **Bollinger Bands:** Calculates the moving average and standard deviation to create upper and lower bands.
   - **Simple Moving Averages (SMA):** Computes SMAs over different windows (20, 50, 200).

2. **Multi-Page Layout:**  
   The sidebar navigation splits the app into three main sections:
   - **Stock Analysis:** Enter a ticker and select data period/interval to view charts and tables.
   - **Options Trading:** Enter a ticker and (optionally) an expiration date to view options chains along with an underlying stock chart.
   - **Notification Subscription:** Subscribe by providing an email and ticker. The app saves subscriptions and can test alert notifications based on current RSI levels.

3. **Parallel Processing:**  
   For efficiency, the app supports screening multiple tickers in parallel using Python’s `concurrent.futures` module.

4. **Email Notifications:**  
   When the RSI crosses set thresholds, the app formats and sends an email using SMTP. (Ensure to configure SMTP settings in the code for email functionality.)

5. **Auto-Refresh Mechanism:**  
   The app uses [`streamlit_autorefresh`](https://pypi.org/project/streamlit-autorefresh/) to re-run the subscription page automatically every 5 minutes—ideal for monitoring live data (in production, consider a background job scheduler).

## Requirements

- Python 3.7+
- [yfinance](https://pypi.org/project/yfinance/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [streamlit](https://streamlit.io/)
- [streamlit-autorefresh](https://pypi.org/project/streamlit-autorefresh/)

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
