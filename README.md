# Stock Data Analysis App

The Stock Data Analysis App is a Streamlit-based application that fetches live stock data from Yahoo Finance (using the yfinance library) and performs technical analysis. The app calculates popular indicators like the Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) and displays the results in interactive charts. You can use the app in three different modes:

1. **Single Ticker Analysis:**  
   Analyze one ticker at a time, view its historical RSI and MACD charts, and check the current indicator values.

2. **RSI Screening for Multiple Tickers (Comma-Separated List):**  
   Enter multiple tickers separated by commas. The app screens the tickers and displays those with a current RSI under 40. Shorter time intervals (e.g., 1m, 2m, 5m, etc.) are available for screening.

3. **Whole Market Screening (CSV Upload):**  
   Upload a CSV file containing a column named `Ticker` with your entire list of tickers. The app fetches live data for all tickers in the CSV and displays those with a current RSI under 40 using short interval options.

## Features

- **Live Data:**  
  Uses the [yfinance](https://pypi.org/project/yfinance/) library to fetch up-to-date stock data from Yahoo Finance.

- **Technical Analysis:**  
  Calculates RSI and MACD indicators, complete with interactive charts.

- **Flexible Screening Options:**  
  Choose from different data periods and intervals. For screening modes, short intervals (e.g., 1m, 2m, 5m, 15m, 30m) are available for finer analysis.

- **Parallel Data Fetching:**  
  Uses parallel processing to speed up screening for multiple tickers.

- **Error Handling:**  
  Logs and displays errors for tickers that cannot be processed, helping you identify issues with data or input.

## How to Use

### 1. Single Ticker Analysis

- **Input:** Enter a stock ticker (e.g., `AAPL`).
- **Options:**  
  - Choose a data period (e.g., `1d`, `5d`, `1mo`, etc.).
  - Choose an interval (e.g., `1m`, `2m`, `5m`, etc.).
  - Optionally check a box to display data only if the current RSI is under 40.
- **Output:**  
  - Displays the current RSI and MACD values.
  - Shows interactive charts for RSI and MACD. The MACD chart is accessible via an expander.

### 2. RSI Screening for Multiple Tickers (Comma-Separated List)

- **Input:** Enter multiple tickers separated by commas (e.g., `AAPL, MSFT, GOOG, AMZN`).
- **Options:**  
  - Select a data period suitable for screening.
  - Select a short interval (e.g., `1m`, `2m`, `5m`, etc.).
- **Output:**  
  - Displays a table with tickers that have a current RSI under 40, along with their MACD and signal values.
  - Any errors during processing are logged and shown in a separate table.

### 3. Whole Market Screening (CSV Upload)

- **Input:** Upload a CSV file that contains a column named `Ticker` with all the tickers you wish to screen.
- **Options:**  
  - Select a data period.
  - Select a short interval for screening.
- **Output:**  
  - Screens all tickers from the CSV and displays those with a current RSI under 40.
  - Provides feedback on processing time and error logs if any tickers fail to load.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/stock-data-analysis-app.git
   cd stock-data-analysis-app
