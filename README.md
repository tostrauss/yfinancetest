# ToFu´s Stock Analysis and Option Trading

**ToFu´s Stock Analysis and Option Trading** is a comprehensive, multi-page web application designed for traders and financial analysts from Tobias Strauss. Built with [Streamlit](https://streamlit.io/) and powered by live data from [Yahoo Finance](https://finance.yahoo.com/), this application provides real-time stock analysis and options trading insights—complete with advanced technical indicators and Black–Scholes-based Greeks calculations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [GitHub Download & Installation](#github-download--installation)
- [Usage](#usage)
- [Pages Overview](#pages-overview)
  - [1. Stock Analysis](#1-stock-analysis)
  - [2. Options Trading](#2-options-trading)
  - [3. Notification Subscription](#3-notification-subscription)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

*ToFu´s Stock Analysis and Option Trading* is an all‑in‑one tool that caters to both equity and options traders. The application delivers:

- **Real-time stock data** with technical indicators such as RSI, MACD, Bollinger Bands, and multiple Simple Moving Averages (SMA20, SMA50, SMA200).
- **Options chain analysis** with detailed Black–Scholes Greeks (Delta, Gamma, Theta, Vega, Rho) and estimated option prices.
- A **visual dashboard** that includes interactive charts and graphs for both stock prices and option pricing.
- An **email notification system** to alert you when key technical indicators (e.g., RSI) breach predefined thresholds.

## Features

- **Real-Time Stock Analysis**  
  - Displays live intraday data (e.g., AAPL with a 1‑day period at 1‑minute intervals).
  - Computes and plots technical indicators: RSI, MACD, Bollinger Bands, and SMAs.
  - Shows daily high/low values and supports multiple intervals and periods.
  
- **Options Trading Analysis**  
  - Retrieves available options expiration dates and lets you select one.
  - Fetches options chain data and calculates Black–Scholes Greeks.
  - Provides detailed tables for call and put options.
  - Includes a dual-panel graph:
    - One panel for call options and one for put options.
    - Each panel shows a smooth line plot of Black–Scholes estimated option prices versus strike prices.
    - A vertical dashed line indicates the current underlying price.
  
- **Notification Subscription**  
  - Allows you to subscribe for email alerts when a stock’s RSI crosses critical thresholds (RSI < 35 or RSI > 65).
  - Email alerts include detailed metrics like current price, volume, and key moving averages.
  
- **Robust Error Handling & Troubleshooting**  
  - Detailed error messages and troubleshooting output.
  - Flexible configuration options to suit various trading strategies.

## GitHub Download & Installation

### Download from GitHub

You can download *ToFu´s Stock Analysis and Option Trading* from GitHub in two ways:

1. **Clone the Repository**  
   Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/tostrauss/ToFu-Stock-Analysis-Option-Trading.git
   cd ToFu-Stock-Analysis-Option-Trading

## Installation

Make sure you have Python 3.7 or later installed, then dependencies:
pip install streamlit yfinance pandas numpy matplotlib scripy

## Usage

Just run the following command in your terminal:
streamlit run app.py

## Contributing
More than welcome! Just fork and submit pull requests :)

## License 
This project is licensed under the MIT License. 

## Contact
-Name: Tobias Strauss
-Email: tobias.p.strauss@gmail.com

