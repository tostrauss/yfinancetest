import yfinance as yf
import pandas as pd 

# Define the ticker symbol for OKLO
ticker_symbol = "OKLO"

# Create a Ticker object for OKLO
oklo = yf.Ticker(ticker_symbol)

# 1. Company Information
print("=== Company Information ===")
info = oklo.info
print(info)  # prints a dictionary of key company details

# 2. Historical Market Data
print("\n=== Historical Market Data (Last 1 Year) ===")
# Fetch historical data over the past year. Adjust period as needed ('1mo', '5d', etc.)
history_data = oklo.history(period="1d", interval="1m")
print(history_data)
# Optionally, save the data to CSV:

# 3. Dividends and Splits
print("\n=== Dividends ===")
dividends = oklo.dividends
print(dividends)

print("\n=== Stock Splits ===")
splits = oklo.splits
print(splits)

# 4. Financial Statements
print("\n=== Financial Statements ===")
print("\nBalance Sheet:")
print(oklo.balance_sheet)

print("\nIncome Statement:")
print(oklo.financials)

print("\nCash Flow Statement:")
print(oklo.cashflow)

# 5. Earnings Data (if available)
print("\n=== Earnings Data ===")
# Annual earnings
print("Annual Earnings:")
print(oklo.earnings)
# Quarterly earnings can also be retrieved if needed:
print("Quarterly Earnings:")
print(oklo.quarterly_earnings)

# 6. Options Data (if available)
print("\n=== Options Data ===")
# Check if there are available options (some stocks might not have options data)
if oklo.options:
    # Get the first expiration date available
    first_expiration = oklo.options[0]
    print(f"Options Expiration Date: {first_expiration}")
    
    # Fetch the option chain for the first expiration date
    option_chain = oklo.option_chain(first_expiration)
    print("\nCalls:")
    print(option_chain.calls)
    print("\nPuts:")
    print(option_chain.puts)
else:
    print("No options data available for OKLO.")

