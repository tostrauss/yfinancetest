// Implementation

+Implemented yfinance library using the 'pip install yfinance' command and openend up a new project.
+First File (oklo_data.py)

++ Implementation of yfinance library
- Integrated.

++ Implementation of streamlit framework
- Integrated.

++ Also integrated Pandas, numpy, matplotlib and MIME text service for email-notifications

1. First Page is about analyzing one ticker at a time, view its historical RSI and MACD charts, and check the current indicator values.
2. Second Page is the Options with newly implemented Greeks.
3. Third Page is a email notification/subscription system, where an SMTP Server still needs to be implemented to store data and refresh and to automatically send out a notification when the price drops below a RSI under 35 or hits a RSI over 65.

######
To-Do:

1. Watchlist (Stocks you want to have a look at closer, "add Ticker to watchlist", Compare RSI from Industry to Watchlist)
2. SMTP Server
3. Email Notifications

 “port” your full Streamlit application to use Alpaca as your data source and (hypothetical) broker. In this example, we replace the yfinance‐based data fetching with functions that call Alpaca’s REST API for historical bars (and—for options data—a try/fallback to yfinance if the Alpaca options endpoint isn’t available). We also add helper functions for order submission. (Note that some endpoints—especially for options orders—are shown as hypothetical because Alpaca’s options API is still emerging. You may need to adjust endpoints and parameters per your actual Alpaca account and API documentation.)




