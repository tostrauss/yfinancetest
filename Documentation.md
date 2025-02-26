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

1. Educational Onboarding
* Begin with a “Learn” section or tutorial:
* Basic concepts of trading, differences between asset classes, quick knowledge checks.
* Possibly short videos or interactive slides.

2. Strategy Builder & Simulations ("Header")
* Provide an interface (“Build”) to pick which assets, define signals (e.g., RSI < 30 => Buy), set day-trade toggles, etc.
* Let them run a backtest on historical or even randomized data.

3. Analysis & Visualization
* Use Charts & Indicators to let User see performance, distribution of returns, drawdowns, etc.
* Possibly incorporate advanced “Math” or risk metrics.
* Offer an “Explore” mode for deeper data inspection.

4. Go Live
* Switch from “paper trading” to real brokerage (if desired).
* Display a Portfolio tab to track open trades, overall P/L, and so on.

5. User Profiling
* (Optional) Let them input risk tolerance, goals, or prior experience.
* Could shape the content or disclaimers for each user.


# Potential Next Steps
1. MVP: Start with the “Learn” module and a simple “Build + Backtest” flow. This will give users immediate value in exploring trading ideas.

2. Incremental Features:
* Add advanced indicators or an AI-driven “strategy advisor.”
* Integrate a brokerage for real trading.
* Expand user profile features for more personalized guidance.


UI Flow:
-Onboarding / Profile Setup
-Learn & Quiz
-Build a Strategy (Indicators, etc.)
-Backtest & Evaluate (Charts, P/L, distribution)
-Option to Paper Trade or Connect Real Broker


Website (Starting with Login/Sign-Up) -> 
1. Page = Learn/Lessons 
                (*Asset classes:
                    - Stocks (Bluechip vs. small/med/large cap & pennystocks), (Bid/Ask spread), (Order types), (Dividends)
                    - Bonds (Rolling over), (Auctions)
                 *Types of trades:
                    - Long, Short, Options, Arbitrage
                 *Indicators/Date:
                    - Earnings Reports/ Earnings
                    - Bollinger Bands
                    - News
                    - Candles
                    - Market sentiment 
                    - Projected Growth
                    - Analyst Indicators
                    - Cash Flow
                    - Debt/Equity
                 *Strategies:
                    -Pros/Cons of Daytrading/Swingtrading/Long-Term Trading/Long and short only
                    -Complex Options
                    -Event driven
                    -Algorithmic
                    -Systematic
                    -Value driven
                    -Analyst driven
                    -Industry driven)

Starting off with Login/Sign-Up, creating new account (username, email, password, how much money you want to invest, Risktolerance, Goals) => store the information in a database to set up new account, click on "Learn" -> Pop-up ("Welcome to the learning page. Take the example test to determine your expertise."). Make it personal!!

Alpaca! Paper Trading and possible option for real brokerage!

