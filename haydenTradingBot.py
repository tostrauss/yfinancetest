#Author: Hayden Hall
#Purpose: Trading bot alpaca API integration

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce


APCA_API_BASE_URL = 'https://paper-api.alpaca.markets' #paper trading url

trading_client =TradingClient('','', paper = False) #api-key, secret-key, paper = True to trade

#Get account informations
account = trading_client.get_account()
#Get a list of all positions
portfolio = trading_client.get_all_positions()
for position in portfolio:
    print("{} shares of {}".format(position.qty, position.symbol))

#Check if account is restricted from trading
if account.trading_blocked:
    print("Account is currently restricted from trading")

#Check buying power
print ('${} is available as buying power.'.format(account.buying_power))

#Check current balance vs balance at last market close
balance_change =float(account.equity) - float(account.last_equity)
print(f'Today\'s portfolio balance change is: ${balance_change}')

#Get a list of assets
search_params = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
assets = trading_client.get_all_assets(search_params)

"""

MARKET ORDER FORMAT

"""

# preparing market order
"""market_order_data = MarketOrderRequest(
                    symbol="", #ticker
                    qty=PLACEHOLDER, #number of shares
                    side=OrderSide.BUY, #leave alone
                    time_in_force=TimeInForce.DAY
                    client_order_id='order1',
                    )

# Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
               )
#Get order using its Client order ID
my_order= trading_client.get_order_by_client_id('order1')
print('Gotorder #{}'.format(my_order.id))
"""