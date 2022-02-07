import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from math import floor
from termcolor import colored as cl
import yfinance
from datetime import datetime
from pytz import timezone
import pytz

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)

rsi_upper_bound = 70
rsi_lower_bound = 30
stoch_rsi_upper_bound = 80
stoch_rsi_lower_bound = 20

max_profit = 100
max_loss = 50
total_profit = 0
total_loss = 0

# To control whole of stop loss and profit taking - Master flag
have_stop_loss = False

# To control only profit taking, have_stop_loss flag must still be True for this.
take_profit_flag = False

# To Control 200 EMA check when taking Trades.
# Enabling this will allow taking the following trades only:
##      Only BUY when the PRICE IS ABOVE 200 EMA and OVERSOLD
##      Only SELL when the PRICE IS BELOW 200 EMA and OVERBOUGHT
enable_ema_check = False

def get_historical_data(symbols, interval = None, start_date = None, end_date = None ):
    #api_key = open(r'api_key.txt')
    #api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=B1070L97KPM1MSGC&outputsize=full'
    df = yfinance.download(tickers = symbols, period = "1d", interval = "2m")
    #df = yfinance.download(tickers = symbols, start="2022-02-01", end="2022-02-02", interval = "1m")
    return df

def ema(data, period = 20):
    #return data.ewm(span=period, adjust=False).mean()
    return data.ewm(com=period, adjust=False).mean()

def ema_span(data, period = 20):
    return data.ewm(span=period, adjust=True).mean()

def calculate_ema(data, days, smoothing=2):
    ema = [sum(data[:days]) / days]
    for price in data[days:]:
        ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
    return ema

def calc_rsi(close, ema_lookback = 14):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = ema(up_series, period = ema_lookback-1)
    down_ewm = ema(down_series, period = ema_lookback-1)
    # up_ewm = up_series.ewm(com = ema_lookback - 1, adjust = False).mean()
    # down_ewm = down_series.ewm(com = ema_lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))

    return rsi

def get_rsi(close, ema_lookback = 14):
    rsi = calc_rsi(close, ema_lookback)
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()

    return rsi_df[3:]

def stock_rsi(close, period = 14, column = 'Close'):
#     delta = data[column].diff(1)
#     delta = delta.dropna()
#     up = delta.copy()
#     down = delta.copy()
#     up[up<0] = 0
#     down[down>0] = 0
#     data["up"] = up
#     data["down"] = down
#     AVG_GAIN = ema(data, period, column = 'up')
#     AVG_LOSS = abs(ema(data, period, column = 'down'))
#     RS = AVG_GAIN / AVG_LOSS
#     RSI = 100.0 - (100.0/(1.0+RS))
    rsi = calc_rsi(close, period)
    stock_rsi = ((rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())) * 100
    stock_rsi_df = pd.DataFrame(stock_rsi).rename(columns = {0:'stock_rsi'}).set_index(close.index)
    stock_rsi_df = stock_rsi_df.dropna()
    return stock_rsi_df


nq = get_historical_data("NQ=F")
nq['rsi_20'] = get_rsi(nq['Close'], 14)
nq['stock_rsi_20'] = stock_rsi(nq['Close'], 14)
nq = nq.dropna()

#print(ema(nq['Close'], 200), '200 ema df')
#print(calculate_ema(nq['Close'], 200), '200 ema manual')

# Converting nq timings to pacific and during market hours
start="06:30:00"
End="13:00:00"
market_hours = [datetime.strptime(start,"%H:%M:%S").time(), datetime.strptime(End,"%H:%M:%S").time()]
nq = nq.reset_index()
nq['Date'] = nq['Datetime'].apply(lambda x:x.tz_convert("US/Pacific").date())
nq['Datetime'] = nq['Datetime'].apply(lambda x:x.tz_convert("US/Pacific").time())
nq = nq[nq['Datetime'] > market_hours[0]][nq['Datetime'] < market_hours[1]]
nq['Datetime'] = nq[['Datetime', 'Date']].apply(lambda x:pd.to_datetime(x[1].strftime("%m/%d/%Y") + " " + x[0].strftime("%H:%M:%S")), axis = 1)
nq = nq.set_index('Datetime')
#print(nq)



#plot the charts with rsi
ax1 = plt.subplot2grid((15,1), (0,0), rowspan = 10, colspan = 1)
# ax2 = plt.subplot2grid((15,1), (5,0), rowspan = 4, colspan = 1)
ax3 = plt.subplot2grid((15,1), (11,0), rowspan = 7, colspan = 1)
ax1.plot(nq.index, nq['Close'], linewidth = 2.5)
ax1.set_title('NQ CLOSE PRICE')
# ax2.plot(nq.index, nq['rsi_20'], color = 'orange', linewidth = 2.5)
# ax2.axhline(rsi_lower_bound, linestyle = '--', linewidth = 1.5, color = 'grey')
# ax2.axhline(rsi_upper_bound, linestyle = '--', linewidth = 1.5, color = 'grey')
# ax2.set_title('NQ RELATIVE STRENGTH INDEX')
ax3.plot(nq.index, nq['stock_rsi_20'], color = 'orange', linewidth = 2.5)
ax3.axhline(20, linestyle = '--', linewidth = 1.5, color = 'grey')
ax3.axhline(80, linestyle = '--', linewidth = 1.5, color = 'grey')
ax3.set_title('NQ Stochastic RSI')
plt.show()

investment_value = 5000
account_min = 2000
stop_loss = 15
take_profit = 5
take_profit_2 = 10
number_of_contracts = 2
handle_value = 2

trades = []


def implement_rsi_strategy(ema_prices, date_indices, prices, rsi, rsi_lower_bound, rsi_upper_bound):
    buy_price = []
    sell_price = []
    rsi_signal = []
    size_open = []
    signal = 0
    relative_signal = 0
    bp =  0
    sp = 0
    buy_date = 0
    sell_date = 0
    max_draw_down = 0
    max_pull_up = 0

    for i in range(len(rsi)):
        if rsi[i-1] < rsi_lower_bound and rsi[i] > rsi_lower_bound:
            # if buy signal received for the first time and there are no long positions open
            # buy only when oversold and is above 200 ema
            if signal != 1 and relative_signal <= 0 and (True if not enable_ema_check else enable_ema_check and ema_prices[i] < prices[i]):
            #if signal != 1 and relative_signal <= 0:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1

                # if no position is open, then open a new position
                if relative_signal == 0:
                    # open 2 long poositions
                    relative_signal = 2
                    bp = prices[i]
                    buy_date = date_indices[i]
                    size_open.append([date_indices[i], 'FULL'])
                # if a position is already open and is a short position, then close the existing position
                elif relative_signal < 0:
                    relative_signal = 0
                    size_open.append([date_indices[i], 'FULL'])
                    trades.append(['SHORT', sell_date, sp, date_indices[i], prices[i], max_draw_down, max_pull_up, sp - prices[i], 'FULL','RSI'])
                    bp = 0
                    sp = 0
                    sell_date = 0
                    buy_date = 0
                    max_draw_down = 0
                    max_pull_up = 0

                rsi_signal.append(relative_signal)
            else:
                size_open.append([date_indices[i], ''])
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(relative_signal)
        elif rsi[i-1] > rsi_upper_bound and rsi[i] < rsi_upper_bound:
            # if sell signal received for the first time and there are no short positions open
            if signal != -1 and relative_signal >= 0 and (True if not enable_ema_check else enable_ema_check and ema_prices[i] > prices[i]):
            #if signal != -1 and relative_signal >= 0:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1

                # if no position is open, then open a new position
                if relative_signal == 0:
                    # open 2 short poositions
                    relative_signal = -2
                    sp = prices[i]
                    sell_date = date_indices[i]
                    size_open.append([date_indices[i], 'FULL'])
                # if a position is already open and is a long position, then close the existing position
                elif relative_signal > 0:
                    relative_signal = 0
                    size_open.append([date_indices[i], 'FULL'])
                    trades.append(['LONG', buy_date, bp, date_indices[i], prices[i], max_draw_down, max_pull_up, prices[i] - bp, 'FULL','RSI'])
                    sp = 0
                    bp = 0
                    sell_date = 0
                    buy_date = 0
                    max_draw_down = 0
                    max_pull_up = 0

                rsi_signal.append(relative_signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                size_open.append([date_indices[i], ''])
                rsi_signal.append(relative_signal)
        elif have_stop_loss:
            # if neither a buy signal nor a sell signal
            # Check for number of handles up or down. i.e max_draw_down and max_pull_up
            # Close 1 position if max_pull_up == 5 handles achieved or close 2 positions if max_pull_up == 15

            # Check which position are we in at the moment.
            # if relative_signal> 0: it is LONG position
            if relative_signal > 0:
                # If LONG, then bp will have buy_price
                # Calculate max_draw_down if the price goes down since we are LONG
                if max_draw_down <  bp - prices[i]:
                    max_draw_down = bp - prices[i]
                    # If max_draw_down >= 15 then close 2 positions.
                    # Since we are LONG, now we need to sell and update the signal to -1 (Overrides the values given by rsi)
                    if max_draw_down >= 15:
                        buy_price.append(np.nan)
                        sell_price.append(prices[i])
                        signal = 0

                        # Since we are LONG and got a SELL signal and closing all. Update relative_signal = 0. i.e no more open positions
                        relative_signal = 0
                        rsi_signal.append(relative_signal)
                        size_open.append([date_indices[i], 'FULL'])
                        trades.append(['LONG', buy_date, bp, date_indices[i], prices[i], max_draw_down, max_pull_up, prices[i] - bp, 'FULL','LIMIT'])
                        bp = 0
                        sp = 0
                        sell_date = 0
                        buy_date = 0
                        max_draw_down = 0
                        max_pull_up = 0
                    else:
                        # if max_draw_down is not 15 or more, then we sit and watch the drama, while we dance assuming we get profits
                        size_open.append([date_indices[i], ''])
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        rsi_signal.append(relative_signal)
                # Calculate max_pull_up only if the price goes up
                elif max_pull_up < prices[i] - bp:
                    max_pull_up = prices[i] - bp
                    # Close all positions if we are more than 10 handles
                    if max_pull_up >= 10 and take_profit_flag:
                        print("inside partial 1")
                        size_open.append([date_indices[i], 'FULL'])
                        trades.append(['LONG', buy_date, bp, date_indices[i], prices[i], max_draw_down, max_pull_up, prices[i] - bp, 'FULL','LIMIT'])
                        buy_price.append(np.nan)
                        sell_price.append(prices[i])
                        signal = 0
                        # Re-calculate the relative_signal to reflect how many open positions
                        relative_signal = 0
                        rsi_signal.append(relative_signal)
                        bp = 0
                        sp = 0
                        sell_date = 0
                        buy_date = 0
                        max_draw_down = 0
                        max_pull_up = 0
                    # Close 1 position if we reached 5 handles. YAYYY! its a profit
                    # This is to be executed only when we have 2 positions opened
                    elif max_pull_up >= 5 and relative_signal >1 and take_profit_flag:
                        print("inside partial 2")
                        size_open.append([date_indices[i], 'PARTIAL'])
                        trades.append(['LONG', buy_date, bp, date_indices[i], prices[i], max_draw_down, max_pull_up, prices[i] - bp, 'PARTIAL','LIMIT'])
                        buy_price.append(np.nan)
                        sell_price.append(prices[i])
                        # Re-calculate the relative_signal to reflect how many open positions
                        relative_signal = relative_signal - 1
                        rsi_signal.append(relative_signal)
                    else:
                        # if max_pull_up is not 5 then as usual we drink coke and dance
                        size_open.append([date_indices[i], ''])
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        rsi_signal.append(relative_signal)
                # if we are less than both previously calculated max_pull_up and max_draw_down, we wait
                else:
                    size_open.append([date_indices[i], ''])
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(relative_signal)
            # if relative_signal < 0: it is SHORT position
            elif relative_signal< 0:
                # If SHORT, then sp will have sell_price
                # Calculate max_pull_up if the price goes down
                if max_pull_up <  sp - prices[i]:
                    max_pull_up = sp - prices[i]
                    # Close all positions if we are more than 10 handles
                    if max_pull_up >= 10 and take_profit_flag:
                        print("inside partial 3")
                        size_open.append([date_indices[i], 'FULL'])
                        trades.append(['SHORT', sell_date, sp, date_indices[i], prices[i], max_draw_down, max_pull_up, sp - prices[i], 'FULL','LIMIT'])
                        buy_price.append(prices[i])
                        sell_price.append(np.nan)
                        signal = 0
                        # Re-calculate the relative_signal to reflect how many open positions
                        relative_signal = 0
                        rsi_signal.append(relative_signal)
                        bp = 0
                        sp = 0
                        sell_date = 0
                        buy_date = 0
                        max_draw_down = 0
                        max_pull_up = 0
                    # Close 1 position if we reached 5 handles. YAYYY! its a profit
                    # This is to be executed only when we have 2 positions opened
                    elif max_pull_up >= 5 and relative_signal < 1 and take_profit_flag:
                        print("inside partial 4")
                        size_open.append([date_indices[i], 'PARTIAL'])
                        trades.append(['SHORT', sell_date, sp, date_indices[i], prices[i], max_draw_down, max_pull_up, sp - prices[i], 'PARTIAL','LIMIT'])
                        buy_price.append(prices[i])
                        sell_price.append(np.nan)
                        # Re-calculate the relative_signal to reflect how many open positions
                        relative_signal = relative_signal + 1
                        rsi_signal.append(relative_signal)
                    else:
                        # if max_pull_up is not 5 then as usual we wait
                        size_open.append([date_indices[i], ''])
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        rsi_signal.append(relative_signal)
                # Calculate max_draw_down if the price goes up, since we are SHORT
                elif max_draw_down < prices[i] - sp:
                    max_draw_down = prices[i] - sp
                    # If max_draw_down >= 15 then close 2 positions.
                    # Since we are SHORT, now we need to buy and update the signal to 1 (Overrides the values given by rsi)
                    if max_draw_down >= 15:
                        buy_price.append(prices[i])
                        sell_price.append(np.nan)

                        signal = 0
                        # Since we are SHORT and got a BUY signal and closing all. Update relative_signal = 0. i.e no more open positions
                        relative_signal = 0
                        rsi_signal.append(relative_signal)
                        size_open.append([date_indices[i], 'FULL'])
                        trades.append(['SHORT', sell_date, sp, date_indices[i], prices[i], max_draw_down, max_pull_up, sp - prices[i], 'FULL','LIMIT'])
                        bp = 0
                        sp = 0
                        sell_date = 0
                        buy_date = 0
                        max_draw_down = 0
                        max_pull_up = 0
                    else:
                        # if max_draw_down is not 15 or more, then we sit and watch the drama, while we dance assuming we get profits
                        size_open.append([date_indices[i], ''])
                        buy_price.append(np.nan)
                        sell_price.append(np.nan)
                        rsi_signal.append(relative_signal)
                # if we are less than both previously calculated max_pull_up and max_draw_down, we wait
                else:
                    size_open.append([date_indices[i], ''])
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    rsi_signal.append(relative_signal)
            else:
                # if neither a buy signal nor a sell signal and we dont have any positions open, Then dont rush, sit tight, you are in good shape.
                size_open.append([date_indices[i], ''])
                #size_open.append([date_indices[i].tz_localize('UTC'), ''])
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                rsi_signal.append(relative_signal)
        else:
            size_open.append([date_indices[i], ''])
            #size_open.append([date_indices[i].tz_localize('UTC'), ''])
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            rsi_signal.append(relative_signal)

    return buy_price, sell_price, rsi_signal, size_open

# Generate buy price/sell price/rsi array
#buy_price, sell_price, rsi_signal = implement_rsi_strategy(nq.index, nq['Close'], nq['rsi_20'], rsi_lower_bound, rsi_upper_bound)
ema_200 = ema_span(nq['Close'], 100)
ema_df = pd.DataFrame(ema_200).rename(columns = {'Close':'ema_200'}).set_index(nq.index)
print(ema_df)
buy_price, sell_price, rsi_signal, size_open = implement_rsi_strategy(ema_df['ema_200'], nq.index, nq['Close'], nq['stock_rsi_20'], stoch_rsi_lower_bound, stoch_rsi_upper_bound)

size_open_df = pd.DataFrame(size_open).rename(columns = {0:'Datetime',1:'Trade Style'})
size_open_df = size_open_df.set_index('Datetime')
#Plot the graph with the buy and sell prices
ax1 = plt.subplot2grid((15,1), (0,0), rowspan = 10, colspan = 1)
# ax2 = plt.subplot2grid((15,1), (5,0), rowspan = 4, colspan = 1)
ax3 = plt.subplot2grid((15,1), (11,0), rowspan = 7, colspan = 1)
ax1.plot(nq['Close'], linewidth = 2.5, color = 'skyblue', label = 'NQ')
ax1.plot(ema_df['ema_200'], linewidth = 2.5, color = 'yellow', label = 'ema_200')
ax1.plot(size_open_df.index, buy_price, marker = '^', markersize = 10, color = 'green', label = 'BUY SIGNAL')
ax1.plot(size_open_df.index, sell_price, marker = 'v', markersize = 10, color = 'r', label = 'SELL SIGNAL')
ax1.set_title('NQ RSI TRADE SIGNALS')
# ax2.plot(nq.index, nq['rsi_20'], color = 'orange', linewidth = 2.5)
# ax2.axhline(rsi_lower_bound, linestyle = '--', linewidth = 1.5, color = 'grey')
# ax2.axhline(rsi_upper_bound, linestyle = '--', linewidth = 1.5, color = 'grey')
ax3.plot(size_open_df.index, nq['stock_rsi_20'], color = 'orange', linewidth = 2.5)
ax3.axhline(stoch_rsi_lower_bound, linestyle = '--', linewidth = 1.5, color = 'grey')
ax3.axhline(stoch_rsi_upper_bound, linestyle = '--', linewidth = 1.5, color = 'grey')
plt.show()



# # Initializing the position array to 0
# position = []
# for i in range(len(rsi_signal)):
#     position.append(0)
#
# #Updating the position array to get the cumulative positions at every timestamp
# for i in range(len(nq['Close'])):
#     if rsi_signal[i] == 0:
#         position[i] = position[i-1]
#     else:
#         position[i] = rsi_signal[i]

stoch_rsi = nq['stock_rsi_20']
close_price = nq['Close']
rsi_signal = pd.DataFrame(rsi_signal).rename(columns = {0:'rsi_signal'}).set_index(size_open_df.index)
# position = pd.DataFrame(position).rename(columns = {0:'rsi_position'}).set_index(size_open_df.index)
trades_df = pd.DataFrame(trades).rename(columns = {0:'Position',1:'Open Date', 2:'Open',3:'Close Date', 4:'Close',5:'Max Draw Down',6:'Max Pull Up', 7:'Profit/Loss', 8:'Trade Style',9:'Signal Type'})

print(trades_df)
# frames = [close_price, stoch_rsi, rsi_signal]
# strategy = pd.concat(frames, join = 'inner', axis = 1)

nq_ret = pd.DataFrame(np.diff(nq['Close'])).rename(columns = {0:'returns'})
rsi_strategy_ret = []

for i in range(len(nq_ret)):
    returns = nq_ret['returns'][i]*rsi_signal['rsi_signal'][i] * handle_value
    rsi_strategy_ret.append(returns)

rsi_strategy_ret_df = pd.DataFrame(rsi_strategy_ret).rename(columns = {0:'rsi_returns'})

# rsi_investment_ret = []
#
# for i in range(len(rsi_strategy_ret_df['rsi_returns'])):
#     returns = number_of_contracts*rsi_strategy_ret_df['rsi_returns'][i]
#     rsi_investment_ret.append(returns)

# rsi_investment_ret_df = pd.DataFrame(rsi_investment_ret).rename(columns = {0:'investment_returns'})
total_investment_ret = round(sum(rsi_strategy_ret_df['rsi_returns']), 2)
profit_percentage = floor((total_investment_ret/investment_value)*100)
print(cl('Profit gained from the RSI strategy by investing $5000 in NQ : {}'.format(total_investment_ret), attrs = ['bold']))
print(cl('Profit percentage of the RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))
