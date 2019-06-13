

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data


def get_rolling_mean(values, window):
    # Return rolling mean of given values, using specified window size.
    return pd.rolling_mean(values, window=window)

def get_rolling_std(values, window):
    # Return rolling standard deviation of given values, using specified window size.
    return pd.rolling_std(values, window=window)

def get_bollinger_bands(rm, rstd):
    # Return upper and lower Bollinger Bands.
    upper_band = rm + rstd*1  # I use 1x std to calculate the bollinger bands
    lower_band = rm - rstd*1
    return upper_band, lower_band

def get_share_no(leverage, share_count, price, cash, action):
    # calculate how many shares to buy or sell based on current holding, cash and given leverage
    if action == 'BUY':
        if share_count >= 0:
            share_no = int((leverage * cash + (leverage - 1) * share_count * price) / price)
        if share_count < 0:
            share_no = int((leverage * cash + (leverage + 1) * share_count * price) / price) * (-1)

    if action == 'SELL':
        if share_count <= 0:
            share_no = int((leverage * cash + (leverage + 1) * share_count * price) / price)
        if share_count > 0:
            share_no = int((leverage * cash + (leverage - 1) * share_count * price) / price) * (-1)
    return share_no

def get_leverage(share_count, price, cash):
    # calculate current leverage
    if share_count >= 0:
        leverage = share_count*price / (share_count*price + cash)
    if share_count < 0:
        leverage = (-1) * share_count*price / (share_count*price + cash)
    return leverage

def buy(orders, symbol, date, price, share_count, cash, leverage):
    # execute a buy order with given share amount, cash and leverage
    share_no = get_share_no(leverage, share_count, price, cash, action='BUY')
    orders.loc[len(orders)] = [date, symbol, 'BUY', share_no]
    cash = cash - price * share_no
    share_count += share_no
    return orders, share_count, cash

def sell(orders, symbol, date, price, share_count, cash, leverage):
    # execute a sell order with given share amount, cash and leverage
    share_no = get_share_no(leverage, share_count, price, cash, action='SELL')
    orders.loc[len(orders)] = [date, symbol, 'SELL', share_no]
    cash = cash + price * share_no
    share_count -= share_no
    return orders, share_count, cash

def cover_all(orders, symbol, date, price, share_count, cash):
    # execute an order that covers all short selling
    orders.loc[len(orders)] = [date, symbol, 'BUY', -share_count]
    cash = cash + price * share_count
    share_count = 0
    return orders, share_count, cash

def sell_all(orders, symbol, date, price, share_count, cash):
    # execute an order that sells all long position
    orders.loc[len(orders)] = [date, symbol, 'SELL', share_count]
    cash = cash + price * share_count
    share_count = 0
    return orders, share_count, cash


def test_run():
    # Read data
    start_date = pd.to_datetime('2007-12-31')
    end_date = pd.to_datetime('2009-12-31')
    start_val = 10000
    symbols = ['IBM']

    read_start_date = start_date - pd.DateOffset(150)
    dates = pd.date_range(read_start_date, end_date)
    df = get_data(symbols, dates)  # get more data to calculate indicators from beginning of the start date

    # Compute Bollinger Bands
    sma = get_rolling_mean(df[symbols], window=99)  # WINDOW = 99
    rstd = get_rolling_std(df[symbols], window=99)  # WINDOW = 99
    rm_sma = get_rolling_mean(sma, window=5)  # use rolling SMA to determine the trend

    upper_band, lower_band = get_bollinger_bands(sma, rstd)

    rm10 = get_rolling_mean(df[symbols], window=10)  # get 10-day MA
    rm20 = get_rolling_mean(df[symbols], window=20)  # get 20-day MA

    df = pd.concat([df, sma, rm10, rm20, rm_sma, upper_band, lower_band], axis=1)
    df.columns = ['SPY', symbols[0], 'SMA', 'RM10', 'RM20', 'RM_SMA', 'UB', 'LB']
    df = df.truncate(before=start_date)

    orders = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    share_count = 0
    cash = start_val
    indicator = 'N'

# Strategy:
    for (i_row, row), (i_prev, prev)in zip(df.iterrows(), df.shift(1).iterrows()):
        if prev[symbols[0]] <= prev['SMA'] and row[symbols[0]] > row['SMA']:  # cross up SMA
            indicator = 'S'
            if share_count < 0:  # if there is a short position, cover all.
                orders, share_count, cash = cover_all(orders, symbols[0], i_row, row[symbols[0]], share_count, cash)
                plt.axvline(i_row, color='k')

        if prev[symbols[0]] >= prev['SMA'] and row[symbols[0]] < row['SMA'] and share_count >= 0:  # cross down SMA
            indicator = 'W'
            if share_count > 0:  # if there is a long position, sell all
                orders, share_count, cash = sell_all(orders, symbols[0], i_row, row[symbols[0]], share_count, cash)
                plt.axvline(i_row, color='k')

        if prev[symbols[0]] <= prev['UB'] and row[symbols[0]] > row['UB']:  # cross up UB
            indicator = 'VS'

        if row[symbols[0]] > row['UB'] and row['SMA'] > row['RM_SMA']*1.001 and indicator == 'VS':
            # stock in "very strong" region and SMA start to go up
            if get_leverage(share_count, row[symbols[0]], cash) < 1:
                orders, share_count, cash = buy(orders, symbols[0], i_row, row[symbols[0]], share_count, cash, leverage=2.0)
                plt.axvline(i_row, color='g')

        if prev[symbols[0]] >= prev['UB'] and row[symbols[0]] < row['UB']:  # cross down UB
            if share_count > 0 and (row['RM10'] < row['RM20'] or row['SMA'] < row['RM_SMA']):
                # if stock trend is moving down, sell it
                orders, share_count, cash = sell_all(orders, symbols[0], i_row, row[symbols[0]], share_count, cash)
                plt.axvline(i_row, color='k')

        if prev[symbols[0]] >= prev['LB'] and row[symbols[0]] < row['LB']:  # cross down LB
            indicator = 'VW'

        if row[symbols[0]] < row['LB'] and row['SMA'] < row['RM_SMA']*0.999 and indicator == 'VW':
            # stock in "very weak" region and SMA start to go down
            if get_leverage(share_count, row[symbols[0]], cash) < 1:
                orders, share_count, cash = sell(orders, symbols[0], i_row, row[symbols[0]], share_count, cash, leverage=2.0)
                plt.axvline(i_row, color='r')

        if prev[symbols[0]] <= prev['LB'] and row[symbols[0]] > row['LB']: # cross up LB
            if share_count < 0 and (row['RM10'] > row['RM20'] or row['SMA'] > row['RM_SMA']):
                # if stock trend is moving up, cover it
                orders, share_count, cash = cover_all(orders, symbols[0], i_row, row[symbols[0]], share_count, cash)
                plt.axvline(i_row, color='k')


    file_dir = os.path.join('orders', 'orders_test.csv')
    orders.to_csv(file_dir)

    plt.plot(df.index, df[symbols], label='IBM', color='b')
    plt.plot(df.index, df['SMA'], label='SMA', color='c')
    plt.plot(df.index, df['UB'], label='Bollinger Bands', color='m')
    plt.plot(df.index, df['LB'], label='', color='m')
    plt.plot(df.index, df['RM10'], label='RM10', color='r')
    plt.plot(df.index, df['RM20'], label='RM20', color='y')


    plt.title('My Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='lower left', fontsize=10)
    plt.show()


if __name__ == "__main__":
    test_run()

