"""Bollinger Bands."""

import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data


def get_rolling_mean(values, window):
    # Return rolling mean of given values, using specified window size.
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    # Return rolling standard deviation of given values, using specified window size.
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    # Return upper and lower Bollinger Bands.
    upper_band = rm + rstd*2
    lower_band = rm - rstd*2
    return upper_band, lower_band


def test_run():
    # Read data
    dates = pd.date_range('2007-12-31', '2009-12-31')
    symbols = ['IBM']
    df = get_data(symbols, dates)

    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = get_rolling_mean(df[symbols], window=20)

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df[symbols], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)


    df = pd.concat([df, rm, upper_band, lower_band], axis=1)
    df.columns = ['SPY', symbols[0], 'SMA', 'UB', 'LB']
    orders = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    share_count = 0


    for (i_row, row), (i_prev, prev)in zip(df.iterrows(), df.shift(1).iterrows()):
        if prev[symbols[0]] <= prev['LB'] and row[symbols[0]] > row['LB'] and share_count == 0: # cross up LB
            orders.loc[len(orders)] = [i_row, symbols[0], 'BUY', '100']
            share_count += 100
            plt.axvline(i_row, color='g')
        if prev[symbols[0]] <= prev['SMA'] and row[symbols[0]] > row['SMA'] and share_count == 100: # cross up SMA
            orders.loc[len(orders)] = [i_row, symbols[0], 'SELL', '100']
            share_count -= 100
            plt.axvline(i_row, color='k')
        if prev[symbols[0]] >= prev['UB'] and row[symbols[0]] < row['UB'] and share_count == 0: # cross down UB
            orders.loc[len(orders)] = [i_row, symbols[0], 'SELL', '100']
            share_count -= 100
            plt.axvline(i_row, color='r')
        if prev[symbols[0]] >= prev['SMA'] and row[symbols[0]] < row['SMA'] and share_count == -100: # cross down SMA
            orders.loc[len(orders)] = [i_row, symbols[0], 'BUY', '100']
            share_count += 100
            plt.axvline(i_row, color='k')

    file_dir = os.path.join('orders', 'orders.csv')
    orders.to_csv(file_dir)

    plt.plot(df.index, df[symbols], label='IBM', color='b')
    plt.plot(df.index, df['SMA'], label='SMA', color='c')
    plt.plot(df.index, df['UB'], label='Bollinger Bands', color='m')
    plt.plot(df.index, df['LB'], label='', color='m')

    plt.title('Bollinger Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='lower left')
    plt.show()



if __name__ == "__main__":
    test_run()

