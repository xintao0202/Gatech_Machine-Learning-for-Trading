import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data

import numpy as np
%pylab
%matplotlib inline


def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

#  simple moving average
def SMA(df, periods=12):
    return pd.rolling_mean(df, window=periods)

# better for low <0.9, not for high
def test_SMA_run():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    symbols = ['AAPL']
    df = get_data(symbols, dates)

    #print df

    # compute SMA
    sma = SMA(df[symbols], 21)

    # compute price_sma
    price_sma = df[symbols]/sma

    #print price_sma




    df = pd.concat([df, sma, price_sma], axis=1)
    df.columns = ['SPY', symbols[0], 'SMA', 'Price_SMA']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df.index, df[symbols], label='Price', color='black')
    ax1.plot(df.index, df['SMA'], label='SMA', color='green')

    ax2.plot(df.index, df['Price_SMA'], label='Price/SMA', color='red', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_ylim([50, 250])

    ax2.set_ylabel('Price/SMA')
    ax2.set_ylim([0.7, 1.2])
    ax2.axhline(y = 0.9)
    ax2.axhline(y = 1.1)

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')
    plt.show()



if __name__ == "__main__":
    test_SMA_run()