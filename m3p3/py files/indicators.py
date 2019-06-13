import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data

#indicator #1: Trend indicator MACD (Moving average covergence/divergence)
#compute using fast and slow exponetial moving average

def MACD(price, fast=12, slow=26, signal_period=9):
    ema_fast = pd.ewma(price, span = fast)
    ema_slow = pd.ewma(price, span = slow)
    macd = ema_fast - ema_slow
    signal = pd.ewma(macd, signal_period)
    macd_H = macd - signal
    df=pd.concat([ema_fast, ema_slow, macd,signal,macd_H],axis=1)
    df.columns = ['EMA Fast', 'EMA Slow','MACD Line', 'Signal Line','MACD Histogram']
    return df

# Calculate relative strength of n days
def RSI(price,lookback=14):
    dates = price.shape[0]
    RS = pd.DataFrame(columns=price.columns, index=price.index)
    rsi=RS
    for index in range(dates):
        for sym in price.columns:
            df_gain=0
            df_loss=0
            for prev_day in range (index-lookback+1,index+1):
                delta = price.ix[prev_day,sym]-price.ix[prev_day-1,sym]
                #print delta
                if delta >= 0:
                    df_gain=df_gain+delta
                else:
                    df_loss = df_loss+(-1*delta)
            if df_loss==0:
               rsi.ix[index][sym]=100
            else:
                RS=(df_gain/lookback)/(df_loss/lookback)
                rsi.ix[index][sym] = 100 - (100 / (1 + RS))
    rsi.columns=["RSI"]
    return rsi

def BB(price, lookback=21):
    dates = price.shape[0]
    sma=pd.rolling_mean(price,window=lookback) #middle band is SMA (simple moving average)
    smd=pd.rolling_std(price,window=lookback) # smd is the simple moving stdev
    middle_band=sma
    upper_band=sma+smd*2
    lower_band=sma-smd*2
    df=pd.concat([upper_band,middle_band,lower_band], axis=1)
    df.columns=["Upper Band","Middle Band", "Lower Band"]
    return df




# better for low <0.9, not for high
def test_MACD_run():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms] # only price of each stock in portfolio
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later


    # compute SMA
    macd = MACD(prices_port[syms], 12,26)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(prices_port.index, prices_port, label='Stock price', color='black')
    ax1.plot(macd.index, macd['EMA Fast'], label='EMA Fast', color='cyan')
    ax1.plot(macd.index, macd['EMA Slow'], label='EMA Slow', color='green')
    ax2.plot(macd.index, macd['MACD Line'], label='MACD', color='blue', linewidth=2.0)
    ax2.plot(macd.index, macd['Signal Line'], label='Signal', color='red', linewidth=2.0)
    ax2.fill_between(macd.index, macd['MACD Histogram'], 0, alpha=0.5, facecolor='darkslategrey', edgecolor='darkslategrey', label='Histogram')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MACD Indicator')
    ax1.legend(loc='lower left')
    ax1.set_ylabel('Stock Price and EMAs')
    ax2.legend(loc='lower right')

    plt.title('MACD')
    plt.show()


def test_RSI_run():
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms]  # only price of each stock in portfolio
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    #compute RSI
    rsi=RSI(prices_port)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(prices_port.index, prices_port, label='Stock price', color='black')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Stock price')
    ax2.plot(rsi.index, rsi, label='RSI', color='red',linewidth=2.0)
    ax2.set_ylabel('RSI')
    ax2.axhline(y=30)
    ax2.axhline(y=70)
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')
    plt.title('RSI')
    plt.show()

def test_BB_run():
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms]  # only price of each stock in portfolio
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    # compute Bollinger band
    bb = BB(prices_port)
    df=pd.concat([prices_port,bb],axis=1)
    plot_data(df, title="Bollinger Bands and stock price", ylabel="Stock price", xlabel="Date")

if __name__ == "__main__":
    test_MACD_run()
    test_RSI_run()
    test_BB_run()