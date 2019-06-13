

import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
from indicators import MACD, RSI,BB
from best_possible_strategy import BestPossible
from marketsim import compute_portvals,find_leverage
import datetime as dt



def test_run():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms]  # only price of each stock in portfolio
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    bb = BB(prices_port,lookback=20)
    macd=MACD(prices_port)
    rsi=RSI(prices_port,lookback=14)
    df = pd.concat([prices_port, bb,macd,rsi], axis=1)
    df = df.fillna(method='bfill')
    #print df
    #plot_data(df, title="Bollinger Bands and stock price", ylabel="Stock price", xlabel="Date")
    orders = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])

    holding = 0 #holding =200 long; holding=-200 short
    window=dt.datetime(2008,1,1) # window should more than 21
    for (i_row, row), (i_prev, prev)in zip(df.iterrows(), df.shift(1).iterrows()):
        if prev['MACD Line'] >= 0 and row['MACD Line'] < 0 and (i_row-window).days>21 and holding>-200:
            orders.loc[len(orders)] = [i_row, syms[0], 'SELL', str(200+holding)]
            holding -= 200+holding
            window = i_row
            plt.axvline(i_row, color='r')
        if prev['MACD Line'] <= 0 and row['MACD Line'] > 0 and (i_row-window).days>21 and holding<200:
            orders.loc[len(orders)] = [i_row, syms[0], 'BUY', str(200-holding)] #max buy
            holding += 200-holding
            window=i_row
            plt.axvline(i_row, color='g')
        if prev['MACD Line'] >= prev['Signal Line'] and row['MACD Line'] < row['Signal Line'] and (i_row - window).days > 21 and holding > -200 and row['RSI']>70:
            orders.loc[len(orders)] = [i_row, syms[0], 'SELL', str(200 + holding)]
            holding -= 200 + holding
            window = i_row
            plt.axvline(i_row, color='r')
        if prev['MACD Line'] <= prev['Signal Line'] and row['MACD Line'] > row['Signal Line'] and (i_row-window).days>21 and holding<200 :
            orders.loc[len(orders)] = [i_row, syms[0], 'BUY', str(200-holding)] #max buy
            holding += 200-holding
            window=i_row
            plt.axvline(i_row, color='g')

        if prev[syms[0]] <= prev['Lower Band'] and row[syms[0]] > row['Lower Band'] and (i_row-window).days>21 and holding<200: # cross up Lower Band
            orders.loc[len(orders)] = [i_row, syms[0], 'BUY', str(200-holding)] #max buy
            holding += 200-holding
            window=i_row
            plt.axvline(i_row, color='g')
        if prev[syms[0]] <= prev['Middle Band'] and row[syms[0]] > row['Middle Band'] and (i_row-window).days>21 and holding>-200 and row['RSI']>70 : # cross up Middle Band
            orders.loc[len(orders)] = [i_row, syms[0], 'SELL', str(200+holding)]
            holding -= 200+holding
            window = i_row
            plt.axvline(i_row, color='r')
        # if prev[syms[0]] >= prev['Upper Band'] and row[syms[0]] < row['Upper Band'] and (i_row-window).days>21 and holding>-200 and row['RSI']>70: # cross down Upper Band
        #     orders.loc[len(orders)] = [i_row, syms[0], 'SELL', str(200+holding)]
        #     holding -= 200+holding
        #     window = i_row
        #     plt.axvline(i_row, color='r')
        if prev[syms[0]] >= prev['Middle Band'] and row[syms[0]] < row['Middle Band'] and (i_row-window).days>21 and holding<200: # cross down Middle Band
            orders.loc[len(orders)] = [i_row, syms[0], 'BUY', str(200-holding)]
            holding += 200-holding
            window =i_row
            plt.axvline(i_row, color='g')

    file_dir = os.path.join('orders', 'orders.csv')
    orders.to_csv(file_dir)





    Compare_df=BestPossible()
    benchmark = Compare_df['Benchmark']
    of = "./orders/orders.csv"
    sv = 100000
    portvals = compute_portvals(orders_file=of, start_val=sv)
    portvals = portvals / portvals.ix[0, :]
    benchmark = benchmark / benchmark.ix[0, :]
    #plot_data(plot_df, title="Benchmark vs. Rule-based portfolio", ylabel="Normalized price", xlabel="Date")
    prices_port=prices_port/prices_port.ix[0,:]
    plt.plot(prices_port.index, portvals, label='Rule-based portfolio', color='b')
    plt.plot(prices_port.index, prices_port, label='APPL', color='m')
    plt.plot(benchmark.index, benchmark, label='Benchmark', color='k')
    plt.title('Benchmark vs. Rule-based portfolio')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    test_run()

