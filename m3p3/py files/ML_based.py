import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
from indicators import MACD, RSI,BB
from best_possible_strategy import BestPossible
from marketsim import compute_portvals,find_leverage
import datetime as date
import DecisionTreeClassification as dt
import math
import numpy as np




def test_run():
    #construct indicator and decision
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms]  # only price of each stock in portfolio

    bb = BB(prices_port, lookback=20)
    bb=bb.fillna(method='bfill')
    macd = MACD(prices_port)
    rsi = RSI(prices_port, lookback=14)
    df = pd.concat([prices_port, bb, macd, rsi], axis=1)
    df = df.fillna(method='bfill')

    indicators = pd.DataFrame(columns=['MACD_ZERO', 'MACD_SIGNAL', 'ACROSS_BAND', 'BB_value', 'RSI','Decision'])

    indicators['BB_value'] = (df[syms[0]] - bb['Middle Band']) / (bb['Upper Band'] - bb['Middle Band'])
    indicators['MACD_ZERO']=macd['MACD Line']
    indicators['MACD_SIGNAL'] = macd['Signal Line']
    indicators['ACROSS_BAND'] = bb['Middle Band']
    indicators['RSI'] = rsi['RSI']
    indicators2 = indicators[['MACD_SIGNAL', 'MACD_ZERO']]
    indicators2 = (indicators2 - indicators2.mean(axis=0)) / indicators2.std(axis=0)
    indicators2['Color']='k'
    # construct indicators
    # for (i_row, row), (i_prev, prev) in zip(df.iterrows(), df.shift(1).iterrows()):
    #     if prev['MACD Line'] >= 0 and row['MACD Line'] < 0:  # MACD prev>0,now<0, SELL MACD_ZERO=-1
    #         indicators.ix[i_row,'MACD_ZERO']=-1
    #     if prev['MACD Line'] <= 0 and row['MACD Line'] > 0:  # MACD prev<0,now>0, BUY MACD_ZERO=1
    #         indicators.ix[i_row, 'MACD_ZERO'] = 1
    #     if prev['MACD Line'] >= prev['Signal Line'] and row['MACD Line'] < row['Signal Line']: # MACD prev>Signal Line,now<Signal line, SELL MACD_Signal=-1
    #         indicators.ix[i_row,'MACD_SIGNAL']=-1
    #     if prev['MACD Line'] <= prev['Signal Line'] and row['MACD Line'] > row['Signal Line']:  # MACD prev<Signal Line,now>Signal line, BUY MACD_Signal=1
    #         indicators.ix[i_row, 'MACD_SIGNAL'] = 1
    #     if prev[syms[0]] <= prev['Lower Band'] and row[syms[0]] > row['Lower Band']:  # cross up Lower Band, BUY Acroo_band=1
    #         indicators.ix[i_row, 'ACROSS_BAND'] = 1
    #     if prev[syms[0]] >= prev['Upper Band'] and row[syms[0]] < row['Upper Band']: # cross down Upper Band SELL Acroo_band=-1
    #         indicators.ix[i_row, 'ACROSS_BAND'] = -1
    #     if row['RSI']>70:     #RSI>70 overbought, likely to sell it
    #         indicators.ix[i_row, 'RSI'] = -1
    #     if row['RSI'] < 30:   #RSI<30, oversold, likely to buy it.
    #         indicators.ix[i_row, 'RSI'] = 1
    # construct decision

    indicators = (indicators - indicators.mean(axis=0)) / indicators.std(axis=0)

    YBUY=0.05
    YSELL=-0.05
    for (i_row, row), (i_future, future) in zip(df.iterrows(), df.shift(-21).iterrows()):
        if (future[syms[0]]/row[syms[0]]-1)> YBUY: # if 21 days return exceed YBUY, then BUY Decision=1
            indicators.ix[i_row, 'Decision'] = 1
            indicators2.ix[i_row, 'Color'] = 'g'
        if (future[syms[0]] / row[syms[0]] - 1) < YSELL:  # if 21 days return less than YSELL, then SELL Decision=-1
            indicators.ix[i_row, 'Decision'] = -1
            indicators2.ix[i_row, 'Color'] = 'r'

    indicators=indicators.fillna(0)


    #print indicators
    file_dir = os.path.join('orders', 'indicators.csv')
    file_dir2 = os.path.join('orders', 'Training data.csv')
    indicators.to_csv(file_dir, header=False, index=False)

    indicators2.to_csv(file_dir2)

def TestLearner():
    inf = open('orders/indicators.csv')
    data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])
    predY_list = [0] * data.shape[0]

    for i in range(10):  # bag 10 times
        learner = dt.DTclass(leaf_size=5, verbose=False)
        train_rows = int(round(1.0 * data.shape[0]))
        test_rows = data.shape[0] - train_rows
        Xtrain = data[:train_rows, 0:-1]
        Ytrain = data[:train_rows, -1]
        Xtest = Xtrain
        Ytest = Ytrain
        learner.addEvidence(Xtrain, Ytrain)  # training step
        predY = learner.query(Xtest)  # query
        predY_list=predY_list+predY
    predY=predY_list/10.0



    #rmse = math.sqrt(((Ytrain - predY) ** 2).sum() / Ytrain.shape[0])
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms]  # only price of each stock in portfolio
    prices_port['decision']= predY
    #print prices_port
    orders = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    holding = 0  # holding =200 long; holding=-200 short
    window = date.datetime(2008, 1, 1)  # window should more than 21
    for (i_row, row) in  prices_port.iterrows():
        if row['decision']<0 and (i_row-window).days>21 and holding>-200:
            orders.loc[len(orders)] = [i_row, syms[0], 'SELL', str(200 + holding)]
            holding -= 200 + holding
            window = i_row
            plt.axvline(i_row, color='r')
        if  row['decision']>0 and (i_row-window).days>21 and holding<200:
            orders.loc[len(orders)] = [i_row, syms[0], 'BUY', str(200 - holding)]  # max buy
            holding += 200 - holding
            window = i_row
            plt.axvline(i_row, color='g')

    file_dir = os.path.join('orders', 'orders_ML.csv')
    orders.to_csv(file_dir)
    Compare_df = BestPossible()
    benchmark = Compare_df['Benchmark']
    of_ML = "./orders/orders_ML.csv"
    of = "./orders/orders.csv"
    sv = 100000
    portvals_RB = compute_portvals(orders_file=of, start_val=sv)
    portvals_RB = portvals_RB / portvals_RB.ix[0, :]
    benchmark = benchmark / benchmark.ix[0, :]
    portvals_ML = compute_portvals(orders_file=of_ML, start_val=sv)
    portvals_ML = portvals_ML / portvals_ML.ix[0, :]
    # plot_data(plot_df, title="Benchmark vs. Rule-based portfolio", ylabel="Normalized price", xlabel="Date")
    prices_port = prices_port / prices_port.ix[0, :]
    plt.plot(prices_port.index, portvals_RB, label='Rule-based portfolio', color='b')
    plt.plot(prices_port.index, portvals_ML, label='ML-based portfolio', color='g')
    #plt.plot(prices_port.index, prices_port, label='APPL', color='m')
    plt.plot(benchmark.index, benchmark, label='Benchmark', color='k')
    plt.title('Benchmark vs. Rule-based portfolio')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='lower right')
    plt.show()
if __name__ == "__main__":
    #test_run()
    TestLearner()