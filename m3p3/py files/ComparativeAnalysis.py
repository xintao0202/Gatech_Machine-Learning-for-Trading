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
    Traindates = pd.date_range('2008-1-1', '2009-12-31')
    TestDates = pd.date_range('2010-1-1', '2011-12-31')
    syms = ['AAPL']
    prices_train = get_data(syms, Traindates)
    prices_test = get_data(syms, TestDates)
    prices_train = prices_train[syms]
    prices_test = prices_test[syms]

    bb_train = BB(prices_train, lookback=20)
    bb_train=bb_train.fillna(method='bfill')
    bb_test = BB(prices_test, lookback=20)
    bb_test = bb_test.fillna(method='bfill')

    macd_train = MACD(prices_train)
    macd_test = MACD(prices_test)

    rsi_train = RSI(prices_train, lookback=14)
    rsi_test = RSI(prices_test, lookback=14)
    df_train = pd.concat([prices_train, bb_train, macd_train,  rsi_train], axis=1)
    df_train = df_train.fillna(method='bfill')
    df_test = pd.concat([prices_test, bb_test, macd_test, rsi_test], axis=1)
    df_test = df_test.fillna(method='bfill')

    indicators_train = pd.DataFrame(columns=['MACD_ZERO', 'MACD_SIGNAL', 'ACROSS_BAND', 'BB_value', 'RSI','Decision'])
    indicators_test = pd.DataFrame(columns=['MACD_ZERO', 'MACD_SIGNAL', 'ACROSS_BAND', 'BB_value', 'RSI','Decision'])


    indicators_train['BB_value'] = (df_train[syms[0]] - bb_train['Middle Band']) / (bb_train['Upper Band'] - bb_train['Middle Band'])
    indicators_train['MACD_ZERO']=macd_train['MACD Line']
    indicators_train['MACD_SIGNAL'] = macd_train['Signal Line']
    indicators_train['ACROSS_BAND'] = bb_train['Middle Band']
    indicators_train['RSI'] =  rsi_train['RSI']
    indicators_train = (indicators_train - indicators_train.mean(axis=0)) / indicators_train.std(axis=0)

    indicators_test['BB_value'] = (df_test[syms[0]] - bb_test['Middle Band']) / (bb_test['Upper Band'] - bb_test['Middle Band'])
    indicators_test['MACD_ZERO'] = macd_test['MACD Line']
    indicators_test['MACD_SIGNAL'] = macd_test['Signal Line']
    indicators_test['ACROSS_BAND'] = bb_test['Middle Band']
    indicators_test['RSI'] = rsi_test['RSI']
    indicators_test = (indicators_test - indicators_test.mean(axis=0)) / indicators_test.std(axis=0)

    YBUY=0.05
    YSELL=-0.05
    for (i_row, row), (i_future, future) in zip(df_train.iterrows(), df_train.shift(-21).iterrows()):
        if (future[syms[0]]/row[syms[0]]-1)> YBUY: # if 21 days return exceed YBUY, then BUY Decision=1
            indicators_train.ix[i_row, 'Decision'] = 1
        if (future[syms[0]] / row[syms[0]] - 1) < YSELL:  # if 21 days return less than YSELL, then SELL Decision=-1
            indicators_train.ix[i_row, 'Decision'] = -1

    indicators_train=indicators_train.fillna(0)

    for (i_row, row), (i_future, future) in zip(df_test.iterrows(), df_test.shift(-21).iterrows()):
        if (future[syms[0]] / row[syms[0]] - 1) > YBUY:  # if 21 days return exceed YBUY, then BUY Decision=1
            indicators_test.ix[i_row, 'Decision'] = 1
        if (future[syms[0]] / row[syms[0]] - 1) < YSELL:  # if 21 days return less than YSELL, then SELL Decision=-1
            indicators_test.ix[i_row, 'Decision'] = -1

    indicators_test = indicators_test.fillna(0)


    #print indicators
    file_dir_train = os.path.join('orders', 'indicators_train.csv')
    file_dir_test= os.path.join('orders', 'indicators_test.csv')
    indicators_train.to_csv(file_dir_train, header=False, index=False)
    indicators_test.to_csv(file_dir_test, header=False, index=False)



    inf_train = open('orders/indicators_train.csv')
    inf_test = open('orders/indicators_test.csv')
    data_train = np.array([map(float, s.strip().split(',')) for s in inf_train.readlines()])
    data_test = np.array([map(float, s.strip().split(',')) for s in inf_test.readlines()])
    predY_list = [0] * data_test.shape[0]

    for i in range(10):  # bag 10 times
        learner = dt.DTclass(leaf_size=5, verbose=False)
        train_rows = int(round(1.0 * data_train.shape[0]))
        test_rows = int(round(1.0 * data_test.shape[0]))
        Xtrain = data_train[:train_rows, 0:-1]
        Ytrain = data_train[:train_rows, -1]
        Xtest = data_test[:train_rows, 0:-1]
        Ytest = data_test[:train_rows, -1]
        learner.addEvidence(Xtrain, Ytrain)  # training step
        predY = learner.query(Xtest)  # query
        predY_list=predY_list+predY
    predY=predY_list/10.0





    prices_test['decision']= predY
    #print prices_port
    orders = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
    holding = 0  # holding =200 long; holding=-200 short
    window = date.datetime(2008, 1, 1)  # window should more than 21
    for (i_row, row) in  prices_test.iterrows():
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

    file_dir = os.path.join('orders', 'orders_ML_test.csv')
    orders.to_csv(file_dir)

    of_ML_test = "./orders/orders_ML_test.csv"
    #of = "./orders/orders.csv"
    sv = 100000

    #calculate benchmark of testing data
    dfprices = pd.DataFrame(prices_test)
    dfprices['Cash'] = sv - 200 * dfprices.ix[0, 'AAPL']
    dfprices['Holding'] = 200
    dfprices['Stock values'] = dfprices['Holding'] * dfprices['AAPL']
    dfprices['Port_val'] = dfprices['Cash'] + dfprices['Stock values']
    dfprices['Benchmark'] = dfprices['Port_val'] / dfprices['Port_val'].ix[0, :]
    benchmark = dfprices['Benchmark']

    #calculate rule based on test period
    orders_test = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])

    holding_RB = 0  # holding =200 long; holding=-200 short
    window_RB = date.datetime(2010, 1, 1)  # window should more than 21
    for (i_row, row), (i_prev, prev) in zip(df_test.iterrows(), df_test.shift(1).iterrows()):
        if prev['MACD Line'] >= 0 and row['MACD Line'] < 0 and (i_row - window_RB).days > 21 and holding_RB > -200:
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'SELL', str(200 + holding_RB)]
            holding_RB -= 200 + holding_RB
            window_RB = i_row
        if prev['MACD Line'] <= 0 and row['MACD Line'] > 0 and (i_row - window_RB).days > 21 and holding_RB < 200:
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'BUY', str(200 - holding_RB)]  # max buy
            holding_RB += 200 - holding_RB
            window_RB = i_row
        if prev['MACD Line'] >= prev['Signal Line'] and row['MACD Line'] < row['Signal Line'] and (
            i_row - window_RB).days > 21 and holding_RB > -200 and row['RSI'] > 70:
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'SELL', str(200 + holding_RB)]
            holding_RB -= 200 + holding_RB
            window_RB = i_row
        if prev['MACD Line'] <= prev['Signal Line'] and row['MACD Line'] > row['Signal Line'] and (
            i_row - window_RB).days > 21 and holding_RB < 200:
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'BUY', str(200 - holding_RB)]  # max buy
            holding_RB += 200 - holding_RB
            window_RB = i_row

        if prev[syms[0]] <= prev['Lower Band'] and row[syms[0]] > row['Lower Band'] and (
            i_row - window_RB).days > 21 and holding_RB < 200:  # cross up Lower Band
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'BUY', str(200 - holding_RB)]  # max buy
            holding_RB += 200 - holding_RB
            window_RB = i_row
        if prev[syms[0]] <= prev['Middle Band'] and row[syms[0]] > row['Middle Band'] and (
            i_row - window_RB).days > 21 and holding_RB > -200 and row['RSI'] > 70:  # cross up Middle Band
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'SELL', str(200 + holding_RB)]
            holding_RB -= 200 + holding_RB
            window_RB = i_row
        # if prev[syms[0]] >= prev['Upper Band'] and row[syms[0]] < row['Upper Band'] and (i_row-window).days>21 and holding>-200 and row['RSI']>70: # cross down Upper Band
        #     orders.loc[len(orders)] = [i_row, syms[0], 'SELL', str(200+holding)]
        #     holding -= 200+holding
        #     window = i_row
        #     plt.axvline(i_row, color='r')
        if prev[syms[0]] >= prev['Middle Band'] and row[syms[0]] < row['Middle Band'] and (
            i_row - window_RB).days > 21 and holding_RB < 200:  # cross down Middle Band
            orders_test.loc[len(orders_test)] = [i_row, syms[0], 'BUY', str(200 - holding_RB)]
            holding_RB += 200 - holding_RB
            window_RB = i_row

    file_dir_RB_test = os.path.join('orders', 'orders_RB_test.csv')
    orders_test.to_csv(file_dir_RB_test)


    portvals_RB = compute_portvals(start_date = date.datetime(2010, 1, 1), end_date = date.datetime(2011, 12, 31),orders_file=file_dir_RB_test, start_val=sv)
    portvals_RB = portvals_RB / portvals_RB.ix[0, :]
    benchmark = benchmark / benchmark.ix[0, :]
    portvals_ML = compute_portvals(start_date = date.datetime(2010, 1, 1), end_date = date.datetime(2011, 12, 31),orders_file=of_ML_test, start_val=sv)
    portvals_ML = portvals_ML / portvals_ML.ix[0, :]
    prices_test = prices_test / prices_test.ix[0, :]
    plt.plot(prices_test.index, portvals_RB, label='Rule-based portfolio', color='b')
    plt.plot(prices_test.index, portvals_ML, label='ML-based portfolio Out of Sample', color='g')
    #plt.plot(prices_port.index, prices_port, label='APPL', color='m')
    plt.plot(benchmark.index, benchmark, label='Benchmark', color='k')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.show()
if __name__ == "__main__":
    test_run()
    #TestLearner()