import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
from indicators import MACD, RSI,BB
from best_possible_strategy import BestPossible
from marketsim import compute_portvals,find_leverage
import datetime as date
import numpy as np
import DecisionTreeClassification as dt



def test_run():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms]  # only price of each stock in portfolio
    macd = MACD(prices_port)
    df_RB=pd.concat([macd['MACD Line'], macd['Signal Line']], axis=1)
    df_RB = (df_RB - df_RB.mean(axis=0)) / df_RB.std(axis=0)
    df_RB['Color']='k'
    holding = 0  # holding =200 long; holding=-200 short
    window = date.datetime(2008, 1, 1)  # window should more than 21
    for (i_row, row), (i_prev, prev) in zip(df_RB.iterrows(), df_RB.shift(1).iterrows()):
        if prev['MACD Line'] >= 0 and row['MACD Line'] < 0  and holding > -200:
            holding -= 200 + holding
            window = i_row
            df_RB.ix[i_row,'Color']='r'
        if prev['MACD Line'] <= 0 and row['MACD Line'] > 0  and holding<200:
            holding += 200-holding
            window=i_row
            df_RB.ix[i_row, 'Color'] = 'g'
        if prev['MACD Line'] >= prev['Signal Line'] and row['MACD Line'] < row['Signal Line']  and holding > -200:
            holding -= 200 + holding
            window = i_row
            df_RB.ix[i_row, 'Color'] = 'r'
        if prev['MACD Line'] <= prev['Signal Line'] and row['MACD Line'] > row['Signal Line']  and holding<200 :
            holding += 200-holding
            window=i_row
            df_RB.ix[i_row, 'Color'] = 'g'


    #print df_RB
    df_RB=pd.DataFrame(dict(a=df_RB['MACD Line'],b=df_RB['Signal Line'], label=df_RB['Color']))
    groups=df_RB.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.a, group.b,marker='o', linestyle='', ms=12, label='MACD Line', color=name)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('MACD Line Value')
    ax.set_ylabel('Signal Line Value')
    plt.show()

    df_train= pd.read_csv(os.path.join('orders', 'Training data.csv'))
    df_train = pd.DataFrame(dict(a= df_train['MACD_ZERO'], b= df_train['MACD_SIGNAL'], label=df_train['Color']))

    groups =  df_train.groupby('label')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.a, group.b, marker='o', linestyle='', ms=12, label=name, color=name)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('MACD Line Value')
    ax.set_ylabel('Signal Line Value')
    plt.show()

    df_train['Color2']='k'

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
        predY_list = predY_list + predY
    predY = predY_list / 10.0
    df_train['Decision']=predY
    holding = 0

    for (i_row, row) in   df_train.iterrows():
        if row['Decision']<0  and holding>-200:
            holding -= 200 + holding
            df_train.ix[i_row, 'Color2'] = 'r'
        if  row['Decision']>0 and holding<200:
            holding += 200 - holding
            df_train.ix[i_row, 'Color2'] = 'g'

    df_test=df_train.copy()

    groups = df_train.groupby('Color2')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.a, group.b, marker='o', linestyle='', ms=12, label=name, color=name)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel('MACD Line Value')
    ax.set_ylabel('Signal Line Value')
    plt.show()

if __name__ == "__main__":
        test_run()