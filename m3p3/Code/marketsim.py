"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


# from analysis.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def author():
    return 'xtao41'  # replace tb34 with your Georgia Tech username.


def find_leverage(dftrades, dfprices):
    dfholdings = np.cumsum(dftrades)
    # print dfholdings,'dfholding'

    dfvalues = dfprices.ix[-1,] * dfholdings.ix[-1,]  # stock values to current
    dfvalues = dfvalues.drop('Cash')
    # print dfvalues,'dfvalues'
    # print dfvalues
    long = np.sum(dfvalues[dfvalues > 0])  # how much long
    short = np.sum(dfvalues[dfvalues < 0])  # how much short
    cash = dfholdings.ix[-1, 'Cash'] + dfprices.ix[0, 'Cash']
    # print cash,'cash'
    lev = (abs(long) + abs(short)) / (long + short + cash)
    # print lev,'lev'
    return lev


def compute_portvals(start_date = dt.datetime(2008, 1, 1), end_date = dt.datetime(2009, 12, 31), orders_file="./orders/orders-short.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # sort the orders file
    orders_sorted = pd.read_csv(orders_file).sort(columns='Date').reset_index()

    # get symbols
    sym = list(set(orders_sorted['Symbol']))

    # get start and end date

    

    # get prices for each day, trading days only! and drop SPY if any
    dates = pd.date_range(start_date, end_date)
    prices = get_data(sym, dates).drop('SPY', axis=1)
    dfprices = pd.DataFrame(prices)
    dates = dfprices.index & dates  # trading days only
    # dfprices=dfprices.fillna(0)

    # add cash column
    cash = start_val
    dfprices['Cash'] = cash

    # construct another DF represents changes of the stocks and cashes
    dftrades = pd.DataFrame(index=dates, columns=list(dfprices))
    dftrades = dftrades.fillna(0)

    for row in range(0, len(orders_sorted)):
        date = orders_sorted.ix[row, 'Date']
        order = orders_sorted.ix[row, 'Order']
        symbol = orders_sorted.ix[row, 'Symbol']
        shares = orders_sorted.ix[row, 'Shares']
        i = dftrades.index.get_loc(date)
        if order == 'BUY':
            dftrades.ix[date, symbol] += shares
            dftrades.ix[date, 'Cash'] -= dfprices.ix[date, symbol] * shares
            if find_leverage(dftrades.ix[0:i + 1, ], dfprices.ix[0:i + 1, ]) > 1.5:
                dftrades.ix[date, symbol] -= shares
                dftrades.ix[date, 'Cash'] += dfprices.ix[date, symbol] * shares
        if order == 'SELL':
            dftrades.ix[date, symbol] -= shares
            dftrades.ix[date, 'Cash'] += dfprices.ix[date, symbol] * shares

            if find_leverage(dftrades.ix[0:i + 1, ], dfprices.ix[0:i + 1, ]) > 1.5:
                dftrades.ix[date, symbol] += shares
                dftrades.ix[date, 'Cash'] -= dfprices.ix[date, symbol] * shares
    # print dfprices.head
    # print dftrades.head()

    # construct a DF represents the holding (how many shares and cash each day)
    dfholding = pd.DataFrame(index=dates, columns=list(dfprices))
    dfholding = dfholding.fillna(0)
    dfholding.ix[0] = dftrades.ix[0]
    dfholding.ix[0, 'Cash'] = start_val + dftrades.ix[0, 'Cash']
    for row in range(1, len(dfholding)):
        dfholding.ix[row,] = dfholding.ix[row - 1,] + dftrades.ix[row,]

    # construct a DF represents the values of the stocks in profolio
    dfvalue = dfprices * dfholding
    dfvalue.ix[:, 'Cash'] = dfvalue.ix[:, 'Cash'] / dfprices.ix[:, 'Cash']
    portvals = dfvalue.sum(axis=1)

    # print portvals.head()

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008,1,1)
    # end_date = dt.datetime(2008,6,1)
    # portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    # portvals = portvals[['IBM']]  # remove SPY

    return portvals


# def test_code():
#     # this is a helper function you can use to test your code
#     # note that during autograding his function will not be called.
#     # Define input parameters
#
#     of = "./orders/orders-short.csv"
#     sv = 1000000
#
#     # Process orders
#     portvals = compute_portvals(orders_file="./orders/orders.csv", start_val=sv)
#     # print (portvals), "portval"
#     if isinstance(portvals, pd.DataFrame):
#         portvals = portvals[portvals.columns[0]]  # just get the first column
#     else:
#         "warning, code did not return a DataFrame"
#
#     # Get portfolio stats
#     # Here we just fake the data. you should use your code from previous assignments.
#     start_date = dt.datetime(2011, 1, 5)
#     end_date = dt.datetime(2011, 1, 20)
#     # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
#     # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]
#
#     cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)
#
#     # Simulate a $SPX-only reference portfolio to get stats
#     prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
#     prices_SPX = prices_SPX[['$SPX']]  # remove SPY
#     portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
#     cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(portvals_SPX)
#
#     # Compare portfolio against $SPX
#     print "Date Range: {} to {}".format(start_date, end_date)
#     print
#     print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
#     print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
#     print
#     print "Cumulative Return of Fund: {}".format(cum_ret)
#     print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
#     print
#     print "Standard Deviation of Fund: {}".format(std_daily_ret)
#     print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
#     print
#     print "Average Daily Return of Fund: {}".format(avg_daily_ret)
#     print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
#     print
#     print "Final Portfolio Value: {}".format(portvals[-1])
#
#     # Plot computed daily portfolio value
#     # df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
#     # plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


# if __name__ == "__main__":
#     test_code()
