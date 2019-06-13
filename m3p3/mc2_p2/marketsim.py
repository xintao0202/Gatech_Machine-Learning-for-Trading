"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # TODO: Your code here
    df_orders = pd.read_csv(orders_file)  # read csv file
    df_orders = df_orders.sort(columns='Date')  # sort orders by date
    df_orders = df_orders.reset_index()

    symbols = list(set(df_orders['Symbol']))  # get symbols in orders, remove duplicate

    df_prices = get_data(symbols, pd.date_range(start_date, end_date))  # get prices of the symbols
    df_prices = df_prices.drop('SPY', axis=1)  # drop 'SPY'

    index_trade = pd.date_range(start_date, end_date)
    df_trade = pd.DataFrame(index=index_trade, columns=[symbols])  # construct a df_trade to record orders by date
    df_trade = df_trade.fillna(0)
    cash = start_val

    df_position = np.cumsum(df_trade)  # construct a df_position to record the stock holdings everyday


    for order_no in range (0, len(df_orders)):  # read each order

        order = df_orders.ix[order_no, ]   # read the current order

        df_trade, df_position, cash = execute_order(order, df_prices, df_trade, df_position, cash)
            # get the result of order execution (depending on leverage check)

    df_stockValue = df_prices * df_position  # construct a df_stockValue to record the value of stock holdings
    df_stockValue = df_stockValue.dropna()
    df_stockValue['Value'] = np.sum(df_stockValue, axis=1)

    df_trade['OrderValue'] = np.sum(df_prices * df_trade, axis=1)
    df_trade = df_trade.dropna(subset=['OrderValue'])
    df_trade['Cash'] = 0
    df_trade.ix[0, 'Cash'] = start_val - df_trade.ix[0, 'OrderValue']
        # calculate cash remaining everyday
    for row in range(1, len(df_trade)):
        df_trade.ix[row, 'Cash'] = df_trade.ix[row-1, 'Cash'] - df_trade.ix[row, 'OrderValue']

    df_trade['StockValue'] = df_stockValue['Value']
    df_trade['Portvals'] = df_trade['Cash'] + df_trade['StockValue']  # calculate portfolio value everyday

    portvals = df_trade[['Portvals']]

    return portvals


def execute_order(order, df_prices, df_trade, df_position, cash):

    df_trade_before = df_trade.copy()  # store old df_trade
    df_position_before = df_position.copy()  # store old df_position
    cash_before = cash  # store old cash

    date = order['Date']
    symbol = order['Symbol']

    current_leverage = get_leverage(date, df_trade, df_prices, cash)  # get old leverage

    # fill df_trade according to 'BUY' or 'SELL', calculate order value
    if order['Order'] == 'BUY':
        df_trade.ix[date, symbol] = df_trade.ix[date, symbol] + order['Shares']
        order_value = np.sum(df_prices.ix[date, symbol] * order['Shares'])
    if order['Order'] == 'SELL':
        df_trade.ix[date, symbol] = df_trade.ix[date, symbol] + order['Shares'] * (-1)
        order_value = np.sum(df_prices.ix[date, symbol] * order['Shares'] * (-1))

    df_position = np.cumsum(df_trade)   # calculate new df_position

    cash = cash - order_value    # calculate new cash amount

    new_leverage = get_leverage(date, df_trade, df_prices, cash)   # get new leverage

    # check leverage, if safisfied, return new df_trade, df_position, cash; otherwise, return old ones
    if new_leverage <= 2:
        return df_trade, df_position, cash
    elif new_leverage > 2 and new_leverage < current_leverage:
        return df_trade, df_position, cash
    else:
        return df_trade_before, df_position_before, cash_before


def get_leverage(date, df_trade, df_prices, cash):
    df_position = np.cumsum(df_trade)
    stock_value = df_prices.ix[date, ] * df_position.ix[date, ]  # calculate current stock values

    # get sum of long position and short position
    sum_long = np.sum(stock_value[stock_value>0])
    sum_short = np.sum(stock_value[stock_value<0])

    # get leverage
    leverage = (sum_long + (-1)*sum_short) / ((sum_long - (-1)*sum_short) + cash)

    return leverage


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2007-12-31'
    end_date = '2009-12-31'
    orders_file = os.path.join("orders", "orders_test.csv")
    start_val = 10000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
