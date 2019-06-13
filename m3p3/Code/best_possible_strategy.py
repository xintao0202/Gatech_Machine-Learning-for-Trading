import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data

def Best_strategy(prices_port):
    daily_rets =prices_port.pct_change(1)
    daily_rets.ix[0] = 0
    port_val=np.copy(prices_port)
    port_val[0] = 100000
    for i in range (1, len(daily_rets)):
        if daily_rets.ix[i] is 0.0:
            port_val[i]=port_val[i-1]
        else:
            port_val[i] = port_val[i - 1]+200*abs((prices_port.ix[i]-prices_port.ix[i-1]))
    return port_val

def compute_portfolio_stats (port_val):
    daily_rets = port_val.pct_change(1)
    daily_rets.ix[0] = 0
    dr = daily_rets[1:]  # daily return excluding first row value (0)
    cr = (port_val.ix[-1, :] / port_val.ix[0, :]) - 1  # Cumulative Return
    adr = dr.mean()
    sddr = dr.std()
    return cr, adr, sddr


def BestPossible():
    # Read data
    dates = pd.date_range('2008-1-1', '2009-12-31')
    start_val=100000
    syms = ['AAPL']
    prices_all = get_data(syms, dates)
    prices_port = prices_all[syms] # only price of each stock in portfolio
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    dfprices = pd.DataFrame(prices_port)
    dfprices['Cash'] = start_val-200*dfprices.ix[0,'AAPL']
    dfprices['Holding']=200
    dfprices['Stock values']=dfprices['Holding']*dfprices['AAPL']
    dfprices['Port_val']=dfprices['Cash']+dfprices['Stock values']
    dfprices['Benchmark'] = dfprices['Port_val']/dfprices['Port_val'].ix[0,:]
    dfprices['Best']=Best_strategy(prices_all[syms])
    dfprices['Best possible portfolio'] = dfprices['Best'] / dfprices['Best'].ix[0, :]
    df=pd.concat([dfprices['Benchmark'],dfprices['Best possible portfolio']], keys=['Benchmark', 'Best possible portfolio'], axis=1)
    plot_data(df, title="Benchmark vs. Best possible portfolio", ylabel="Normalized price", xlabel="Date")
    crBM, adrBM, sddrBM=compute_portfolio_stats (dfprices['Benchmark'])
    crBest, adrBest, sddrBest = compute_portfolio_stats(dfprices['Best possible portfolio'])
    print "statistics of Benchmark", crBM, adrBM, sddrBM
    print "statistics of Best Stategy", crBest, adrBest, sddrBest
    return df

if __name__ == "__main__":
    BestPossible()