# better for low <0.9, not for high
def test_Basket_strategy_run():
    # Read data
    long_dates = pd.date_range('2007-11-1', '2009-12-31')
    dates = pd.date_range('2008-1-1', '2009-12-31')
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    symbols = ['AAPL']
    df = get_data(symbols, long_dates)

    df_STO = pd.DataFrame(index=long_dates)
    for symbol in symbols:
        df_STO_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                    parse_dates=True, na_values=['nan'])

        df_STO = df_STO.join(df_STO_temp)
    #print df_STO.head()
    df_STO = df_STO.dropna()
    #print df_STO.head()
    #print df_STO.shape[0]

    #print df

    # compute SMA
    sma_window = 12
    sma = SMA(df[symbols], sma_window)
    #mask = (sma.index >= start_date) & (sma.index <= end_date)
    #sma = sma.loc[mask]
    #sma = sma.fillna(method='bfill')
    ######################################## compute price_sma
    price_sma = df[symbols]/sma
    # generate orders_file and signal for price_sma
    price_sma.columns = ['Price/SMA']
    price_sma_mask = (price_sma.index >= start_date) & (price_sma.index <= end_date)
    price_sma = price_sma.loc[price_sma_mask]
    #price_sma['Order'] = np.repeat('NoAction',price_sma.shape[0])
    #price_sma['Signal'] = np.repeat('NoAction',price_sma.shape[0])
    #price_sma = price_sma.fillna(method='bfill')
    #print price_sma

    # compute STO_KD
    sto_kd = STO_KD(df_STO, periods=14)
    df_STO = pd.concat([df_STO, sto_kd], axis=1)
    mask_sto_kd = (sto_kd.index >= start_date) & (sto_kd.index <= end_date)
    sto_kd = sto_kd.loc[mask_sto_kd]
    #print sto_kd.head()
    #print sto_kd.shape[0]

    # compute BB and %B
    # 1. Compute rolling mean
    rm = get_rolling_mean(df[symbols], window=20)

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df[symbols], window=20)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)

    # get B
    bollinger_B = get_bollinger_bands_B_indicator(df, rm, rstd)
    bollinger_B = bollinger_B[symbols]
    mask_BB = (bollinger_B.index >= start_date) & (bollinger_B.index <= end_date)
    bollinger_B = bollinger_B.loc[mask_BB]
    bollinger_B.columns = ['bollinger_B']
    #print bollinger_B.head()

    ########################## need a basket here, store all price_sma, sto_kd, B, Order and Signal, used for add vertical line
    df_basket = pd.concat([price_sma, bollinger_B, sto_kd], axis=1)
    # add 'Order' and 'Signal column to basket df
    df_basket['Order'] = np.repeat('NoAction',df_basket.shape[0])
    df_basket['Signal'] = np.repeat('NoAction',df_basket.shape[0])
    #print df_basket

    orders = []
    share_hold = 0
    cash = 0
    sv = 100000
    #print df.ix[dates, symbols].head()
    print df['AAPL'].ix[df.shape[0]-1]
    #print df
    print price_sma.index[0].date()

    #print df.head()
    #print df.shape[0]
    #print price_sma.shape[0]
    #print price_sma.head()
    #print price_sma['Price/SMA'].size

    # start from  sma_window-1 since the beginning is backfilled
    day=0
    print df_basket['Price/SMA'].ix[day].item()
    while(day < df_basket.shape[0]-1):
        # close the position first
        if(share_hold != 0):
            cash += df['AAPL'].ix[day]*share_hold
            if(share_hold > 0):
                #price_sma.loc[price_sma.index[day],'Order']='SELL'
                #price_sma.loc[price_sma.index[day], 'Signal']='EXIT'
                # need to sell first
                orders.append([df_basket.index[day].date(), 'AAPL', 'SELL', 200])
            elif(share_hold < 0):
                #price_sma.loc[price_sma.index[day],'Order']='BUY'
                #price_sma.loc[price_sma.index[day], 'Signal']='EXIT'
                # need to buy first
                orders.append([df_basket.index[day].date(), 'AAPL', 'BUY', 200])
            share_hold = 0

        #Overboght
        if(((df_basket['Price/SMA'].ix[day] > 1.05) & (df_basket['bollinger_B'].ix[day] > 1)) |
           ((df_basket['%D'].ix[day] > 80) & (df_basket['bollinger_B'].ix[day] > 1)) |
           ((df_basket['Price/SMA'].ix[day] > 1.05) & (df_basket['%D'].ix[day] > 80))):
        #if((df_basket['Price/SMA'].ix[day] > 1.05) & (df_basket['bollinger_B'].ix[day] > 1) & (df_basket['%D'].ix[day] > 80)):
            # 'BUY'
            orders.append([df_basket.index[day].date(), 'AAPL', 'BUY', 200])
            df_basket.loc[df_basket.index[day],'Order']='BUY'
            df_basket.loc[df_basket.index[day], 'Signal']='LONG'
            share_hold += 200
            cash -= df['AAPL'].ix[day]*200
            # hold for 21 days
            if (day < df_basket.shape[0]-21):
                day += 21
            else:
                day = df_basket.shape[0]-1

        elif(((df_basket['Price/SMA'].ix[day] < 0.95) & (df_basket['bollinger_B'].ix[day] < 0)) |
           ((df_basket['%D'].ix[day] < 20) & (df_basket['bollinger_B'].ix[day] < 0)) |
           ((df_basket['Price/SMA'].ix[day] < 0.95) & (df_basket['%D'].ix[day] < 20))):
        #elif((df_basket['Price/SMA'].ix[day] < 0.95) & (df_basket['bollinger_B'].ix[day] < 0) & (df_basket['%D'].ix[day] < 20)):
            # 'SELL'
            orders.append([df_basket.index[day].date(), 'AAPL', 'SELL', 200])
            df_basket.loc[df_basket.index[day],'Order']='SELL'
            df_basket.loc[df_basket.index[day], 'Signal']='SHORT'
            share_hold -= 200
            cash += df['AAPL'].ix[day]*200
            # hold for 21 days
            if (day < df_basket.shape[0]-21):
                day += 21
            else:
                day = df_basket.shape[0]-1

        else:
            day += 1

    # last day close the position
    #day += 1
    if(share_hold != 0):
            cash += df['AAPL'].ix[day]*share_hold
            if(share_hold > 0):
                #price_sma.loc[price_sma.index[day],'Order']='SELL'
                #price_sma.loc[price_sma.index[day], 'Signal']='EXIT'
                # need to sell first
                orders.append([df_basket.index[day].date(), 'AAPL', 'SELL', 200])
            elif(share_hold < 0):
                #price_sma.loc[price_sma.index[day],'Order']='BUY'
                #price_sma.loc[price_sma.index[day], 'Signal']='EXIT'
                # need to buy first
                orders.append([df_basket.index[day].date(), 'AAPL', 'BUY', 200])
            share_hold = 0




    #print price_sma

    #print share_hold
    #sv = share_hold*df['AAPL'].ix[df.shape[0]-1] + cash
    #print cash
    #print sv
    #print price_sma
    # save to csv
    #price_sma.to_csv('./testpart3.csv')
    #save orders
    df_basket.to_csv('./df_basket.csv')
    with open("./basket_orders.csv",'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow(['Date','Symbol', 'Order', 'Shares'])
        wr.writerows(orders)
    #print orders

    of = "./basket_orders.csv"
    df_orders = pd.read_csv(of)
    start_date = df_orders['Date'].min()
    end_date = df_orders['Date'].max()

    benchmark = "./benchmark_order.csv"
    benchmark_orders = pd.read_csv(benchmark)
    #print benchmark_orders
    start_date_benchmark = benchmark_orders['Date'].min()
    end_date_benchmark = benchmark_orders['Date'].max()
    #print start_date_benchmark
    #print end_date_benchmark
    #start_date = '2008-1-1'
    #end_date = '2009-12-31'

    #df = pd.concat([df, sma, price_sma], axis=1)
    #df.columns = ['SPY', symbols[0], 'SMA', 'Price_SMA', 'Order', 'Signal']

    long_days=df_basket[df_basket['Signal']=='LONG'].index.values
    #print long_days
    short_days=df_basket[df_basket['Signal']=='SHORT'].index.values
    #print df

#     orders = df[['Order']][(df['Order'] == 'BUY') | (df['Order'] == 'SELL')]
#     orders['Symbol']=symbols*orders.shape[0]
#     orders['Shares']=np.repeat(200,orders.shape[0])
#     output=orders[['Symbol','Order','Shares']]
#     output.to_csv("./orders2.csv",index_label='Date')

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    portvals.columns = ['strategy']
    nomalized_portvals = portvals / portvals.ix[0, :]
    #print portvals

    benchmarkvals = compute_portvals(orders_file = benchmark, start_val = sv)
    benchmarkvals.columns = ['benchmark']
    nomalized_benchmarkvals = benchmarkvals / benchmarkvals.ix[0, :]
    print benchmarkvals.head()

    #df_AAPL = df['AAPL']
    #print df_AAPL.head()
    #print sv
    #print portvals
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = calculate_portfolio_statistics(portvals, 0.0, 252)

    # Get portfolio stats for SPY
    prices_SPY = get_data(['SPY'], pd.date_range(start_date, end_date))
    #print prices_SPY
    prices_SPY = prices_SPY[['SPY']]
    portvals_SPY = calculate_portfolio_value(prices_SPY, [1.0], sv)
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = calculate_portfolio_sta

    # Plot computed daily portfolio value
    #df_temp = pd.concat([portvals, prices_SPY['SPY']], keys=['Portfolio', 'SPY'], axis=1)
    #df_temp = pd.concat([portvals, benchmarkvals], axis=1)
    #nomalized_df = df_temp / df_temp.ix[0, :]
    #print nomalized_df


    #plot_normalized_prices(df_temp)

    fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()

    ax1.plot(nomalized_benchmarkvals.index, nomalized_benchmarkvals['benchmark'], label='Benchmark', color='black')
    ax1.plot(nomalized_portvals.index, nomalized_portvals['strategy'], label='rule-based portfolio', color='blue')

#     ax2.plot(df.index, df['Price_SMA'], label='Price/SMA', color='red', linestyle='--')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Price')
    #ax.set_xticks((10,100,1000))
#     ax1.set_ylim([50, 250])

#     ax2.set_ylabel('Price/SMA')
    ax1.set_ylim([0, 2.0])
    plot_start_date = '2007-12-25'
    plot_end_date = '2009-12-31'
    ax1.set_xlim([plot_start_date, plot_end_date])
    ax1.axhline(y = 1.0, linestyle='--', color='purple')
#     ax2.axhline(y = 1.1)
    for ld in long_days:
        plt.axvline(x=ld,color='g')
    for sd in short_days:
        plt.axvline(x=sd,color='red')

    ax1.legend(loc='upper right')
#     ax2.legend(loc='lower right')
    fig.autofmt_xdate()
    plt.show()



if __name__ == "__main__":
    test_Basket_strategy_run()