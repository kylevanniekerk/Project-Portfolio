# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 08:34:10 2021

@author: Admin
"""

import numpy as np 
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import t



tickers = {'ETF' : ['XIC', 'VUN', 'AVUV', 'XEF.TO', 'AVDV', 'XEC'],
           'US Stocks' : ['MSFT', 'DIS', 'GOOGL', 'GOOG', 'DHR', 'MA', 'VRSK', 'LIN', 'AXP', 'COST', 'WM', 'CLX', 'AAPL', 
                          'V', 'AMZN', 'AMAT', 'PG', 'MU', 'LRCX', 'HOLX', 'TMO', 'PRGO', 'MSI', 'ICE', 'SCHW', 'AMT', 'NVDA', 
                          'HD', 'GPS', 'ECL', 'REGN', 'USFD', 'IBM', 'TFX', 'XYL', 'CERN', 'CMI', 'BABA', 'VZ', 'JKHY', 'RSG', 
                          'DG', 'NTRS', 'IPG', 'TRMB', 'SJM', 'SWK', 'KKR', 'MRK', 'JNJ', 'JPM'], 
           }

def remove(D, a):
        values = D.pop(a)
        for v in values:
            D[v].remove(a)

def dataMGMT(dictionary):
    stock_list = []
    etf_list = []
    for key, i in dictionary.items():
        if key == 'ETF':
            for n in i:
                # iterate for every stock indices
                # Retrieve data from Yahoo! Finance
                tickerDataETF = yf.Ticker(n)
                tickerDfETF = tickerDataETF.history(period='1d', start='2011-01-01', end='2021-02-27')
                # Save historical data 
                tickerDfETF['ticker'] = n # don't forget to specify the index
                tickerDfETF['asset class'] = key
                tickerDfETF['Returns'] = tickerDfETF['Close']/tickerDfETF['Close'].shift(1)
                etf_list.append(tickerDfETF)
        else:
            for n in i:
                tickerDatastock = yf.Ticker(n)
                tickerDfstock = tickerDatastock.history(period='1d', start='2011-01-01', end='2021-02-27')
                tickerDfstock['ticker'] = n # don't forget to specify the index
                tickerDfstock['asset class'] = key
                tickerDfstock['Returns'] = tickerDfstock['Close']/tickerDfstock['Close'].shift(1)
                stock_list.append(tickerDfstock)
                
    # Concatenate all data
    msi_etf = pd.concat(etf_list, axis = 0)
    msi_stock = pd.concat(stock_list, axis = 0)
    
    ETF_df = msi_etf.groupby(['Date', 'ticker'])['Close'].first().unstack()
    
    # Transform the data to be ticker column-wise
    Gross_rets = msi_stock.groupby(['Date', 'ticker'])['Returns'].first().unstack()
    Gross_rets = Gross_rets.bfill()
    rets = Gross_rets - 1
    
    rets.head()
    
    
    rets_df = pd.DataFrame(rets, index = rets.index, columns = rets.columns)
    
    rets_df.head()
    
    prices_df = msi_stock.groupby(['Date', 'ticker'])['Close'].first().unstack()
    prices_df = prices_df.bfill()
   
    # T-Statistic test           
    T = len(rets_df)
    l_etf = []
    tcrit = t.pdf(1 - 0.01, T-1)
    
    null = 0.10
    se = np.std(rets_df) / np.sqrt(T)
    tstat = (np.mean(rets_df) - 0) / se
    p_val = 2*(1-t.cdf(abs(tstat), T-1))
    for i, n in enumerate(tstat):
        if abs(n) > tcrit:
            l_etf.append(tstat.index[i])
    
    for key, i in dictionary.items():
        if key != 'ETF':
            for k in i:
                if k not in l_etf:
                    prices_df = prices_df.drop(k, axis = 1)
                    tickers[key].remove(k)
    
    return prices_df, p_val, tstat, tcrit, rets_df, ETF_df

prices_df, p_val, tstat, tcrit, rets_df, ETF_df = dataMGMT(tickers)