#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:14:58 2021

@author: kyle
"""
import numpy as np 
import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import t

tickers = { 'Govenment' : ['XGB.TO', 'VGV.TO'],
           'Candian Stocks' : ['BESG.TO', 'XCG.TO', 'XCV.TO', 'XCSR.TO', 'XESG.TO', 'GGRO.TO' ],
           'US Stocks' : ['ESGV', 'VUG', 'VV', 'MGC', 'MGC', 'MGV', 'VOO', 'VOOG', 'VOOV', 'VTV', 'VIOO', 'VIOG', 'VIOV', 'VB', 'VBK', 'VBR', 'FCMO.TO', 'ESG.TO', 'XUS.TO', 'XUSR.TO', 'XSUS.TO', 'XMTM.TO'],
           'US reits' : ['VNQ', 'XRE.TO'],
           'International' : ['VSGX', 'VEA', 'VWO'],
           'Developed & Emerging Markets' : ['VWOB', 'VGT', 'VDE', 'XEM.TO', 'VIU.TO', 'VEE.TO', 'VMO.TO', 'VVL.TO', 'VGRO.TO', 'EMB', 'ESGE'] 
           }

stock_list = []
for i in tickers:
    for n in tickers[i]: # iterate for every stock indices
        # Retrieve data from Yahoo! Finance
        tickerData = yf.Ticker(n)
        tickerDf1 = tickerData.history(period='1d', start='2011-01-01', end='2021-02-27')
        # Save historical data 
        tickerDf1['ticker'] = n # don't forget to specify the index
        tickerDf1['asset class'] = i
        tickerDf1['Returns'] = tickerDf1['Close']/tickerDf1['Close'].shift(1)
        stock_list.append(tickerDf1)
# Concatenate all data
msi = pd.concat(stock_list, axis = 0)
# Transform the data to be ticker column-wise

# Transform the data to be ticker column-wise
Gross_rets = msi.groupby(['Date', 'ticker'])['Returns'].first().unstack()
Gross_rets = Gross_rets.bfill()
rets = Gross_rets - 1
# Fill null values with the values on the row before

rets.head()
T = len(rets)
#rets = rets.fillna(method='bfill')
xrets = rets.values - rets['XGB.TO'].values.reshape(T,1)
xrets_df = pd.DataFrame(xrets, index = rets.index, columns = rets.columns)

xrets_df.head()

prices_df = msi.groupby(['Date', 'ticker'])['Close'].first().unstack()
prices_df = prices_df.bfill()
XGB = prices_df['XGB.TO']
VGV = prices_df['VGV.TO']
# T-Statistic test 
def remove(D, a):
        values = D.pop(a)
        for v in values:
            D[v].remove(a)
            
def t_test(df, null, tickers, df2):
    T = len(df)
    l_etf = []
    tcrit = t.pdf(1 - 0.01, T-1)
    
    null = null
    se = np.std(df) / np.sqrt(T)
    tstat = (np.mean(df) - null) / se
    p_val = 2*(1-t.cdf(abs(tstat), T-1))
    for i, n in enumerate(tstat):
        if abs(n) > tcrit:
            l_etf.append(tstat.index[i])
    
    for i in tickers:
        for k in tickers[i]:
            if k not in l_etf:
                df = df.drop(k, axis = 1)
                df2 = df2.drop(k, axis = 1)
                tickers[i].remove(k)
    return df, tickers, p_val, tstat, df2
     
xrets_df, tickers, p_val, tstat, prices_df = t_test(xrets_df, 0, tickers, prices_df)
prices_df['XGB.TO'] = XGB
prices_df['VGV.TO'] = VGV
# =============================================================================
# fig, axes = plt.subplots(3,2, figsize=(12, 8),sharex=True)
# pagoda = ["#965757", "#D67469", "#4E5A44", "#A1B482", '#EFE482', "#99BFCF"] # for coloring
# for i, k in enumerate(tickers.keys()):
# # Iterate for each region
#     ax = axes[int(i/2), int(i%2)]
#     for j,x in enumerate(tickers[k]):
#         # Iterate and plot for each stock index in this region
#         ax.plot(rets.index, rets[x], marker='', linewidth=1, color = pagoda[j])
#        # ax.legend([tickers[t]], loc='upper left', fontsize=7)
#         ax.set_title(k, fontweight='bold')
# fig.text(0.5,0, "Year", ha="center", va="center", fontweight ="bold")
# fig.text(0,0.5, "Price Change/Return (%)", ha="center", va="center", rotation=90, fontweight ="bold")
# fig.suptitle("Price Change/Return for Major Stock Indices based on 2010", fontweight ="bold",y=1.05, fontsize=14)
# fig.tight_layout()
# =============================================================================











