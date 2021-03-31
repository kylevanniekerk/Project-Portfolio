#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:38:56 2021

@author: kyle
"""
import uvicorn 
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from Data_generation import dataMGMT, ETF_df
from portfolio_construction import cointegration, equal_p, efficient_frontier_p, bot_up_opti, spo, analyze, regime_prediction, etf_strat


web = FastAPI()

# =============================================================================
# tickers = { 'Govenment' : ["XGB.TO", "VGV.TO"],
#            'Candian Stocks' : ["BESG.TO", "XCG.TO", "XCV.TO", "XCSR.TO", "XESG.TO", "GGRO.TO" ],
#            'US Stocks' : ["ESGV", "VUG", "VV", "MGC", "MGC", "MGV", "VOO", "VOOG", "VOOV", "VTV", "VIOO", "VIOG", "VIOV", "VB", "VBK", "VBR", "FCMO.TO", "ESG.TO", "XUS.TO", "XUSR.TO", "XSUS.TO", "XMTM.TO"],
#            'US reits' : ["VNQ", "XRE.TO"],
#            'International' : ["VSGX", "VEA", "VWO"],
#            'Developed & Emerging Markets' : ["VWOB", "VGT", "VDE", "XEM.TO", "VIU.TO", "VEE.TO", "VMO.TO", "VVL.TO", "VGRO.TO", "EMB", "ESGE"] 
#            }
# =============================================================================

class userinput(BaseModel):
    investment_amount: int = 90000
    weights_ETF: list = [0.06, 0.1, 0.3, 0.08, 0.16, 0.3]
    symbols: Dict[str, list] = {'ETF Portfolio': ["XIC.TO", "VUN.TO", "AVUV", "XEF.TO", "AVDV", "XEC.TO"],
                                'US Stocks': ['MSFT', 'DIS', 'GOOGL', 'GOOG', 'DHR', 'MA', 'VRSK', 'LIN', 'AXP', 'COST', 'WM', 'CLX', 'AAPL', 
                          'V', 'AMZN', 'AMAT', 'PG', 'MU', 'LRCX', 'HOLX', 'TMO', 'PRGO', 'MSI', 'ICE', 'SCHW', 'AMT', 'NVDA', 
                          'HD', 'GPS', 'ECL', 'REGN', 'USFD', 'IBM', 'TFX', 'XYL', 'CERN', 'CMI', 'BABA', 'VZ', 'JKHY', 'RSG', 
                          'DG', 'NTRS', 'IPG', 'TRMB', 'SJM', 'SWK', 'KKR', 'MRK', 'JNJ', 'JPM'],}
    
@web.get("/")
async def Home():
    return {"This is where the Portfolio management API rests. Built by Kyle van Niekerk"}

@web.post("/data_generation")
async def data(user : userinput):
    donation = pd.DataFrame()
    df, p_val, tstat, tcrit = dataMGMT(user.symbols)
    
    cointegrated, X_train, X_test = cointegration(user.symbols)
    regime_predictions = regime_prediction(cointegrated, X_train, X_test, user.investment_amount)
    equal, equal_sharpe = equal_p(cointegrated, regime_predictions, X_test, user.investment_amount)
    p_efficient_frontier, p_efficient_frontier_sharpe = efficient_frontier_p(equal, cointegrated, regime_predictions, X_test, user.investment_amount)
    bottom_up, bottom_up_sharpe = bot_up_opti(cointegrated, regime_predictions, X_test, user.investment_amount)
    spo_portfolio, spo_portfolio_sharpe = spo(cointegrated, regime_predictions, X_test, user.investment_amount)
    portfolio_sharpe, portfolio_values = analyze(spo_portfolio, spo_portfolio_sharpe, bottom_up, bottom_up_sharpe, p_efficient_frontier, p_efficient_frontier_sharpe, equal, equal_sharpe)
    etf_portfolio, etf_portfolio_sharpe = etf_strat(ETF_df, user.investment_amount)
    
    donation['Equally Weighted'] = equal['Total Portfolio Value'] * 0.1 
    donation['Efficient Frontier'] = p_efficient_frontier['Total Portfolio Value'] * 0.1
    donation['Bottom Up'] = bottom_up['Total Portfolio Value'] * 0.1
    donation['SPO'] = spo_portfolio['Total Portfolio Value'] * 0.1
    donation['ETF Portfolio'] = etf_portfolio['Total Portfolio Value'] * 0.1
    return portfolio_sharpe, portfolio_values, donation , equal, p_efficient_frontier, bottom_up, spo_portfolio

@web.get("/portfolio")
async def portfolio_visualize(equal, p_efficient_frontier, bottom_up, spo_portfolio, etf_portfolio, donation, portfolio_sharpe) :
    fig, ax = plt.subplots(1, 3, figsize = (10,6))
    equal['Total Portfolio Value after donation'] = equal['Total Portfolio Value'] - donation['Equally Weighted'] 
    p_efficient_frontier['Total Portfolio Value after donation'] = p_efficient_frontier['Total Portfolio Value'] - donation['Efficient Frontier'] 
    bottom_up['Total Portfolio Value after donation'] = bottom_up['Total Portfolio Value'] - donation['Bottom Up'] 
    spo_portfolio['Total Portfolio Value after donation'] = spo_portfolio['Total Portfolio Value'] - donation['SPO']
    etf_portfolio['Total Portfolio Value after donation'] = etf_portfolio['Total Portfolio Value'] - donation['ETF Portfolio']
    
    ax[0].plot(equal['Total Portfolio Value after donation'], p_efficient_frontier['Total Portfolio Value after donation'], bottom_up['Total Portfolio Value after donation'], spo_portfolio['Total Portfolio Value after donation'])
    ax[0].set_title('4 different Portfolio Equity Curves')
    
    ax[1].plot(donation)
    ax[1].set_title('Donation Curve')
    
    ax[2].bar(x = portfolio_sharpe.index, height = portfolio_sharpe['Sharpe Ratio'])
    ax[2].set_title('Sharpe Ratios for different Portfolios')
    ax[2].set_xticklabels(portfolio_sharpe.index, rotation = 90)
    
    fig.subtitle('Portfolio Analysis')
    
    fig.show()
    
    return 

if __name__ == "__main__":
    uvicorn.run(web, host="0.0.0.0", port=8000)
    
# uvicorn port_api:web --reload