#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 13:45:33 2021

@author: kyle
"""
# Data analysis and manipulation
import numpy as np
import pandas as pd 

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Statistics and machine learning
from statsmodels.tsa.api import adfuller
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM 
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import linregress
import scipy.optimize as sco

from Data_generation import xrets_df, prices_df, tickers

# =============================================================================
# # Load Data from Excel spreadsheet
# fundamental_data = pd.ExcelFile('SPO_Data.xlsx')
# # Identifying the sheet that will be used. 
# features = fundamental_data.parse('Fundamentals')
# features = features.drop('Unnamed: 0', axis = 1)
# 
# # Drop column Name as it is redundant because we already have the symbol
# features = features.drop('Name', axis = 1)
# features.head()
# 
# # Identifying value of K to use for K-Means Clustering using elbow technique
# def find_k(features):
#     errors = []
#     # Itering over possible values of K
#     for k in range(1,51):
#         model = KMeans(n_clusters = k)
#         model.fit(features)
#         errors.append(sum(np.min(cdist(features, model.cluster_centers_, 'euclidean'), axis = 1)))
# 
#     # Plot the graph and Identify elbow
#     with plt.style.context(['classic','ggplot']):
#         plt.figure(figsize = (10,6))
#         plt.plot(errors)
#         plt.xlabel('Clusters')
#         plt.ylabel('Errors')
#         plt.title('Elbow technique')
#     return plt.show()
#         
# # Make copy of features DataFrame where symbol is the index
# copy_features = features.copy()
# copy_features = copy_features.reindex(index = copy_features['Symbol'], columns = copy_features.columns)
# 
# # Adding data back into the Dataframe after resetting the index
# copy_features['P/E'] = features['P/E'].values
# copy_features['EPS'] = features['EPS'].values
# copy_features['MarketCap'] = features['MarketCap'].values
# copy_features = copy_features.drop('Symbol', axis = 1)
# copy_features.head()
# 
# # See if there are any missing values
# copy_features.isnull().sum()
# 
# # Use function to see graph of KMeans Clustering Elbow technique, While filling missing values with Zero
# find_k(copy_features.fillna(0))
# 
# # K = 15 is the value I will be using to look for tradable relationships
# k_means = KMeans(n_clusters = 15, random_state = 101)
# k_means.fit(copy_features.fillna(0))
# 
# # Checking the labels of the clusetrs
# copy_features['Cluster'] = k_means.labels_
# copy_features.head()
# copy_features.tail()
# 
# # Create Dataframe
# clusters = pd.DataFrame()
# # Group Clusters together that are greater than 1 
# clusters = pd.concat(i for clusters, i in copy_features.groupby(copy_features['Cluster']) if len(i) > 1)
# clusters.head()
# clusters.tail()
# =============================================================================

# Create a function to identify each possible pairs 
def create_pairs(list_of_symbols):
    pairs = []
    # intialize placeholders for the symbols in each pair
    x = 0
    y = 0
    for count, symbol in enumerate(list_of_symbols):
        for nxcount, nxsymbol in enumerate(list_of_symbols):
            x = symbol
            y = nxsymbol
            if x != y:
                pairs.append([x,y])
    return pairs

# create list of grouped clusters

all_pairs = []
for i in tickers:
    list_of_symbols = []
    for k in tickers[i]:
        list_of_symbols.append(k)
    pairs = create_pairs(list_of_symbols)
    all_pairs.append(pairs)   

# The function to parse the training and testing data from one another
# over the period January 4th 2018 - June 12th 2018
def parse_data(data, percent):
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
  # Itering over the data
    for count, symbol in enumerate(data):
        parsable = data[data[symbol].notnull()][symbol]
        # Parsing out the data for the training data
        copy_train, copy_test = train_test_split(parsable, test_size = percent)
        X_train[symbol] = copy_train.values
        X_test[symbol] = copy_test.values
    return X_train, X_test

X_train_xrets, X_test_xrets = parse_data(xrets_df, 0.33)
X_train, X_test = parse_data(prices_df, 0.33)

def cointegrated(all_pairs, X_train):
    # creating a list to hold cointegrated pairs
    cointegrated = []
    # iterate over each pair in possible pairs list; pair is a list of our 2 stock symbols
    for count, allo in enumerate(all_pairs):
        for pair in all_pairs[count]: 
            # getting data for each stock in pair from training_df
            ols = linregress(X_train[str(pair[1])], X_train[str(pair[0])]) #note scipy's linregress takes in Y then X
            # storing slope or hedge ratio in variable
            slope = ols[0]
            # creating spread
            spread = X_train[str(pair[1])] - (slope * X_train[str(pair[0])])
            # testing spread for cointegration
            cadf = adfuller(spread,1)
            # checking to see if spread is cointegrated, if so then store pair in cointegrated list
            if cadf[0] < cadf[4]['1%']:
                print('Pair Cointegrated at 99% Confidence Interval')
                # appending the X and Y of pair
                cointegrated.append([pair[0],pair[1]])
            elif cadf[0] < cadf[4]['5%']:
                print('Pair Cointegrated at 95% Confidence Interval')
                # appending the X and Y of pair
                cointegrated.append([pair[0],pair[1]])
            elif cadf[0] < cadf[4]['10%']:
                print('Pair Cointegrated at 90% Confidence Interval')
                cointegrated.append(pair[0],pair[1])
            else:
                print('Pair Not Cointegrated ')

    return cointegrated 

cointegrated = cointegrated(all_pairs, X_train)

class statarbi(object):
    def __init__(self, df1, df2, ma, floor, ceiling, beta_lookback, exit_zscore = 0):
        # Setting the attributes
        self.df1 = df1 # Array of the prices for X
        self.df2 = df2 # Array of the prices for Y
        self.ma = ma # The lookback period
        self.floor = floor # The Threshold to buy for the z-score
        self.ceiling = ceiling # The threshold to sell for the z-score 
        self.Close = 'Close Long' # Used as a close signal for longs
        self.Cover = 'Cover Short' # Used as a close signal for shorts
        self.exit_zscore = exit_zscore
        self.beta_lookback = beta_lookback # The lookback for the hedge ratio
    
    def spread(self):
        # Create new Dataframe
        self.df = pd.DataFrame(index = range(0, len(self.df1)))
        self.df['X'] = self.df1
        self.df['Y'] = self.df2
        
        # Calculate the beta of the pairs
        ols = linregress(self.df['Y'], self.df['X'])
        self.df['Beta'] = ols[0]
        # Calculate the spread 
        self.df['Spread'] = self.df['Y'] - (self.df['Beta'].rolling(window = self.beta_lookback).mean() * self.df['X'])
        return self.df.head()

    def signal_generation(self):
        # Creating z-score
        self.df['Z-Score'] = (self.df['Spread'] - self.df['Spread'].rolling(window = self.ma).mean()) / self.df['Spread'].rolling(window = self.ma).std()
        self.df['Prior Z-score'] = self.df['Z-Score'].shift(1)
        # Creating Buy and Sell signals where to long, short and exit
        self.df['Longs'] = (self.df['Z-Score'] <= self.floor) * 1.0 # Buy the spread
        self.df['Shorts'] = (self.df['Z-Score'] >= self.ceiling) * 1.0 # Short the spread
        self.df['Exit'] = (self.df['Z-Score'] <= self.exit_zscore) * 1.0
        # track positions with for loop
        self.df['Long Market'] = 0.0
        self.df['Short Market'] = 0.0 
        # Setting variables to track whether or to be long while iterating
        self.long_market = 0 
        self.short_market = 0 
        # Determing when to trade
        for i, value in enumerate(self.df.iterrows()):
            if value[1]['Longs'] == 1.0:
                self.long_market = 1
            elif value[1]['Shorts'] == 1.0:
                self.short_market = 1
            elif value[1]['Exit'] == 1.0:
                self.long_market = 0
                self.short_market = 0
            self.df.iloc[i]['Long Market'] = self.long_market
            self.df.iloc[i]['Short Market'] = self.short_market
        return self.df.head()
    
    def returns(self, allocation, pair_number):
        '''
        Parameters
        ----------
        allocation : The amount of Capital for each pair
        pair_number : String to annotate the plots
        '''
        self.allocation = allocation
        self.pair = pair_number
        
        self.portfolio = pd.DataFrame(index = self.df.index)
        self.portfolio['Positions'] = self.df['Long Market'] - self.df['Short Market']
        self.portfolio['X'] = 1.0 * self.df['X'] * self.portfolio['Positions']
        self.portfolio['Y'] = self.df['Y'] * self.portfolio['Positions']
        self.portfolio['Total'] = self.portfolio['X'] + self.portfolio['Y'] 
        # Creating a stream of returns
        self.portfolio['Returns'] = self.portfolio['Total'].pct_change()
        self.portfolio['Returns'] = self.portfolio['Returns'].fillna(0.0)
        self.portfolio['Returns'] = self.portfolio['Returns'].replace([np.inf, -np.inf], 0.0)
        self.portfolio['Returns'] = self.portfolio['Returns'].replace(-1.0, 0.0)
        # Calculating the Metrics
        self.mu = (self.portfolio['Returns'].mean())
        self.sigma = (self.portfolio['Returns'].std())
        self.portfolio['Win'] = np.where(self.portfolio['Returns'] > 0, 1, 0)
        self.portfolio['Loss'] = np.where(self.portfolio['Returns'] < 0, 1 ,0)
        self.wins = self.portfolio['Win'].sum()
        self.losses = self.portfolio['Loss'].sum()
        self.tot_trades = self.wins + self.losses
        # Calculating the Sharpe ratio with an interest rate of 0.75  
        interest_rate_assumption = 0.75 # Risk free Rate
        self.sharpe = (self.mu - interest_rate_assumption) / self.sigma
        # win loss ration
        self.win_loss = (self.wins / self.losses)
        self.prob_win = (self.wins / self.tot_trades)
        self.prob_loss = (self.losses / self.tot_trades)
        self.avg_return_win = (self.portfolio['Returns'] > 0).mean()
        self.avg_return_loss = (self.portfolio['Returns'] < 0).mean()
        # Calculating the Payout ratio
        self.payout_ratio=(self.avg_return_win/self.avg_return_loss)
        # Creating the Equity Curve
        self.portfolio['Returns'] = (self.portfolio['Returns'] + 1.0).cumprod()
        self.portfolio['Trade Returns'] = (self.portfolio['Total'].pct_change())
        self.portfolio['Portfolio Value'] = (self.allocation * self.portfolio['Returns'])
        self.portfolio['Portfolio Returns'] = self.portfolio['Portfolio Value'].pct_change()
        self.portfolio['Initial Value'] = self.allocation
        
        with plt.style.context(['ggplot', 'seaborn-paper']):
        # Plotting Portfolio Value
            plt.plot(self.portfolio['Portfolio Value'])
            plt.plot(self.portfolio['Initial Value'])
            plt.title('%s Strategy Return' %(self.pair))
            plt.legend(loc = 0)
            plt.show()
        return

equal = pd.DataFrame()
for k, pair in enumerate(cointegrated):
    pair_arbi = statarbi(X_test[pair[0]], X_test[pair[1]], 10, -2, 2, 10)
    pair_arbi.spread()
    pair_arbi.signal_generation()
    pair_arbi.returns(90000/len(cointegrated), str(pair))
    equal[k] = pair_arbi.portfolio['Portfolio Value']
    
equal['Cash'] = 10000   
equal['Total Portfolio Value'] =   equal.sum(axis = 1)
equal['Returns'] = np.log(equal['Total Portfolio Value'] / equal['Total Portfolio Value'].shift(1))

# Mean, sigma and Sharpe
equal_mu = equal['Returns'].mean()
equal_sigma = equal['Returns'].std()

# In as of December 2017, the fed funds rate was 1.5%. We'll use this as our interest rate assumption. 
rate = 0.015
equal_sharpe = round((equal_mu - rate) / equal_sigma, 2)

plt.figure(figsize = (10,6))
plt.plot(equal['Total Portfolio Value'])
plt.title('Equally Weighted Portfolio Equity Curve')
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Efficient Frontier 

returns = pd.DataFrame()
for i in equal:
    returns[i] = np.log(equal[i] / equal[i].shift(1))

returns = returns.drop('Cash', axis = 1)
returns = returns.drop('Returns', axis = 1)
returns = returns.drop('Total Portfolio Value', axis = 1)

def efficient_frontier(returns, rate = 0.015):
    # Create lists with returns, variance and sharpe ratios
    p_returns = []
    p_volatility = []
    p_sharpe = []
    
    for i in range(500):
        # assign weights
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        # Getting returns
        current_return = np.sum(returns.mean() * weights) * 252
        p_returns = np.append(p_returns, current_return)
        # Getting variances
        variance = np.dot(weights, np.dot(returns.cov() * 252 , weights))
        # Volatility
        volatility = np.sqrt(variance)
        p_volatility = np.append(p_volatility, volatility)
        # Sharpe
        ratio = (current_return - rate)/volatility
        p_sharpe = np.append(p_sharpe, ratio)
        
        p_returns = np.array(p_returns)
        p_volatility = np.array(p_volatility)
        p_sharpe = np.array(p_sharpe)
        # plot to find efficient returns
        plt.figure(figsize = (10,6))
        plt.scatter(p_volatility, p_returns, c = p_sharpe, marker = 'o')
        plt.xlabel('Expected Volatility')
        plt.ylabel('Efficient Frontier')
        plt.colorbar(label = 'Sharpe Ratio')
        plt.show()
        
    return 

efficient_frontier(returns.fillna(0))

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)
 
def stats(weights, rate = 0.015):
    weights = np.array(weights)
    p_returns = np.sum(returns.mean()*weights)*252
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    p_sharpe = (p_returns - rate) / p_volatility

    return np.array([p_returns,p_volatility,p_sharpe])   

stats(weights)
# function for optimization
def minimize(weights):
    return -stats(weights)[2]

minimize(weights)

# Finding the optimal weights
def optimal_weights(weights):
    # variables for optimization
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for x in range(len(returns.columns)))
    starting_weights = len(returns.columns) * [1 / len(returns.columns)]
    most_optimal = sco.minimize(minimize, starting_weights, method='SLSQP', bounds = bounds, constraints = constraints)
    best_weights = most_optimal['x'].round(3)
    return best_weights, print('Weights:',best_weights)

optimal_weights = optimal_weights(weights)

investment =  90000
p_efficient_frontier = pd.DataFrame()
for q, pair in enumerate(cointegrated):
    pair_arbi_frontier = statarbi(X_test[pair[0]], X_test[pair[1]], 10, -2, 2, 10)
    pair_arbi_frontier.spread()
    pair_arbi_frontier.signal_generation()
    pair_arbi_frontier.returns(round(investment * optimal_weights[0][q],2), str(pair))
    p_efficient_frontier[q] = pair_arbi_frontier.portfolio['Portfolio Value']
    
p_efficient_frontier['Cash'] = 10000   
p_efficient_frontier['Total Portfolio Value'] =   p_efficient_frontier.sum(axis = 1)
p_efficient_frontier['Returns'] = np.log(p_efficient_frontier['Total Portfolio Value'] / p_efficient_frontier['Total Portfolio Value'].shift(1))

p_efficient_frontier_mu = p_efficient_frontier['Returns'].mean()
p_efficient_frontier_sigma = p_efficient_frontier['Returns'].std()
#recall that we initialized our interest assumption earlier
p_efficient_frontier_sharpe = (p_efficient_frontier_mu - rate) / p_efficient_frontier_sigma

plt.figure(figsize=(10,6))
plt.plot(p_efficient_frontier['Total Portfolio Value'])
plt.title('Efficient Frontier Portfolio Equity Curve')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class GMM_randomforests(object):
    def __init__(self, historical_rets_train, historical_rets_test, base_portfolio_rets, gmm_components, df, base_portfolio_df):
        '''
    
        Parameters
        ----------
        base_portfolio_rets : this is our figurative live data; ie returns; 5/01/18-6/12/18 from either
            Equally Weighted or Efficient Frontier Portfolios dependent upon implementation
            of Bottom Up or Stereoscopic Portfolio Optimization (SPO) Framework.
            
            ex. data over 01/04/18-04/30/18
          
            we would first split this 80/20 the 80% is our training set the 20% is our testing set
          
            we would then do another split on our training set created above this is so that 
            if we can better understand the historical regimes and recalibrate our models if 
            necessary before actually predicting our 5/1/18-6/12/18 testing set.
          
            in this ex. our gmm_training_train is 80% of the period 01/04/18-4/30/18
            our gmm_training_test is 20% of the period 01/04/18-4/30/18 and our
            gmm_test_actual is 05/01/18-6/12/18
            
        gmm_components : type:int; for number of components for GMM
        
        df : The entire dataframe containing prior trading history; the dataframe from either Equally Weighted or
            Efficient Frontier Portfolios; Our Random Forests Implementation will take this dataframe created by
            our statarb class(i.e. from the prior portfolios) and add our features to it. It will then use these
            features to predict the regimes of our test period. Recall that our Equally Weighted and Efficient
            Frontier Portfolios were constructed over our assessment period of 5/1/18 to 6/12/18. We will then
            be able to store our predictions in a varible for our test period. These predictions will be passed
            into a new statarb object as a parameter and be used to create the Bottom Up and SPO Framework Portfolios.
        
        base_portfolio_df : (i.e.adbe_antm.df,etc) Note: for the Bottom Up Implementation this df would be the Equally Weighted df but for the
            SPO Framework df this would be the df from the Efficient Frontier implementation

        -------
        
        '''
        self.historical_rets_train = historical_rets_train
        self.historical_rets_test = historical_rets_test
        self.base_portfolio_rets = base_portfolio_rets
        self.gmm_components = gmm_components
        self.max_iter = 300
        self.random_state = 0
        self.df = df
        #self.total_training_start=total_training_start
        #self.total_training_end=total_training_end
        self.base_portfolio_df = base_portfolio_df
        self.volatility = self.historical_rets_train.rolling(window = 5).std()
        self.negative_volatility = np.where(self.historical_rets_train < 0 , self.historical_rets_train.rolling(window = 5).std(), 0)
        
    def make_GMM(self):
        model_kwds = dict(n_components = self.gmm_components, max_iter = self.max_iter, n_init = 100, random_state = 1)
        gmm = GM(**model_kwds)
        return gmm
    
    def analyze_historical_regimes(self):
        # Create the Guassian mixture mdoel
        self.gmm = self.make_GMM()
        # instantiating the Xtrain as the gmm_training_train; (the 80% of total training period)
        self.gmm_Xtrain = np.array(self.historical_rets_train).reshape(-1, 1)
        # Fitting the GMM on the Training Set(note this is the internal training set within the broader training set)
        self.gmm.fit(self.gmm_Xtrain.astype(int))
        #Making predictions on the historical period; ie. the gmm_training_train
        self.gmm_historical_predictions = self.gmm.predict(self.gmm_Xtrain.astype(int))
        #Making Predictions on the gmm_training_test (i.e. the 20% of total training period;)
        self.gmm_Xtest = np.array(self.historical_rets_test).reshape(-1,1)
        self.gmm_training_test_predictions = self.gmm.predict(self.gmm_Xtest.astype(int))
        #Fitting the Model on ACTUAL data we want to Predict Regimes For
        self.gmm_Actual = np.array(self.base_portfolio_rets).reshape(-1,1)
        self.base_portfolio_predictions = self.gmm.predict(self.gmm_Actual)
        return 
    
    def historical_regime_returns_volatility(self, plotTitle):
        self.plotTitle = plotTitle
        data = pd.DataFrame({'Volatility' : self.volatility, 'Regime': self.gmm_historical_predictions, 'Returns': self.historical_rets_train})

        with plt.style.context(['classic','seaborn-paper']):
            fig,ax = plt.subplots(figsize = (15,10), nrows = 1, ncols = 2)
            
        left = 0.125 # the left side of the subplots of the figure
        right = 0.9 # the right side of the subplots of the figure
        bottom = .125 # the bottom of the subplots of the figure
        top = 0.9 # the top of the subplots of the figure
        wspace = .5 # the amount of width reserved for blank space between subplots
        hspace = 1.1 # the amount of height reserved for white space between subplots 
        
        # function that adjusts subplots using the above paramters
        plt.subplots_adjust(
        left = left,
        bottom = bottom,
        right = right,
        top = top,
        wspace = wspace,
        hspace = hspace
        )
        
        plt.suptitle(self.plotTitle, y = 1, fontsize=20)
        
        plt.subplot(121)
        sns.swarmplot(x = 'Regime', y = 'Volatility', data = data)#,ax=ax[0][0])
        plt.title('Regime to Volatility')
        
        plt.subplot(122)
        sns.swarmplot(x = 'Regime', y = 'Returns', data = data)#, ax=ax[0][1])
        plt.title('Regime to Returns')
        plt.tight_layout()
        plt.show()
        return
        
    def train_random_forests(self):
        # adding features to the Dataframe assumption is that the df is over the entire period
        # getting vix to add as feature
        #self.VIX = pdr.get_data_yahoo('^VIX',start=self.total_training_start,end=self.total_training_end)
        # creating features
        #self.df['VIX']=self.VIX['Close']
        
        self.df['6 X Vol'] = self.df['X'].rolling(window=6).std()
        self.df['6 Y Vol'] = self.df['Y'].rolling(window=6).std()
        self.df['6 Spread Vol'] = self.df['Spread'].rolling(window=6).std()
        self.df['6 Z-Score Vol'] = self.df['Z-Score'].rolling(window=6).std()
        
        self.df['12 X Vol'] = self.df['X'].rolling(window=12).std()
        self.df['12 Y Vol'] = self.df['Y'].rolling(window=12).std()
        self.df['12 Spread Vol'] = self.df['Spread'].rolling(window=12).std()
        self.df['12 Z-Score Vol'] = self.df['Z-Score'].rolling(window=12).std()
        
        self.df['15 X Vol'] = self.df['X'].rolling(window=15).std()
        self.df['15 Y Vol'] = self.df['Y'].rolling(window=15).std()
        self.df['15 Spread Vol'] = self.df['Spread'].rolling(window=15).std()
        self.df['15 Z-Score Vol'] = self.df['Z-Score'].rolling(window=15).std()

        #self.base_portfolio_df['VIX']=self.VIX['Close']
        self.base_portfolio_df['6 X Vol'] = self.df['X'].rolling(window=6).std()
        self.base_portfolio_df['6 Y Vol'] = self.df['Y'].rolling(window=6).std()
        self.base_portfolio_df['6 Spread Vol'] = self.df['Spread'].rolling(window=6).std()
        self.base_portfolio_df['6 Z-Score Vol'] = self.df['Z-Score'].rolling(window=6).std()
        
        self.base_portfolio_df['12 X Vol'] = self.df['X'].rolling(window=12).std()
        self.base_portfolio_df['12 Y Vol'] = self.df['Y'].rolling(window=12).std()
        self.base_portfolio_df['12 Spread Vol'] = self.df['Spread'].rolling(window=12).std()
        self.base_portfolio_df['12 Z-Score Vol'] = self.df['Z-Score'].rolling(window=12).std()
        
        self.base_portfolio_df['15 X Vol'] = self.df['X'].rolling(window=15).std()
        self.base_portfolio_df['15 Y Vol'] = self.df['Y'].rolling(window=15).std()
        self.base_portfolio_df['15 Spread Vol'] = self.df['Spread'].rolling(window=15).std()
        self.base_portfolio_df['15 Z-Score Vol'] = self.df['Z-Score'].rolling(window=15).std()

        #replacing na values
        self.df.fillna(0, inplace=True)
        self.df = self.df.drop(['X','Y','Longs','Shorts','Exit','Long Market','Short Market'], axis = 1)
        # Creating X_train for RF over the historical Period; Will train over the historical period, ie self.historical_training_start/end then predict
        self.RF_X_train = self.df[:len(self.gmm_historical_predictions)][['6 X Vol', '6 Y Vol','6 Spread Vol','6 Z-Score Vol','12 X Vol','12 Y Vol','12 Spread Vol','12 Z-Score Vol','15 X Vol','15 Y Vol','15 Spread Vol','15 Z-Score Vol']]
        # Removiing unnecessary columns
        #self.RF_X_train.drop(['X','Y','Longs','Shorts','Exit','Long Market','Short Market'], inplace = True, axis = 1)
        #setting Y_Train for the RF to the predictions of GMM over historical period
        self.RF_Y_TRAIN = self.gmm_historical_predictions
        self.base_portfolio_df = self.base_portfolio_df.drop(['X','Y','Longs','Shorts','Exit','Long Market','Short Market'], axis = 1)
        self.RF_X_TEST = self.base_portfolio_df[['6 X Vol','6 Y Vol','6 Spread Vol','6 Z-Score Vol','12 X Vol','12 Y Vol','12 Spread Vol','12 Z-Score Vol','15 X Vol','15 Y Vol','15 Spread Vol','15 Z-Score Vol']]
        #dropping unnecessary columns from train data
        #self.RF_X_TEST.drop(['X','Y','Longs','Shorts','Exit','Long Market','Short Market'], inplace = True, axis = 1)
        # Predictions for the x test over the internal testing period
        self.RF_Y_TEST = self.base_portfolio_predictions #regime predictions for base portfolio
        # Building the random forest and check accuracy
        self.RF_Model = RF(n_estimators = 100)
        #training the random forests model on assessment period data
        self.RF_Model.fit(self.RF_X_train.fillna(0),self.RF_Y_TRAIN)
        #Making predictions for base portfolio period
        self.RF_BASE_PORTFOLIO_PREDICTIONS = self.RF_Model.predict(self.RF_X_TEST.fillna(0))
        #Checking Precision of Predictions
        print(confusion_matrix(self.RF_Y_TEST,self.RF_BASE_PORTFOLIO_PREDICTIONS))
        print('\n')
        print(classification_report(self.RF_Y_TEST,self.RF_BASE_PORTFOLIO_PREDICTIONS))
        
        return 

historical = pd.DataFrame()
hist_rets = pd.DataFrame()
for k, pair in enumerate(cointegrated):
    hist_pair_arbi = statarbi(X_train[pair[0]], X_train[pair[1]], 10, -2, 2, 10)
    hist_pair_arbi.spread()
    hist_pair_arbi.signal_generation()
    hist_pair_arbi.returns(90000/len(cointegrated), str(pair))
    historical[k] = hist_pair_arbi.portfolio['Portfolio Value']
    hist_rets[k] = hist_pair_arbi.portfolio['Returns']

hist_rets_train, hist_rets_test = train_test_split(hist_rets, test_size = 0.33)
 
regime_predictions = pd.DataFrame()
for p, pair in enumerate(cointegrated):
    hist_pair_arbi = statarbi(X_train[pair[0]], X_train[pair[1]], 10, -2, 2, 10)
    hist_pair_arbi.spread()
    hist_pair_arbi.signal_generation()
    hist_pair_arbi.returns(90000/len(cointegrated), str(pair))
    pair_arbi = statarbi(X_test[pair[0]], X_test[pair[1]], 10, -2, 2, 10)
    pair_arbi.spread()
    pair_arbi.signal_generation()
    pair_arbi.returns(90000/len(cointegrated), str(pair))
    gmm_pair = GMM_randomforests(hist_rets_train[p], hist_rets_test[p], pair_arbi.portfolio['Returns'], 2, hist_pair_arbi.df, pair_arbi.df)
    gmm_pair.analyze_historical_regimes()
    gmm_pair.historical_regime_returns_volatility(str(pair))
    gmm_pair.train_random_forests()
    regime_predictions[p] = gmm_pair.base_portfolio_predictions

class statarb_update(object):
#np.seterr(divide='ignore',invalid='ignore')

    def __init__(self, df1, df2, ptype, ma, floor, ceiling, beta_lookback, regimePredictions, p2Objective, avoid1 = 1, target1 = 0, exit_zscore = 0):
        #setting the attributes of the data cleaning object
        self.df1 = df1 #the complete dataframe of X
        self.df2 = df2 # the comlete dataframe of Y
        self.df = pd.DataFrame(index=df1.index) #creates a new dataframe in the create_spread method
        self.ptype = ptype #the portfolio type 1= standard implementation 2=machine learning implementation
        self.ma = ma# the moving average period for the model
        self.floor = floor #the buy threshold for the z-score
        self.ceiling = ceiling #the sell threshold for the z-score
        self.Close = 'Close Long' #used as close signal for longs
        self.Cover = 'Cover Short' #used as close signal for shorts
        self.exit_zscore = exit_zscore #the z-score
        self.beta_lookback = beta_lookback #the lookback of beta for hedge ratio
        self.regimePredictions = regimePredictions #the regime predictions from GMM for p2=2 implementation
        self.avoid1 = avoid1 #the regime to avoid
        self.target1 = target1 #the regime to target
        self.p2Objective = p2Objective # the objective of p2 implementation; can be 'Avoid','Target',or 'None';
        
    #create price spread
    def create_spread(self):
        if self.ptype == 1:   
            #setting the new dataframe values for x and y of the closing
            #prices of the two dataframes passed in
            self.df['X'] = self.df1
            self.df['Y'] = self.df2
            #calculating the beta of the pairs
            self.ols = linregress(self.df['Y'],self.df['X'])
            #setting the hedge ratio
            self.df['Hedge Ratio'] = self.ols[0]
            
            self.df['Spread'] = self.df['Y'] - (self.df['Hedge Ratio']*self.df['X'])

        elif self.ptype == 2:
            #setting the new dataframe values for x and y of the closing
            #prices of the two dataframes passed in
            self.df['X'] = self.df1
            self.df['Y'] = self.df2
            #calculating the beta of the pairs
            self.ols = linregress(self.df['Y'],self.df['X'])
            #setting the hedge ratio
            self.df['Hedge Ratio'] = self.ols[0]
            #creating spread
            self.df['Spread'] = self.df['Y'] - (self.df['Hedge Ratio']*self.df['X'])
            #creating the z-score
            self.df['Z-Score'] = (self.df['Spread'] - self.df['Spread'].rolling(window = self.ma).mean()) / self.df['Spread'].rolling(window = self.ma).std()
            #Creating the features columns
            self.df['6 X Vol'] = self.df['X'].rolling(window = 6).std()
            self.df['6 Y Vol'] = self.df['Y'].rolling(window = 6).std()
            self.df['6 Spread Vol'] = self.df['Spread'].rolling(window = 6).std()
            self.df['6 Z-Score Vol'] = self.df['Z-Score'].rolling(window = 6).std()
            
            self.df['12 X Vol'] = self.df['X'].rolling(window = 12).std()
            self.df['12 Y Vol'] = self.df['Y'].rolling(window = 12).std()
            self.df['12 Spread Vol'] = self.df['Spread'].rolling(window = 12).std()
            self.df['12 Z-Score Vol'] = self.df['Z-Score'].rolling(window = 12).std()
            
            self.df['15 X Vol'] = self.df['X'].rolling(window = 15).std()
            self.df['15 Y Vol'] = self.df['Y'].rolling(window = 15).std()
            self.df['15 Spread Vol'] = self.df['Spread'].rolling(window = 15).std()
            self.df['15 Z-Score Vol'] = self.df['Z-Score'].rolling(window = 15).std()
            #Creating the Regime Prediction Column
            self.df['Regime'] = 0
            self.df['Regime'] = self.regimePredictions.astype(int)

        return

    def generate_signals(self):
       # Creating z-score
        self.df['Z-Score'] = (self.df['Spread'] - self.df['Spread'].rolling(window = self.ma).mean()) / self.df['Spread'].rolling(window = self.ma).std()
        self.df['Prior Z-score'] = self.df['Z-Score'].shift(1)
        if self.ptype == 1:   
            # Creating Buy and Sell signals where to long, short and exit
            self.df['Longs'] = (self.df['Z-Score'] <= self.floor) * 1.0 # Buy the spread
            self.df['Shorts'] = (self.df['Z-Score'] >= self.ceiling) * 1.0 # Short the spread
            self.df['Exit'] = (self.df['Z-Score'] <= self.exit_zscore) * 1.0
            # track positions with for loop
            self.df['Long Market'] = 0.0
            self.df['Short Market'] = 0.0 
            # Setting variables to track whether or to be long while iterating
            self.long_market = 0 
            self.short_market = 0 
            # Determing when to trade
            for i, value in enumerate(self.df.iterrows()):
                if value[1]['Longs'] == 1.0:
                    self.long_market = 1
                elif value[1]['Shorts'] == 1.0:
                    self.short_market = 1
                elif value[1]['Exit'] == 1.0:
                    self.long_market = 0
                    self.short_market = 0
                self.df.iloc[i]['Long Market'] = self.long_market
                self.df.iloc[i]['Short Market'] = self.short_market
    
        elif self.ptype == 2:
            self.df['Longs'] = (self.df['Z-Score'] <= self.floor)*1.0 #buy the spread
            self.df['Shorts'] = (self.df['Z-Score'] >= self.ceiling)*1.0 #short the spread
            self.df['Exit']=(self.df['Z-Score'] <= self.exit_zscore)*1.0
            #tracking positions via for loop implementation
            self.df['Long Market'] = 0.0
            self.df['Short Market'] = 0.0
            #Setting Variables to track whether or not to be long while iterating over df
            self.long_market = 0
            self.short_market = 0
            #Determining when to trade
            for i, value in enumerate(self.df.iterrows()):
                if self.p2Objective == 'Avoid':
                    if value[1]['Regime'] != self.avoid1:
                        #Calculate longs
                        if value[1]['Longs'] == 1.0:
                            self.long_market = 1
                        elif value[1]['Shorts'] == 1.0:
                            self.short_market = 1
                        elif value[1]['Exit'] == 1.0:
                            self.long_market = 0
                            self.short_market = 0
                self.df.iloc[i]['Long Market'] = value[1]['Longs']#self.long_market
                self.df.iloc[i]['Short Market'] = value[1]['Shorts']#self.short_market
                
                if self.p2Objective == 'Target':
                    if value[1]['Regime'] == self.target1:
                        #Calculate longs
                        if value[1]['Longs'] == 1.0:
                            self.long_market = 1
                        elif value[1]['Shorts'] == 1.0:
                            self.short_market = 1
                        elif value[1]['Exit'] == 1.0:
                            self.long_market = 0
                            self.short_market = 0
                self.df.iloc[i]['Long Market'] = value[1]['Longs']#self.long_market
                self.df.iloc[i]['Short Market'] = value[1]['Shorts']#self.short_market

                if self.p2Objective == 'None':
                    #Calculate longs
                    if value[1]['Longs'] == 1.0:
                        self.long_market = 1                     
                        #Calculate Shorts
                    elif value[1]['Shorts'] == 1.0:
                        self.short_market = 1                        
                    elif value[1]['Exit'] == 1.0:
                        self.long_market = 0
                        self.short_market = 0

                self.df.iloc[i]['Long Market'] = value[1]['Longs']#self.long_market
                self.df.iloc[i]['Short Market'] = value[1]['Shorts']#self.short_market

        return self.df

    def create_returns(self, allocation,pair_number):
        if self.ptype==1:
            
            self.allocation = allocation
            self.pair = pair_number
            self.portfolio = pd.DataFrame(index=self.df.index)
            self.portfolio['Positions'] = self.df['Long Market'] - self.df['Short Market']
            self.portfolio['X'] = -1.0 * self.df['X'] * self.portfolio['Positions']
            self.portfolio['Y'] = self.df['Y' ] * self.portfolio['Positions']
            self.portfolio['Total'] = self.portfolio['X'] + self.portfolio['Y']
            #creating a percentage return stream
            self.portfolio['Returns'] = self.portfolio['Total'].pct_change()
            self.portfolio['Returns'].fillna(0.0,inplace=True)
            self.portfolio['Returns'].replace([np.inf,-np.inf],0.0,inplace=True)
            self.portfolio['Returns'].replace(-1.0,0.0,inplace=True)
            #calculating metrics
            self.mu = (self.portfolio['Returns'].mean())
            self.sigma = (self.portfolio['Returns'].std())
            self.portfolio['Win'] = np.where(self.portfolio['Returns'] > 0,1,0)
            self.portfolio['Loss'] = np.where(self.portfolio['Returns'] < 0,1,0)
            self.wins = self.portfolio['Win'].sum()
            self.losses = self.portfolio['Loss'].sum()
            self.total_trades = self.wins + self.losses
            #calculating sharpe ratio with interest rate of
            #interest_rate_assumption=0.75
            #self.sharp = (self.mu - interest_rate_assumption) / self.sigma
            #win loss ratio;
            self.win_loss_ratio = (self.wins / self.losses)
            #probability of win
            self.prob_of_win = (self.wins / self.total_trades)
            #probability of loss
            self.prob_of_loss = (self.losses / self.total_trades)
            #average return of wins
            self.avg_win_return = (self.portfolio['Returns'] > 0).mean()
            #average returns of losses
            self.avg_loss_return = (self.portfolio['Returns'] < 0).mean()
            #calculating payout ratio
            self.payout_ratio = (self.avg_win_return/self.avg_loss_return)
            #calculate equity curve
            self.portfolio['Returns'] = (self.portfolio['Returns'] + 1.0).cumprod()
            self.portfolio['Trade Returns'] = (self.portfolio['Total'].pct_change()) #non cumulative Returns
            self.portfolio['Portfolio Value'] = (self.allocation * self.portfolio['Returns'])
            self.portfolio['Portfolio Returns'] = self.portfolio['Portfolio Value'].pct_change()
            self.portfolio['Initial Value'] = self.allocation
            
            with plt.style.context(['ggplot','seaborn-paper']):
                #Plotting Portfolio Value
                plt.plot(self.portfolio['Portfolio Value'])
                plt.plot(self.portfolio['Initial Value'])
                plt.title('%s Strategy Returns '%(self.pair))
                plt.legend(loc=0)
                plt.show()

        elif self.ptype==2:
            
            self.allocation = allocation
            self.pair = pair_number
            self.portfolio = pd.DataFrame(index=self.df.index)
            self.portfolio['Positions'] =self.df['Longs'] - self.df['Shorts']
            self.portfolio['X'] = -1.0 * self.df['X'] * self.portfolio['Positions']
            self.portfolio['Y'] = self.df['Y' ] * self.portfolio['Positions']
            self.portfolio['Total'] = self.portfolio['X'] + self.portfolio['Y']
            #creating a percentage return stream
            self.portfolio.fillna(0.0,inplace=True)
            self.portfolio['Returns'] = self.portfolio['Total'].pct_change()
            self.portfolio['Returns'].fillna(0.0, inplace = True)
            self.portfolio['Returns'].replace([np.inf, -np.inf], 0.0, inplace = True)
            self.portfolio['Returns'].replace(-1.0,0.0, inplace = True)
            #calculating metrics
            self.mu = (self.portfolio['Returns'].mean())
            self.sigma = (self.portfolio['Returns'].std())
            self.portfolio['Win'] = np.where(self.portfolio['Returns'] > 0,1,0)
            self.portfolio['Loss'] = np.where(self.portfolio['Returns'] < 0,1,0)
            self.wins = self.portfolio['Win'].sum()
            self.losses = self.portfolio['Loss'].sum()
            self.total_trades = self.wins + self.losses
            #calculating sharpe ratio with interest rate of
            #interest_rate_assumption=0.75
            #self.sharp = (self.mu - interest_rate_assumption) / self.sigma
            #win loss ratio;
            self.win_loss_ratio = (self.wins / self.losses)
            #probability of win
            self.prob_of_win = (self.wins / self.total_trades)
            #probability of loss
            self.prob_of_loss = (self.losses / self.total_trades)
            #average return of wins
            self.avg_win_return = (self.portfolio['Returns'] > 0).mean()
            #average returns of losses
            self.avg_loss_return = (self.portfolio['Returns'] < 0).mean()
            #calculating payout ratio
            self.payout_ratio = (self.avg_win_return/self.avg_loss_return)
            #calculate equity curve
            self.portfolio['Returns'] = (self.portfolio['Returns'] + 1.0).cumprod()
            self.portfolio['Trade Returns'] = (self.portfolio['Total'].pct_change()) #non cumulative Returns
            self.portfolio['Portfolio Value'] = (self.allocation * self.portfolio['Returns'])
            self.portfolio['Portfolio Returns'] = self.portfolio['Portfolio Value'].pct_change()
            self.portfolio['Initial Value'] = self.allocation
            
            with plt.style.context(['ggplot','seaborn-paper']):
                #Plotting Portfolio Value
                plt.plot(self.portfolio['Portfolio Value'])
                plt.plot(self.portfolio['Initial Value'])
                plt.title('%s Strategy Returns '%(self.pair))
                plt.legend(loc = 0)
                plt.show()

        return 
    
bottom_up = pd.DataFrame()
for t, pair in enumerate(cointegrated):
    bot_pair_arbi = statarb_update(X_test[pair[0]], X_test[pair[1]], 2, 10, -2, 2, 10, regime_predictions[t], 'Target')
    bot_pair_arbi.create_spread()
    bot_pair_arbi.generate_signals()
    bot_pair_arbi.create_returns(investment/len(cointegrated), str(pair))
    bottom_up[t] = bot_pair_arbi.portfolio['Portfolio Value']
    
bottom_up['Cash'] = 10000   
bottom_up['Total Portfolio Value'] =   bottom_up.sum(axis = 1)
bottom_up['Returns'] = np.log(bottom_up['Total Portfolio Value'] / bottom_up['Total Portfolio Value'].shift(1))

bottom_up_mu = bottom_up['Returns'].mean()
bottom_up_sigma = bottom_up['Returns'].std()
#recall that we initialized our interest assumption earlier
bottom_up_sharpe = (bottom_up_mu - rate) / bottom_up_sigma

plt.figure(figsize=(10,6))
plt.plot(bottom_up['Total Portfolio Value'])
plt.title('Bottom Up Portfolio Equity Curve')
plt.show()  

spo_portfolio = pd.DataFrame()
for n, pair in enumerate(cointegrated):
    spo_pair_arbi = statarb_update(X_test[pair[0]], X_test[pair[1]], 2, 6, -2, 2, 6, regime_predictions[n], 'Target')
    spo_pair_arbi.create_spread()
    spo_pair_arbi.generate_signals()
    spo_pair_arbi.create_returns(round(investment * optimal_weights[0][n],2), str(pair))
    spo_portfolio[n] = spo_pair_arbi.portfolio['Portfolio Value']
    
spo_portfolio['Cash'] = 10000   
spo_portfolio['Total Portfolio Value'] =   spo_portfolio.sum(axis = 1)
spo_portfolio['Returns'] = np.log(spo_portfolio['Total Portfolio Value'] / spo_portfolio['Total Portfolio Value'].shift(1))

spo_portfolio_mu = spo_portfolio['Returns'].mean()
spo_portfolio_sigma = spo_portfolio['Returns'].std()
#recall that we initialized our interest assumption earlier
spo_portfolio_sharpe = (spo_portfolio_mu - rate) / spo_portfolio_sigma

plt.figure(figsize = (10,6))
plt.plot(spo_portfolio['Total Portfolio Value'])
plt.title('SPO Portfolio Equity Curve')
plt.show() 

#list to hold portfolio names
names = ['Equally Weighted','Efficient Frontier','Bottom Up','SPO Framework']
#variable to hold column name
column_name = 'Sharpe Ratio'
#list to hold Sharpe Ratios
sharpes = [equal_sharpe, p_efficient_frontier_sharpe, bottom_up_sharpe, spo_portfolio_sharpe]
#creating dataframe to compare Sharpe Ratios of Portfolios
portfolio_assessment = pd.DataFrame({column_name:sharpes},index = names)

#creating list to hold ending values of portfolios
#We pass in 1 into the tail method because it represents the last index position
portfolio_values = [equal['Total Portfolio Value'].tail(1).values.astype(int), p_efficient_frontier['Total Portfolio Value'].tail(1).values.astype(int), bottom_up['Total Portfolio Value'].tail(1).values.astype(int),spo_portfolio['Total Portfolio Value'].tail(1).values.astype(int)]
#creating dataframe to hold ending value of portfolios
pd.DataFrame({'Ending Portfolio Values':portfolio_values},index=names)
 







































