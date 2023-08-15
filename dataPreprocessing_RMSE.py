# -*- coding: utf-8 -*-
"""
"""
#%%
import sys
print(sys.version, sys.platform, sys.executable)
%reset -f
#Import libraries
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from numpy import mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from mlxtend.plotting import plot_decision_regions
from IPython.display import display, HTML
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn import preprocessing
import smogn
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error  as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score            as R2
from sklearn.svm import SVR
import sklearn.metrics as metrics
import matplotlib.pyplot   as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
import random
from xgboost import XGBRegressor
import h2o
from h2o.automl import H2OAutoML
import joblib
from tqdm import tqdm
from denseweight import DenseWeight
#%%
#Create a function to o assign the corresponding percentile to each value
def asignar_percentil(valor):
    for i, p in enumerate(percentiles):
        if valor <= p:
            return i + 10
    return 100
#%%
#Define paths
#-----------------------------
user = 1 # (1) WHL 
#-----------------------------
if user == 1:  # 
    read     = 'C:/w_2023/meanTest/_data(temp)/'
    write    = 'C:/w_2023/meanTest/_data(write)/'

if user == 2:
    chl_data = 'Pcloud data path'  
#%%
#An initial list of countries is defined, containing data from each of the twelve countroies and a list 
#named samples with values from 0 to 4
#There is a loop that iterates through each country and inside a nested loop that iterates
#through each sample. In these loops are constructed paths for reading CSV files based on the
#courrent country and sample number. 
#In this section we create distinct Dataframes designated for training and testing data 
#Rows with missing values (Nah) are removed from trainning and testing data and prepare the
#resulting dataframe for a next use.

#countries = ['arg', 'bol', 'bra', 'chi', 'col', 'cri', 'ecu', 'hti', 'mex', 'per', 'pry', 'ury']
samples = [0]
# specfs  = [2,3]

countries = ['cri']
for country in countries:  
    # for specif in specfs:   
    for sample in samples:
    # for i in range(5):
        print(country + ': sample ' + str(sample))

        file1 = read + 'data_tr_' + country + '_' + str(sample) + '.csv'
        dtr = pd.read_csv(file1, header=0)
        file2 = read + 'data_te_' + country + '_' + str(sample) + '.csv'
        dte = pd.read_csv(file2, header=0)
        
        dte     = dte.dropna(how='any')  
        dtr     = dtr.dropna(how='any')  
        
        dte = dte.reset_index(drop=True)
        dtr = dtr.reset_index(drop=True)

        #In this section there is a preprocessing data, following the next steps:  
            #1) New dataframe called ID_test is created where id and lny0 are extracted. 
            #2) y_train is generated using lny0 column from the dtr dataframe
            #3) x_train is generated from the dtr dataframe, excluding lny0 and id.
            #4) x_test is generated from the dte dataframe, excluding lny0 and id. 
    
        ID_test = pd.DataFrame(dte[['id', 'lny0']], index=dte.index)
                    
        y_train = pd.DataFrame(dtr['lny0'], columns=['lny0'], index=dtr.index)
        y_train = y_train.dropna() 
        X_train = dtr.drop(['lny0','id'], axis=1)
        X_train = X_train.dropna()          
        X_test = dte.drop(['lny0','id'], axis=1)
     
        '''
        *------------------------------------------------------------------------------
        (B) oversampling [smogn]
        *------------------------------------------------------------------------------
        '''
        #In this section we make an oversampling using the synthetic minority over sampling
        #technique for regression (SMOGN).
        # First we apply the StandardScaler to scale the features in x_train and x_test to 
        # have zero mean and unit variance.
        # The train dataframe is created by concatenating y_train and x_train.
        
        #SMOGN oversampling technique is applied to balance the target variable distribution, by 
        #using the smoter function from the smogn library.
        #This technique focuses on generating synthetic samples for the minority class 
        #(which corresponds to lower values of the target variable) to achieve more balanced 
        #distribution of the target variable through the creation of synthetic instances that closely resemble existing instances.
        #It should be noted that when applying smogn, a sampling method of 
        # was chosen because the degree of modification applied to the data distribution is lower. 
        #Additionally, a high relevance coefficient is used because the phi relevance will be broader, which could lead 
        #to classifying more observations as rare due to the nature of the data.
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, 'C:/w_2023/scaler.pkl') 
        X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
        scaler = joblib.load('C:/w_2023/scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
        train = pd.concat([y_train, X_train], axis=1)
        train = train.reset_index(drop=True)
        train = train.dropna()
        train_short = train.head(n=4000)

        random.seed(666)
        data_smogn = smogn.smoter(
            data=train,  
            y='lny0', 
            k=5,  
            samp_method='balance',  
            rel_method= 'auto',
            replace = False,
            drop_na_col=True,  
            drop_na_row=True,  
            rel_coef = 2.5,
            pert = 0.02,
            ) 

        #Here the oversampled data is extracted
        X_train_sm = data_smogn.drop(['lny0'], axis=1)
        y_train_sm = data_smogn['lny0']
        y_train_sm =y_train_sm.dropna()

        #Export the oversampled data to Stata and CSV files for further analysis.
        data_smogn.to_stata(write + '_pySMOGN(' + country + ')_te' + str(sample) + '.dta')
        data_smogn.to_csv(write + '_pySMOGN(' + country + ')_te' + str(sample)   + '.csv')

