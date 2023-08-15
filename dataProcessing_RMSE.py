# -*- coding: utf-8 -*-
#%%
#Create a function to o assign the corresponding percentile to each value
def asignar_percentil(valor):
    for i, p in enumerate(percentiles):
        if valor <= p:
            return i + 10
    return 100
#%%
#Read the files saved before, corresponding to the combinations of country and sample
#and display the first rows of the data if the file is found and read successfully.

#countries = ['arg', 'bol', 'bra', 'chi', 'col', 'cri', 'ecu', 'hti', 'mex', 'per', 'pry', 'ury']
samples = [4]
countries =['cri']
base_path = 'C:/w_2023/meanTest/_data(write)/'

for country in countries:
    for sample in samples:
        file_name = f'_pySMOGN({country})_te{sample}.dta'
        file_path = base_path + file_name 
        try:
            data = pd.read_stata(file_path)
            print(data.head())
            
        except FileNotFoundError:
            print(f'file no found: {file_path}')
      
        '''
        *------------------------------------------------------------------------------
        (C) train regressors 
        *------------------------------------------------------------------------------
        '''
        # In this section, you'll find code snippets that train diverse regression models. 
        # These snippets not only store the trained models in variables but also encompass 
        # different types of regressor models—with and without oversampling. 
        # This approach involves using distinct subsets of the training data: X_train_sm and y_train_sm 
        # for oversampled data, and X_train and y_train for non-oversampled data. 
        # This comparison allows us to determine which models yield superior results.


        # [UNTuned Random Forest with Oversampling ] -------------------------------------------------------
        # Initializes an instance of the RandomForestRegressor class with a parameter 
        # min_samples_split=3. Fits (trains) the model using the oversampled training data X_train_sm 
        # and y_train_sm.
  
        Aurf      = RandomForestRegressor(min_samples_split=3)
        Aurf.fit(X_train_sm, np.ravel(y_train_sm))

        # [UNTuned Random Forest without Oversampling ] -----------------------------

        urf      = RandomForestRegressor(min_samples_split=3)
        urf.fit(X_train, np.ravel(y_train))
    

        # [OLS Linear Regression with Oversampling] -----------------------------------------------------------------------

        Aols = LinearRegression()
        Aols.fit(X_train_sm, np.ravel(y_train_sm))

        # [OLS,Linear Regression without Oversampling] -----------------------------------------------------------------------

        ols = LinearRegression()
        ols.fit(X_train, np.ravel(y_train))
  
        # [XGBRegressor with Oversampling]  -----------------------------------------------------------------------
  
        # This code section conducts a randomized search to discover the best hyperparameters for an XGBoost 
        # regression model. The aim is to minimize the negative mean squared error, 
        # which gauges prediction accuracy. 
        # The search explores various combinations of hyperparameters like learning rate, tree depth, and more.
        # Once the search is done, the code saves the optimal hyperparameters in best_params_a. 
        # It also stores the corresponding best model configuration in best_xgb_a.
  
        xgb_a = XGBRegressor()
        param_distributions = {
           'n_estimators': randint(100, 500),
           'learning_rate': uniform(0.01, 0.2),
           'max_depth': randint(3, 8),
           'subsample': uniform(0.8, 0.2),
           'colsample_bytree': uniform(0.8, 0.2),
           }
  
        random_search = RandomizedSearchCV(estimator=xgb_a, 
                                      param_distributions=param_distributions, 
                                      scoring='neg_mean_squared_error', 
                                      cv=2, n_iter=100,
                                      verbose=True)
   
        with tqdm(total=random_search.n_iter, desc="Hyperparameter Search", unit="combination", leave=True) as pbar:
         random_search.fit(X_train, y_train)
         pbar.update(random_search.n_iter)

        best_params_a = random_search.best_params_
        best_xgb_a = random_search.best_estimator_
  
  
        # [XGBRegressor without  Oversampling -----------------------------------------------------------------------
        xgb_wo = XGBRegressor()
        param_distributions = {
           'n_estimators': randint(100, 500),
           'learning_rate': uniform(0.01, 0.2),
           'max_depth': randint(3, 8),
           'subsample': uniform(0.8, 0.2),
           'colsample_bytree': uniform(0.8, 0.2),
           }

        random_search = RandomizedSearchCV(estimator=xgb_wo, 
                                      param_distributions=param_distributions, 
                                      scoring='neg_mean_squared_error', 
                                      cv=2, n_iter=100,
                                      verbose=True)
        with tqdm(total=random_search.n_iter, desc="Hyperparameter Search", unit="combination", leave=True) as pbar:
           random_search.fit(X_train, y_train)
           pbar.update(random_search.n_iter)
      
        best_params_wo = random_search.best_params_
        best_xgb_wo = random_search.best_estimator_
        
        '''
        *------------------------------------------------------------------------------
        (D) Predict
        *------------------------------------------------------------------------------
        '''
        # This code generates predictions for multiple regression models, 
        # each trained with different data processing techniques, and stores 
        # the predictions in separate DataFrames. 
      
        # Untuned RF w/ oversampling 
        yhat_Aurf      = pd.DataFrame(Aurf.predict(X_test))
        yhat_Aurf.columns = ['yhat_Aurf']
        
        # Untuned RF wo/ oversampling 
        yhat_urf      = pd.DataFrame(urf.predict(X_test))
        yhat_urf.columns = ['yhat_urf']
        
        # OLS w/ oversampling
        yhat_Aols     = pd.DataFrame(Aols.predict(X_test))
        yhat_Aols.columns = ['yhat_Aols']
        
        # OLS wo/ oversampling
        yhat_ols     = pd.DataFrame(ols.predict(X_test))
        yhat_ols.columns = ['yhat_ols']
        
        #XGB regress w/oversampling
        yhat_Axgb = pd.DataFrame(best_xgb_a.predict(X_test))
        yhat_Axgb.columns =['yhat_Axgb']

        #XGB regress wo/oversampling
        yhat_xgb = pd.DataFrame(best_xgb_wo.predict(X_test))
        yhat_xgb.columns =['yhat_xgb']
        
        '''
        *----------------------------------------------------------------------
        (E) Evaluation and Analysis of Prediction Results
        *----------------------------------------------------------------------
        '''

        # This code processes prediction results from different regression 
        # models and observed data. 
        # It calculates squared differences, differences from oversampling, 
        # saves the modified DataFrame to files
        # Creates a copy of the DataFrame, and calculates percentiles 
        # for the observed values. 

        ID_test['yhat_Aurf']  = yhat_Aurf['yhat_Aurf']
        ID_test['yhat_urf']   = yhat_urf['yhat_urf']
        ID_test['yhat_Aols']  = yhat_Aols['yhat_Aols']
        ID_test['yhat_ols']   = yhat_ols['yhat_ols']
        ID_test['yhat_Axgb']  = yhat_Axgb['yhat_Axgb']
        ID_test['yhat_xgb']   = yhat_xgb['yhat_xgb']
                 
        # Difference with observed
            
        ID_test['sqr_Aurf'] = (ID_test['lny0'] - ID_test['yhat_Aurf'])**2
        ID_test['sqr_urf'] = (ID_test['lny0'] - ID_test['yhat_urf'])**2
        ID_test['sqr_Aols'] = (ID_test['lny0'] - ID_test['yhat_Aols'])**2
        ID_test['sqr_ols'] = (ID_test['lny0'] - ID_test['yhat_ols'])**2
        ID_test['sqr_Axgb'] = (ID_test['lny0'] - ID_test['yhat_Axgb'])**2
        ID_test['sqr_xgb'] = (ID_test['lny0'] - ID_test['yhat_xgb'])**2
            
        # Difference from oversampling
        # This section calculates differences between predicted and observed 
        # values from regression models, recording them in the ID_test DataFrame.
        # It generates percentiles for observed values, and appends a 
        # percentiles column using a custom function. 
            
        ID_test['dif_RF'] = ID_test['yhat_Aurf'] - ID_test['yhat_urf']
        ID_test['dif_OLS'] = ID_test['yhat_Aols'] - ID_test['yhat_ols']
        ID_test['dif_XGB'] = ID_test['yhat_Axgb'] - ID_test['yhat_xgb']
                 
        ID_test.to_stata(write + '_py(' + country + ')_te' + str(sample) + '.dta')
        ID_test.to_csv(write + '_py(' + country + ')_te' + str(sample) + '.csv') 
        
       

        
        
        