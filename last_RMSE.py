# -*- coding: utf-8 -*-

#Read the files saved before, corresponding to the combinations of country and sample
#and display the first rows of the data if the file is found and read successfully.

# Definir la función asignar_percentil si aún no está definida
def asignar_percentil(valor):
    for i, p in enumerate(percentiles):
        if valor <= p:
            return i + 10
    return 10
# Lista de países y muestras
countries = ['cri']
samples = [0, 1]

# Recorrer la lista de países
for country in countries:
    for sample in samples:
        # Leer el archivo de datos correspondiente al país y la muestra
        file_path = f'C:/w_2023/meanTest/_data(write)/_py({country})_te{sample}.dta'
        try:
            data = pd.read_stata(file_path)
        except FileNotFoundError:
            print(f'Archivo no encontrado: {file_path}')
            continue

        combined_df = data.copy()
        percentiles = np.percentile(combined_df['lny0'], range(1, 101, 1))
        combined_df['percentiles'] = combined_df['lny0'].apply(asignar_percentil)
        combined_df.sort_values('percentiles', ascending=False, inplace=True)

        '''
        *----------------------------------------------------------------------
        (F) RMSE by percentile
        *----------------------------------------------------------------------
        '''
        # This section calculates the root mean squared error (RMSE) for different 
        # error columns by grouping the DataFrame based on percentiles. 
        # It calculates the sum and count for each error column within each percentile 
        # group. Then, it computes the RMSE for each error column using the 
        # formula: RMSE = sqrt(sum of squared errors / count).
        
        error_columns = ['sqr_Aurf', 'sqr_urf', 'sqr_Aols', 'sqr_ols', 'sqr_Axgb',
                    'sqr_xgb']

        sum_error_pct = combined_df.groupby('percentiles')[error_columns].agg(['sum', 'count'])
        sum_error_pct.columns = ['sum_' + col[0] if col[1] == 'sum' else 'n_' + col[0] for col in sum_error_pct.columns]
        sum_error_pct['rmse_Aurf'] = np.sqrt(sum_error_pct['sum_sqr_Aurf'] / sum_error_pct['n_sqr_Aurf'])
        sum_error_pct['rmse_urf'] = np.sqrt(sum_error_pct['sum_sqr_urf'] / sum_error_pct['n_sqr_urf'])
        sum_error_pct['rmse_Aols'] = np.sqrt(sum_error_pct['sum_sqr_Aols'] / sum_error_pct['n_sqr_Aols'])
        sum_error_pct['rmse_ols'] = np.sqrt(sum_error_pct['sum_sqr_ols'] / sum_error_pct['n_sqr_ols'])
        sum_error_pct['rmse_Axgb'] = np.sqrt(sum_error_pct['sum_sqr_Axgb'] / sum_error_pct['n_sqr_Axgb'])
        sum_error_pct['rmse_xgb'] = np.sqrt(sum_error_pct['sum_sqr_xgb'] / sum_error_pct['n_sqr_xgb'])
       
        #  It calculates the mean value of each selected column within 
        # each percentile group.
        promedio = combined_df.copy()
        promedio_yhat_columns = ['lny0', 'yhat_Aurf', 'yhat_urf', 'yhat_Aols', 'yhat_ols', 'yhat_Axgb', 'yhat_xgb']
        promedio_yhat_pct = promedio.groupby('percentiles')[promedio_yhat_columns].mean()
        promedio_yhat_pct.sort_values('percentiles', ascending=False, inplace=True)

       # RMSE GLOBAL
        combined_df = ID_test.copy()
        combined_df['one'] = 1
        sum_error_all = combined_df.groupby('one')[['sqr_Aurf',
                                     'sqr_urf',
                                     'sqr_Aols',
                                     'sqr_ols',
                                     'sqr_Axgb',
                                     'sqr_xgb']].agg([('sum', 'sum'), ('n_', 'count')])
        sum_error_all.columns = ['sum_' + col[0] if col[1] == 'sum' else 'n_' + col[0] for col in sum_error_all.columns]
        sum_error_all['rmse_Aurf'] = np.sqrt(sum_error_all['sum_sqr_Aurf']/sum_error_all['n_sqr_Aurf'])
        sum_error_all['rmse_urf'] = np.sqrt(sum_error_all['sum_sqr_urf']/sum_error_all['n_sqr_urf'])
        sum_error_all['rmse_Aols'] = np.sqrt(sum_error_all['sum_sqr_Aols']/sum_error_all['n_sqr_Aols'])
        sum_error_all['rmse_ols'] = np.sqrt(sum_error_all['sum_sqr_ols']/sum_error_all['n_sqr_ols'])
        sum_error_all['rmse_Axgb'] = np.sqrt(sum_error_all['sum_sqr_Axgb']/sum_error_all['n_sqr_Axgb'])
        sum_error_all['rmse_xgb'] = np.sqrt(sum_error_all['sum_sqr_xgb']/sum_error_all['n_sqr_xgb'])

        sum_error_all.to_stata(file_path + '_pyDF(' + country + ')_te' + str(sample) + '.dta')        
#%%
        '''
        *----------------------------------------------------------------------
        (G) Graphics
        *----------------------------------------------------------------------
        '''
        # Density Graphic
        # The plot compares the density distributions of two sets of data: 
        # the observed values and the oversampled values 
        # It helps in assessing how well the oversampling technique approximates 
        # the distribution of the original data.
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))  
        sns.kdeplot(y_train['lny0'], label=f'Observed', fill=True)
        sns.kdeplot(y_train_sm, label=f'sampling from 1.5x median', fill=True, color='orange')
        plt.title(f'Density plots of observed vs. oversampled outcomes', fontsize=16)
        plt.xlabel('lny0', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend()
        plt.show()
        
         # Average Estimated Income
         # The plot visually compares the performance of these models in estimating 
         # income across different percentiles. 
         # It allows to see how well each model's predictions align with the 
         # observed income values at various points in the distribution. 
         
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['yhat_Aols'], label='SMOGN OLS')
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['yhat_ols'], label='OLS')
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['yhat_Aurf'], label='SMOGN RF(u)')
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['yhat_urf'], label='RF(u)')
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['yhat_Axgb'], label='SMOGN XGB')
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['yhat_xgb'], label='XGB')
         plt.plot(promedio_yhat_pct.index, promedio_yhat_pct['lny0'], label='Observed')
         plt.xlabel('Percentil')
         plt.ylabel('Average Estimated Income')
         plt.title('Promedio del ingreso estimado por percentil y modelo - sample 4, 0.05')
         plt.legend()
         plt.grid(True)
         plt.show()
         
    
    
