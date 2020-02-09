########################################################################
# Author : Bastien Girardet
# License : MIT
# Created : 05-08-2019
# Updated : 09-10-2019
# Goal : Bruteforce every combination of model possible.
# Results : Results are saved into a separate csv file
########################################################################


# We import our libraries
import os
import numpy as np
import pandas as pd        
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.optimizers import SGD



# Global variables
btc_path = './models/ts/forecasts/BTC/'
eos_path = './models/ts/forecasts/EOS/'
eth_path = './models/ts/forecasts/ETH/'
iota_path = './models/ts/forecasts/IOTA/'
base_path = './models/ts/forecasts/'
suffix = '.csv'
test_size = 0.2
n_epoch = 100
verbose = 1

# Degree of the parameters lag
# e.g. if 14 => returns-lagged-1, returns-lagged-2, ... , returns-lagged-14
lag_number = 14

def generate_models():

    model_list = []

    # We set our optimizers
    optimizer_list = ['sgd', 'rmsprop']

    # We set our drop out rates
    dropout_rate_list = [0.4,0.5]

    # we define our models
    for each_optimizer in optimizer_list:
        for each_dropout_rate in dropout_rate_list:     
            # Model 1
            model1 = Sequential(name=f'model1-drate-{each_dropout_rate}-opt-{each_optimizer}')
            model1.add(Dense(40, input_dim=113, kernel_initializer='he_uniform', activation='relu'))
            model1.add(Dropout(rate=each_dropout_rate, name='Dropout'))
            model1.add(Dense(20, activation='relu'))
            model1.add(Dense(5, activation='relu'))
            model1.add(Dense(1, activation='linear'))
            model1.compile(loss='mean_squared_error', optimizer=each_optimizer, metrics=['mse'])

            # Model 2 previ 5
            # Reduce layer size from the Model 1
            model2 = Sequential(name=f'model2-drate-{each_dropout_rate}-opt-{each_optimizer}')
            model2.add(Dense(40, input_dim=113, kernel_initializer='he_uniform', activation='relu'))
            model2.add(Dropout(rate=each_dropout_rate, name='Dropout'))
            model2.add(Dense(15, activation='relu'))
            model2.add(Dense(1, activation='linear'))
            model2.compile(loss='mean_squared_error', optimizer=each_optimizer, metrics=['mse'])

            # Model 3 prev. 6
            # Reduce layer size from the Model 1
            model3 = Sequential(name=f'model3-drate-{each_dropout_rate}-opt-{each_optimizer}')
            model3.add(Dense(40, input_dim=113, kernel_initializer='he_uniform', activation='relu'))
            model3.add(Dropout(rate=each_dropout_rate, name='Dropout'))
            model3.add(Dense(15, activation='sigmoid'))
            model3.add(Dense(1, activation='linear'))
            model3.compile(loss='mean_squared_error', optimizer=each_optimizer, metrics=['mse'])

            # Model 4 prev 7
            # Changed the 
            model4 = Sequential(name=f'model4-drate-{each_dropout_rate}-opt-{each_optimizer}')
            model4.add(Dense(40, input_dim=113, kernel_initializer='he_uniform', activation='relu'))
            model4.add(Dense(15, activation='relu'))
            model4.add(Dropout(rate=each_dropout_rate, name='Dropout'))
            model4.add(Dense(1, activation='linear'))
            model4.compile(loss='mean_squared_error', optimizer=each_optimizer, metrics=['mse'])

            # Model 5 prev. 9
            # Reduced layer
            model5 = Sequential(name=f'model5-drate-{each_dropout_rate}-opt-{each_optimizer}')
            model5.add(Dense(40, input_dim=113, kernel_initializer='he_uniform', activation='relu'))
            model5.add(Dropout(rate=each_dropout_rate, name='Dropout'))
            model5.add(Dense(1, activation='linear'))
            model5.compile(loss='mean_squared_error', optimizer=each_optimizer, metrics=['mse'])

            # We add all the models.
            model_list.append(model1)
            model_list.append(model2)
            model_list.append(model3)
            model_list.append(model4)
            model_list.append(model5)

    return model_list

def retrieve_meta_data_from_filename(filename, keys = ['fullpath','filename','crypto', 'model', 'dist', 'p', 'q']):
    """Retrieve all the meta data relevant to the file
    
        Args:
            filename (string) : Name of the file
            keys : Keys to be merged with the values of the filename
    """
    values = filename.split('.')[0].split('-')
    crypto_name = values[0]
    fullpath = os.path.join(base_path,crypto_name)
    fullpath = os.path.join(fullpath,filename)
    values = [fullpath] + [filename] + values

    return dict(zip(keys, values))

def compute_mse(results, col_vol, col_ts, col_ann, test_length):
    # Compute MSE in and out of sample for Time serie and ANN
    # Compute out of sample MSE

    split_date = len(results[col_vol])-len(y_test)

    x_vol_out = results[col_vol][split_date:]
    hat_x_ts_out = results[col_ts][split_date:]
    hat_x_ann_out = results[col_ann][split_date:]

    x_vol_in = results[col_vol][:split_date]
    hat_x_ts_in = results[col_ts][:split_date]
    hat_x_ann_in = results[col_ann][:split_date]

    ts_mse_out = np.mean((hat_x_ts_out-x_vol_out)**2)
    ann_mse_out = np.mean((hat_x_ann_out-x_vol_out)**2)
    
    ts_mse_in = np.mean((hat_x_ts_in-x_vol_in)**2)
    ann_mse_in = np.mean((hat_x_ann_in-x_vol_in)**2)

    print(f"ANN MSE out : {round(ann_mse_out,4)}\tEGARCH MSE out : {round(ts_mse_out,4)}")
    return ann_mse_in, ann_mse_out, ts_mse_in, ts_mse_out

def pick_best_model(ann_mse_out, ts_mse_out):
    """Pick the best model provided mse, usually out of sample
    """
    if ann_mse_out < ts_mse_out:
        return "ANN"
    else:
        return "TS" 

# We define our lists of file to go through
btc_models = [ retrieve_meta_data_from_filename(filename) for filename in os.listdir(btc_path) if filename.endswith( suffix )]
eos_models = [ retrieve_meta_data_from_filename(filename) for filename in os.listdir(eos_path) if filename.endswith( suffix )]
eth_models = [ retrieve_meta_data_from_filename(filename) for filename in os.listdir(eth_path) if filename.endswith( suffix )]
iota_models = [ retrieve_meta_data_from_filename(filename) for filename in os.listdir(iota_path) if filename.endswith( suffix )]

print(btc_models)

crypto_models_list = [btc_models,eos_models,eth_models,iota_models]
history_list = []

crypto_list = []
ann_model_list = []
n_layers_list = []
drop_rate_list = []
optimizer_list = []
ann_mse_in_list = []
ann_mse_out_list = []

ts_model_list = []
ts_dist_list = []
ts_p_list = []
ts_q_list = []
ts_mse_in_list = []
ts_mse_out_list = []
best_model_ts_or_ann_list = []
test_size_list = []

# We go through each models for each crypto
for crypto_models in crypto_models_list:

    for each_ts_model in crypto_models:

        print("="*120)        
        print(f"Processing\tCrypto :{each_ts_model['crypto']}\tModel :{each_ts_model['model']}\tDist :{each_ts_model['dist']}\tp :{each_ts_model['p']}\tq :{each_ts_model['q']}")
        print("-"*120)

        # We define our model name which is going to be used as column name
        model_name = f"{each_ts_model['model']}-{each_ts_model['p']}-{each_ts_model['q']}"
        model_name = f"{model_name.lower()}-{each_ts_model['dist']}"

        # We read our dataset
        df = pd.read_csv(each_ts_model['fullpath'],index_col=0)

        # We can't do anything with the MSE, MSE, and Squared error, so we drop them.
        df = df.drop(['MSE_in','MSE_out','RMSE_in', 'RMSE_out', 'squared_error'], axis=1)

        # We renamte all of our columns
        df.columns=[model_name, 'hvol14', 'momentum', 'rsi', 'macd', 'bblb', 'bbub','returns']

        # We check
        # print(df.head(3))

        # We compute our squared error
        df['squared_error'] = (df['hvol14'] - df[model_name] )**2

        # We recompute MSE for our ts model
        tse_only_mse = df['squared_error'].mean()

        print(f"{model_name} - MSE :{round(tse_only_mse,4)}")

        # We can't do anything with the the current period value, because it is computed daily so we drop them.
        df = df.drop(['squared_error'], axis=1)
        for col in [model_name, 'hvol14', 'momentum', 'rsi', 'macd', 'bblb', 'bbub','returns']:
            for each in range(1,lag_number+1):
                    df[f'{col}-{each}'] = df[col].shift(each)
        df.dropna(inplace=True)
        print(df.head(3))

        # Since we are forecasting with previous date values we can't have the daily values, we drop the columns
        # Except for the ts model which itself is a prediction
        df = df.drop(['momentum', 'rsi', 'macd', 'bblb', 'bbub','returns'], axis=1)

        # We rescale our values
        ss = StandardScaler()
        df_scaled = pd.DataFrame(ss.fit_transform(df),columns = df.columns, index=df.index)


        # We check everything is in order
        print("="*100,'df_scaled',"="*100)
        print(df_scaled.head(3))


        # We define our train/test features and labels 
        X_train = df_scaled.drop('hvol14', axis=1).values
        y_train = df_scaled['hvol14'].values

        # Here 0.2 is important since we need maximum number of values to make a correct prediction.
        # We don't want a shuffle since a previous our next value must be based on the last one.
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size,shuffle=False)

        model_list = generate_models()

        for model in model_list:

            print(f"Processing model {model.name}-{model_name} ")
            print(f"Num layers :{len(model.get_config()['layers'])-1}\tDropout rate :{model.get_layer('Dropout').get_config()['rate']}")

            # we fit the model
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epoch, batch_size=5, verbose=verbose)
            history_list.append(history)

            train_mse = model.evaluate(X_train, y_train, verbose=0)
            test_mse = model.evaluate(X_test, y_test, verbose=0)

            print(f"train_mse : {round(train_mse[1],3)}\ttest_mse : {round(test_mse[1],3)}\ttest RMSE : {round(np.sqrt(test_mse[1]),3)}")

            forecast_df = pd.DataFrame(model.predict(X_test), columns=['forecast_standardized'])
            forecast_df.index  = pd.to_datetime(df.index[len(df)-len(X_test):])
            forecast_df['hvol14_standardized'] = y_test
            df_scaled.index = pd.to_datetime(df_scaled.index)
            results = pd.concat([df_scaled, forecast_df], axis=1, sort=False)

            _, _, ts_mse_in, ts_mse_out = compute_mse(results, 'hvol14', model_name, 'forecast_standardized', len(y_test))


            window_size = len(results['hvol14'])

            test_size =len(y_test)
            best_model = pick_best_model(test_mse[1], ts_mse_out)

            crypto_list.append(each_ts_model['crypto'])
            ann_model_list.append(model.name.split('-')[0])
            n_layers_list.append(len(model.get_config()['layers'])-1)
            drop_rate_list.append(model.get_layer('Dropout').get_config()['rate'])
            optimizer_list.append(model.name.split('-')[4])
            ann_mse_in_list.append(train_mse[1])
            ann_mse_out_list.append(test_mse[1])
            best_model_ts_or_ann_list.append(best_model)
            test_size_list.append(test_size)


            ts_model_list.append(each_ts_model['model'])
            ts_dist_list.append(each_ts_model['dist'])
            ts_p_list.append(each_ts_model['p'])
            ts_q_list.append(each_ts_model['q'])
            ts_mse_in_list.append(ts_mse_in)
            ts_mse_out_list.append(ts_mse_out)
            

            #### PLOTTING ######
            # We define the figs
            fig, axs = plt.subplots(2, 1, figsize=(30,10))

            # We define the box of text
            textstr = f'MSE in={round(train_mse[1],4)}\nMSE out={round(test_mse[1],4)}\nRMSE in={round(np.sqrt(train_mse[1]),4)}\nRMSE out={round(np.sqrt(test_mse[1]),4)}'

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # place a text box in upper left in axes coords
            axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

            # We plot the graph
            axs[0].set_title(f'Predicted values - Heterogenous {model.name} - combined with ts {model_name}')
            axs[0].plot(results['forecast_standardized'][window_size-test_size:], label=f'Deep ANN - Model: {model.name}',color="red")
            axs[0].plot(results[model_name][window_size-test_size:], label=f'{model_name}',color="orange")
            axs[0].plot(results['hvol14'][window_size-test_size:], label='Historical volatility 14 days - Standardized',color="black", alpha=0.7)
            axs[0].legend()
            axs[0].grid()
            axs[1].set_title('Loss / Mean Squared Error')
            axs[1].plot(history.history['loss'], label='train')
            axs[1].plot(history.history['val_loss'], label='test')
            axs[1].legend()
            axs[1].grid()
            
            # Export the results
            plt.savefig(f"./models/ann/v3/{each_ts_model['crypto']}/figs/{model.name}-{model_name}.jpg")

            df_model_results = pd.DataFrame({f'ann':results['forecast_standardized'][window_size-test_size:],f'ts': results[model_name][window_size-test_size:], 'hvol14':results['hvol14'][window_size-test_size:]})
            df_model_results.index  = results['hvol14'][window_size-test_size:].index
            df_model_results.to_csv(f"./models/ann/v3/{each_ts_model['crypto']}/csv/{model.name}-{model_name}.csv")

df = pd.DataFrame({"crypto":crypto_list,"ann_model":ann_model_list,"n_layers":n_layers_list,"drop_rate":drop_rate_list,"optimizer":optimizer_list,"ann_mse_in":ann_mse_in_list,"ann_mse_out":ann_mse_out_list,"best_model_ts_or_ann":best_model_ts_or_ann_list,"test_size":test_size_list,"ts_model":ts_model_list,"ts_dist":ts_dist_list,"ts_p":ts_p_list,"ts_q":ts_q_list,"ts_mse_in":ts_mse_in_list,"ts_mse_out":ts_mse_out_list})
df.to_csv(f"./models/ann/v3/ultimate-answer.csv")