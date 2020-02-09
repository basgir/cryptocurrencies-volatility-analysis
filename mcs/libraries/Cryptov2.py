####################################
#  Author : Bastien Girardet
#  Goal : Fetch the data, assign it to an object and perform time-series analysis, ratios calculation.
#  Creation Date : 15-07-2019
#  Updated : 02-10-2019
#  License : MIT
####################################

# We import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.univariate import arch_model
import os

class Cryptocurrency():
    
    """
    The cryptocurrency object is here to represent an asset class. It can stores data and compute technical indicators.
    
    
    Args:
        name (str): This is the cryptocurrency's name
        data (pandas.DataFrame): This is the data which is provided to the class to store. Technical indicators are computed from it.
        url (str): This is the url of the data to fetch

    
    Attributes:
        name (str) :  Currency's name
        data_url (str): Currency data's url
        data (pandas.DataFrame): Currency data
        
    Methods:
        clean_data
        selling_or_buying
    
    """
    def __init__(self, name, data, path="./data/gen", url=None, ma_correction=False, force_update=False):
        
        print("Cryptocurrency V.0.1")
        self.name = name
        self.data_url = url
        self.ma_correction = ma_correction

        if not os.path.isdir(path):
            print(f"{path} does not exists...")
            print("Creating...")
            os.mkdir(path)
            print("Done!")
            # If the data has already been generated
            existing_file = self.check_if_data_already_exists(path)
        else:
            # If the data has already been generated
            existing_file = self.check_if_data_already_exists(path)

        # If the asset data have already been generated
        # We read the data and set is as default.
        if existing_file and force_update == False:
            
            # We read the data if it exists
            print(f"Data for {self.name} already exists.")
            print("Reading...")
            cwd = os.getcwd()
            data_dir = os.path.join(cwd, path)
            self.data = pd.read_csv(os.path.join(data_dir,existing_file))

            # We set the index
            self.data.index = pd.to_datetime(self.data['Date'])
            
            # We delete the unused Date field
            del self.data['Date']

            # We sort the index because it tends to have recent to old date index format
            self.data.sort_index(inplace=True)

            # We print the shape of the data
            print(f"Done! {self.data.shape[0]} rows and {self.data.shape[1]} columns")

            # When does it starts and when does it end
            # print(f"Starting index at : {self.data.index[0]}\tfinish at : {self.data.index[len(self.data.index['Date'])-1]}")

        # Otherwise we generate it
        else: 
            if url:
                self.data = pd.read_html(self.data_url)[0]
            else:
                self.data = data
            
            self.clean_data()

            # We lag the serie
            self.lag_serie()

            # We compute the returns
            self.compute_returns()

            # We add an index increment 0 to x
            self.set_index_range()

            # Compute the amound of ups and downs
            self.up_moves_and_down_moves()

            # We compute the historical volatilty based on 10, 22 and 44 days.
            self.compute_historical_volatility(14, shift=0)
            self.compute_historical_volatility(22, shift=0)
            self.compute_historical_volatility(44, shift=0)

            # SMA 14 days
            # We compute the SMA for 14 days.
            self.simple_moving_average(14, column='close', shift=5)

            # SMA 28 days
            # We compute the SMA for 28 days.
            self.simple_moving_average(28, column='close',  shift=5)
            
            # We compute the EMA
            # Compute the exponential moving average
            self.exponential_moving_average(12, 'close')
                  
            # Compute the exponential moving average
            self.exponential_moving_average(26, 'close')

            # RSI
            # We compute the Realtive strenght index
            self.relative_strength_index(14)

            # MACD
            # We compute the MACD      
            self.macd()

            # Momentum
            # We compute the momentum over a 10 period range
            self.momentum(10)


            # Bollinger bands
            # We compute the bollinger bands
            self.bollinger_bands(14,0)

            # We compute the realized volatility                      
            # We compute daily volatility and daily annualized volatility
            self.compute_daily_volatility()
            
            self.compute_lagged_serie('daily_volatility_returns', 5)
            self.compute_lagged_serie('returns', 5)

            # We save our generated data
            self.save_data(path)
    
    def check_if_data_already_exists(self, path):
        """Check if the data have already been generated"""
        import os
        from os import listdir
        from os.path import isfile
        cwd = os.getcwd()
        data_path = os.path.join(cwd, path)
        all_files_in_folder = onlyfiles = [f for f in listdir(data_path) if isfile(os.path.join(data_path, f))]
        exists = None

        for file in all_files_in_folder:
            if self.name in file:
                exists = file
                break
        return exists
        
    def set_index_range(self):
        self.data['idx'] = range(0,len(self.data))
        self.data['idx'] = pd.to_numeric(self.data['idx'], downcast='integer')
                  
    def lag_serie(self):
        """Construct the lagged serie of the open and close
        """
        self.data['lagged_open'] = self.data['open'].shift(-1)
        self.data['lagged_close'] = self.data['close'].shift(-1)
        
    def clean_data(self):
        """Clean the data by renaming columns name
        """
        self.data.index = pd.to_datetime(self.data['Date'])
        del self.data['Date']
        self.data.columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
        self.data.sort_index(inplace=True)
    
    def compute_returns(self):
        """ Compute the returns and log returns
        """
        import numpy as np

        print("Compute returns and log returns...")
        self.data['log_price'] = np.log(self.data['close'])
        self.data['log_returns'] = self.data['log_price'].diff()


        self.data['lagged_returns'] = self.data['returns'].shift(-1)
        self.data['returns2'] = self.data['returns'] ** 2
        print("Done!")
        
    def compute_daily_volatility(self):
        print("Computing daily volatility...")
        self.data['daily_volatility_close'] = self.data.apply(lambda x: np.sqrt(0.5 * (x['close'] - 0.5*(x['lagged_close'] + x['close']))**2) , axis=1)
        self.data['daily_volatility_returns'] = self.data.apply(lambda x: np.sqrt(0.5 * (x['returns'] - 0.5*(x['lagged_returns'] + x['returns']))**2) , axis=1)

        # We set the last daily_volatility value to the next day value
        self.data['daily_volatility_returns'].iloc[-1] = self.data['daily_volatility_returns'].iloc[-2]

        self.data['daily_ann_volatility'] = self.data['daily_volatility_returns']* np.sqrt(365)
        print("Done !")
    
    def compute_historical_volatility(self,n_days, shift=0):
        """ Compute historical volatility of a certain n_days range
        Args :
            n_days : days range to compute from
            
        Ouptut :
            Creates the series containing the historical volatility and add it to the data
        """
        print(f"Computing historical volatility {n_days} days")

        # Set the list of volatilities
        volatilities = []
        
        # Go through each rows and return the volatility of n range
        for idx, row in self.data.iterrows():
            current_idx = int(row['idx'])

            if current_idx <= len(self.data['returns']) - n_days:
                current_range = self.data.iloc[current_idx:n_days+current_idx,:]['returns']
                volatilities.append(current_range.std())
                
            else: 
                volatilities.append(np.nan)
    
        self.data['HV_{0}_days'.format(n_days)] = volatilities
        self.data['hvol{0}'] = EOS.data['hvol21'] = EOS.data['stdev21'] * (365**0.5) # Annualize.
        print("Done!")
        
    def up_moves_and_down_moves(self):
        """Compute the up moves and down moves in order to compute the RSI
        
        Source : https://www.macroption.com/rsi/
        """
        
        up_moves = []
        down_moves = []
        
        def compute_up_move(C_t, C_t_1):
 
            delta_C_t = C_t - C_t_1
            r_t = (C_t - C_t_1) / C_t_1 * 100

            if delta_C_t > 0:
                return(np.around(r_t,3))
            else:
                return(0)
            
        def compute_down_move(C_t, C_t_1):
 
            delta_C_t =  C_t - C_t_1
            r_t = (C_t - C_t_1) / C_t_1 * 100

            if delta_C_t < 0:
                return(np.around(np.abs(r_t),3))
            else:
                return(0)
        
        for idx, row in self.data.iterrows():
            current_idx = int(row['idx'])
            if current_idx == len(self.data['close'])-1:
                C_t_1 = self.data['close'].iloc[current_idx]
            else:
                C_t_1 = self.data['close'].iloc[current_idx+1]

            
            C_t = self.data['close'].iloc[current_idx]
            
            actual_diff = C_t - C_t_1
            
            up_moves.append(compute_up_move(C_t,C_t_1))
            down_moves.append(compute_down_move(C_t,C_t_1))
             
        
        self.data['up'] = up_moves
        self.data['down'] = down_moves

    def relative_strength_index(self, n=14):
        """ Compute the RSI function
        """
        print("Computing RSI...")
        prices = self.data['close'][::-1]
        deltas = np.diff(prices)
        seed = deltas[:n+1]

        up = seed[seed>=0].sum()/n

        down = -seed[seed<0].sum()/n

        rs = up/down
        
        rsi = np.zeros_like(prices)

        rsi[:n] = 100. - 100./(1.+rs)

        for i in range(n, len(prices)):
            delta = deltas[i-1]

            if delta>0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(n-1) + upval)/n
            down = (down*(n-1) + downval)/n
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        self.data['rsi'] = rsi[::-1]
        print("Done!")
    
    def macd(self):
        """Compute the moving average convergence divergence (MACD) 
        
        Formula:
            EMA_period 12 - EMA_period_26
            
        Definition:
            The MACD is an indicator of changing trend
        """
               
        print("Computing MACD...")
        # Compute the exponential moving average
        self.exponential_moving_average(12, shift=8)

        # Compute the exponential moving average
        self.exponential_moving_average(26, shift=8)

        self.data['macd'] = self.data['EMA_12_close'] - self.data['EMA_26_close']
        print("Done !")

    def momentum(self, n_days):
        """Compute the momentum over a n_days period.
        """
        
        print("Computing momentum...")
        momentum_list = []
        for idx, row in self.data.iterrows():
            current_idx = int(row['idx'])

            if current_idx <= len(self.data['close']) - n_days:
                
                # Formula C_t - C_t-n
                c_t = self.data['close'][current_idx]
                c_t_n_days = self.data['close'][n_days+current_idx-1]
                momentum_list.append(c_t-c_t_n_days)
                
            else:
                # If the index is out of bound. for example if n_days is 10 and we are at index < 10 we won't be able to compute the momentum 
                momentum_list.append(0)
          
        
        self.data['momentum'.format(n_days)] = momentum_list        
        print("Done !")
    
    def exponential_moving_average(self, n_days, column='close', shift=0):
        """Exponential moving average = [Close - previous EMA] * (2 / n+1) + previous EMA
        
        Args: 
            n_days (int) : Number of days on which the SMA is based
            alpha (float) : weight of the lagged close price
            shift (int) : Number of days which lags the serie
            
        Returns:
            None
        """

        alpha = 2 / (n_days+1)

        self.simple_moving_average(n_days, column, shift=0)
        self.data[f'lagged_SMA_{n_days}_{column}'] = self.data[f'SMA_{n_days}_{column}'].shift(-1)

        import numpy as np
                   
        EMA_list = []
        
        if alpha <=1:
            for idx, row in self.data.iterrows():
                current_idx = int(row['idx'])    

                if current_idx == 0:
                    EMA_list.append(self.data[column][0])
                else:
                    EMA_list.append(alpha * self.data[column][current_idx] + (1-alpha) *  self.data[f'lagged_SMA_{n_days}_{column}'][current_idx] )
                    
        else:
            print("Error alpha must be < 1")

        
        
        self.data[f'EMA_{n_days}_{column}'] = EMA_list
        self.data[f'DEMA_{n_days}_{column}'] = self.data[f'EMA_{n_days}_{column}'].shift(shift)
        
    def compute_lagged_serie(self, column, n_lag):
        print(f"Computing {column} for {n_lag} lag...")
        for lag in range(1, n_lag+1):
            self.data[f"{column}_{lag}_lag"] = self.data[column].shift(-lag)
        print("Done!")

    def simple_moving_average(self, n_days, column='close', shift=0):
        """Compute a simple moving average from a provided number of days
        
        Args: 
            n_days (int) : Number of days on which the SMA is based
            shift (int) : Number of days which lags the serie
            
        Returns:
            None
        """

        means = []
        for idx, row in self.data.iterrows():
            current_idx = int(row['idx'])            
            current_value = 0

            if current_idx <= len(self.data[column]) - n_days:
                current_range = self.data.iloc[current_idx:n_days+current_idx,:][column]
                current_value = current_range.mean()
                means.append(current_range.mean())
                
            else: 
                means.append(np.nan)
          
        self.data[f'SMA_{n_days}_{column}'] = means
        self.data[f'DSMA_{n_days}_{column}'] = self.data[f'SMA_{n_days}_{column}'].shift(shift)
        
    def bollinger_bands(self, n_days, shift=0, delta=2):
        """Compute the bollinger bands for a certain (Displaced) Moving Average
        
        Args :
            n_days : Number of days of the (Displaced) Moving Average (D)MA
            shift : The shift at which the bollinger must be set
            delta : standard deviation multiplication factor
        Return : 
            None
            
        Creates a bollinger bands upper and lower band in the self.data
        """
        
        self.data['boll_bands_upper_band'] = self.data['SMA_{0}_close'.format(n_days, shift)] + delta * self.data['HV_{0}_days'.format(n_days)] * self.data['SMA_{0}_close'.format(n_days, shift)] 
        self.data['boll_bands_lower_band'] = self.data['SMA_{0}_close'.format(n_days, shift)] - delta * self.data['HV_{0}_days'.format(n_days)] * self.data['SMA_{0}_close'.format(n_days, shift)] 
    def plot_RSI(self, start=0, end=700, lim=(20,80)):

        import numpy as np
        import matplotlib.pyplot as plt
        
        # If the end index is bigger than the index itself
        if end > len(self.data['close']):
            end = len(self.data['close'])-1
            
        # Starting index vs ending index
        start_date = self.data.index[end].strftime("%d-%b-%Y")
        end_date = self.data.index[start].strftime("%d-%b-%Y")
        
        # We plot the series 
        t = self.data.iloc[start:end].index
        s1 = self.data['close'].iloc[start:end]
        s2 = self.data['boll_bands_upper_band'].iloc[start:end]
        s3 = self.data['boll_bands_lower_band'].iloc[start:end]
        s4 = self.data['rsi'].iloc[start:end]

        fig, axs = plt.subplots(2, 1, figsize=(30,20))
        axs[0].set_title('{0} Close {1} to {2}'.format(self.name, start_date, end_date))
        axs[0].plot(t, s1, label='Close')
        axs[0].plot(t, s2, label='BB up')
        axs[0].plot(t, s3, label='BB down')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('Close')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(t, s4)
        # Overbought line
        axs[1].axhline(y=lim[1], c='r')
        axs[1].fill_between(t, lim[1], s4, where=s4>lim[1], color='r')

        # Oversold line
        axs[1].axhline(y=lim[0], c='g')
        axs[1].fill_between(t, lim[0], s4, where=s4<lim[0],  color='g')
        axs[1].set_ylabel('RSI')
        axs[1].grid(True)


        fig.tight_layout()
        plt.show()

    def save_data(self, path):
        """We save our computed data to csv file.
        
        Args:
            path (str) : Path of folder where the data will be saved.
        """

        from datetime import datetime
        import os

        # We generate a timestamp
        ts = datetime.timestamp(datetime.now())

        # Generate the path.
        cwd = os.getcwd()
        data_path = os.path.join(cwd,path)

        # Generate the filename
        filename = f"{self.name}-generated.csv"
        filepath = os.path.join(data_path, filename)
        
        # We save the path
        self.data.to_csv(filepath)