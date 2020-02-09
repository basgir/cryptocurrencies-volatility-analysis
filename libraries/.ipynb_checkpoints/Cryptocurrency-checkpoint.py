####################################
#  Author : Bastien Girardet
#  Goal : Fetch the data, assign it to an object and perform time-series analysis, ratios calculation.
#  Creation Date : 15-07-2019
#  Updated : 05-09-2019
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
            print(f"Data for {self.name} already exists.")
            print("Reading...")
            cwd = os.getcwd()
            data_dir = os.path.join(cwd, path)
            self.data = pd.read_csv(os.path.join(data_dir,existing_file))
            print(f"Done! {self.data.shape[0]} rows and {self.data.shape[1]} columns")
            print(f"Starting index at : {self.data['Date'][len(self.data['Date'])-1]}\tfinish at : {self.data['Date'][0]}")

        # Otherwise we generate it
        else: 
            if url:
                self.data = pd.read_html(self.data_url)[0]
            else:
                self.data = data
            
            self.clean_data()
        
            self.win_or_loose()
            
            # We add an index increment 0 to x
            self.set_index_range()
            
            # We compute the ratios and returns
            self.open_on_close()
            self.candle_body_value()
            self.upper_candle_value()
            self.lower_candle_value()
            self.maximum_difference()
            self.max_diff_to_upper_leg_ratio()
            self.max_diff_to_lower_leg_ratio()
            self.body_to_upper_leg_ratio()
            self.body_to_lower_leg_ratio()
            self.body_to_max_diff_ratio()
            self.lag_serie()
            self.compute_return()
            
            # We check for overnight trading
            self.overnight_trading()
            
            # We compute daily volatility and daily annualized volatility
            self.compute_daily_volatility()
            
            # We compute the historical volatilty based on 10, 22 and 44 days.
            self.compute_historical_volatility(10, shift=0)
            self.compute_historical_volatility(30, shift=0)

            # We compute the historical volatilty based on 10, 22 and 44 days. with formula
            self.compute_historical_volatility_DMA(10, shift=0)
            
            # We compute the SMA for 10, 22, and 44 days.
            self.simple_moving_average(10, shift=5)
            
            # Compute the exponential moving average
            self.exponential_moving_average(12, alpha=0.7)
                  
            # Compute the exponential moving average
            self.exponential_moving_average(26, alpha=0.7)
            
            # We compute the MACD      
            self.macd()
                  
            # Compute the amound of ups and downs
            self.up_moves_and_down_moves()
            
            # We compute the Realtive strenght inde
            self.relative_strength_index(14, method="SMA")
            
            # We compute the momentum over a 10 period range
            self.momentum(10)
            
            # We compute the bollinger bands
            self.bollinger_bands(10,5,2)
            
            # We compute the stochastic K
            self.stochastic_k(10)
            
            # We compute the stochastic K
            self.stochastic_d(10)
            
            # We compute the Larry William's R oscillator
            self.larry_william_r(14)
            
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
    
    def lag_serie(self):
        """Construct the lagged serie of the open and close
        """
        
        self.data['lagged_open'] = self.data['open'].shift(-1)
        self.data['lagged_close'] = self.data['close'].shift(-1)
        
    def open_on_close(self):
        self.data['open_close'] = self.data['open'] / self.data['close']
    
    def win_or_loose(self):
        self.data['is_win'] = self.data['open'] < self.data['close']
        self.data['pos']  = self.data['is_win'].apply(lambda x: 1.00 if x else -1.00)
    
    def candle_body_value(self):
        self.data['candle_body_value'] = abs(self.data['open'] - self.data['close'])
        
    def upper_candle_value(self):
        self.data['upper_candle_value'] = self.data.apply(lambda x: (x['high']-x['close']) if x['is_win'] else(x['high']-x['open']), axis=1) 
        
    def lower_candle_value(self):
        self.data['lower_candle_value'] = self.data.apply(lambda x:  (x['open']-x['low']) if x['is_win'] else (x['close']-x['low']), axis=1) 
        
    def maximum_difference(self):
        self.data['max_diff'] = self.data['high']-self.data['low']
    
    def max_diff_to_upper_leg_ratio(self):
        self.data['upper_leg_on_max_diff_ratio'] = self.data['max_diff']  / self.data['upper_candle_value'] 
        
    def max_diff_to_lower_leg_ratio(self):
        self.data['lower_leg_on_max_diff_ratio'] = self.data['max_diff']  / self.data['lower_candle_value']
    
    def body_to_upper_leg_ratio(self):
        self.data['body_to_upper_leg_ratio'] = self.data['candle_body_value']  / self.data['upper_candle_value'] 
        
    def body_to_lower_leg_ratio(self):
        self.data['body_to_lower_leg_ratio'] = self.data['candle_body_value']  / self.data['lower_candle_value']
        
    def body_to_max_diff_ratio(self):
        self.data['body_to_max_diff_ratio'] = self.data['candle_body_value']  / self.data['max_diff'] * self.data['pos']
        
    def clean_data(self):
        self.data.index = pd.to_datetime(self.data['Date'])
        del self.data['Date']
        self.data.columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap']
    
    def compute_return(self):
        self.data['returns'] = (self.data['close']/self.data['lagged_close'])-1
        self.data['log_return'] = np.log(self.data['returns']*100)
    
    def overnight_trading(self):
        """Check for overnight trading; if the close of the previous day doesn't match the current day open"""
        print("Computing overnight trading...")
        overnight_trading = []
        for idx, row in self.data.iterrows():
            if row['idx'] < len(self.data['close'])-1:
                overnight_trading.append(self.data['open'][row['idx']] != self.data['close'][row['idx']+1])
            else:
                overnight_trading.append(np.nan)
        self.data['overnight_trading'] = overnight_trading
        print("Done !")
        
    def compute_daily_volatility(self):
        print("Computing daily volatility")
        self.data['daily_volatility'] = self.data.apply(lambda x: np.sqrt(0.5 * (x['close'] - 0.5*(x['lagged_close'] + x['close']))**2) , axis=1)
        self.data['daily_ann_volatility'] = self.data['daily_volatility']* np.sqrt(365)
        print("Done !")
    
    def compute_historical_volatility(self,n_days, shift=0):
        """ Compute historical volatility of a certain n_days range
        Args :
            n_days : days range to compute from
            
        Ouptut :
            Creates the series containing the historical volatility and add it to the data
        """
        
        # Set the list of volatilities
        volatilities = []
        
        # Go through each rows and return the volatility of n range
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['returns']) - n_days:
                current_range = self.data.iloc[row['idx']:n_days+row['idx'],:]['returns']
                volatilities.append(current_range.std())
                
            else: 
                volatilities.append(np.nan)
    
        self.data['HV_{0}_days'.format(n_days)] = volatilities
        self.data['DHV_{0}_days'.format(n_days)] = self.data['HV_{0}_days'.format(n_days)].shift(shift)
        
        
    def compute_historical_volatility_DMA(self,n_days, shift=0):
        """ Compute historical volatility of a certain n_days range
        Args :
            n_days : days range to compute from
            
        Ouptut :
            Creates 
        """
        
        # Set the list of volatilities
        volatilities = []
        
        # Go through each rows and return the volatility of n range
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['returns']) - n_days:
                
                # FOMULA 
                # sigma = sqrt((1/(n_days-1)) sum_{t=1}^{N}(R_t - \hat{R})^2)
                
                current_range = self.data.iloc[row['idx']:n_days+row['idx'],:]['returns']
                current_range_mean = current_range.mean()
                
                # R_t - \hat{R}
                diff_R_hat_R = current_range - current_range_mean
                
                # (R_t - \hat{R})^2
                squared_R_hat_R = diff_R_hat_R**2
                
                # Sum (R_t - \hat{R})^2
                sum_squared_R_hat_R = np.cumsum(squared_R_hat_R)
                
                # sqrt((1/(n_days-1)) sum_{t=1}^{N}(R_t - \hat{R})^2)
                factor = 1 / (n_days - 1)
                
                sigma = np.sqrt(factor * sum_squared_R_hat_R)
                
                
                volatilities.append(sigma.values[0])
                
            else: 
                volatilities.append(np.nan)
    
        self.data['DMA_HV_{0}_days'.format(n_days)] = volatilities
        self.data['DMA_DHV_{0}_days'.format(n_days)] = self.data['HV_{0}_days'.format(n_days)].shift(shift)
    
    
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
            
            if row['idx'] == len(self.data['close'])-1:
                C_t_1 = self.data['close'].iloc[row['idx']]
            else:
                C_t_1 = self.data['close'].iloc[row['idx']+1]

            
            C_t = self.data['close'].iloc[row['idx']]
            
            actual_diff = C_t - C_t_1
            
            up_moves.append(compute_up_move(C_t,C_t_1))
            down_moves.append(compute_down_move(C_t,C_t_1))
             
        
        self.data['up'] = up_moves
        self.data['down'] = down_moves

    
    def relative_strength_index(self, n_days=14, method="SMA"):
        """Compute the relative strength index (RSI) on a period of n_days
        
        
        Args: 
            n_days (int) : Period on which RSI is computed
            method (string): Method used to compute RSI
            
            
        Definition:
            The RSI bounds are [0,100] if it is low around (0) it is a bearish market and
            if high (100) bullish market.
        """
        
        methods_list = ['SMA', 'EMA' , 'Wilder']
        RSI_list = []
        
        print("Computing RSI...")
        
        
        for idx, row in self.data.iterrows():
            if method == "SMA":
                # average of all ups and downs during the defined period which is index to index + n_days (n_days past days)
                num_up = len(self.data['up'].iloc[row['idx']:row['idx'] + n_days][self.data['up'].iloc[row['idx']:row['idx'] + n_days] != 0])
                num_down = len(self.data['down'].iloc[row['idx']:row['idx'] + n_days][self.data['down'].iloc[row['idx']:row['idx'] + n_days] != 0])
                # print(f"num_up : {num_up}\t num_down : {num_down}")

                avg_up = np.sum(self.data['up'].iloc[row['idx']:row['idx'] + n_days][self.data['up'].iloc[row['idx']:row['idx'] + n_days] != 0])/n_days
                avg_down = np.sum(self.data['down'].iloc[row['idx']:row['idx'] + n_days][self.data['down'].iloc[row['idx']:row['idx'] + n_days] != 0])/n_days   

                # print(f"avg_up : {avg_up}\t avg_down : {avg_up}")

                RS = avg_up / avg_down
                RSI = 100-(100/(1+RS))
                RSI_list.append(RSI)
        
        
        self.data['RSI'] = RSI_list
        print("Done !")
                  
    
    def macd(self):
        """Compute the moving average convergence divergence (MACD) 
        
        Formula:
            EMA_period 12 - EMA_period_26
            
        Definition:
            The RSI bounds are [0,100] if it is low around (0) it is a bearish market and
            if high (100) bullish market.
        """
               
        print("Computing MACD...")
        
        self.data['macd'] = self.data['EMA_12_days'] - self.data['EMA_26_days']
        print("Done !")
    
    
    
    def momentum(self, n_days):
        """Compute the momentum over a n_days period.
        """
        
        print("Computing momentum...")
        momentum_list = []
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['close']) - n_days:
                
                # Formula C_t - C_t-n
                c_t = self.data['close'][row['idx']]
                c_t_n_days = self.data['close'][n_days+row['idx']-1]
                momentum_list.append(c_t-c_t_n_days)
                
            else: 
                momentum_list.append(np.nan)
          
        
        self.data['momentum'.format(n_days)] = momentum_list        
        print("Done !")
    
    def stochastic_k(self, n_days):
        """Stockastic K%
        Stochastic oscillators. These oscillators are clear trend indicatiors for any stock. When stockastic oscillator are increasing, the stock prices are likely to go up and vice-a-versa.
        If the stochastic oscillators at time 't' is greater than the value at time 't-1' then the opinion of trend is 'up' and represented as '+1' and vice a versa
        """
        
        
        print("Computing Stochastic K...")
        stochastics_k = []
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['close']) - n_days:
                

                t_past_days = row['idx'] + n_days
                C_t = self.data['close'].iloc[row['idx']]
                LL_t = np.min(self.data['close'].iloc[row['idx']:t_past_days])
                HH_t = np.max(self.data['close'].iloc[row['idx']:t_past_days])
                
                stochastic_k_perc = (C_t - LL_t) / (HH_t - LL_t ) * 100
                stochastics_k.append(np.around(stochastic_k_perc,2))
                
            else: 
                stochastics_k.append(np.nan)
          
        
        self.data['stochastic_k'] = stochastics_k
        print("Done !")
        
    
    def stochastic_d(self, n_days=10):
        """Stockastic D%
        Stochastic oscillators. These oscillators are clear trend indicatiors for any stock. When stockastic oscillator are increasing, the stock prices are likely to go up and vice-a-versa.
        If the stochastic oscillators at time 't' is greater than the value at time 't-1' then the opinion of trend is 'up' and represented as '+1' and vice a versa
        """
        
        stochastics_d = []
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['close']) - n_days:
                
                
                t_past_days = row['idx'] + n_days
                stochastic_d = self.data['stochastic_k'].iloc[row['idx']:t_past_days].mean()         
                
                stochastics_d.append(np.around(stochastic_d,2))
                
            else: 
                stochastics_d.append(np.nan)
          
        
        self.data['stochastic_d'] = stochastics_d 
    
    
    def larry_william_r(self, n_days):
        """Larry William's R%
        Stochastic oscillators. These oscillators are clear trend indicatiors for any stock. When stockastic oscillator are increasing, the stock prices are likely to go up and vice-a-versa.
        If the stochastic oscillators at time 't' is greater than the value at time 't-1' then the opinion of trend is 'up' and represented as '+1' and vice a versa
        source : https://www.abcbourse.com/apprendre/11_williams.html
        """
        
        will_R_list = []
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['close']) - n_days:
    
                C_t = self.data['close'].iloc[row['idx']]
                L_n = self.data['low'].iloc[n_days]
                H_n = self.data['high'].iloc[n_days]
                
                current_will_R = (H_n - C_t) / (H_n - L_n ) * 100
                will_R_list.append(np.around(current_will_R,2))
                
            else: 
                will_R_list.append(np.nan)
          
        
        self.data['R%'] = will_R_list
        
    def exponential_moving_average(self, n_days, alpha=0.5, shift=0):
        """Exponential moving average = [Close - previous EMA] * (2 / n+1) + previous EMA
        
        Args: 
            n_days (int) : Number of days on which the SMA is based
            alpha (float) : weight of the lagged close price
            shift (int) : Number of days which lags the serie
            
        Returns:
            None
        """
        import numpy as np
        
        def EMA(t):
            if t == 0:
                return(self.data['close'][0])
            else:
                return(self.data['close'][t])
            
        EMA_list = []
        
        if alpha <=1:
        
            for idx, row in self.data.iterrows():
                if row['idx'] == 0:
                    EMA_list.append(self.data['close'][0])
                else:
                    EMA_list.append((1-alpha) * EMA(row['idx']-1) + alpha * self.data['close'][row['idx']])
                    
        else:
            print("Error alpha must be < 1")

        
        
        self.data['EMA_{0}_days'.format(n_days)] = EMA_list
        
    
    def simple_moving_average(self, n_days, shift=0):
        """Compute a simple moving average from a provided number of days
        
        Args: 
            n_days (int) : Number of days on which the SMA is based
            shift (int) : Number of days which lags the serie
            
        Returns:
            None
        """
        
        means = []
        for idx, row in self.data.iterrows():
            if row['idx'] <= len(self.data['close']) - n_days:
                current_range = self.data.iloc[row['idx']:n_days+row['idx'],:]['close']
                means.append(current_range.mean())
                
            else: 
                means.append(np.nan)
          
        self.data['SMA_{0}_days'.format(n_days)] = means
        self.data['DSMA_{0}_days_{1}_days_shift'.format(n_days, shift)] = self.data['SMA_{0}_days'.format(n_days)].shift(shift)
        
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
        
        self.data['boll_bands_upper_band'] = self.data['DSMA_{0}_days_{1}_days_shift'.format(n_days, shift)] + delta * self.data['DHV_{0}_days'.format(n_days)] * self.data['DSMA_{0}_days_{1}_days_shift'.format(n_days, shift)] 
        self.data['boll_bands_lower_band'] = self.data['DSMA_{0}_days_{1}_days_shift'.format(n_days, shift)] - delta * self.data['DHV_{0}_days'.format(n_days)] * self.data['DSMA_{0}_days_{1}_days_shift'.format(n_days, shift)] 
    
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
        filename = f"{self.name}-generated-{ts}.csv"
        filepath = os.path.join(data_path, filename)
        
        # We save the path
        self.data.to_csv(filepath)