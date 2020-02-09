########################################################################
# Author : Bastien Girardet
# License : MIT
# Created : 05-08-2019
# Updated : 09-10-2019
# Goal : Bruteforce and evaluate a list of ARCH models provided a range
#        of parameters.
# Results : Results are saved into a separate csv file
########################################################################

# We import our libraries
import pandas as pd
import numpy as np
import requests as rq
from datetime import datetime
import matplotlib.pyplot as plt
from libraries.Cryptov2 import Cryptocurrency
from arch.univariate import arch_model
from arch.univariate import ARCH, GARCH, EGARCH, MIDASHyperbolic
from arch.univariate import ConstantMean, ZeroMean
from arch.univariate import Normal, SkewStudent, StudentsT

# We gather our currency data with the Cryptocurrency library
BTC = Cryptocurrency('BTC', data=None ,url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20191001')
IOTA = Cryptocurrency('IOTA', data=None ,url = 'https://coinmarketcap.com/currencies/iota/historical-data/?start=20130428&end=20191001')
ETH = Cryptocurrency('ETH', data=None ,url = 'https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end=20191001')
EOS = Cryptocurrency('EOS', data=None ,url = 'https://coinmarketcap.com/currencies/eos/historical-data/?start=20130428&end=20191001')

# We define our list of crypto
crypto_list = [BTC, EOS, ETH, IOTA]

def bruteforce_ts_model(returns, start_p, start_q, max_p, max_q):
    """ This methods bruteforce each possible combination of the ARCH family models. (e.g. ARCH(3), GARCH(3,4), EGARCH(1,3))
        Records its score and save it.

        Args: 
            returns (pandas.Series) : Contains the list of all the returns.
            start_p (int) : Integer who gives the starting point of the range of p parameter
            start_q (int) : Integer who gives the starting point of the range of q parameter
            max_p (int) : Integer who gives the ending point of the range of p parameter
            max_q (int) : Integer who gives the ending point of the range of q parameter

        Output:
            df (pandas.DataFrame) : Dataframe containing all the models and Information criteria
    """

    # We define our list of models to test
    model_types = ['ARCH', 'GARCH', 'EGARCH']

    # We define our list of distribution to test
    dist_types = ['normal', 'studentst', 'skewstudent']

    # We define our list
    AIC_score = []
    BIC_score = []
    LL_score = []
    model_list = []
    mean_model_list = []
    dist_list = []
    q_list = []
    p_list = []

    # We compute the total number of models
    max_iter = max_p * max_q * len(model_types) * len(dist_types)
    current_iter = 0

    # For each model we have
    for model in model_types:

        # For each parameter p
        for each_p in range(start_p,max_p):

            # For each parameter q
            for each_q in range(start_q,max_q):

                # For each distribution type
                for dist in dist_types:
                    
                    # We define our mean model
                    am = ConstantMean(returns)
                    
                    # We define our constant mean
                    mean_model_list.append('ConstantMean')

                    # Our distribution
                    if dist is 'normal':
                        am.distribution = Normal()
                    elif dist is 'studentst':
                        am.distribution = StudentsT()
                    elif dist is 'skewstudent':
                        am.distribution = SkewStudent()

                    # Our volatility process
                    if model is "ARCH":
                        am.volatility = ARCH(p=each_p)
                    elif model is "GARCH":
                        am.volatility = GARCH(p=each_p, q=each_q)
                    elif model is "EGARCH":
                        am.volatility = EGARCH(p=each_p, q=each_q)
                    
                    # We fit our model
                    res = am.fit(update_freq=5, disp='off')

                    # We record our model and distribution
                    model_list.append(model)
                    dist_list.append(dist)

                    # We record the scores
                    AIC_score.append(res.aic)
                    BIC_score.append(res.bic)
                    LL_score.append(res.loglikelihood)

                    # We record the parameters
                    q_list.append(each_q)
                    p_list.append(each_p)

                    # We log the information about each computed model
                    print(f"it: {current_iter}/{max_iter}\tmodel:{model}\tdist:{dist[:6]}\tp:{each_p}\tq:{each_q}\tAIC_score:{round(res.aic,2)}\tBIC_score:{round(res.bic,2)}\tLog Likelihood:{round(res.loglikelihood,2)}")

                    # If a model has been added then we add one to the iterator
                    current_iter += 1

        # For each computed model
        print("="*20,f"{model} finished","="*20)

    # We combine everything to a dataframe
    df = pd.DataFrame({'volatility_model': model_list,'mean_model': mean_model_list,'dist': dist_list, 'p':p_list,'q':q_list, 'AIC_score':AIC_score, 'BIC_score':BIC_score, 'LL_score':LL_score})
    return df

# We generate a timestamp
ts = int(datetime.timestamp(datetime.now()))

for each in crypto_list:
    # We drop drop the first na
    log_returns = each.data['log_returns'].dropna()
    
    # Then we compute all the models possible with a p range from 1 to 10 and q range from 1 to 10
    print(f"Computing models for {each.name}...")
    df_results = bruteforce_ts_model(log_returns, 1, 1, 4, 4)
    print(f"{each.name} models done!")

    df_results.to_csv(f"./models/ts/fits/{each.name}-results.csv")

