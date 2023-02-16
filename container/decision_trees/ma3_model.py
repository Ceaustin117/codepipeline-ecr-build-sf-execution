#Import packages
import pandas as pd
import numpy as np
import os
import json
import pickle
from statsmodels.tsa.arima.model import ARIMA

#Import local function
from train_test_function import train_test

# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name='training'
training_path = os.path.join(input_path, channel_name)

#Import horizons dictionary
#Based on determined forecastability thresholds, which were jointly determined by Michael, Patrick and Mitch, the horizons dictionary maps the determined Fcast_flag to the # of periods the given series is able to be forecast.
    #NOTE: These thresholds were determined based on the data being aggregated to a weekly level. Should this aggregation level change, please address this in the set_forecast_flag() function.
        #The breakdown of forecast flag values is as follows:
            #0: The given series has <20 non-zero values. Non-forecastable
            #1: The given series has between 20 and 48 non-zero values. Forecastable to 1 period (1wk) out
            #2: The given series has between 49 and 120 non-zero values. Forecastable to 4 periods (1mo) out
            #3: The given series has >120 non-zero values. Forecastable to 12 periods (1qtr) out
horizons = {}
with open("horizons.txt") as f:
    horizons = json.load(f)

def MA3(df, target, date, flag):
    """Predicts target variable using average of prior 3 items in series. Will be used for every time series regardless of level of 'forecastability'

    Args:
        df (pandas DataFrame): dataframe used for pipeline procedures
        target (str): target column name found in df
        date (str): date column name found in df

    Returns:
        array: 3-point moving average forecast results equal to length of time series. 
    """
    horiz = df['flag'].mean().astype('int').astype('str')
    train = train_test(df, date)
    print('Inside 3ma Invoked training with {} records'.format(train.shape[0]))
    print(train)
    print('type of df..')
    print(type(df))
    print('type of train')
    print(type(train))
    print('train dtypes')
    print(train.dtypes)
    print("train data index")
    print(train.index)
    if len(train[target]) >= 3:
        model = ARIMA(train[target], order = (0,0,3)).fit()
        pred = np.array([x for x in model.predict(start = 0, end = (len(train) + (horizons[horiz]-1)))])

        # pred.fillna(np.mean(train[target]), inplace = True)
    else:
        pred = np.mean(df[target])

    print("time to return model")
    return model
