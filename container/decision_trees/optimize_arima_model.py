#Import packages
import pandas as pd
import numpy as np
import json
from statsmodels.tsa.stattools import kpss, arma_order_select_ic
from statsmodels.tsa.arima.model import ARIMA

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

#Import local function
from train_test_function import train_test

def optimize_arima(df, date, target, flag):
    """Optimizes an ARIMA model for the input time series. Utilizes the arma_order_select_ic function to optimize p and q

    Args:
        df (pandas DataFrame): dataframe containing the date, target and flag variables
        date (str): column name for date variable found in df
        target (str): column name for target variable found in df
        flag (str): column name for forecast flag variable found in df. Used to determing forecasting horizon

    Returns:
        array-like: in-sample predicted values for given series along with forecast values determined by forecastability flag. Since no forecast goes past available data, returned series is padded with NaN from end of forecast window to last date in data
    """
    
    #list flag as variable (later map to horizons dictionary to determine forecasting horizon)
    horiz = df[flag].mean().astype('int').astype('str')

    #train/test split
    train = train_test(df, date)

    #Determine non-seasonal differencing necessary (d)
    d = 0
    if kpss(train[target])[1] > 0.05:
        d = 0
        p_q = arma_order_select_ic(train[target], max_ar=8, max_ma=6, ic='aic').aic_min_order
    elif kpss(np.nan_to_num(train[target].diff(1),0))[1] > 0.05:
        d = 1
        p_q = arma_order_select_ic(train[target].diff(1), max_ar=8, max_ma=6, ic='aic').aic_min_order
    else:
        d = 2
        p_q = arma_order_select_ic(train[target].diff(2), max_ar=8, max_ma=6, ic='aic').aic_min_order

    model_order = (p_q[0], d, p_q[1])
    model = ARIMA(np.asarray(train[target].values.astype('float64')), order = model_order, enforce_stationarity=True)
    # model_fit = model.fit()
    # predictions = np.array([x for x in model_fit.predict(start = 0, end = (len(train) + (horizons[horiz]-1)))])
    # filler = np.array([np.nan] * (len(test) - (horizons[horiz])))
    # predictions = np.append(predictions, filler)
    return model

# if __name__ == '__main__':
#     print('Test')
    #if __name == main allows below lines to be run when file is run as script. These lines won't be run if imported as a module
    #Unit testing goes here
        #Boundary Conditions to test:
            #All nulls, no rows, dataframe output, input format, etc.
