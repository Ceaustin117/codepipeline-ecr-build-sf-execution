#Import packages
import pandas as pd
import numpy as np
import json
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

#Import horizons dictionary
#Horizons based on determined forecastability thresholds, which were jointly determined by Michael, Patrick and Mitch, the horizons dictionary maps the determined Fcast_flag to the # of periods the given series is able to be forecast.
    #NOTE: These thresholds were determined based on the data being aggregated to a weekly level. Should this aggregation level change, please address this in the set_forecast_flag() function.
        #The breakdown of forecast flag values is as follows:
            #0: The given series has <20 non-zero values. Non-forecastable
            #1: The given series has between 20 and 48 non-zero values. Forecastable to 1 period (1wk) out
            #2: The given series has between 49 and 120 non-zero values. Forecastable to 4 periods (1mo) out
            #3: The given series has >120 non-zero values. Forecastable to 12 periods (1qtr) out
horizons = {}
with open("horizons.txt") as f:
    horizons = json.load(f)

#Import ETS parameters dictionary
ETS_params = {
    "error" : ['add','mul'],
    "trend" : ['add', 'mul', None],
    "dampened_trend" : [True, False],
    "seasonal" : ['add', 'mul', None],
    "seasonal_periods" : [2,4,12,26,52], #52 weeks in a year, same # of seasonal periods consistently confirmed by STL decomp auto-period detection
    "initialization_method" : ['estimated', 'heuristic']
}

#Import local function
from train_test_function import train_test

def evaluate_ETS(data, target, date, e, t, dampened_t, s, periods, init):
    """This function evaluates an ETS model on the given parameters

    Args:
        data (pandas DataFrame): Series or array containing time series to train/test the model
        target (str): column name for target variable found in data
        date (str): column name for date variable found in data
        e (str): model error. "add" or "mul"
        t (str): model trend component. "add" or "mul"
        dampened_t (bool): Determines whether or not the trend component is dampened
        s (str): seasonality in the model. "add" or "mul"
        periods (int): number of periods in a complete seasonal cycle
        init (str): method for initialization of the statespace model. "estimated" or "heuristic"

    Returns:
        rmse (float): calculated root mean squared error of the model
        mape (float): calculated mean absolute percent error of the model
    """
    #train test split
    train, test = train_test(data, date)
    #model
    model = ETSModel(train[target], error=e, trend=t, damped_trend=dampened_t, seasonal=s, seasonal_periods=periods, initialization_method=init)
    results = model.fit(disp=0)
    predictions = [x for x in results.predict(start=len(train), end = len(data)-1)]
    rmse = np.round(np.sqrt(np.mean(np.nan_to_num((test[target] - predictions)**2, 0))) / np.mean(test[target]) * 100, 2)
    mape = np.round(mean_absolute_percentage_error(test[target], predictions),2)
    return rmse, mape


def optimize_ETS(data, target, date, flag, errors, trends, dampened_trends, seasonal, periods, init):
    """Optimizes ETS model based on lists of parameters given to the function. Each combination of given lists will be evaluated on RMSE

    Args:
        data (pandas DataFrame): series or array to be used to train/test the model
        target (str): column name for target variable found in data
        date (str): column name for date variable found in data
        flag (str): column name for forecastability flag variable found in data. To be used for determining forecast horizon
        errors (array_like): array of str values for error argument
        trends (array_like): array of str values for trend argument
        dampened_trends (array_like): array of bool values for determining if trend will be dampened in model
        seasonal (array_like): array of str values for seasonal argument
        periods (array_like): array of int values for determining seasonal period
        init (array_like): array of str values for determining initialization method

    Returns:
        print statement: Prints combination of parameters used, corresponding RMSE and MAPE. Prints best combination after all iterations have been run
    """
    horiz = data[flag].mean().astype('int').astype('str')
    best_params, best_rmse = ['n','n','n','n','n','n'], float('inf')
    for e in errors:
        for t in trends:
            for d_t in dampened_trends:
                for s in seasonal:
                    for p in periods:
                        for i in init:
                            try:
                                rmse, mape = evaluate_ETS(data,target,date,e,t,d_t,s,p,i)
                                if rmse < best_rmse:
                                    best_params = [e,t,d_t,s,p,i]
                                    best_rmse = rmse
                            except:
                                continue
    
    train, test = train_test(data, date)
    model = ETSModel(train[target], error=best_params[0], trend=best_params[1], damped_trend=best_params[2], seasonal=best_params[3], seasonal_periods=best_params[4], initialization_method=best_params[5])
    model_fit = model.fit(disp=0)
    predictions = np.array([x for x in model_fit.predict(start = 0, end = (len(train) + (horizons[horiz]-1)))])
    filler = np.array([np.nan] * (len(test) - (horizons[horiz])))
    predictions = np.append(predictions, filler)

    return best_params, predictions