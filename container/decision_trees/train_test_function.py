import pandas as pd
import numpy as np

#NOTE: Dec. 01, 2021 is used as cutoff as current Iowa data only goes through end of Nov. Thus, using this split cuttoff will give us a full year of test data
def train_test(df, date):
    """Used to separate 'train' dates (Dec. 2018 - Nov. 2021) from 'test' dates (Dec. 2021 - Present (Nov. 2022))

    Args:
        df (pandas DataFrame): datafrmae used for pipeline procedures
        date (str): date column found in df

    Returns:
        pandas DataFrame: train set of years 2019- end of Nov. 2021
        pandas DataFrame: test set of Dec. 2021 - Present
    """
    train, test = df.loc[df[date] < pd.to_datetime('12-01-2021', format='%m-%d-%Y')], df.loc[df[date] >= pd.to_datetime('12-01-2021', format='%m-%d-%Y')]
    if len(train) == 0 or len(test) == 0:
        train_size = np.round((0.7*len(df)),0).astype('int')
        train, test = df[0:train_size], df[train_size:]
    return train, test