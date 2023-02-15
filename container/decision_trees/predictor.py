# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import sys
import signal
import traceback

import flask

import pandas as pd

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            print("time_to_decode_rb")
            with open(os.path.join(model_path, 'decision-tree-model.pkl'), 'rb') as inp:
                cls.model = pickle.load(inp,encoding="latin1")
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        print("get model")
        clf = cls.get_model()
        print("predict with model")
        return clf.predict(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        print('???text/csv')
        print("raw csv")
        print(data)
        s = StringIO(data)
        data = pd.read_csv(s, header=0)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    
    print("data columns before fixing!")
    print(data.columns)
    print("data dtypes before fixing!")
    print(data.dtypes)
    print("row 1 before fixing")
    print(data.iloc[0])
    print("row 2 before fixing")
    print(data.iloc[1])
    data.columns=data.iloc[0]
    data=data.drop(data.index[[0]])
    data = data.rename(columns=str).rename(columns={'nan':'new_lbl'})
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['date'] = pd.to_datetime(data['date'], errors='coerce', format = '%Y-%m-%d')
    data['flag'] = 3
    print("data columns after fixing!")
    print(data.columns)
    print("data dtypes after fixing!")
    print(data.dtypes)
    print("row 1 after fixing")
    print(data.iloc[0])
    print("row 2 after fixing")
    print(data.iloc[1])
    print("row 3 after fixing")
    print(data.iloc[2])
    print('Invoked with {} records'.format(data.shape[0]))
    print("!!!!time to predict!!!")
    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = StringIO()
    pd.DataFrame({'results':predictions}).to_csv(out, header=0, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')
