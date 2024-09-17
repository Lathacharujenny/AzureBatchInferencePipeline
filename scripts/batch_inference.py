# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import joblib
import numpy as np
from azureml.core import Model

def init():
    global model
    model_path = Model.get_model_path('Diabetes_model')
    model = joblib.load(model_path)
    
def run(mini_batch):
    resultList = []
    for f in mini_batch:
        data = np.genfromtxt(f, delimiter=',')
        prediction = model.predict(data.reshape(1,-1))
        resultList.append('{}:{}'.format(os.path.basename(f), prediction[0]))
    return resultList
# -


