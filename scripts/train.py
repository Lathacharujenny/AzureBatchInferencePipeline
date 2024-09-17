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

from azureml.core import Experiment
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from azureml.core import Workspace
import pandas as pd
import numpy as np
import joblib

ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='mslearn-train-diabetes')
run = experiment.start_logging()
print('Start Experiment: ', run.name)

data = pd.read_csv('data/diabetes.csv')
X, y = data[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, data['Diabetic'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier().fit(X_train, y_train)
y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print('Accuracy: ', accuracy)
run.log('Accuracy', np.float(accuracy))
auc = roc_auc_score(y_test, y_hat)
print('AUC: ', auc)
run.log('AUC', np.float(auc))

model_file = 'diabetes.joblib'
joblib.dump(model, model_file)
run.upload_file(name='outputs/'+model_file, path_or_stream='./'+model_file)


run.register_model(model_path='outputs/diabetes.joblib', 
                   tags={'Training context': 'Inline Training'},
                  model_name='Diabetes_model',
                  properties={'AUC': run.get_metrics()['AUC'],
                             'Accuracy': run.get_metrics()['Accuracy']})


