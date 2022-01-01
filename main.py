# Lib & Dependencies
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn import (
    feature_extraction, feature_selection, decomposition, linear_model,
    model_selection, metrics, svm
)
import requests
from scipy import stats

# Data for training
train_data = pd.read_csv("X_train.csv")
# Data for which you will submit your prediction
test_data = pd.read_csv("X_test.csv")
# Targets to be predicted
train_targets = pd.read_csv("y_train.csv")

df_train = pd.concat([train_data, train_targets], axis=1)
del df_train['id']
df_train.to_csv('train.csv', index=False)
#If you don't want to work with time serie
# train_data = train_data.drop(columns=['Moons', 'id'])
# test_data = test_data.drop(columns=['Moons', 'id'])
