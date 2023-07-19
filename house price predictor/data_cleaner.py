import pandas as pd
import numpy as np
from scipy import stats
import os

import xgboost as xgb
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime

data = pd.read_csv("train_new.csv")
#data = data[::-1]
x = 0

for index, row in data.iterrows():
    if pd.isnull(row["locality"]):
        print("locality missing on line:" + str(index))
        x += 1
print(x)


while x < 20:
    for index, row in data.iterrows():
        if pd.isnull(row["locality"]):
            # Define your condition here
            if row["postcode"] == data.loc[index - 1, "postcode"] and not pd.isnull(data.loc[index - 1, "locality"]):
                data.loc[index, "locality"] = data.loc[index - 1, "locality"]

            elif row["postcode"] == data.loc[index + 1, "postcode"] and not pd.isnull(data.loc[index + 1, "locality"]):
                data.loc[index, "locality"] = data.loc[index + 1, "locality"]
    x += 1

a = 0

data.to_csv("train_new.csv")
