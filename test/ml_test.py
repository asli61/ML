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

data = pd.read_csv("train.csv")

# print(data)


data.loc[data['POSTED_BY'] == 'Owner', 'POSTED_BY'] = 0.0
data.loc[data['POSTED_BY'] == 'Dealer', 'POSTED_BY'] = 0.5
data.loc[data['POSTED_BY'] == 'Builder', 'POSTED_BY'] = 1.0
data.loc[data['BHK_OR_RK'] == 'BHK', 'BHK_OR_RK'] = 1
data.loc[data['BHK_OR_RK'] == 'RK', 'BHK_OR_RK'] = 0

data = data.loc[data['TARGET(PRICE_IN_LACS)'] < 15000]
data = data.loc[data['SQUARE_FT'] < 2e5]

data = data.drop(['ADDRESS'], axis=1)

y = data['TARGET(PRICE_IN_LACS)']
X = data.drop(['TARGET(PRICE_IN_LACS)', 'POSTED_BY'], axis=1)



# X['POSTED_BY'] = X['POSTED_BY'].astype('float')
X['BHK_OR_RK'] = X['BHK_OR_RK'].astype('bool')

print(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#####################################################

score_dict = {}
score_dict_test = {}


def score_dict_add(score_dict, model, mse, mae, r2, y_pred):
    score_dict[model] = {"R2 Score": r2,
                         "Mean Squared Error": mse,
                         "Mean Absolute Error": mae,
                         "y_pred": y_pred}
    return score_dict


# Creating the XGBoost regressor
xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Training the model
xgb_regressor.fit(X_train, y_train)

# Making predictions on the test set
y_pred = xgb_regressor.predict(X_test)

# Evaluating the model's performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

score_dict = score_dict_add(score_dict, "XGBoost Regression", mse, mae, r2, y_pred)

print(score_dict)
