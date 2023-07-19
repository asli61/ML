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

data = pd.read_csv("train.csv")

# print(data)

data.loc[data['property_type'] == 'D', 'property_type'] = 3
data.loc[data['property_type'] == 'S', 'property_type'] = 2
data.loc[data['property_type'] == 'T', 'property_type'] = 1
data.loc[data['property_type'] == 'F', 'property_type'] = 0
data.loc[data['new_build'] == 'Y', 'new_build'] = 1
data.loc[data['new_build'] == 'N', 'new_build'] = 0
data.loc[data['estate_type'] == 'F', 'estate_type'] = 1
data.loc[data['estate_type'] == 'L', 'estate_type'] = 0

#data = data.loc[data['TARGET(PRICE_IN_LACS)'] < 15000]
#data = data.loc[data['SQUARE_FT'] < 2e5]

data = data.drop(['unique_id'], axis=1)
data = data.drop(['saon'], axis=1)
data = data.drop(['paon'], axis=1)
data = data.drop(['postcode'], axis=1)
data = data.drop(['street'], axis=1)
#data = data.drop(['transaction_category'], axis=1)
data = data.drop(['linked_data_uri"08860560-9147-4237-A9D8-725FBA1CE458"'], axis=1)

y = data['price_paid']
X = data.drop(['price_paid'], axis=1)

X['new_build'] = X['new_build'].astype('bool')

print(X.head())######################################################################

# Get the list of categorical columns
categorical_cols = ['property_type', 'estate_type', 'locality', 'town', 'district', 'county', 'transaction_category']

# Apply one-hot encoding to categorical columns
#X_encoded = pd.get_dummies(X, columns=categorical_cols)

X = pd.get_dummies(X, columns=categorical_cols)

#print(X_encoded)
print(X)


#X['postcode'] = X['postcode'].astype('category')
'''X['property_type'] = X['property_type'].astype('category')
X['estate_type'] = X['estate_type'].astype('category')
X['street'] = X['street'].astype('category')
X['locality'] = X['locality'].astype('category')
X['town'] = X['town'].astype('category')
X['district'] = X['district'].astype('category')
X['county'] = X['county'].astype('category')
X['transaction_category'] = X['transaction_category'].astype('category')'''

X['deed_date'] = pd.to_datetime(X['deed_date'], format="%d/%m/%Y")

reference_date = datetime(1970, 1, 1)
print(X['deed_date'])
X['deed_date'] = (X['deed_date'] - reference_date).dt.days

print(X['deed_date'])

print("type of date_object =", type(X['deed_date']))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
xgb_regressor = xgb.XGBRegressor(n_estimators=100)

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


input_postcode = input("What is your postcode? ")

input_
