import pandas as pd
import numpy as np
import requests
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

data = pd.read_csv("train_final.csv")

# print(data)

#data.loc[data['property_type'] == 'D', 'property_type'] = 3
#data.loc[data['property_type'] == 'S', 'property_type'] = 2
#data.loc[data['property_type'] == 'T', 'property_type'] = 1
#data.loc[data['property_type'] == 'F', 'property_type'] = 0
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
data = data.drop(['locality'], axis=1)
data = data.drop(['town'], axis=1)
data = data.drop(['county'], axis=1)
data = data.drop(['transaction_category'], axis=1)
data = data.drop(['linked_data_uri"08860560-9147-4237-A9D8-725FBA1CE458"'], axis=1)
#data = data.drop(['transaction_category'], axis=1)

y = data['price_paid']
X = data.drop(['price_paid'], axis=1)

X['new_build'] = X['new_build'].astype('bool')
X['estate_type'] = X['estate_type'].astype('bool')

print(X.head())######################################################################

# Get the list of categorical columns
categorical_cols = ['property_type', 'district']

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

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

input_floor_area = ""

while isinstance(input_floor_area, str):
    try:
        input_floor_area = int(input("Floor Area?"))
    except ValueError:
        input_floor_area = ""

apiSuccess = False

while not apiSuccess:
    input_postcode = input("What is your postcode?: ")

    apiResponse = requests.get("https://api.postcodes.io/postcodes/" + input_postcode)

    if apiResponse.status_code == 200:
        longitude = apiResponse.json()["result"]["longitude"]
        latitude = apiResponse.json()["result"]["latitude"]
        district = apiResponse.json()["result"]["admin_district"]
        apiSuccess = True
    else:
        print(f"There's a {apiResponse.status_code} error with your request")

input_property_type = ""

while not (input_property_type == "D" or input_property_type == "S" or input_property_type == "T" or input_property_type == "F"):
    input_property_type = input("What is the type of property? ((D)etached/(S)emi-detached/(T)erraced/(F)lat): ").upper()


input_new_build = ""

while not (input_new_build == "Y" or input_new_build == "N"):
    input_new_build = input("Is the property new? (Y/N): ").upper()

if(input_new_build == "Y"):
    input_new_build = True
elif(input_new_build == "F"):
    input_new_build = False


input_estate_type = ""

while not (input_estate_type == "F" or input_estate_type == "L"):
    input_estate_type = input("Is the property freehold or leasehold? (F/L): ").upper()



'''longitude = apiResponse.json()["result"]["longitude"]
latitude = apiResponse.json()["result"]["latitude"]
district = apiResponse.json()["result"]["admin_district"]

input_property_type = ""
input_new_build = ""
input_estate_type = ""'''




input_data = {
    'floor_area': input_floor_area,
    'property_type': input_property_type,
    'new_build': input_new_build,
    'estate_type': input_estate_type,
    'longitude': longitude,
    'latitude': latitude,
    'district': district,
}

input_df = pd.DataFrame(input_data, index=[0])

# Convert 'new_build' and 'estate_type' columns to boolean
input_df['new_build'] = input_df['new_build'].astype(bool)
input_df['estate_type'] = input_df['estate_type'].astype(bool)

# Apply one-hot encoding to categorical columns
input_df = pd.get_dummies(input_df, columns=['property_type', 'district'])

a = pd.to_datetime("19/07/2022", format="%d/%m/%Y")

input_df['deed_date'] = pd.to_datetime("19/07/2023", format="%d/%m/%Y")
input_df['deed_date'] = (input_df['deed_date'] - reference_date).dt.days

missing_cols = set(X.columns) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0

# Ensure the column order is consistent
input_df = input_df[X.columns]

x = xgb_regressor.predict(input_df)

print("Predicted price is:" + str(x))
