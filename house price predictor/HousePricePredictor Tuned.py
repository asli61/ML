import pandas as pd
import numpy as np
from scipy import stats
import os

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

from hyperopt import fmin, tpe, hp

data = pd.read_csv("train.csv")

# Your data preprocessing steps...

# Replace the property_type, new_build, and estate_type with numerical values
data.loc[data['property_type'] == 'D', 'property_type'] = 3
data.loc[data['property_type'] == 'S', 'property_type'] = 2
data.loc[data['property_type'] == 'T', 'property_type'] = 1
data.loc[data['property_type'] == 'F', 'property_type'] = 0
data.loc[data['new_build'] == 'Y', 'new_build'] = 1
data.loc[data['new_build'] == 'N', 'new_build'] = 0
data.loc[data['estate_type'] == 'F', 'estate_type'] = 1
data.loc[data['estate_type'] == 'L', 'estate_type'] = 0

# Drop unnecessary columns
data = data.drop(['unique_id', 'saon', 'paon', 'postcode', 'street', 'linked_data_uri"08860560-9147-4237-A9D8-725FBA1CE458"'], axis=1)

# Convert 'new_build' to bool type
data['new_build'] = data['new_build'].astype('bool')

# Separate the target variable (price_paid) from the features
y = data['price_paid']
X = data.drop(['price_paid'], axis=1)

# Convert categorical columns to one-hot encoded features
categorical_cols = ['property_type', 'estate_type', 'locality', 'town', 'district', 'county', 'transaction_category']
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Convert deed_date to days since reference_date
reference_date = datetime(1970, 1, 1)
X_encoded['deed_date'] = pd.to_datetime(X_encoded['deed_date'], format="%d/%m/%Y")
X_encoded['deed_date'] = (X_encoded['deed_date'] - reference_date).dt.days

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

# Define the expanded hyperparameter space to search
space = {
    'n_estimators': hp.choice('n_estimators', range(50, 200, 10)),
    'max_depth': hp.choice('max_depth', range(3, 12)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'gamma': hp.uniform('gamma', 0, 5),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
}



def objective(params):
    # Create the XGBoost regressor with the given hyperparameters
    xgb_regressor = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        min_child_weight=params['min_child_weight'],
        reg_alpha=params['reg_alpha'],
        reg_lambda=params['reg_lambda'],
    )

    # Train the model
    xgb_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = xgb_regressor.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("mse:",mse)
    print("r2:",r2)
    # Return the error (minimize MSE)
    return mse

# Hyperparameter tuning using Bayesian optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200)

# Save the best hyperparameters to a file (e.g., "best_hyperparameters.pkl")
import pickle
with open("best_hyperparameters.pkl", "wb") as f:
    pickle.dump(best, f)

# Load the best hyperparameters from the file for future use
with open("best_hyperparameters.pkl", "rb") as f:
    best = pickle.load(f)

# Get the best hyperparameters
best_n_estimators = 50 + 10 * best['n_estimators']
best_max_depth = 3 + best['max_depth']
best_learning_rate = best['learning_rate']
best_gamma = best['gamma']
best_subsample = best['subsample']
best_colsample_bytree = best['colsample_bytree']
best_min_child_weight = 1 + best['min_child_weight']
best_reg_alpha = best['reg_alpha']
best_reg_lambda = best['reg_lambda']

# Train the final XGBoost model with the best hyperparameters
xgb_regressor = xgb.XGBRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    learning_rate=best_learning_rate,
    gamma=best_gamma,
    subsample=best_subsample,
    colsample_bytree=best_colsample_bytree,
    min_child_weight=best_min_child_weight,
    reg_alpha=best_reg_alpha,
    reg_lambda=best_reg_lambda,
)

xgb_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:")
print("n_estimators:", best_n_estimators)
print("max_depth:", best_max_depth)
print("learning_rate:", best_learning_rate)
print("gamma:", best_gamma)
print("subsample:", best_subsample)
print("colsample_bytree:", best_colsample_bytree)
print("min_child_weight:", best_min_child_weight)
print("reg_alpha:", best_reg_alpha)
print("reg_lambda:", best_reg_lambda)

print("Final Model Performance:")
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
