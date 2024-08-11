import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib

# import dataset
with open('./results/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('./results/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)


# Initialize the XGBoost regressor
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)


# Tune the parameter
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001]
}

random_search = RandomizedSearchCV(
    estimator=model_xgb,
    param_distributions=param_grid, 
    n_iter=10, cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

random_search.fit(X_train, y_train)
best_model_xgb = random_search.best_estimator_


# save model
joblib.dump(best_model_rf, './results/model_xgb.sav')