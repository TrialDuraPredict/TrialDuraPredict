import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib

# import dataset
with open('./results/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('./results/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)


# Initialize the model
model_rf = RandomForestRegressor(random_state=42)

# Tune the parameter
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_grid, 
    n_iter=10, cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

random_search.fit(X_train, y_train)
best_model_rf = random_search.best_estimator_

# save model
joblib.dump(best_model_rf, './results/model_rf.sav')