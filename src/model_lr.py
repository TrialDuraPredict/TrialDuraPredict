import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib

# import dataset
with open('./results/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('./results/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)


# modeling training
model_ridge = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10.0]}

grid_search = GridSearchCV(
    estimator=model_ridge,
    param_grid=param_grid, cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)
best_model_ridge = grid_search.best_estimator_


# save model
joblib.dump(best_model_ridge, './results/model_lr.sav')