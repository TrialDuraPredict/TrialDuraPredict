import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import time

# import dataset
with open('./results_example/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('./results_example/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)


# modeling training
model_ridge = Ridge()
param_grid = {'alpha': [0.1, 1.0, 10.0]}

grid_search = GridSearchCV(
    estimator=model_ridge,
    param_grid=param_grid, cv=5,
    scoring='neg_mean_squared_error'
)

start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

# Print results
print("Total Time for Randomized Search CV: {:.2f} mins".format((end_time - start_time)/60))
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# show details for each iteration
for idx, (mean_test_score, params) in enumerate(zip(grid_search.cv_results_['mean_test_score'],
                                                    grid_search.cv_results_['params'])):
    print(f"Iteration {idx+1}: Score = {mean_test_score}, Params = {params}")
    
# save the best model
best_model_lr = grid_search.best_estimator_
joblib.dump(best_model_lr, './results_example/model_lr.sav')