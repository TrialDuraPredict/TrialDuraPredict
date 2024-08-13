import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import time

# import dataset
with open('./results/X_train_pca.pkl', 'rb') as file:
    X_train_pca = pickle.load(file)

with open('./results/y_train.pkl', 'rb') as file:
    y_train = pickle.load(file)

# Initialize the model
model_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Tune the parameter
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=model_rf,
    param_distributions=param_grid, 
    n_iter=5, cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

start_time = time.time()
random_search.fit(X_train_pca, y_train)
end_time = time.time()

# Print results
print("Total Time for Randomized Search CV: {:.2f} mins".format((end_time - start_time)/60))
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# show details for each iteration
for idx, (mean_test_score, params) in enumerate(zip(random_search.cv_results_['mean_test_score'],
                                                    random_search.cv_results_['params'])):
    print(f"Iteration {idx+1}: Score = {mean_test_score}, Params = {params}")
    
# save the best model
best_model_rf = random_search.best_estimator_
joblib.dump(best_model_rf, './results/model_rf.sav')