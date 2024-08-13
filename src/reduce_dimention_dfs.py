import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA


with open('./results/X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)
    
with open('./results/X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)

with open('./results/X_incompleted.pkl', 'rb') as file:
    X_incompleted = pickle.load(file)
    
# reduce the dimensions for X_train, making the training more efficient
# criteria for selection of PCA components is >70% variability
pca = PCA(n_components=50)
X_train_description_pca = pca.fit_transform(X_train[:, 1:769])
X_test_description_pca = pca.transform(X_test[:, 1:769])
X_incompleted_description_pca = pca.transform(X_incompleted[:, 1:769])

pca = PCA(n_components=50)
X_train_inclusion_pca = pca.fit_transform(X_train[:, 769:1537])
X_test_inclusion_pca = pca.transform(X_test[:, 769:1537])
X_incompleted_inclusion_pca = pca.transform(X_incompleted[:, 769:1537])

pca = PCA(n_components=50)
X_train_exclusion_pca = pca.fit_transform(X_train[:, 1537:2305])
X_test_exclusion_pca = pca.transform(X_test[:, 1537:2305])
X_incompleted_exclusion_pca = pca.transform(X_incompleted[:, 1537:2305])

pca = PCA(n_components=50)
X_train_treatment_pca = pca.fit_transform(X_train[:, 2305:3073])
X_test_treatment_pca = pca.transform(X_test[:, 2305:3073])
X_incompleted_treatment_pca = pca.transform(X_incompleted[:, 2305:3073])

pca = PCA(n_components=50)
X_train_disease_pca = pca.fit_transform(X_train[:, 3073:3841])
X_test_disease_pca = pca.transform(X_test[:, 3073:3841])
X_incompleted_disease_pca = pca.transform(X_incompleted[:, 3073:3841])

pca = PCA(n_components=50)
X_train_measure_pca = pca.fit_transform(X_train[:, 3841:4609])
X_test_measure_pca = pca.transform(X_test[:, 3841:4609])
X_incompleted_measure_pca = pca.transform(X_incompleted[:, 3841:4609])

pca = PCA(n_components=50)
X_train_timeframe_pca = pca.fit_transform(X_train[:, 4609:5377])
X_test_timeframe_pca = pca.transform(X_test[:, 4609:5377])
X_incompleted_timeframe_pca = pca.transform(X_incompleted[:, 4609:5377])

# get pca reduction for train, test and incompleted dataset
X_train_pca = np.concatenate((X_train[:, [0]], X_train_description_pca, X_train_inclusion_pca,
                              X_train_exclusion_pca, X_train_treatment_pca, X_train_disease_pca,
                              X_train_measure_pca, X_train_timeframe_pca), axis=1)

X_test_pca = np.concatenate((X_test[:, [0]], X_test_description_pca, X_test_inclusion_pca,
                              X_test_exclusion_pca, X_test_treatment_pca, X_test_disease_pca,
                              X_test_measure_pca, X_test_timeframe_pca), axis=1)

X_incompleted_pca = np.concatenate((X_incompleted[:, [0]], X_incompleted_description_pca, X_incompleted_inclusion_pca,
                              X_incompleted_exclusion_pca, X_incompleted_treatment_pca, X_incompleted_disease_pca,
                              X_incompleted_measure_pca, X_incompleted_timeframe_pca), axis=1)

# save this dataset
with open('./results/X_train_pca.pkl', 'wb') as file:
    pickle.dump(X_train_pca, file)
    
with open('./results/X_test_pca.pkl', 'wb') as file:
    pickle.dump(X_test_pca, file)

with open('./results/X_incompleted_pca.pkl', 'wb') as file:
    pickle.dump(X_incompleted_pca, file)