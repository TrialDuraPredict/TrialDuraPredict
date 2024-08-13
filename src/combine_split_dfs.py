# this script is to combine all of the outputs of the data cleaning scripts into one dataframe
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler


# combine all embeddings into one dataframe
# 9 cols in total
def combine_pickle_files(directory_path):
    df_list = []
    file_names = ['description2embedding.pkl',
                  'eligibility2embedding.pkl',
                  'treatment2embedding.pkl',
                  'disease2embedding.pkl',
                  'outcome2embedding.pkl']

    for file_name in file_names:
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "rb") as f:
            p = pd.DataFrame(pickle.load(f))\
                .set_index('nctId')
            df_list.append(p)

    combined_df = pd.concat(df_list, axis=1, join="inner")
    combined_df.reset_index(inplace=True)

    return combined_df


# separate each embedding group into multiple columns
# 5378 cols = 2 + 768 * 7
def separate_cols(combined_df):
    separated_df = pd.concat([combined_df[['nctId', 'enrollment']],
            combined_df['description_embedding'].apply(pd.Series),
            combined_df['inclusion_embedding'].apply(pd.Series),
            combined_df['exclusion_embedding'].apply(pd.Series),
            combined_df['treatment_embedding'].apply(pd.Series),
            combined_df['disease_embedding'].apply(pd.Series),
            combined_df['measures_embedding'].apply(pd.Series),
            combined_df['timeframes_embedding'].apply(pd.Series)], axis=1)
    
    return separated_df


# add duration time into the dataset
# 5379 cols
def add_duration(separated_df):
    duration_df = pd.read_csv('./results/trial_duration.csv')
    durAdded_df = pd.merge(separated_df, duration_df, how="inner", on="nctId")

    return durAdded_df


# split dataset to train, test and incompleted
# X dataset has cols of 5377
def train_test_incomplete_split(durAdded_df, directory_path):
    train_ids_path = os.path.join(directory_path, "train_ids.csv")
    test_ids_path = os.path.join(directory_path, "test_ids.csv")
    incompleted_ids_path = os.path.join(directory_path, "incompleted_ids.csv")

    train_ids = pd.read_csv(train_ids_path).nctId.tolist()
    test_ids = pd.read_csv(test_ids_path).nctId.tolist()
    incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

    train_df = durAdded_df[durAdded_df.nctId.isin(train_ids)]
    test_df = durAdded_df[durAdded_df.nctId.isin(test_ids)]
    incompleted_df = durAdded_df[durAdded_df.nctId.isin(incompleted_ids)]
    
    X_train = train_df.iloc[:, 1:-1]
    X_test = test_df.iloc[:, 1:-1]
    X_incompleted = incompleted_df.iloc[:, 1:-1]
    y_train = train_df['durationMonths']
    y_test = test_df['durationMonths']
    y_incompleted = incompleted_df['durationMonths']

    return X_train, y_train, X_test, y_test, X_incompleted, y_incompleted


# scale enrollment in X_test and X_incompleted based on X_train
def standardize_X(X_train, X_test, X_incompleted):
    scaler = StandardScaler()
    X_train_enroll_scaled = scaler.fit_transform(X_train[['enrollment']])
    X_train_scaled = np.concatenate([X_train_enroll_scaled, X_train.iloc[:, 1:].values], axis=1)

    X_test_enroll_scaled = scaler.transform(X_test[['enrollment']])
    X_test_scaled = np.concatenate([X_test_enroll_scaled, X_test.iloc[:, 1:].values], axis=1)

    X_incompleted_enroll_scaled = scaler.transform(X_incompleted[['enrollment']])
    X_incompleted_scaled = np.concatenate([X_incompleted_enroll_scaled, X_incompleted.iloc[:, 1:].values], axis=1)
    
    return X_train_scaled, X_test_scaled, X_incompleted_scaled


def main():
    directory_path = "./results"

    combined_df = combine_pickle_files(directory_path)
    separated_df = separate_cols(combined_df)
    durAdded_df = add_duration(separated_df)
    durAdded_df.dropna(inplace=True)
    
    X_train, y_train, X_test, y_test, X_incompleted, y_incompleted = \
        train_test_incomplete_split(durAdded_df, directory_path)
    
    X_train_scaled, X_test_scaled, X_incompleted_scaled = \
        standardize_X(X_train, X_test, X_incompleted)
        
    with open(os.path.join(directory_path, 'X_train.pkl'), 'wb') as file:
        pickle.dump(X_train_scaled, file)
    
    with open(os.path.join(directory_path, 'y_train.pkl'), 'wb') as file:
        pickle.dump(y_train, file)

    with open(os.path.join(directory_path, 'X_test.pkl'), 'wb') as file:
        pickle.dump(X_test_scaled, file)
        
    with open(os.path.join(directory_path, 'y_test.pkl'), 'wb') as file:
        pickle.dump(y_test, file)

    with open(os.path.join(directory_path, 'X_incompleted.pkl'), 'wb') as file:
        pickle.dump(X_incompleted_scaled, file)
        
    with open(os.path.join(directory_path, 'y_incompleted.pkl'), 'wb') as file:
        pickle.dump(y_incompleted, file)


if __name__ == "__main__":
    main()
