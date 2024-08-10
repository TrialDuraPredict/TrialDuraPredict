# this script is to combine all of the outputs of the data cleaning scripts into one dataframe
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import re


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
            p = pd.DataFrame(pickle.load(f))
            df_list.append(p)

    combined_df = pd.concat(df_list, axis=1, join="inner")
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    return combined_df


def add_duration(directory_path):
    duration = os.path.join(directory_path, "trial_duration.csv")
    duration_df = pd.read_csv(duration)

    combined_df = combine_pickle_files(directory_path)
    combined_df = pd.merge(combined_df, duration_df, how="inner", on="nctId")

    return combined_df


def train_test_incomplete_split(combined_df, directory_path):
    train_ids_path = os.path.join(directory_path, "train_ids.csv")
    test_ids_path = os.path.join(directory_path, "test_ids.csv")
    incompleted_ids_path = os.path.join(directory_path, "incompleted_ids.csv")

    train_ids = pd.read_csv(train_ids_path).nctId.tolist()
    test_ids = pd.read_csv(test_ids_path).nctId.tolist()
    incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

    train_df = combined_df[combined_df.nctId.isin(train_ids)]
    test_df = combined_df[combined_df.nctId.isin(test_ids)]
    incompleted_df = combined_df[combined_df.nctId.isin(incompleted_ids)]

    return train_df, test_df, incompleted_df


def main():
    directory_path = "./data_example"

    combined_df = add_duration(directory_path)
    train_df, test_df, incompleted_df = train_test_incomplete_split(combined_df, directory_path)

    with open(os.path.join(directory_path, 'train_df.pkl'), 'wb') as file:
        pickle.dump(train_df, file)

    with open(os.path.join(directory_path, 'test_df.pkl'), 'wb') as file:
        pickle.dump(test_df, file)

    with open(os.path.join(directory_path, 'incompleted_df.pkl'), 'wb') as file:
        pickle.dump(incompleted_df, file)


if __name__ == "__main__":
    main()
