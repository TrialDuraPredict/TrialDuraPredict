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

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pkl"):
            if file_name == "embeddings_df.pkl":
                continue

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


def main():
    directory_path = "./data_example"
    output_path = "./data_example/embeddings_df.pkl"

    embedding_data = add_duration(directory_path)
    with open(output_path, "wb") as file:
        pickle.dump(embedding_data, file)


if __name__ == "__main__":
    main()
