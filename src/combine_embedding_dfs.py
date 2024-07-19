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
    combined_df = pd.DataFrame()

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "rb") as f:
                content = pickle.load(f)
                if isinstance(content, pd.DataFrame):
                    combined_df = pd.concat([combined_df, content], ignore_index=True)
    return combined_df


def main():
    directory_path = "./data_example"
    output_path = "./data_example/embeddings_df.pkl"

    embedding_data = combine_pickle_files(directory_path)
    with open(output_path, "wb") as file:
        pickle.dump(embedding_data, file)


if __name__ == "__main__":
    main()
