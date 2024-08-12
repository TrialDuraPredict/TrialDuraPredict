# this script is to generate a dataframe with study descriptions for use in the DashBoard
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import re


def extract_description(input_path):
    all_ids = get_all_ids(input_path)

    description_df_list = []

    for study_id in tqdm(all_ids):
        filepath = os.path.join("./data/ctg-studies.json", f"{study_id}.json")

        try:
            with open(filepath, "r") as file:
                file_data = json.load(file)
                description_extracted = {}

                try:
                    description_extracted["nctId"] = file_data["protocolSection"][
                        "identificationModule"
                    ]["nctId"]
                except:
                    description_extracted["nctId"] = np.nan

                try:
                    description_extracted["description"] = file_data["protocolSection"][
                        "descriptionModule"
                    ]["detailedDescription"]
                except:
                    try:
                        description_extracted["description"] = file_data[
                            "protocolSection"
                        ]["descriptionModule"]["briefSummary"]
                    except:
                        description_extracted["description"] = np.nan
                desc_df = pd.DataFrame(
                    {
                        "nctId": [description_extracted["nctId"]],
                        "description": [description_extracted["description"]],
                    }
                )
                description_df_list.append(desc_df)

        except Exception as e:
            print(f"An unexpected error occurred while processing {study_id}: {e}")

    description_df = pd.concat(description_df_list, join="inner").reset_index(drop=True)
    duration = os.path.join(input_path, "trial_duration.csv")
    duration_df = pd.read_csv(duration)

    description_df = pd.merge(description_df, duration_df, how="inner", on="nctId")
    return description_df


# generate the train-test-incompleted ids
def get_all_ids(input_path):
    train_ids_path = os.path.join(input_path, "train_ids.csv")
    test_ids_path = os.path.join(input_path, "test_ids.csv")
    incompleted_ids_path = os.path.join(input_path, "incompleted_ids.csv")

    train_ids = pd.read_csv(train_ids_path).nctId.tolist()
    test_ids = pd.read_csv(test_ids_path).nctId.tolist()
    incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

    all_ids = train_ids + test_ids + incompleted_ids

    return all_ids


def main():
    input_path = "./results"
    output_path = "./results/description_df.csv"

    description_df = extract_description(input_path)
    description_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
