# Generate dataframe for reference when creating dashboard for key search words
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import re

input_path = "./data_example"

train_ids_path = os.path.join(input_path, "train_ids.csv")
test_ids_path = os.path.join(input_path, "test_ids.csv")
incompleted_ids_path = os.path.join(input_path, "incompleted_ids.csv")

train_ids = pd.read_csv(train_ids_path).nctId.tolist()
test_ids = pd.read_csv(test_ids_path).nctId.tolist()
incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

all_ids = train_ids + test_ids + incompleted_ids

df_list = []

for study_id in tqdm(all_ids):
    filepath = os.path.join(input_path, "ctg-studies.json", f"{study_id}.json")

    try:
        with open(filepath, "r") as file:
            file_data = json.load(file)

            nctID = file_data["protocolSection"]["identificationModule"]["nctId"]
            status = file_data["protocolSection"]["statusModule"]["overallStatus"]
            dis = file_data["protocolSection"]["conditionsModule"]["conditions"]
            inter_type = file_data["protocolSection"]["armsInterventionsModule"][
                "interventions"
            ][0]["type"]
            inter_name = file_data["protocolSection"]["armsInterventionsModule"][
                "interventions"
            ][0]["name"]

            # important features extracted from SHAP for potential extra keywords to enhance searching
            # feats =

            file_data = pd.DataFrame(
                {
                    "nctID": nctID,
                    "status": status,
                    "disease": dis,
                    "intervention_type": inter_type,
                    "intervention_name": inter_name,
                    #   'features': feats
                }
            )
            df_list.append(file_data)
            # print(file_data)
    except Exception as e:
        print(f"An unexpected error occurred while processing {study_id}: {e}")

combined_df = pd.concat(df_list, join="inner")
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

output_path = "./data_example/keyword_df.csv"
combined_df.to_csv(output_path, index=False)
