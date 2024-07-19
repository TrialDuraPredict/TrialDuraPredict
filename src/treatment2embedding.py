# this script is to parse raw data to
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import re


# Extracts the study ID and treatment type from the given file data.
def extract_treatment(file_data):
    treatment_extracted = {}

    try:
        treatment_extracted["nctId"] = file_data["protocolSection"][
            "identificationModule"
        ]["nctId"]
    except:
        treatment_extracted["nctId"] = np.nan

    # Treatment Type
    interventions = file_data["protocolSection"]["armsInterventionsModule"][
        "interventions"
    ]
    try:
        for t in interventions:
            try:
                treatment_extracted["treatment_type"].append(t["type"])
            except:
                treatment_extracted["treatment_type"] = [t["type"]]

        if type(treatment_extracted["treatment_type"]) == list:
            treatment_extracted["treatment_type"] = ", ".join(
                treatment_extracted["treatment_type"]
            )
    except:
        treatment_extracted["treatment_type"] = np.nan

    return treatment_extracted


# Generates embeddings for treatment type using ***.
def treatment2embedding(input_path):
    embedding_data = []

    # Load BioBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # generate the train-test-incompleted ids
    train_ids_path = os.path.join(input_path, "train_ids.csv")
    test_ids_path = os.path.join(input_path, "test_ids.csv")
    incompleted_ids_path = os.path.join(input_path, "incompleted_ids.csv")

    train_ids = pd.read_csv(train_ids_path).nctId.tolist()
    test_ids = pd.read_csv(test_ids_path).nctId.tolist()
    incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

    all_ids = train_ids + test_ids + incompleted_ids

    # generate embedding
    for study_id in tqdm(all_ids):
        filepath = os.path.join(input_path, "ctg-studies.json", f"{study_id}.json")

        try:
            with open(filepath, "r") as file:
                file_data = json.load(file)
                treatment_extracted = extract_treatment(file_data)
                nctId = treatment_extracted["nctId"]
                treatment = treatment_extracted["treatment_type"]

                if pd.notna(treatment):
                    # Tokenize and encode the eligibility criteria
                    inputs = tokenizer(
                        treatment,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    outputs = model(**inputs)
                    treatment_embedding = (
                        outputs.last_hidden_state[:, 0, :].detach().numpy()
                    )

                    # append nctId and embedding data
                    embedding_data.append(
                        {
                            "nctId": nctId,
                            "treatment_embedding": treatment_embedding,
                        }
                    )
                else:
                    print(f"No valid description found for {study_id}")

        except Exception as e:
            print(f"An unexpected error occurred while processing {study_id}: {e}")

    return embedding_data


def main():
    input_path = "./data_example"
    output_path = "./data_example/treatment2embedding.pkl"

    embedding_data = treatment2embedding(input_path)
    with open(output_path, "wb") as file:
        pickle.dump(embedding_data, file)


if __name__ == "__main__":
    main()
