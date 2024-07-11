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


# Extracts the study ID and eligibility criteria from the given file data.
def extract_eligibility(file_data):
    eligibility_extracted = {}

    try:
        eligibility_extracted["nctId"] = file_data["protocolSection"][
            "identificationModule"
        ]["nctId"]
    except:
        eligibility_extracted["nctId"] = np.nan

    # Eligibility Criteria, splitting inclusion and exclusion criteria
    try:
        eligibility = re.split(
            (r"(?i)exclusion\b.\bcriteria\b"),
            file_data["protocolSection"]["eligibilityModule"]["eligibilityCriteria"],
        )
    except:
        eligibility_extracted["inclusion_criteria"] = np.nan
        eligibility_extracted["exclusion_criteria"] = np.nan

    try:
        eligibility_extracted["inclusion_criteria"] = eligibility[0]
        eligibility_extracted["exclusion_criteria"] = eligibility[1]
    except:
        try:
            eligibility_extracted["inclusion_criteria"] = eligibility[0]
            eligibility_extracted["exclusion_criteria"] = np.nan
        except:
            eligibility_extracted["inclusion_criteria"] = np.nan
            eligibility_extracted["exclusion_criteria"] = np.nan

    return eligibility_extracted


# Generates embeddings for elgibility criteria using BioBERT.
def eligibility2embedding(input_path):
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
                eligibility_extracted = extract_eligibility(file_data)
                nctId = eligibility_extracted["nctId"]
                inclusion = eligibility_extracted["inclusion_criteria"]
                exclusion = eligibility_extracted["exclusion_criteria"]

                if pd.notna(inclusion):
                    # Tokenize and encode the eligibility criteria
                    inputs = tokenizer(
                        inclusion,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    outputs = model(**inputs)
                    inclusion_embedding = (
                        outputs.last_hidden_state[:, 0, :].detach().numpy()
                    )

                    inputs = tokenizer(
                        exclusion,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    )
                    outputs = model(**inputs)
                    exclusion_embedding = (
                        outputs.last_hidden_state[:, 0, :].detach().numpy()
                    )

                    # append nctId and embedding data
                    embedding_data.append(
                        {
                            "nctId": nctId,
                            "inclusion_embedding": inclusion_embedding,
                            "exclusion_embedding": exclusion_embedding,
                        }
                    )
                else:
                    print(f"No valid description found for {study_id}")

        except Exception as e:
            print(f"An unexpected error occurred while processing {study_id}: {e}")

    return embedding_data


def main():
    input_path = "./data_example"
    output_path = "./data_example/eligibility2embedding.pkl"

    embedding_data = eligibility2embedding(input_path)
    with open(output_path, "wb") as file:
        pickle.dump(embedding_data, file)


if __name__ == "__main__":
    main()
