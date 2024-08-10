# this script is for disease embedding
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Extracts the study ID and disease from the given file data.
def extract_disease(file_data):
    disease_extracted = {}

    try:
        disease_extracted["nctId"] = file_data["protocolSection"][
            "identificationModule"
        ]["nctId"]
    except:
        disease_extracted["nctId"] = np.nan

    try:
        disease_extracted["disease"] = file_data["protocolSection"][
            "conditionsModule"
        ]["conditions"]
    except:
        disease_extracted["description"] = np.nan

    return disease_extracted


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


# Generate embeddings for a list of text entries.
def generate_embeddings(tokenizer, model, texts):
    embeddings = np.zeros((len(texts), 768))
    for i, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        outputs = model(**inputs)
        embeddings[i] = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings.mean(axis=0)


# Generates embeddings for study descriptions using BioBERT.
def disease2embedding(input_path):
    embedding_data = []
    all_ids = get_all_ids(input_path)

    # Load BioBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # generate embedding
    for study_id in tqdm(all_ids):
        filepath = os.path.join("./data/ctg-studies.json", f"{study_id}.json")

        try:
            with open(filepath, "r") as file:
                file_data = json.load(file)
                disease_extracted = extract_disease(file_data)
                nctId = disease_extracted["nctId"]
                diseases = disease_extracted["disease"]

                if isinstance(diseases, list):
                    disease_embedding = generate_embeddings(
                        tokenizer,
                        model,
                        diseases
                    )
                    
                    embedding_data.append(
                        {
                            "nctId": nctId,
                            "disease_embedding": disease_embedding,
                        }
                    )
                else:
                    print(f"No valid disease found for {study_id}")

        except Exception as e:
            print(f"An unexpected error occurred while processing {study_id}: {e}")

    return embedding_data


def main():
    input_path = "./results"
    output_path = "./results/disease2embedding.pkl"

    embedding_data = disease2embedding(input_path)
    with open(output_path, "wb") as file:
        pickle.dump(embedding_data, file)


if __name__ == "__main__":
    main()
