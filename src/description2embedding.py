# this script is to parse raw data to 
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch


# Extracts the study ID and description from the given file data.
def extract_description(file_data):
    description_extracted = {}
    
    try:
        description_extracted['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        description_extracted['nctId'] = np.nan
    
    try:
        description_extracted['description'] = file_data['protocolSection']['descriptionModule']['detailedDescription']
    except:
        try:
            description_extracted['description'] = file_data['protocolSection']['descriptionModule']['briefSummary']
        except:
            description_extracted['description'] = np.nan
    
    return description_extracted


# Generates embeddings for study descriptions using BioBERT.
def description2embedding(input_path):
    embedding_data = []

    # Load BioBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # generat the train-test-incompleted ids
    train_ids_path = os.path.join(input_path, 'train_ids.csv')
    test_ids_path = os.path.join(input_path, 'test_ids.csv')
    incompleted_ids_path = os.path.join(input_path, 'incompleted_ids.csv')

    train_ids = pd.read_csv(train_ids_path).nctId.tolist()
    test_ids = pd.read_csv(test_ids_path).nctId.tolist()
    incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

    all_ids = train_ids + test_ids + incompleted_ids

    # generate embedding
    for study_id in tqdm(all_ids):
        filepath = os.path.join(input_path, 'ctg-studies.json', f'{study_id}.json')

        try:
            with open(filepath, 'r') as file:
                file_data = json.load(file)
                description_extracted = extract_description(file_data)
                nctId = description_extracted['nctId']
                description = description_extracted['description']

                if pd.notna(description):
                    # Tokenize and encode the description
                    inputs = tokenizer(description, return_tensors='pt', padding=True,
                                    truncation=True, max_length=512)
                    outputs = model(**inputs)
                    description_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

                    # append nctId and embedding data
                    embedding_data.append({'nctId': nctId,
                                           'description_embedding': description_embedding})
                else:
                    print(f'No valid description found for {study_id}')

        except Exception as e:
            print(f"An unexpected error occurred while processing {study_id}: {e}")
    
    return embedding_data


def main():
    input_path = './data_example'
    output_path = './data_example/description2embedding.pkl'

    embedding_data = description2embedding(input_path)
    with open(output_path, 'wb') as file:
        pickle.dump(embedding_data, file)


if __name__ == '__main__':
    main()


