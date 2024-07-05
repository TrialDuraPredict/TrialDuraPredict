# this script is to parse raw data to 
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer, AutoModel
import torch


# collect description info from study file
def collect_description(file_data):
    collected_data = {}
    try:
        collected_data['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        collected_data['nctId'] = np.nan
    
    try:
        collected_data['description'] = file_data['protocolSection']['descriptionModule']['detailedDescription']
    except:
        try:
            collected_data['description'] = file_data['protocolSection']['descriptionModule']['briefSummary']
        except:
            collected_data['description'] = np.nan
    
    return collected_data


# Load BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


# transfer description to embeddings by BioBERT
input_path = './data_example/ctg-studies.json'
output_path = './data_example/description2embedding.pkl'

json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
final_data = []

for json_file in tqdm(json_files[:10]):
    filepath = os.path.join(input_path, json_file)

    try:
        with open(filepath, 'r') as file:
            file_data = json.load(file)
            collected_data = collect_description(file_data)
            description = collected_data['description']

            # Tokenize and encode the description
            inputs = tokenizer(description, return_tensors='pt', padding=True,
                               truncation=True, max_length=512)
            outputs = model(**inputs)
            description_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()

            # update the description to embeddings
            collected_data['description'] = description_embedding
            final_data.append(collected_data)

    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_file}: {e}")


# save data
with open(output_path, 'wb') as file:
    pickle.dump(final_data, file)


