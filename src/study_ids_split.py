import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# Extracts the study ID, status, and study type from the given file data
def extract_study_id(file_data):
    id_extracted = {}
    
    try:
        id_extracted['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        id_extracted['nctId'] = np.nan
    
    try:
        id_extracted['status'] = file_data['protocolSection']['statusModule']['overallStatus']
    except:
        id_extracted['status'] = np.nan
    
    try:
        id_extracted['studyType'] = file_data['protocolSection']['designModule']['studyType']
    except:
        id_extracted['studyType'] = np.nan
    
    return id_extracted


# Determines whether the study should be included in the train-test-incompleted dataset.
def is_includable(id_extracted):
    # parse train-test-incompleted dataset
    status = id_extracted['status']
    studyType = id_extracted['studyType']

    is_valid_status = status in ['NOT_YET_RECRUITING', 'RECRUITING', 'COMPLETED',
                                'ACTIVE_NOT_RECRUITING']
    is_interventional = studyType == 'INTERVENTIONAL'

    return is_valid_status and is_interventional


# generate tain_test_incompleted dataset
# output is a pandas dataframe, with columns: nctID, status, studyType
def generate_train_test_incompleted_ids(input_path):
    train_test_incompleted_ids = []
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]

    for json_file in tqdm(json_files):
        filepath = os.path.join(input_path, json_file)

        with open(filepath, 'r') as file:
            file_data = json.load(file)
            id_extracted= extract_study_id(file_data)

            if is_includable(id_extracted):
                train_test_incompleted_ids.append(id_extracted)

    return pd.DataFrame(train_test_incompleted_ids)


# split datasets
def split_and_save_datasets(train_test_incompleted_ids, output_dir):
    incompleted_ids = train_test_incompleted_ids[train_test_incompleted_ids.status != 'COMPLETED']
    completed_ids = train_test_incompleted_ids[train_test_incompleted_ids.status == 'COMPLETED']

    train_ids, test_ids = train_test_split(completed_ids, test_size=0.1, random_state=21)

    train_ids.to_csv(os.path.join(output_dir, 'train_ids.csv'), index=False)
    test_ids.to_csv(os.path.join(output_dir, 'test_ids.csv'), index=False)
    incompleted_ids.to_csv(os.path.join(output_dir, 'incompleted_ids.csv'), index=False)


def main():
    input_path = './data_example/ctg-studies.json'
    output_dir = './data_example/'

    train_test_incompleted_ids = generate_train_test_incompleted_ids(input_path)
    split_and_save_datasets(train_test_incompleted_ids, output_dir)


if __name__ == '__main__':
    main()