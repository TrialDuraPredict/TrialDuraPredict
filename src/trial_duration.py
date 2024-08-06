import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from datetime import datetime


# Extracts the study ID, start date and completion date from the given file data
def extract_study_dates(file_data):
    date_extracted = {}
    
    try:
        date_extracted['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        date_extracted['nctId'] = np.nan
    
    try:
        date_extracted['startDate'] = file_data['protocolSection']['statusModule']['startDateStruct']['date']
    
    except:
        date_extracted['startDate'] = np.nan
    
    try:
        date_extracted['completionDate'] = file_data['protocolSection']['statusModule']['primaryCompletionDateStruct']['date']
    except:
        date_extracted['completionDate'] = np.nan
    
    return date_extracted


# Transforms a date string to a datetime object
def transform_date(input_date):
    try:
        input_date = datetime.strptime(input_date, '%Y-%m-%d')
    except:
        try:
            input_date = datetime.strptime(input_date, '%Y-%m')
        except:
            input_date = np.nan
    
    return input_date


# Calculates the duration between two dates in months.
def calculate_duration(start_date, completion_date):
    start_date = transform_date(start_date)
    completion_date = transform_date(completion_date)

    # duration calculated as months
    try:
        durationMonth = (completion_date - start_date).days / 30
    except:
        durationMonth = np.nan

    return durationMonth


def generate_duration_dataset(input_path):
    duration_data = []

    # generat the train-test-incompleted ids
    train_ids_path = os.path.join(input_path, 'train_ids.csv')
    test_ids_path = os.path.join(input_path, 'test_ids.csv')
    incompleted_ids_path = os.path.join(input_path, 'incompleted_ids.csv')

    train_ids = pd.read_csv(train_ids_path).nctId.tolist()
    test_ids = pd.read_csv(test_ids_path).nctId.tolist()
    incompleted_ids = pd.read_csv(incompleted_ids_path).nctId.tolist()

    all_ids = train_ids + test_ids + incompleted_ids

    # generate duration
    for study_id in tqdm(all_ids):
        filepath = os.path.join(input_path, 'ctg-studies.json', f'{study_id}.json')

        with open(filepath, 'r') as file:
            file_data = json.load(file)
            date_extracted= extract_study_dates(file_data)

            start_date = date_extracted['startDate']
            completion_date = date_extracted['completionDate']
            date_extracted['durationMonths'] = calculate_duration(start_date, completion_date)

            del date_extracted['startDate']
            del date_extracted['completionDate']

            duration_data.append(date_extracted)

    return pd.DataFrame(duration_data)


def main():
    input_path = './data_example/'
    output_path = './data_example/trial_duration.csv'

    duration = generate_duration_dataset(input_path)
    duration.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()