# this script is to parse raw data to 
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from datetime import datetime
import re


# to filer out the useless info from each file data 
def extract_data(file_data):
    data_extracted = {}
    try:
        data_extracted['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        data_extracted['nctId'] = np.nan
    
    # extract clinical status info
    try:
        data_extracted['status'] = file_data['protocolSection']['statusModule']['overallStatus']
    except:
        data_extracted['status'] = np.nan
    
    # extract description info
    try:
        data_extracted['description'] = file_data['protocolSection']['descriptionModule']['detailedDescription']
    except:
        try:
            data_extracted['description'] = file_data['protocolSection']['descriptionModule']['briefSummary']
        except:
            data_extracted['description'] = np.nan
    
    # extract diesease info
    try:
        data_extracted['disease'] = file_data['protocolSection']['conditionsModule']['conditions']
    except:
        data_extracted['disease'] = np.nan
    
    # extract clinical phase info
    try:
        data_extracted['phases'] = file_data['protocolSection']['designModule']['phases']
    except:
        data_extracted['phases'] = np.nan
    
    # extract intervention info
    try:
        interventions = file_data['protocolSection']['armsInterventionsModule']['interventions']
        data_extracted['intervention_type'] = [i['type'] for i in interventions]
        data_extracted['intervention_name'] = [i['name'] for i in interventions]
    except:
        data_extracted['intervention_type'] = np.nan
        data_extracted['intervention_name'] = np.nan
        
    # extract outcome info (measure & timeframe)
    try:
        outcomes = file_data["protocolSection"]["outcomesModule"]["primaryOutcomes"]
        data_extracted["outcome_measures"] = [o['measure'] for o in outcomes]
        data_extracted["outcome_timeframes"] = [o['timeFrame'] for o in outcomes]
    except:
        data_extracted["outcome_measures"] = np.nan
        data_extracted["outcome_timeframes"] = np.nan
    
    # extract inclusion and exclusion criteria
    try:
        eligibility = re.split(
            (r"(?i)exclusion\b.\bcriteria\b"),
            file_data["protocolSection"]["eligibilityModule"]["eligibilityCriteria"],
            1,
        )
    except:
        data_extracted["inclusion_criteria"] = np.nan
        data_extracted["exclusion_criteria"] = np.nan

    try:
        data_extracted["inclusion_criteria"] = eligibility[0]
        data_extracted["exclusion_criteria"] = eligibility[1]
    except:
        try:
            data_extracted["inclusion_criteria"] = eligibility[0]
            data_extracted["exclusion_criteria"] = np.nan
        except:
            data_extracted["inclusion_criteria"] = np.nan
            data_extracted["exclusion_criteria"] = np.nan
    
    return data_extracted

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


# transfer files to cleaned data
input_path = 'results'
output_path = './results/keywords_data.csv'

# extract clinical keywords
all_ids = get_all_ids(input_path)
combined_df = []

for study_id in tqdm(all_ids):
    filepath = os.path.join("./data/ctg-studies.json", f"{study_id}.json")

    try:
        with open(filepath, 'r') as file:
            file_data = json.load(file)
            data_extracted = extract_data(file_data)
            combined_df.append(data_extracted)
            
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_file}: {e}")

combined_df = pd.DataFrame(combined_df)

# add clinical duration
duration_df = pd.read_csv('results/trial_duration.csv')
combined_df = pd.merge(combined_df, duration_df, how="inner", on="nctId")

# save dataset
combined_df[['nctId', 'status', 'phases', 'description', 'inclusion_criteria', 
             'exclusion_criteria', 'intervention_type', 'intervention_name',
             'disease', 'outcome_measures', 'outcome_timeframes', 'durationMonths']]\
    .to_csv(output_path, index=False)
