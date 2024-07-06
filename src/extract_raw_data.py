# this script is to parse raw data to 
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from datetime import datetime


# to filer out the useless info from each file data 
def extract_data(file_data):
    data_extracted = {}
    try:
        data_extracted['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        data_extracted['nctId'] = np.nan
    
    try:
        data_extracted['status'] = file_data['protocolSection']['statusModule']['overallStatus']
    except:
        data_extracted['status'] = np.nan
        
    try:
        data_extracted['startDate'] = file_data['protocolSection']['statusModule']['startDateStruct']['date']
    
    except:
        data_extracted['startDate'] = np.nan
    
    try:
        data_extracted['completionDate'] = file_data['protocolSection']['statusModule']['completionDateStruct']['date']
    except:
        data_extracted['completionDate'] = np.nan
    
    try:
        data_extracted['description'] = file_data['protocolSection']['descriptionModule']['detailedDescription']
    except:
        try:
            data_extracted['description'] = file_data['protocolSection']['descriptionModule']['briefSummary']
        except:
            data_extracted['description'] = np.nan
    
    try:
        data_extracted['condition'] = file_data['protocolSection']['conditionsModule']['conditions']
    except:
        data_extracted['condition'] = np.nan
    
    try:
        data_extracted['studyType'] = file_data['protocolSection']['designModule']['studyType']
    except:
        data_extracted['studyType'] = np.nan
    
    try:
        data_extracted['phases'] = file_data['protocolSection']['designModule']['phases']
    except:
        data_extracted['phases'] = np.nan
    
    try:
        data_extracted['enrollment'] = file_data['protocolSection']['designModule']['enrollmentInfo']['count']
    except:
        data_extracted['enrollment'] = np.nan
    
    try:
        data_extracted['primaryPurpose'] = file_data['protocolSection']['designModule']['designInfo']['primaryPurpose']
    except:
        data_extracted['primaryPurpose'] = np.nan
    
    try:
        data_extracted['interventions'] = file_data['protocolSection']['armsInterventionsModule']['interventions']
    except:
        data_extracted['interventions'] = np.nan
    
    try:
        data_extracted['primaryOutcomes'] = file_data['protocolSection']['outcomesModule']['primaryOutcomes']
    except:
        data_extracted['primaryOutcomes'] = np.nan
    
    try:
        data_extracted['eligibilityCriteria'] = file_data['protocolSection']['eligibilityModule']['eligibilityCriteria']
    except:
        data_extracted['eligibilityCriteria'] = np.nan
    
    return data_extracted


# transfer files to cleaned data
input_path = './data_example/ctg-studies.json'
output_path = './data_example/raw_data_extracted.csv'

json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
final_data = []

for json_file in tqdm(json_files):
    filepath = os.path.join(input_path, json_file)

    try:
        with open(filepath, 'r') as file:
            file_data = json.load(file)
            data_extracted = extract_data(file_data)
            final_data.append(data_extracted)

    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_file}: {e}")

final_data = pd.DataFrame(final_data)
final_data.to_csv(output_path, index=False)

