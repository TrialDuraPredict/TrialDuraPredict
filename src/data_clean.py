# this script is to parse raw data to 
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from datetime import datetime


# to filer out the useless info from each file data 
def collect_data(file_data):
    data_collected = {}
    try:
        data_collected['nctId'] = file_data['protocolSection']['identificationModule']['nctId']
    except:
        data_collected['nctId'] = np.nan
    
    try:
        data_collected['status'] = file_data['protocolSection']['statusModule']['overallStatus']
    except:
        data_collected['status'] = np.nan
        
    try:
        data_collected['startDate'] = file_data['protocolSection']['statusModule']['startDateStruct']['date']
    
    except:
        data_collected['startDate'] = np.nan
    
    try:
        data_collected['completionDate'] = file_data['protocolSection']['statusModule']['completionDateStruct']['date']
    except:
        data_collected['completionDate'] = np.nan
    
    try:
        data_collected['description'] = file_data['protocolSection']['descriptionModule']['detailedDescription']
    except:
        try:
            data_collected['description'] = file_data['protocolSection']['descriptionModule']['briefSummary']
        except:
            data_collected['description'] = np.nan
    
    try:
        data_collected['condition'] = file_data['protocolSection']['conditionsModule']['conditions']
    except:
        data_collected['condition'] = np.nan
    
    try:
        data_collected['studyType'] = file_data['protocolSection']['designModule']['studyType']
    except:
        data_collected['studyType'] = np.nan
    
    try:
        data_collected['phases'] = file_data['protocolSection']['designModule']['phases']
    except:
        data_collected['phases'] = np.nan
    
    try:
        data_collected['enrollment'] = file_data['protocolSection']['designModule']['enrollmentInfo']['count']
    except:
        data_collected['enrollment'] = np.nan
    
    try:
        data_collected['primaryPurpose'] = file_data['protocolSection']['designModule']['designInfo']['primaryPurpose']
    except:
        data_collected['primaryPurpose'] = np.nan
    
    try:
        data_collected['interventions'] = file_data['protocolSection']['armsInterventionsModule']['interventions']
    except:
        data_collected['interventions'] = np.nan
    
    try:
        data_collected['primaryOutcomes'] = file_data['protocolSection']['outcomesModule']['primaryOutcomes']
    except:
        data_collected['primaryOutcomes'] = np.nan
    
    try:
        data_collected['eligibilityCriteria'] = file_data['protocolSection']['eligibilityModule']['eligibilityCriteria']
    except:
        data_collected['eligibilityCriteria'] = np.nan
    
    return data_collected


def transform_date(input_date):
    try:
        input_date = datetime.strptime(input_date, '%Y-%m-%d')
    except:
        try:
            input_date = datetime.strptime(input_date, '%Y-%m')
        except:
            input_date = np.nan
    
    return input_date


def calculate_duration(start_date, completion_date):
    start_date = transform_date(start_date)
    completion_date = transform_date(completion_date)

    # duration calculated as months
    try:
        durationMonth = (completion_date - start_date).days//30
    except:
        durationMonth = np.nan

    return durationMonth


def update_data(data_collected):
    # calculate the trial duration
    data_collected['durationMonth'] = \
        calculate_duration(data_collected['startDate'], data_collected['completionDate'])
    
    # get trial start year
    try:
        data_collected['startYear'] = \
            transform_date(data_collected['startDate']).year
    except:
        data_collected['startYear'] = np.nan
    
    return data_collected


# transfer files to cleaned data
input_path = './data_example/ctg-studies.json'
output_path = './data_example/data_cleaned.csv'

json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
final_data = []

for json_file in tqdm(json_files):
    filepath = os.path.join(input_path, json_file)

    try:
        with open(filepath, 'r') as file:
            file_data = json.load(file)
            data_collected = collect_data(file_data)
            data_updated = update_data(data_collected)
            
            final_data.append(data_updated)

            # filter data to be appended
            # cond_status = data_updated['status'] == 'COMPLETED'
            # cond_studyType = data_updated['studyType'] == 'INTERVENTIONAL'

            #if cond_status and cond_studyType:
            #    final_data.append(data_updated)

    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_file}: {e}")

final_data = pd.DataFrame(final_data)
final_data.to_csv(output_path, index=False)

