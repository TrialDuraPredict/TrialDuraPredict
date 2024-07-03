# this script is to parse raw data to 
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import pickle


# to filer out the useless info from each file data 
def filter_data(data):
    data_ = {}
    try:
        data_['nctId'] = data['protocolSection']['identificationModule']['nctId']
    except:
        data_['nctId'] = np.nan
    
    try:
        data_['status'] = data['protocolSection']['statusModule']['overallStatus']
    except:
        data_['status'] = np.nan
        
    try:
        data_['startDate'] = data['protocolSection']['statusModule']['startDateStruct']['date']
    except:
        data_['startDate'] = np.nan
    
    try:
        data_['completionDate'] = data['protocolSection']['statusModule']['completionDateStruct']['date']
    except:
        data_['completionDate'] = np.nan
    
    try:
        data_['description'] = data['protocolSection']['descriptionModule']['detailedDescription']['date']
    except:
        data_['description'] = np.nan
    
    try:
        data_['condition'] = data['protocolSection']['conditionsModule']['condition']
    except:
        data_['condition'] = np.nan
    
    try:
        data_['studyType'] = data['protocolSection']['designModule']['studyType']
    except:
        data_['studyType'] = np.nan
    
    try:
        data_['phases'] = data['protocolSection']['designModule']['phases']
    except:
        data_['phases'] = np.nan
    
    try:
        data_['enrollment'] = data['protocolSection']['designModule']['enrollmentInfo']['count']
    except:
        data_['enrollment'] = np.nan
    
    try:
        data_['allocation'] = data['protocolSection']['designModule']['designInfo']['allocation']
    except:
        data_['allocation'] = np.nan
    
    try:
        data_['interventionModel'] = data['protocolSection']['designModule']['designInfo']['interventionModel']
    except:
        data_['interventionModel'] = np.nan
    
    try:
        data_['primaryPurpose'] = data['protocolSection']['designModule']['designInfo']['primaryPurpose']
    except:
        data_['primaryPurpose'] = np.nan
    
    try:
        data_['interventions'] = data['protocolSection']['armsInterventionsModule']['interventions']
    except:
        data_['interventions'] = np.nan
    
    try:
        data_['primaryOutcomes'] = data['protocolSection']['outcomesModule']['primaryOutcomes']
    except:
        data_['primaryOutcomes'] = np.nan
    
    try:
        data_['eligibilityCriteria'] = data['protocolSection']['eligibilityModule']['eligibilityCriteria']
    except:
        data_['eligibilityCriteria'] = np.nan
    
    try:
        data_['location'] = data['protocolSection']['contactsLocationsModule']['locations']['country']
    except:
        data_['location'] = np.nan
    
    return data_


# read data from json files once per time
def trials_to_clean_data(input_path, output_path):
    json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
    data_cleaned = []
    
    for json_file in tqdm(json_files):
        filepath = os.path.join(input_path, json_file)

        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                # filter data
                data_filtered = filter_data(data)
                data_cleaned.append(data_filtered)

        except FileNotFoundError:
            print(f"Error: The file {json_file} was not found.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {json_file}: {e}")
    
    data_cleaned = pd.DataFrame(data_cleaned)
    data_cleaned.to_csv(output_path)

    
def main():
    input_path = input("Please enter the path to the input ctg-studies.json: ")
    output_path = input("Please enter the path to save the output CSV file: ")
    
    trials_to_clean_data(input_path, output_path)
    
    
if __name__ == '__main__':
    main()