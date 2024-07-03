# data were downloaded as separate json files (each json file represents single study)
# on 2-July from https://clinicaltrials.gov/ 
# raw dataset include 500,046 studies, and example dataset includes 1,000 studies

import json
import os
from tqdm import tqdm
import pickle

directory_path = './example_data/ctg-studies.json'
json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]

# read data from json files once per time
df = []
for json_file in tqdm(json_files):
    filepath = os.path.join(directory_path, json_file)
    
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            # collect useful info from each study, to be updated
            data_ = data['protocolSection']['statusModule']['overallStatus']
            df.append(data_)

    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {json_file}: {e}")


# write parsed data to file
with open('example_data_clean.pkl', 'wb') as file:
    pickle.dump(df, file)



