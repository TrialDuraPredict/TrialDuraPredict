

# to filer out the useless info from each file data 
def collect_IDs(file_data):
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
        data_collected['studyType'] = file_data['protocolSection']['designModule']['studyType']
    except:
        data_collected['studyType'] = np.nan
    
    return data_collected


# split the study ID as tran, test and incompleted groups
input_path = './data_example/ctg-studies.json'
output_train_path = './data_example/train.pkl'
output_test_path = './data_example/test.pkl'
output_incompleted_path = './data_example/incompleted.pkl'

json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
train_data = []
test_data = []
incompleted_data = []

for json_file in tqdm(json_files):
    filepath = os.path.join(input_path, json_file)

    try:
        with open(filepath, 'r') as file:
            file_data = json.load(file)
            collected_data = collect_IDs(file_data)
            
            # 
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