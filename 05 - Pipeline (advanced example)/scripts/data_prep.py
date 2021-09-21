# code in data_prep.py
from azureml.core import Run
import argparse
import os
from azureml.core import Dataset
from azureml.core import Workspace, Dataset
import azureml.dataprep
import pandas as pd



# Get the experiment run context
run = Run.get_context()

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--raw-ds', type=str, dest='raw_data')
parser.add_argument('--dataset', dest='dataset', required=True)
parser.add_argument('--datadir', dest='datadir', required=True)
args = parser.parse_args()

# Get input dataset as dataframe
raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()

# data cleaning function
def preprocessing(file_name):
    data = file_name
    # importation 
    #data.drop(columns=['Unnamed: 0'], inplace=True)

    # missing values traitement
    data['Training_Completed'] = data['Training_Completed'].interpolate(method='linear', direction = 'forward').round(0)
    data['Social_Media'] = data['Social_Media'].interpolate(method='linear', direction = 'forward').round(0)

    #  Code postal
    data['Dpt'] = data['Code_postal'].astype('str').str[0:2]
    del data['Code_postal']

    #  Grouping 
    data[data['usageMetric2'] > 3]['usageMetric2'] <- 4
    data[data['behaviorPattern2'] > 3]['behaviorPattern2'] <- 4
    data[data['Access_Level'] > 6]['Access_Level'] <- 6
    data[data['Social_Media'] > 2]['Social_Media'] <- 3
    data[data['behaviorPattern2'] > 2]['behaviorPattern2'] <- 3


    # Email grouping & one hot encoding
    dic_email = {"ehow.com": "others", "google.co.uk":"others","nsw.gov.au":"others"}
    data['Email_Domain'].replace(dic_email)
    encoded_columns = pd.get_dummies(data['Email_Domain'])
    data = data.join(encoded_columns)
    del data['Email_Domain']
    
    return data

prepped_df = preprocessing(raw_df)


# save it to transfer the data to the second pipeline
p = args.datadir
os.makedirs(p, exist_ok=True)
prepped_df.to_csv(os.path.join(p,args.dataset), index=False)
