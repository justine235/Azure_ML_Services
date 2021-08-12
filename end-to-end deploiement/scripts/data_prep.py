# code in data_prep.py
from azureml.core import Run
import argparse
import os

#import pandas as pd 
# Get the experiment run context
run = Run.get_context()

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--raw-ds', type=str, dest='raw_data')
parser.add_argument('--output-dir', type=str, dest='output_dir')
args = parser.parse_args()
output_folder = args.output_dir

# Get input dataset as dataframe
print(run.input_datasets['raw_data'])
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

    return data

prepped_df = preprocessing(raw_df)


# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'prepped_data.csv')
prepped_df.to_csv(output_path)
