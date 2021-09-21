import argparse
import json
import os
import azureml.core
from azureml.core import Workspace, Experiment, Model
from azureml.core import Run, run
from shutil import copy2
from azureml.core.model import Model
import pandas as pd
from sklearn import metrics
from joblib import dump, load
import pickle
from azureml.core.dataset import Dataset
from azureml.core.run import get_run
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id
from interpret_community import TabularExplainer
from azureml.interpret import ExplanationClient
import joblib
from azureml.core.datastore import Datastore

summaryid = [0]
summarymetric = [0]

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_datastore_path', dest='data_datastore_path')
parser.add_argument('--model_path', dest='model_path')
parser.add_argument('--pipeline_version', type=str, default='v0_1',help='Pipeline or Code Version.')
parser.add_argument('--run_version',default='v0_1', type=str,help='Pipeline Execusion Version')
parser.add_argument('--aml_artifact_store_name', type=str, default="artifactstore_dev",help='The name of the datastore where pipeline artifacts are stored')
parser.add_argument('--raw_dataset_version', type=str, default='v0_1', help='Version of raw data (for tags).')


args = parser.parse_args()



# -------------------------- GET MODEL -------------------------------#

run = Run.get_context()
ws = run.experiment.workspace

# retrieve best child
pipeline_runid = run.get_details()['properties']['azureml.pipelinerunid']
print('pipelineid', pipeline_runid)
pipeline_run = ws.get_run(pipeline_runid)

for child_run in pipeline_run.get_children():
    listid = child_run.id
    print('id parent', listid)
    for  child_run2 in child_run.get_children():
        print('id pipeline step',child_run2)
        for  child_run3 in child_run2.get_children():
            listid = child_run3.id
            summaryid.append(listid)
            list_metric = child_run3.get_metrics()['Test Precison']
            summarymetric.append(list_metric)

maxi = summarymetric.index(max(summarymetric))
maxi_id = summaryid[maxi]

RAW_DATASET_VERSION = "v01"
BEST_CHILD_RUN_ID = maxi_id
RUN_VERSION = "v01"
PIPELINE_VERSION = "v01"

print('start register model')
for child_run3 in child_run2.get_children():
    if child_run3.id == maxi_id:
        registered_model = child_run3.register_model(model_name='rf',
                                                     model_path=os.path.join('./outputs/model/saved_model.pkl'),
                                                      tags={'raw data version': RAW_DATASET_VERSION,
                                                            'pipeline version': PIPELINE_VERSION,
                                                            'run version': RUN_VERSION,
                                                            'best hp trial version': BEST_CHILD_RUN_ID} )
        MODEL_DIR = args.model_path
        print('all',os.listdir(MODEL_DIR))
        MODEL_CHILD_DIR = os.path.join(MODEL_DIR)
        MODEL_NAME = maxi_id + 'model.joblib'
        print('the best model is : ', MODEL_NAME)
        MODEL_PATH = os.path.join(MODEL_CHILD_DIR, MODEL_NAME)

        # Load model 
        model = joblib.load(MODEL_PATH)
        print("ok it's saved")



# -------------------------- EVALUATION -------------------------------#
# target column
TARGET_COL = "EmployeeTargeted"
# Critical columns for fairness
FAIRNESS_COLS = ['Gender', 'Dpt']


data_folder = args.data_datastore_path
print('data', data_folder)
print(os.listdir(data_folder))
train = pd.read_csv(os.path.join(data_folder, 'data_eval_output.csv'))
test = pd.read_csv(os.path.join(data_folder, 'data_train_output.csv'))
print(test.head(5))

X_train = train.drop(TARGET_COL, axis=1)
del X_train['Unnamed: 0']
y_train = train[[TARGET_COL]]

X_test = test.drop(TARGET_COL, axis=1)
del X_test['Unnamed: 0']
y_test = test[[TARGET_COL]]

print('shape train', X_train.shape)
print('shape test', X_test.shape)
print('columns train',X_train.columns)
print('columns test',X_test.columns)
print('head train', X_train.head(3))
print('head test', X_test.head(3))

y_test_pred = model.predict(X_test)

cm_plot = metrics.plot_confusion_matrix(model, X_test, y_test)

eval_metrics = {
    "Test Accuracy": metrics.accuracy_score(y_test, y_test_pred),
    "Test Recall": metrics.recall_score(y_test, y_test_pred),
    "Test Precison": metrics.precision_score(y_test, y_test_pred),
    "Test F1 Score": metrics.f1_score(y_test, y_test_pred)
}

# Model Accuracy, how often is the classifier correct?
print("Test Accuracy:", metrics.accuracy_score(y_test, y_test_pred))
# Recall
print("Test Recall:", metrics.recall_score(y_test, y_test_pred))
# Precision
print("Test Precison:", metrics.precision_score(y_test, y_test_pred))
# F1score
print("Test F1 Score:", metrics.f1_score(y_test, y_test_pred))






run.complete()