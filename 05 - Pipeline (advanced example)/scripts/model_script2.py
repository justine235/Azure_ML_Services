from azureml.core import Run,Model
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import joblib
from joblib import dump
#from fairlearn.metrics._group_metric_set import _create_group_metric_set
#from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id



parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', dest='input_dir', required=True)
parser.add_argument('--dataset', dest='dataset', required=True)
parser.add_argument('--n_estimators', type=int)
parser.add_argument('--criterion', type=str, default="gini")
parser.add_argument('--max_depth', type=int)
parser.add_argument('--model-path', dest='model_path', required=True)
parser.add_argument('--metrics-path', dest='metrics_path', required=True)
args = parser.parse_args()



# Get input dataset as dataframe
run = Run.get_context()
run.log('Split criterion', np.str(args.criterion))
print(np.str(args.criterion))

# reading data without hyperdrive from datastore
#df = run.input_datasets['prepared_ds'].to_pandas_dataframe()

# reading data from the previous pipeline with fileoutputconfig
#data_directory = args.input_dir
#output_path = os.path.join(data_directory,'file/data_prep_output.csv')
#df = pd.read_csv(output_path)

df = pd.read_csv(args.dataset)
print(df.head(5))

# splitting data
X = df.drop(columns=['EmployeeTargeted'])
y = df.filter(['EmployeeTargeted'])



X_train, X_test, y_train, y_test  = train_test_split(X, y,test_size = 0.2, random_state=0, stratify=y)

print('shape train', X_train.shape)
print('shape test', X_test.shape)
print('columns train',X_train.columns)
print('columns test',X_test.columns)
print('head train', X_train.head(3))
print('head test', X_test.head(3))

# save test data dataset
output_dir = args.input_dir
os.makedirs(os.path.join(output_dir), exist_ok=True)
print('location second', os.path.join(output_dir))
pd.concat([X_test,y_test], axis=1).to_csv(os.path.join(output_dir,'data_eval_output.csv'))
pd.concat([X_train,y_train], axis=1).to_csv(os.path.join(output_dir,'data_train_output.csv'))


# Model 
rf = RandomForestClassifier(class_weight="balanced", 
                            n_estimators = args.n_estimators,
                            max_depth = args.max_depth, random_state=0,
                            criterion = args.criterion).fit(X_train,y_train)

y_test_pred = rf.predict(X_test)


# Model Testing
print("Test Accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_test_pred)))
run.log('Test Accuracy2', np.float(metrics.accuracy_score(y_test, y_test_pred)))
print("Train Recall: {:.2f}".format(metrics.recall_score(y_test, y_test_pred)))
print("Train Precison: {:.2f}".format(metrics.precision_score(y_test, y_test_pred)))
print("Train F1 Score: {:.2f}".format(metrics.f1_score(y_test, y_test_pred)))

run.log('Test Accuracy', np.float(metrics.accuracy_score(y_test, y_test_pred)))
run.log('Test Recall', np.float(metrics.recall_score(y_test, y_test_pred)))
run.log('Test Precison', np.float(metrics.precision_score(y_test, y_test_pred)))
run.log('Test F1 Score', np.float(metrics.f1_score(y_test, y_test_pred)))

print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, y_test_pred))
cm_plot_train = metrics.plot_confusion_matrix(rf, X_train, y_train)
cm_plot_val = metrics.plot_confusion_matrix(rf, X_test, y_test)
cm_plot_train.figure_.savefig("confusion_matrix_train.png")
cm_plot_val.figure_.savefig("confusion_matrix_val.png")


# get run id 
run = Run.get_context()
RUNID = run.get_details()['runId']
MODEL_NAME = RUNID + 'model.joblib'
print(MODEL_NAME)

# store in the blob / datastore
model_dir = args.model_path
os.makedirs(os.path.join(model_dir), exist_ok=True)
joblib.dump(rf,os.path.join(model_dir,MODEL_NAME))

# store in the local execution (run history)
os.makedirs('./outputs/model', exist_ok=True)
joblib.dump(rf,'./outputs/model/saved_model.pkl') 
print('model registered')
run.log_image(name='Confusion Matrix Train Dataset', path="confusion_matrix_train.png")
run.log_image(name='Confusion Matrix Val Dataset', path="confusion_matrix_val.png")


# get metrics & store the metrics in the datastore
metrics_dir = args.metrics_path
metrics = run.get_metrics()
df = pd.DataFrame(list(metrics.items()),columns = ['metrics','value']) 
os.makedirs(os.path.join(metrics_dir,RUNID), exist_ok=True)
df.to_csv(os.path.join(metrics_dir,RUNID,'metricsoutput.csv'))
cm_plot_val.figure_.savefig(os.path.join(metrics_dir,RUNID,'confusion_matrix_val.png'))




run.complete()