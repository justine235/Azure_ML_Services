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
#from fairlearn.metrics._group_metric_set import _create_group_metric_set
#from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id



parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', dest='input_dir', required=True)
parser.add_argument('--dataset', dest='dataset', required=True)
parser.add_argument('--datadir', dest='datadir', required=True)
parser.add_argument('--n_estimators', type=int)
parser.add_argument('--criterion', type=str, default="gini")
parser.add_argument('--max_depth', type=int)
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


#sensitive feature
print(X["Gender"].value_counts().to_dict())
sensitive_features = X[["Gender"]]


X_train, X_test, y_train, y_test, sensitive_features_train, sensitive_features_test = train_test_split(X, y, 
    sensitive_features,test_size = 0.2, random_state=0, stratify=y)


# save test part for later
test_data = args.input_dir
os.makedirs(os.path.join(test_data,'file'), exist_ok=True)
print('location second', os.path.join(test_data,'file'))
X_test.to_csv(os.path.join(test_data,'file/data_eval_output.csv'))


X_train = X_train.reset_index(drop=True)
sensitive_features_train = sensitive_features_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
sensitive_features_test = sensitive_features_test.reset_index(drop=True)

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


# store in the blob
os.makedirs(os.path.join(test_data,'model'), exist_ok=True)
print('location second', os.path.join(test_data,'model'))
joblib.dump(rf,os.path.join(test_data,'model/saved_model.pkl'))

# store in the local execution (run history)
os.makedirs('./outputs/model', exist_ok=True)
joblib.dump(rf,'./outputs/model/saved_model.pkl') 
print('model registered')


# get metrics & store the metrics in the blob
metrics = run.get_metrics()
df = pd.DataFrame(list(metrics.items()),columns = ['metrics','value']) 
os.makedirs(os.path.join(test_data,'metrics'), exist_ok=True)
df.to_csv(os.path.join(test_data,'metrics/metricsoutput.csv'))



#sf = { 'gender': sensitive_features_test.Gender}
#dash_dict_all = _create_group_metric_set(y_true=y_test,
#                                         predictions=y_test_pred,
#                                         sensitive_features=sf,
#                                         prediction_type='binary_classification')

run.complete()