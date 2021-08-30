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



parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', dest='input_dir', required=True)
parser.add_argument('--n_estimators', type=int)
parser.add_argument('--criterion', type=str, default="gini")
parser.add_argument('--max_depth', type=int)
#parser.add_argument('--output-dir', dest='output_dir')
args = parser.parse_args()



# Get input dataset as dataframe
run = Run.get_context()
run.log('Split criterion', np.str(args.criterion))
print(np.str(args.criterion))
#without hyperdrive
#df = run.input_datasets['prepared_ds'].to_pandas_dataframe()
#with hyperdrive

df = pd.read_csv(os.path.join(args.input_dir,"data_prep_output.csv"))
run.log('print arg',df.head(1))
print(df.head(1))

X = df.drop(columns=['EmployeeTargeted'])
y = df.filter(['EmployeeTargeted'])
#sensitive feature
A = df[["Gender"]]
X_train, X_test, y_train, y_test,A_train, A_test = train_test_split(X, y, A, test_size=0.30)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)
 

# Model 
rf = RandomForestClassifier(class_weight="balanced", 
                            n_estimators = args.n_estimators,
                            max_depth = args.max_depth, random_state=0,
                            criterion = args.criterion).fit(X_train,y_train)

y_test_pred = rf.predict(X_test)


# Model Testing
print("Test Accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_test_pred)))
run.log('Test Accuracy2', np.float(metrics.accuracy_score(y_test, y_test_pred)))
# Recall
print("Train Recall: {:.2f}".format(metrics.recall_score(y_test, y_test_pred)))
# Precision
print("Train Precison: {:.2f}".format(metrics.precision_score(y_test, y_test_pred)))
# F1score
print("Train F1 Score: {:.2f}".format(metrics.f1_score(y_test, y_test_pred)))
run.log('Test Accuracy', np.float(metrics.accuracy_score(y_test, y_test_pred)))
run.log('Test Recall', np.float(metrics.recall_score(y_test, y_test_pred)))
run.log('Test Precison', np.float(metrics.precision_score(y_test, y_test_pred)))
run.log('Test F1 Score', np.float(metrics.f1_score(y_test, y_test_pred)))

print("Confusion Matrix: ")
print(metrics.confusion_matrix(y_test, y_test_pred))

os.makedirs('./outputs/model', exist_ok=True)
# files saved in the "./outputs" folder are automatically uploaded into run history
joblib.dump(rf,'./outputs/model/saved_model.pkl') 
print('model registered')


sf = { 'gender': sensitive_features_test.Gender}
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id

dash_dict_all = _create_group_metric_set(y_true=y_test,
                                         predictions=dominant_all_ids,
                                         sensitive_features=sf,
                                         prediction_type='binary_classification')

run.complete()