from azureml.core import Run
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
from raiwidgets import FairnessDashboard
import joblib
from fairlearn.widget import FairlearnDashboard


parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', dest='input_dir', required=True)
parser.add_argument('--n_estimators', type=int)
parser.add_argument('--criterion', type=str, default="gini")
parser.add_argument('--max_depth', type=int)
args = parser.parse_args()


# Get input dataset as dataframe
run = Run.get_context()
run.log('Split criterion', np.str(args.criterion))

#without hyperdrive
#df = run.input_datasets['prepared_ds'].to_pandas_dataframe()
#with hyperdrive
df = pd.read_csv(os.path.join(args.input_dir,"prepped_data.csv"))
run.log('print arg',df.head(1))


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


# Model saving
os.makedirs('outputs', exist_ok=True)
joblib.dump(rf, filename='outputs/model.pkl')


FairlearnDashboard(sensitive_features=X_test[['Gender']],
                   sensitive_feature_names=['Gender'],
                   y_true=y_test,
                   y_pred=y_test_pred)

# display fairness
#FairnessDashboard(sensitive_features=X_test, 
#                  #sensitive_feature_names=['Gender'],
#                  y_true=y_test,
#                  y_pred=rf.predict(X_test))

run.complete()