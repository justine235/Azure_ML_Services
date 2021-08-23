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

# Get input dataset as dataframe
run = Run.get_context()
#df = run.input_datasets['prepared_ds']
df = run.input_datasets['prepared_ds'].to_pandas_dataframe()
#df = args.input_data.to_pandas_dataframe()
print('ici',df)


X = df.drop(columns=['EmployeeTargeted']).values
y = df.filter(['EmployeeTargeted']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Model 
rf= RandomForestClassifier(n_estimators=500, class_weight="balanced")
rf.fit(X_train,y_train)
y_test_pred=rf.predict(X_test)


# Model Testing
print("Test Accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_test_pred)))

run.complete()