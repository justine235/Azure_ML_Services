from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from fairlearn.widget import FairlearnDashboard
from azureml.core import Workspace
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate')
args = parser.parse_args()
reg = args.reg_rate

# Prepare the dataset
df = pd.read_csv('data/train_dataset.csv')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df.drop(columns=['Email_Domain']))
X = df.drop(columns=['EmployeeTargeted'])
y = df.filter(['EmployeeTargeted'])

A = X[["Gender"]]
X = X.drop(labels=['Gender'],axis = 1)

X_train, X_test, y_train, y_test,A_train, A_test = train_test_split(X, y, A, test_size=0.30)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)


# Train a logistic regression model
model = RandomForestClassifier(class_weight="balanced").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)

# Save the trained model / Export model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

# (Optional) View this model in Fairlearn's fairness dashboard, and see the disparities which appear:
FairlearnDashboard(sensitive_features=A_test, 
                   sensitive_feature_names=['Gender'],
                   y_true=y_test,
                   y_pred={"model": model.predict(X_test)})

