from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import metrics

# Get the experiment run context
run = Run.get_context()

# Set parameters
parser = argparse.ArgumentParser()
parser.add_argument('--criterion', type=str, default="gini")
args = parser.parse_args()

# Prepare the dataset
df = pd.read_csv('data/train_dataset.csv')
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df.drop(columns=['Email_Domain']))
X = df.drop(columns=['EmployeeTargeted']).values
y = df.filter(['EmployeeTargeted']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
model = RandomForestClassifier(class_weight="balanced", criterion = args.criterion, random_state=0).fit(X_train, y_train)

# predicting over training & testing datasets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Model Training
print("Train Accuracy: {:.2f}".format(metrics.accuracy_score(y_train, y_train_pred)))
# Recall
print("Train Recall: {:.2f}".format(metrics.recall_score(y_train, y_train_pred)))
# Precision
print("Train Precison: {:.2f}".format(metrics.precision_score(y_train, y_train_pred)))
# F1score
print("Train F1 Score: {:.2f}".format(metrics.f1_score(y_train, y_train_pred)))

run.log('Train Accuracy', np.float(metrics.accuracy_score(y_train, y_train_pred)))
run.log('Train Recall', np.float(metrics.recall_score(y_train, y_train_pred)))
run.log('Train Precison', np.float(metrics.precision_score(y_train, y_train_pred)))
run.log('Train F1 Score', np.float(metrics.f1_score(y_train, y_train_pred)))


# Model Testing
print("Train Accuracy: {:.2f}".format(metrics.accuracy_score(y_test, y_test_pred)))
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


# Save the trained model / Export model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()