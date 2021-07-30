
# Read dataset
df = pd.read_csv('data/train_dataset.csv')

# data cleaning
def preprocessing(file_name):
        
    # importation 
    data = pd.read_csv(file_name)
    data.drop(columns=['Unnamed: 0'], inplace=True)

    # create a dictionnary actionable for renaming necessary columns 
    #old_names = ['Access Level', 'BD877Training Completed', 'Department Code', 'Email Domain', 'EmployeeTargetedOverPastYear', 
    #             'Gender (code)','Social Media Activity (Scaled)','fraudTraining Completed']

    # new_names = ['Access_Level', 'Training_Completed', 'Code_postal','Email_Domain', 'EmployeeTargeted',
    ##              'Gender','Social_Media','fraudTraining']

    #data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    #data.head(3)

    # missing values traitement
    data['Training_Completed'] = data['Training_Completed'].interpolate(method='linear', direction = 'forward').round(0)
    value_nan_Email_Domain = data['Email_Domain'].value_counts().idxmax()
    data['Email_Domain'].replace(to_replace = np.nan, value = value_nan_Email_Domain, inplace=True)
    data['Social_Media'] = data['Social_Media'].interpolate(method='linear', direction = 'forward').round(0)
    data

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

df = preprocessing(data)
