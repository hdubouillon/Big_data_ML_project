import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def import_data(input_filepath):
    # Import the data from raw directory
    data = pd.read_csv(input_filepath)
    return data

def data_preprocessing(data):
    # Deletion of the outliers of DAYS_EMPLOYED
    data = data[data["DAYS_EMPLOYED"] != 365243]
    return data

def feature_engineering(data):
    # Label encoding of columns with 2 or fewer unique categories
    le = LabelEncoder()
    for col in data:
        if data[col].dtype == 'object':
            if len(list(data[col].unique())) <= 2:
                le.fit(data[col])
                data[col] = le.transform(data[col])

    # One-hot encoding of the other categorical variables
    data = pd.get_dummies(data)

    # Median imputation of missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(data)
    data = pd.DataFrame(imputer.transform(data), columns=data.columns)

    # Drop the columns of the id
    try:
        data = data.drop(['SK_ID_CURR'], axis='columns')
    except:
        None
    
    # Return the data
    return data

def export_data(data, output_filepath):
    # Export the data to precessed directory
    data.to_csv(output_filepath, index=False, compression='zip')


# Train data
train_input_filepath = 'data/raw/application_train.csv.zip'
train_output_filepath = 'data/processed/application_train.csv.zip'
train = import_data(train_input_filepath)
train = data_preprocessing(train)
train = feature_engineering(train)
export_data(train, train_output_filepath)

# Test data
test_input_filepath = 'data/raw/application_train.csv.zip'
test_output_filepath = 'data/processed/application_train.csv.zip'
test = import_data(test_input_filepath)
test = data_preprocessing(test)
test = feature_engineering(test)
export_data(test, test_output_filepath)