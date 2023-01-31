import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

import xgboost
from xgboost.sklearn import XGBClassifier


def import_data(input_filepath):
    # Import the data from processed directory
    data = pd.read_csv(input_filepath)
    return data

def split_data(data):
    # Splitting the data in train and test set
    X = data.drop(['TARGET'], axis='columns')
    y = data['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def models_training(X_train, y_train):
    # Train the models (here a random forest classifier and XGBoost classifier)
    rf = RandomForestClassifier(random_state=0, n_estimators=450, n_jobs=-1)
    xgb = xgboost.XGBClassifier(learning_rate =0.15,
        n_estimators=300,
        max_depth=8,
        n_jobs=-1,
        random_state=0)
    models = [rf, xgb]
    for model in models:
        model.fit(X_train, y_train)
    # Return the fitted models
    return models

def export_model(models):
    # Export the models
    path = 'models/'
    model_file_names = ['rf.pkl', 'xgb.pkl'] 
    for model, model_file_name  in zip(models, model_file_names):
        joblib.dump(model, path + model_file_name, compress=9)


input_filepath = 'data/processed/application_train.csv.zip'

data = import_data(input_filepath)
X_train, X_test, y_train, y_test = split_data(data)
models = models_training(X_train, y_train)
export_model(models)