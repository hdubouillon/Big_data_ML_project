import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost


def import_data(input_filepath):
    # Import the data from processed directory
    print('Import', input_filepath)
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
    model_name = 'random forest classifier'
    for model in models:
        print('Training of:', model_name)        
        #mlflow.set_tracking_uri("http://localhost:5000")
        #with mlflow.start_run():
            #model.fit(X_train, y_train)
            #mlflow.log_params(model.get_params())
        model.fit(X_train, y_train)
        model = 'XGBoost classifier'
    # Return the fitted models
    return models

def export_model(models):
    # Export the models    
    path = 'models/'
    model_file_names = ['rf.pkl', 'xgb.pkl']
    model_name = 'random forest classifier'
    for model, model_file_name  in zip(models, model_file_names):
        print('Export of:', model_name)
        joblib.dump(model, path + model_file_name, compress=9)
        model_name = 'XGBoost classifier'

def export_test_data_for_dataviz(X_test, y_test, X_test_filepath, y_test_filepath):
    # Export X_test, y_test for dataviz
    print('Export X_test for dataviz')
    X_test.to_csv(X_test_filepath, index=False, compression='zip')
    print('Export y_test for dataviz')
    y_test.to_csv(y_test_filepath, index=False, compression='zip')


input_filepath = 'data/processed/application_train.csv.zip'
X_test_filepath = 'models/X_test.csv.zip'
y_test_filepath = 'models/y_test.csv.zip'

data = import_data(input_filepath)
X_train, X_test, y_train, y_test = split_data(data)
models = models_training(X_train, y_train)
export_model(models)
export_test_data_for_dataviz(X_test, y_test, X_test_filepath, y_test_filepath)