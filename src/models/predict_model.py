import pandas as pd

import joblib


def import_data(input_filepath):
    # Import the data from precessed directory
    data = pd.read_csv(input_filepath)
    return data

def load_models(rf_filepath, xgb_filepath):
    # Import the two models
    rf = joblib.load(rf_filepath)
    xgb = joblib.load(xgb_filepath)
    models = [rf, xgb]
    return models

def make_prediction(X_test, models):
    # Make predictions from the two models
    rf = models[0]
    xgb = models[1]
    rf_y_pred = (rf.predict_proba(X_test) <= 0.8)[:,0].astype(int)
    xgb_y_pred = (xgb.predict_proba(X_test) <= 0.786)[:,0].astype(int)
    return rf_y_pred, xgb_y_pred

def export_prediction(rf_y_pred, xgb_y_pred):
    # Export the predictions of the two models    
    path = 'models/'
    predictions = [rf_y_pred, xgb_y_pred]
    prediction_file_names = ['rf_y_pred.csv', 'xgb_y_pred.csv'] 
    for prediction, prediction_file_name  in zip(predictions, prediction_file_names):
        prediction.to_csv(path + prediction_file_name, index=False, compression='zip')



input_filepath = 'data/processed/application_test.csv.zip'
output_filepath = 'models/application_test.csv.zip'
rf_filepath = 'models/rf.pkl'
xgb_filepath = 'models/xgb.pkl'

X_test = import_data(input_filepath)
models = load_models(rf_filepath, xgb_filepath)
rf_y_pred, xgb_y_pred = make_prediction(X_test, models)
export_prediction(rf_y_pred, xgb_y_pred)


