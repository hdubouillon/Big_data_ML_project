import pandas as pd
import joblib


def import_data(input_filepath):
    # Import the data from precessed directory
    print('Import', input_filepath)
    data = pd.read_csv(input_filepath)
    return data

def load_models(rf_filepath, xgb_filepath):
    # Import the two models
    print('Load of the random forest classifier model')
    rf = joblib.load(rf_filepath)
    print('Load of the XGBoost classifier model')
    xgb = joblib.load(xgb_filepath)
    models = [rf, xgb]
    return models

def make_prediction(test, models):
    # Make predictions from the two models
    print('Make prediction of both models:')
    rf = models[0]
    xgb = models[1]
    rf_y_pred = (rf.predict_proba(test) <= 0.8)[:,0].astype(int)
    rf_y_pred = pd.DataFrame(rf_y_pred, columns=['rf_y_pred'])
    xgb_y_pred = (xgb.predict_proba(test) <= 0.786)[:,0].astype(int)
    xgb_y_pred = pd.DataFrame(xgb_y_pred, columns=['xgb_y_pred'])
    return rf_y_pred, xgb_y_pred

def export_prediction(rf_y_pred, xgb_y_pred, output_filepath):
    # Export the predictions of the two models
    predictions = [rf_y_pred, xgb_y_pred]
    prediction_file_names = ['rf_y_pred.csv.zip', 'xgb_y_pred.csv.zip']
    model_name = 'random forest classifier'
    for prediction, prediction_file_name  in zip(predictions, prediction_file_names):
        print('Export of the prédiction of:', model_name) 
        prediction.to_csv(output_filepath + prediction_file_name, index=False, compression='zip')
        model_name = 'XGBoost classifier'



input_filepath_application_test = 'data/processed/application_test.csv.zip'
output_filepath_application_test = 'models/application_test_'

input_filepath_X_test = 'data/processed/application_test.csv.zip'
output_filepath_X_test = 'models/X_test_'

rf_filepath = 'models/rf.pkl'
xgb_filepath = 'models/xgb.pkl'

application_test = import_data(input_filepath_application_test)
X_test = import_data(input_filepath_X_test)

models = load_models(rf_filepath, xgb_filepath)

rf_y_pred, xgb_y_pred = make_prediction(application_test, models)
export_prediction(rf_y_pred, xgb_y_pred, output_filepath_application_test)

#
rf_y_pred, xgb_y_pred = make_prediction(X_test, models)
export_prediction(rf_y_pred, xgb_y_pred, output_filepath_X_test)



