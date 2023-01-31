import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import xgboost


def import_data(y_test_filepath):
    # Import y_test for dataviz
    print('Import y_test')
    y_test = pd.read_csv(y_test_filepath)
    return y_test


def import_predictions(X_test_rf_y_pred_filepath, X_test_xgb_y_pred_filepath):
    # Import the predictions of both models
    print('Import X_test_rf_y_pred')
    X_test_rf_y_pred = pd.read_csv(X_test_rf_y_pred_filepath)
    print('Import X_test_xgb_y_pred')
    X_test_xgb_y_pred = pd.read_csv(X_test_xgb_y_pred_filepath)
    return X_test_rf_y_pred, X_test_xgb_y_pred


def load_xgb(xgb_filepath):
    # Load XGBoost for the feature importance
    print('Load of the XGBoost classifier model')
    xgb = joblib.load(xgb_filepath)
    return xgb


def plot_feature_importance(xgb):
    # Plot feature importance for XGBoost
    ax = xgboost.plot_importance(xgb, max_num_features = 50)
    fig = ax.figure
    fig.set_size_inches(10, 20)
    plt.show()


def shape_graphs(xgb):
    X_train_path = 'data/processed/application_train.csv.zip'
    X_train = pd.read_csv(X_train_path)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_train)
    # Explanations for a specific point
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])
    # Explanations for multiple points
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values[:100,:], X_train.iloc[:100,:])
    # Summary plot for each class
    shap.summary_plot(shap_values, X_train)


y_test_filepath = 'models/y_test.csv.zip'

X_test_rf_y_pred_filepath = 'models/X_test_rf_y_pred.csv.zip'
X_test_xgb_y_pred_filepath = 'models/X_test_xgb_y_pred.csv.zip'

xgb_filepath = 'models/xgb.pkl'

y_test = import_data(y_test_filepath)
X_test_rf_y_pred, X_test_xgb_y_pred = import_predictions(X_test_rf_y_pred_filepath, X_test_xgb_y_pred_filepath)
xgb = load_xgb(xgb_filepath)
plot_feature_importance(xgb)
shape_graphs(xgb)




