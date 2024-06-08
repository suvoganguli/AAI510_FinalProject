from typing import List

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from evaluation.ModelEvaluator import ModelEvaluator
from preprocessing.ColumnEncoder import ColumnEncoder
from preprocessing.ColumnSelector import ColumnSelector
from preprocessing.DataCleaner import DataCleaner
from preprocessing.DataImputer import DataImputer
from tuners.HistGradientBoostingRegressorTuner import HistGradientBoostingRegressorTuner


print("Preprocessing data.")
data_path: str = "../data/listings-full.csv"
data: pd.DataFrame = pd.read_csv(data_path)
data = DataCleaner.perform_base_cleaning(data)

data = DataImputer.remove_outliers_iqr(data, ["price"])
train_data, val_data, test_data = DataCleaner.split_train_val_test(data)

train_data = ColumnEncoder.mean_encode_columns(train_data, ColumnSelector.get_categorical_features(train_data), "price")
val_data = ColumnEncoder.mean_encode_columns(val_data, ColumnSelector.get_categorical_features(val_data), "price")
test_data = ColumnEncoder.mean_encode_columns(test_data, ColumnSelector.get_categorical_features(test_data), "price")

train_data = DataImputer.impute_missing_values(train_data, data.columns, SimpleImputer(strategy="median"))
val_data = DataImputer.impute_missing_values(val_data, data.columns, SimpleImputer(strategy="median"))

x_train, y_train, x_val, y_val, x_test, y_test = DataCleaner.perform_x_y_split(train_data, val_data, test_data)

print("Finding top columns.")
#top_columns: List[str] = ColumnSelector.recursive_feature_elimination(x_train, y_train, 20)
top_columns = ['host_response_rate', 'host_acceptance_rate', 'host_listings_count', 'host_total_listings_count', 'neighbourhood_cleansed', 'property_type', 'accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'maximum_minimum_nights', 'minimum_nights_avg_ntm', 'availability_30', 'availability_90', 'review_scores_cleanliness', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'latitude', 'longitude']
print(f"Top columns: {top_columns}")
x_train = x_train[top_columns]
x_val = x_val[top_columns]
x_test = x_test[top_columns]

print("Performing hyperparameter tuning.")
study = optuna.create_study(direction='minimize')
study.optimize(HistGradientBoostingRegressorTuner.tune(x_train, y_train, x_val, y_val), n_trials=10)

print("Training model.")
model = HistGradientBoostingRegressor(**study.best_params, random_state=1)
model.fit(x_train, y_train)

print("Evaluating model.")
val_preds = model.predict(x_val)
print(ModelEvaluator.get_key_metrics(y_val, val_preds))
ModelEvaluator.plot_predictions_vs_actuals(y_val, val_preds)
# Best result: {'mse': 2507.887673913802, 'rmse': 50.078814621692096, 'mae': 34.43816491582932, 'r2': 0.7294924558015126}