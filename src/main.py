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

median_encode_columns = ['neighbourhood_cleansed']
one_hot_columns = ['neighbourhood_group_cleansed', 'property_type']
label_encode_columns = ['host_response_time']
frequency_encode_columns = ['room_type']

print("Preprocessing data.")
data_path: str = "../data/listings-full.csv"
data: pd.DataFrame = pd.read_csv(data_path)
data = DataCleaner.perform_base_cleaning(data)

data = DataImputer.remove_outliers_iqr(data, ["price"])
data = ColumnEncoder.one_hot_encode_columns(data, one_hot_columns)
data = ColumnEncoder.label_encode_columns(data, label_encode_columns)

train_data, val_data, test_data = DataCleaner.split_train_val_test(data)

train_data = ColumnEncoder.median_encode_columns(train_data, median_encode_columns, "price")
val_data = ColumnEncoder.median_encode_columns(val_data, median_encode_columns, "price")
test_data = ColumnEncoder.median_encode_columns(test_data, median_encode_columns, "price")

train_data = ColumnEncoder.frequency_encode_columns(train_data, frequency_encode_columns)
val_data = ColumnEncoder.frequency_encode_columns(val_data, frequency_encode_columns)
test_data = ColumnEncoder.frequency_encode_columns(test_data, frequency_encode_columns)

train_data = DataImputer.impute_missing_values(train_data, data.columns, SimpleImputer(strategy="median"))
val_data = DataImputer.impute_missing_values(val_data, data.columns, SimpleImputer(strategy="median"))

x_train, y_train, x_val, y_val, x_test, y_test = DataCleaner.perform_x_y_split(train_data, val_data, test_data)

print("Finding top columns.")
top_columns: List[str] = ColumnSelector.recursive_feature_elimination(x_train, y_train, 50)
# ['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_listings_count', 'host_total_listings_count', 'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'review_scores_rating', 'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms', 'reviews_per_month', 'longitude', 'dist_times_square', 'dist_central_park', 'dist_empire_state_building', 'dist_statue_of_liberty', 'dist_brooklyn_bridge', 'dist_coney_island', 'dist_high_line', 'Entire condo', 'Entire loft', 'Entire serviced apartment', 'Private room in rental unit', 'Private room in resort', 'Room in boutique hotel', 'Room in hotel']
print(f"Top columns: {top_columns}")
x_train = x_train[top_columns]
x_val = x_val[top_columns]
x_test = x_test[top_columns]

print("Performing hyperparameter tuning.")
study = optuna.create_study(direction='minimize')
study.optimize(HistGradientBoostingRegressorTuner.tune(x_train, y_train, x_val, y_val), n_trials=100)
print(f"Best hyperparameters: {study.best_params}")

print("Training model.")
model = HistGradientBoostingRegressor(**study.best_params, random_state=1)
model.fit(x_train, y_train)

print("Evaluating model.")
val_preds = model.predict(x_val)
print(ModelEvaluator.get_key_metrics(y_val, val_preds))
ModelEvaluator.plot_predictions_vs_actuals(y_val, val_preds)
# Best result: {'mse': 2302.032455378782, 'rmse': 47.979500366081155, 'mae': 32.3882659703898}