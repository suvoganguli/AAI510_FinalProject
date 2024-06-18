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

mean_impute_columns = ['host_listings_count', 'room_type', 'accommodates', 'minimum_nights', 'maximum_minimum_nights',
                       'minimum_maximum_nights', 'maximum_maximum_nights', 'maximum_nights_avg_ntm',
                       'number_of_reviews_ltm', 'number_of_reviews_l30d', 'calculated_host_listings_count',
                       'dist_empire_state_building', 'dist_laguardia', 'amenity_wifi', 'amenity_carbon_monoxide_alarm',
                       'amenity_hot_water', 'amenity_dishes_and_silverware', 'amenity_hair_dryer', 'amenity_heating',
                       'amenity_extra_pillows_and_blankets']
most_frequent_impute_columns = ['host_response_time', 'host_is_superhost', 'minimum_nights_avg_ntm', 'availability_60',
                                'availability_365', 'calculated_host_listings_count_shared_rooms', 'dist_high_line',
                                'dist_st_georges_theatre', 'amenity_iron', 'amenity_bed_linens',
                                'amenity_dedicated_workspace', 'amenity_microwave', 'amenity_first_aid_kit']
constant_impute_columns = ['neighbourhood_cleansed', 'maximum_nights', 'minimum_minimum_nights', 'availability_90',
                           'number_of_reviews', 'dist_central_park', 'dist_brooklyn_bridge', 'amenity_smoke_alarm',
                           'amenity_essentials', 'amenity_refrigerator', 'amenity_tv', 'amenity_fire_extinguisher',
                           'amenity_self_check-in', 'amenity_oven', 'amenity_free_street_parking',
                           'amenity_coffee_maker', 'amenity_cleaning_products']

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

train_data = DataImputer.impute_missing_values(train_data, mean_impute_columns, SimpleImputer(strategy="mean"))
train_data = DataImputer.impute_missing_values(train_data, most_frequent_impute_columns,
                                               SimpleImputer(strategy="most_frequent"))
train_data = DataImputer.impute_missing_values(train_data, constant_impute_columns,
                                               SimpleImputer(strategy="constant", fill_value=-9999))
train_data = DataImputer.impute_missing_values(train_data, data.columns, SimpleImputer(strategy="median"))

val_data = DataImputer.impute_missing_values(val_data, mean_impute_columns, SimpleImputer(strategy="mean"))
val_data = DataImputer.impute_missing_values(val_data, most_frequent_impute_columns,
                                             SimpleImputer(strategy="most_frequent"))
val_data = DataImputer.impute_missing_values(val_data, constant_impute_columns,
                                             SimpleImputer(strategy="constant", fill_value=-9999))
val_data = DataImputer.impute_missing_values(val_data, data.columns, SimpleImputer(strategy="median"))

x_train, y_train, x_val, y_val, x_test, y_test = DataCleaner.perform_x_y_split(train_data, val_data, test_data)

print("Finding top columns.")
top_columns: List[str] = ColumnSelector.recursive_feature_elimination(x_train, y_train, 60)
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
