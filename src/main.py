from typing import List

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from evaluation.ModelEvaluator import ModelEvaluator
from preprocessing.ColumnEncoder import ColumnEncoder
from preprocessing.ColumnSelector import ColumnSelector
from preprocessing.DataImputer import DataImputer
from preprocessing.DataSplitter import DataSplitter
from tuners.HistGradientBoostingRegressorTuner import HistGradientBoostingRegressorTuner

data_path: str = "../data/listings-full.csv"
data: pd.DataFrame = pd.read_csv(data_path)
data = data[data["price"].notna()]
data["price"] = data["price"].str.replace("$", "").str.replace(",", "").astype(float)

data = DataSplitter.remove_non_training_columns(data)
data = DataImputer.remove_outliers(data, ["price"])
data = DataImputer.impute_missing_values(data, data.columns, SimpleImputer(fill_value=-9999, strategy="constant"))
train_data, val_data, test_data = DataSplitter.split_train_val_test(data)

train_data = ColumnEncoder.mean_encode_columns(train_data, ColumnSelector.get_categorical_features(train_data), "price")
val_data = ColumnEncoder.mean_encode_columns(val_data, ColumnSelector.get_categorical_features(val_data), "price")
test_data = ColumnEncoder.mean_encode_columns(test_data, ColumnSelector.get_categorical_features(test_data), "price")

x_train, y_train, x_val, y_val, x_test, y_test = DataSplitter.perform_x_y_split(train_data, val_data, test_data)

top_columns: List[str] = ColumnSelector.select_k_best(x_train, y_train, 10)
x_train = x_train[top_columns]
x_val = x_val[top_columns]
x_test = x_test[top_columns]

study = optuna.create_study(direction='minimize')
study.optimize(HistGradientBoostingRegressorTuner.tune(x_train, y_train, x_val, y_val), n_trials=10)

model = HistGradientBoostingRegressor(**study.best_params, random_state=1)
model.fit(x_train, y_train)
val_preds = model.predict(x_val)
print(ModelEvaluator.get_key_metrics(y_val, val_preds))
ModelEvaluator.plot_predictions_vs_actuals(y_val, val_preds)
