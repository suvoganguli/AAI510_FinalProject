from math import sqrt
from typing import Callable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


class HistGradientBoostingRegressorTuner:

    @staticmethod
    def tune(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series) -> Callable:
        """
        This function is used to tune the hyperparameters of the Histogram-based Gradient Boosting Regression Model.

        :param x_train: Training data features.
        :param y_train: Training data target values.
        :param x_val: Validation data features.
        :param y_val: Validation data target values.
        :return The objective function for the Optuna tuning library.
        """

        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)
            max_iter = trial.suggest_int("max_iter", 100, 1000)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
            l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 8, 256)
            max_bins = trial.suggest_int("max_bins", 32, 255)
            early_stopping = trial.suggest_categorical("early_stopping", [True, False])
            validation_fraction = trial.suggest_float("validation_fraction", 0.05, 0.2)
            tol = trial.suggest_float("tol", 1e-6, 1e-3)

            tune_model = HistGradientBoostingRegressor(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                max_leaf_nodes=max_leaf_nodes,
                max_bins=max_bins,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                tol=tol,
                random_state=1
            )
            tune_model.fit(x_train, y_train)
            model_val_pred = tune_model.predict(x_val)
            return sqrt(mean_squared_error(y_val, model_val_pred))

        return objective
