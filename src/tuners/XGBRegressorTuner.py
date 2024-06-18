from math import sqrt

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


class XGBRegressorTuner:
    @staticmethod
    def tune(x_train, y_train, x_val, y_val):
        """
        This function is used to tune the hyperparameters of the Extreme Gradient Boosting Regressor Model.

        :param x_train: Training data features.
        :param y_train: Training data target values.
        :param x_val: Validation data features.
        :param y_val: Validation data target values.
        :return The objective function for the Optuna tuning library.
        """

        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
            }

            model = XGBRegressor(**param, random_state=1)
            model.fit(x_train, y_train)
            model_val_pred = model.predict(x_val)
            return sqrt(mean_squared_error(y_val, model_val_pred))

        return objective
