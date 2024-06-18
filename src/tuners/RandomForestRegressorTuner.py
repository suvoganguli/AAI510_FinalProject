from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class RandomForestRegressorTuner:
    @staticmethod
    def tune(x_train, y_train, x_val, y_val):
        """
        This function is used to tune the hyperparameters of the Random Forest Regressor Model.

        :param x_train: Training data features.
        :param y_train: Training data target values.
        :param x_val: Validation data features.
        :param y_val: Validation data target values.
        :return The objective function for the Optuna tuning library.
        """

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 2, 32),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }

            model = RandomForestRegressor(**params, random_state=1)
            model.fit(x_train, y_train)
            model_val_pred = model.predict(x_val)
            return sqrt(mean_squared_error(y_val, model_val_pred))

        return objective
