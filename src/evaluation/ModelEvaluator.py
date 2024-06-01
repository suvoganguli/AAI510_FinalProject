from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:

    @staticmethod
    def get_key_metrics(actual_values: pd.Series, predicted_values: pd.Series) -> Dict:
        """
        Calculates and returns the key metrics (mse, rmse, mae) used to evaluate a model's predictions.

        :param actual_values: The actual known values.
        :param predicted_values: The predicted values produced by the model.
        :return: A dictionary containing the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute
        Error (MAE), and R-Squared score (R^2).
        """
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = root_mean_squared_error(actual_values, predicted_values)
        mae = mean_absolute_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    @staticmethod
    def plot_predictions_vs_actuals(actual_values: pd.Series, predicted_values: pd.Series,
                                    title: str = "Predictions vs Actuals",
                                    x_label: str = "Actual Values", y_label: str = "Predicted Values"):
        """
        Plots the predicted values against the actual values.

        :param actual_values: The actual known values.
        :param predicted_values: The predicted values produced by the model.
        :param title: The title of the plot. Defaults to "Predictions vs Actuals".
        :param x_label: The label for the x-axis. Defaults to "Actual Values".
        :param y_label: The label for the y-axis. Defaults to "Predicted Values".
        :return None: The function displays a plot but does not return any values.
        """

        plt.figure(figsize=(10, 6))
        plt.scatter(actual_values, predicted_values, alpha=0.5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'k--', lw=4)
        plt.show()
