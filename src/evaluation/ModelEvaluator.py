from math import sqrt
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelEvaluator:

    @staticmethod
    def get_key_metrics(actual_values: pd.Series, predicted_values: pd.Series) -> Dict:
        """
        Calculates and returns the key metrics (mse, rmse, mae) used to evaluate a model's predictions.

        :param actual_values: The actual known values.
        :param predicted_values: The predicted values produced by the model.
        :return: A dictionary containing the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute
        Error (MAE).
        """
        mse = mean_squared_error(actual_values, predicted_values)
        rmse = sqrt(mean_squared_error(actual_values, predicted_values))
        mae = mean_absolute_error(actual_values, predicted_values)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }

    @staticmethod
    def plot_predictions_vs_actuals(actual_values: pd.Series, predicted_values: pd.Series,
                                    title: str = "Predictions vs Actuals",
                                    x_label: str = "Actual Values", y_label: str = "Predicted Values") -> None:
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

    @staticmethod
    def plot_residuals(actual_values: pd.Series, predicted_values: pd.Series,
                       title: str = "Residuals vs Actuals",
                       x_label: str = "Actual Values", y_label: str = "Residuals") -> None:
        """
        Plots the residuals (the difference between actual and predicted values) against the actual values.

        :param actual_values: The actual known values.
        :param predicted_values: The predicted values produced by the model.
        :param title: The title of the plot. Defaults to "Residuals vs Actuals".
        :param x_label: The label for the x-axis. Defaults to "Actual Values".
        :param y_label: The label for the y-axis. Defaults to "Residuals".
        :return None: The function displays a plot but does not return any values.
        """
        residuals = actual_values - predicted_values
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_values, residuals, alpha=0.5)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.hlines(y=0, xmin=actual_values.min(), xmax=actual_values.max(), colors='k', linestyles='--', lw=4)
        plt.show()
