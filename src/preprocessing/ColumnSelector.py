from typing import List
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE
from sklearn.ensemble import GradientBoostingRegressor


class ColumnSelector:
    @staticmethod
    def variance_threshold_selector(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """
        Selects the columns in the DataFrame whose variance exceeds a certain threshold.

        :param data: A pandas DataFrame containing the columns to be selected.
        :param threshold: The minimum variance required for a column to be selected.
        :return: A pandas DataFrame with the selected columns.
        """
        selector = VarianceThreshold(threshold)
        selector.fit(data)
        return data[data.columns[selector.get_support(indices=True)]]

    @staticmethod
    def select_k_best(data: pd.DataFrame, labels: List[str], k: int) -> pd.DataFrame:
        """
        Selects the k columns that are most strongly related to the output variable.

        :param data: A pandas DataFrame containing the columns to be selected.
        :param labels: A list of output variables.
        :param k: The number of columns to select.
        :return: A pandas DataFrame with the selected columns.
        """
        selector = SelectKBest(chi2, k=k)
        selector.fit(data, labels)
        return data[data.columns[selector.get_support(indices=True)]]

    @staticmethod
    def recursive_feature_elimination(x_train: pd.DataFrame, y_train: pd.Series,
                                      n_features_to_select: int) -> pd.DataFrame:
        """
        Recursively eliminates less important features.

        :param x_train: A pandas DataFrame containing training data features.
        :param y_train: A pandas Series containing the target values.
        :param n_features_to_select: The number of features to be selected.
        :return: A pandas DataFrame with the selected columns.
        """
        model = GradientBoostingRegressor()
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
        rfe.fit(x_train, y_train)
        return x_train[x_train.columns[rfe.get_support(indices=True)]]
