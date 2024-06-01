from typing import List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, chi2, RFE


class ColumnSelector:

    @staticmethod
    def select_k_best(x_train: pd.DataFrame, y_train: pd.Series, k: int) -> List[str]:
        """
        Selects the k columns that are most strongly related to the output variable.

        :param x_train: A pandas DataFrame containing feature columns.
        :param y_train: A pandas Series containing the target values.
        :param k: The number of columns to select.
        :return: A pandas DataFrame with the selected columns.
        """
        selector = SelectKBest(chi2, k=k)
        selector.fit(x_train, y_train.to_list())
        return x_train.columns[selector.get_support(indices=False)].to_list()

    @staticmethod
    def recursive_feature_elimination(x_train: pd.DataFrame, y_train: pd.Series,
                                      n_features_to_select: int) -> List[str]:
        """
        Recursively eliminates less important features.

        :param x_train: A pandas DataFrame containing training data features.
        :param y_train: A pandas Series containing the target values.
        :param n_features_to_select: The number of features to be selected.
        :return: A pandas DataFrame with the selected columns.
        """
        model = RandomForestRegressor()
        rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
        rfe.fit(x_train, y_train)
        return x_train.columns[rfe.get_support(indices=True)]

    @staticmethod
    def get_categorical_features(data: pd.DataFrame) -> List[str]:
        """
        Returns a list of all the columns in the provided DataFrame that have categorical data

        :param data: A pandas DataFrame
        :return: A list of column names that have categorical data
        """
        return data.select_dtypes(include=['object']).columns.tolist()
