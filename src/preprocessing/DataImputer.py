from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DataImputer:

    @staticmethod
    def impute_missing_values(data: pd.DataFrame, columns_to_impute: List[str], imputer: SimpleImputer) -> pd.DataFrame:
        """
        Imputes the missing values in the specified columns of the data DataFrame.

        :param data: A pandas DataFrame containing the columns to be imputed.
        :param columns_to_impute: A list of column names to be imputed.
        :param imputer: The sklearn imputer to use for imputing the values (ex: SimpleImputer())
        :return: A pandas DataFrame with the missing values in the specified columns imputed.
        """
        data_copy = data.copy()
        data_copy[columns_to_impute] = imputer.fit_transform(data_copy[columns_to_impute])
        return data_copy

    @staticmethod
    def remove_outliers_zscore(data: pd.DataFrame, columns_to_check: List[str], threshold: float = 3) -> pd.DataFrame:
        """
        Removes outliers from the specified columns of the data DataFrame.

        :param data: A pandas DataFrame containing the columns to be checked for outliers.
        :param columns_to_check: A list of column names to be checked for outliers.
        :param threshold: The threshold Z-score above which a data point is considered an outlier.
                          Defaults to 3.
        :return: A pandas DataFrame with the outliers removed from the specified columns.
        """
        data_copy = data.copy()
        for column in columns_to_check:
            z_scores = np.abs((data_copy[column] - data_copy[column].mean()) / data_copy[column].std())
            data_copy = data_copy[~(z_scores > threshold)]
        return data_copy

    @staticmethod
    def remove_outliers_iqr(data: pd.DataFrame, columns_to_check: List[str], multiplier: float = 1.5) -> pd.DataFrame:
        """
        Removes outliers from the specified columns of the data DataFrame using the Interquartile Range Method.
        Any data point that is below Q1 - (multiplier * IQR) or above Q3 + (multiplier * IQR) is considered an outlier.

        :param data: A pandas DataFrame containing the columns to be checked for outliers.
        :param columns_to_check: A list of column names to be checked for outliers.
        :param multiplier: The multiplier for the IQR. Defaults to 1.5.
        :return: A pandas DataFrame with the outliers removed from the specified columns.
        """
        data_copy = data.copy()
        for column in columns_to_check:
            q1 = data_copy[column].quantile(0.25)
            q3 = data_copy[column].quantile(0.75)
            iqr = q3 - q1
            data_copy = data_copy[
                (data_copy[column] > (q1 - multiplier * iqr)) & (data_copy[column] < (q3 + multiplier * iqr))]
        return data_copy

    @staticmethod
    def cap_outliers(data: pd.DataFrame, columns_to_check: List[str], lower_percentile: float = 1,
                     upper_percentile: float = 99) -> pd.DataFrame:
        """
        Caps outliers of the specified columns of the data DataFrame at the upper and lower percentiles.

        :param data: A pandas DataFrame containing the columns to be checked for outliers.
        :param columns_to_check: A list of column names to be checked for outliers.
        :param lower_percentile: The lower percentile at which to cap the outliers. Defaults to 1.
        :param upper_percentile: The upper percentile at which to cap the outliers. Defaults to 99.
        :return: A pandas DataFrame with the outliers in the specified columns capped.
        """
        data_copy = data.copy()
        for column in columns_to_check:
            lower_cap = data_copy[column].quantile(lower_percentile / 100)
            upper_cap = data_copy[column].quantile(upper_percentile / 100)
            data_copy[column] = np.clip(data_copy[column], lower_cap, upper_cap)
        return data_copy
