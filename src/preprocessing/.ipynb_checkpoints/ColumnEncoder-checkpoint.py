from typing import List

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ColumnEncoder:

    @staticmethod
    def mean_encode_columns(data: pd.DataFrame, columns_to_encode: List[str], target_column: str) -> pd.DataFrame:
        """
        Encodes the specified columns in the data DataFrame using mean encoding (target encoding).
        For each specified column, the function calculates the average value of the target column
        for each category in the column, and replaces each category with its calculated mean value.

        :param data: A pandas DataFrame containing the columns to be mean-encoded.
        :param columns_to_encode: A list of column names to be mean-encoded.
        :param target_column: The name of the target column.
        :return: A pandas DataFrame with the mean-encoded values in the specified columns.
        """
        data_copy = data.copy()
        for column in columns_to_encode:
            mean_values = data_copy.groupby(column)[target_column].mean()
            data_copy[column] = data_copy[column].map(mean_values)
        return data_copy

    @staticmethod
    def one_hot_encode_columns(data: pd.DataFrame, columns_to_encode: List[str]) -> pd.DataFrame:
        """
        Encodes the specified columns in the given DataFrame using frequency encoding.
        Frequency encoding replaces each category in a column with its frequency of occurrence within that column.

        :param data: A pandas DataFrame containing the columns to be frequency-encoded.
        :param columns_to_encode: A list of column names in the data that should be frequency-encoded.
        :return: A pandas DataFrame with the frequency-encoded values in the specified columns.
        """
        data_copy = data.copy()
        for column in columns_to_encode:
            one_hot_cols = pd.get_dummies(data_copy[column])
            data_copy = data_copy.drop(column, axis=1)
            data_copy = pd.concat([data_copy, one_hot_cols], axis=1)
        return data_copy

    @staticmethod
    def label_encode_columns(data: pd.DataFrame, columns_to_encode: List[str]) -> pd.DataFrame:
        """
        Encodes the specified columns of the given DataFrame using a label encoder, transforming the categorical/nominal
        values in the specified columns to integer values (starting from 0, and increasing in increments of 1 per unique
        category).

        :param data: A pandas DataFrame containing the columns to be label-encoded.
        :param columns_to_encode: A list of column names in the data that should be label-encoded.
        :return: A pandas DataFrame with the label-encoded values in the specified columns.
        """
        data_copy = data.copy()
        label_encoder = LabelEncoder()
        for column in columns_to_encode:
            data_copy[column] = label_encoder.fit_transform(data_copy[column])
        return data_copy

    @staticmethod
    def frequency_encode_columns(data: pd.DataFrame, columns_to_encode: List[str]) -> pd.DataFrame:
        """
        Encodes the specified columns in the given DataFrame using frequency encoding.
        Frequency encoding replaces each category in a column with its frequency of occurrence within that column.

        :param data: A pandas DataFrame containing the columns to be frequency-encoded.
        :param columns_to_encode: A list of column names in the data that should be frequency-encoded.
        :return: A pandas DataFrame with the frequency-encoded values in the specified columns.
        """
        data_copy = data.copy()
        for column in columns_to_encode:
            encoding = data_copy.groupby(column).size() / len(data_copy)
            data_copy[column] = data_copy[column].map(encoding)
        return data_copy
