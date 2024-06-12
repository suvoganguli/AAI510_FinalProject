from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class ColumnScaler:

    @staticmethod
    def standard_scale_columns(data: pd.DataFrame, columns_to_scale: List[str], group_by=None) -> pd.DataFrame:
        """
        Scales the specified columns in the DataFrame using StandardScaler (Z-Score normalization).
        If the group_by parameter is provided, each column value will be scaled relative to other values within the same
        grouping.

        :param data: A pandas DataFrame containing the columns to be scaled.
        :param columns_to_scale: A list of column names to be scaled.
        :param group_by: Column name to group by. Default is None.
        :return: A pandas DataFrame with the scaled values in the specified columns.
        """
        data_copy = data.copy()
        scaler = StandardScaler()

        if group_by:
            data_copy[columns_to_scale] = data_copy.groupby(group_by)[columns_to_scale].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)))
        else:
            data_copy[columns_to_scale] = scaler.fit_transform(data_copy[columns_to_scale])

        return data_copy

    @staticmethod
    def min_max_scale_columns(data: pd.DataFrame, columns_to_scale: List[str], group_by=None) -> pd.DataFrame:
        """
        Scales the specified columns in the DataFrame using MinMaxScaler.
        If the group_by parameter is provided, each column value will be scaled relative to other values within the same
        grouping.

        :param data: A pandas DataFrame containing the columns to be scaled.
        :param columns_to_scale: A list of column names to be scaled.
        :param group_by: Column name to group by. Default is None.
        :return: A pandas DataFrame with the scaled values in the specified columns.
        """
        data_copy = data.copy()
        scaler = MinMaxScaler()

        if group_by:
            data_copy[columns_to_scale] = data_copy.groupby(group_by)[columns_to_scale].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)))
        else:
            data_copy[columns_to_scale] = scaler.fit_transform(data_copy[columns_to_scale])

        return data_copy

    @staticmethod
    def robust_scale_columns(data: pd.DataFrame, columns_to_scale: List[str], group_by=None) -> pd.DataFrame:
        """
        Scales the specified columns in the DataFrame using RobustScaler.
        If the group_by parameter is provided, each column value will be scaled relative to other values within the same
        grouping.

        :param data: A pandas DataFrame containing the columns to be scaled.
        :param columns_to_scale: A list of column names to be scaled.
        :param group_by: Column name to group by. Default is None.
        :return: A pandas DataFrame with the scaled values in the specified columns.
        """
        data_copy = data.copy()
        scaler = RobustScaler()

        if group_by:
            data_copy[columns_to_scale] = data_copy.groupby(group_by)[columns_to_scale].transform(
                lambda x: scaler.fit_transform(x.values.reshape(-1, 1)))
        else:
            data_copy[columns_to_scale] = scaler.fit_transform(data_copy[columns_to_scale])

        return data_copy
