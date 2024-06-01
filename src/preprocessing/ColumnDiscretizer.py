import pandas as pd


class ColumnDiscretizer:
    @staticmethod
    def equal_width_binning(data: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
        """
        Divides the range of the specified column in the DataFrame into intervals of equal width.

        :param data: A pandas DataFrame containing the column to be binned.
        :param column: The column name to be binned.
        :param bins: The number of equal-width bins to create.
        :return: A pandas DataFrame with the binned column.
        """
        data_copy = data.copy()
        data_copy[column] = pd.cut(data_copy[column], bins)
        return data_copy

    @staticmethod
    def equal_freq_binning(data: pd.DataFrame, column: str, bins: int) -> pd.DataFrame:
        """
        Divides the range of the specified column in the DataFrame into intervals of equal frequency.

        :param data: A pandas DataFrame containing the column to be binned.
        :param column: The column name to be binned.
        :param bins: The number of equal-frequency bins to create.
        :return: A pandas DataFrame with the binned column.
        """
        data_copy = data.copy()
        data_copy[column] = pd.qcut(data_copy[column], bins)
        return data_copy

    @staticmethod
    def quartile_binning(data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Divides the range of the specified column in the DataFrame into quartiles.

        :param data: A pandas DataFrame containing the column to be binned.
        :param column: The column name to be binned.
        :return: A pandas DataFrame with the binned column.
        """
        return ColumnDiscretizer.equal_freq_binning(data, column, 4)
