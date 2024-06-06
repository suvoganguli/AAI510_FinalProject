from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataCleaner:

    @staticmethod
    def perform_base_cleaning(data: pd.DataFrame) -> pd.DataFrame:
        """
        This method removes any records that don't have a price, converts the price from string to float, and removes
        non-training columns.

        :param data: A DataFrame that has the raw data to be cleaned
        :return: A DataFrame with the cleaned data
        """
        data_copy = DataCleaner.remove_non_training_columns(data.copy())
        data_copy = data_copy[data["price"].notna()]
        data_copy["price"] = data_copy["price"].str.replace("$", "").str.replace(",", "").astype(float)
        data_copy["host_response_rate"] = data_copy["host_response_rate"].str.replace("%", "").astype(float)
        data_copy["host_acceptance_rate"] = data_copy["host_acceptance_rate"].str.replace("%", "").astype(float)
        data_copy["host_is_superhost"] = data_copy["host_is_superhost"].map({'t': 1, 'f': 0})
        return data_copy

    @staticmethod
    def remove_non_training_columns(data: pd.DataFrame) -> pd.DataFrame:
        """
        This method removes columns from the DataFrame that are not necessary for the training process.

        :param data: A DataFrame with the base data
        :return: A DataFrame with only the necessary columns for the training process
        """
        return data[
            ["host_response_time", "host_response_rate", "host_acceptance_rate", "host_is_superhost",
             "host_listings_count", "host_total_listings_count", "neighbourhood_cleansed",
             "neighbourhood_group_cleansed", "property_type", "room_type", "accommodates", "bathrooms", "bedrooms",
             "price", "minimum_nights", "maximum_nights", "minimum_minimum_nights", "maximum_minimum_nights",
             "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
             "availability_30", "availability_60", "availability_90", "availability_365", "number_of_reviews",
             "number_of_reviews_ltm", "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
             "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
             "review_scores_location", "review_scores_value", "calculated_host_listings_count",
             "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
             "calculated_host_listings_count_shared_rooms", "reviews_per_month", "latitude", "longitude"]]

    @staticmethod
    def split_train_val_test(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        This method splits the input data into training, validation, and testing data.

        :param data: The pandas DataFrame to be split
        :return: a tuple containing three DataFrames: training data, validation data, and testing data
        """
        train_data, test_val_data = train_test_split(data, test_size=0.4, random_state=1)
        val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=1)
        return train_data, val_data, test_data

    @staticmethod
    def perform_x_y_split(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        This method splits the given training, validation and test datasets into features (X) and the target
        variable (y) for each set.

        :param train_data: A pandas DataFrame containing the training data
        :param val_data: A pandas DataFrame containing the validation data
        :param test_data: A pandas DataFrame containing the test data

        :return: A tuple with 6 components:
                    - x_train: Features for the training data
                    - y_train: Target variable for the training data
                    - x_val: Features for the validation data
                    - y_val: Target variable for the validation data
                    - x_test: Features for the test data
                    - y_test: Target variable for the test data
        """
        x_train: pd.DataFrame = train_data.drop(columns=["price"])
        x_val: pd.DataFrame = val_data.drop(columns=["price"])
        x_test: pd.DataFrame = test_data.drop(columns=["price"])
        y_train: pd.Series = train_data["price"]
        y_val: pd.Series = val_data["price"]
        y_test: pd.Series = test_data["price"]

        return x_train, y_train, x_val, y_val, x_test, y_test
