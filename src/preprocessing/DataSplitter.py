from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:

    @staticmethod
    def remove_non_training_columns(data: pd.DataFrame) -> pd.DataFrame:
        return data[
            ["host_listings_count", "host_total_listings_count", "neighbourhood_cleansed",
             "neighbourhood_group_cleansed", "property_type", "room_type", "accommodates", "bathrooms", "bedrooms",
             "price", "minimum_nights", "maximum_nights", "minimum_minimum_nights", "maximum_minimum_nights",
             "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
             "availability_30", "availability_60", "availability_90", "availability_365", "number_of_reviews",
             "number_of_reviews_ltm", "number_of_reviews_l30d", "review_scores_rating", "review_scores_accuracy",
             "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
             "review_scores_location", "review_scores_value", "calculated_host_listings_count",
             "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
             "calculated_host_listings_count_shared_rooms", "reviews_per_month"]]

    @staticmethod
    def split_train_val_test(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_data, test_val_data = train_test_split(data, test_size=0.4, random_state=1)
        val_data, test_data = train_test_split(test_val_data, test_size=0.5, random_state=1)
        return train_data, val_data, test_data

    @staticmethod
    def perform_x_y_split(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[
            pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        x_train: pd.DataFrame = train_data.drop(columns=["price"])
        x_val: pd.DataFrame = val_data.drop(columns=["price"])
        x_test: pd.DataFrame = test_data.drop(columns=["price"])
        y_train: pd.Series = train_data["price"]
        y_val: pd.Series = val_data["price"]
        y_test: pd.Series = test_data["price"]

        return x_train, y_train, x_val, y_val, x_test, y_test
