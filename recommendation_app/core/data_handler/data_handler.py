from typing import List

import pandas as pd
from sklearn import preprocessing


class DataHandler:
    """Feature Engineers the dataframe."""

    def __init__(self, df: pd.DataFrame) -> None:
        """__init__ method.

        Args:
            df (pd.DataFrame): pd.DataFrame.
        """
        self.df = df

    def normalize(self, features: List) -> pd.DataFrame:
        """Normalize a list of features  from the DataFrame inplace.

        Args:
            df (pd.DataFrame): DataFrame to normalize the columns.
            features (List): List of DataFrame column names.

        Returns:
            pd.DataFrame: DataFrame with normalized columns.
        """
        normalized_arr = preprocessing.normalize(self.df[features], axis=0)
        self.df[features] = normalized_arr
        return self.df

    def one_hot_encode(self, features: List) -> pd.DataFrame:
        """One Hot Encode a list of features from the DataFrame inplace.

        Args:
            df (pd.DataFrame): DataFrame to one hot encode the columns.
            features (List): List of DataFrame column names.

        Returns:
            pd.DataFrame: DataFrame with one hot encoded columns.
        """
        for feature in features:
            ohe_df = pd.get_dummies(self.df[feature])
            ohe_df.reset_index(drop=True, inplace=True)
            self.df = pd.concat([self.df, ohe_df], axis=1)
            self.df.drop(columns=feature, inplace=True)
        return self.df
