import string, sys
from typing import Union, Tuple, List

import pandas as pd
from pandas.api.types import is_float_dtype, is_string_dtype

from sklearn.preprocessing import OneHotEncoder


def normalize_string_value(value: str) -> str:
    """
    Applies the normalization transformations.
    """
    new_name = value.lower()

    for c in list(new_name):
        if c in string.ascii_lowercase or c in string.digits:
            continue
        new_name = new_name.replace(c, "_")

    return new_name


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the column names, by removing spaces and special characters and replacing them with underscores.
    """
    renamed_cols = {}
    for column in df.columns:
        renamed_cols[column] = normalize_string_value(column)

    df.rename(columns=renamed_cols, inplace=True)

    return df


def remove_columns_missing(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """
    Removes columns with missing values that have equal or more than the provided threshold
    """
    total_obs = len(df)
    missing = df.isnull().sum().apply(lambda v: v / total_obs)

    drop_columns = set(missing[missing > threshold].index)
    print("remove_columns_missing:", drop_columns, file=sys.stderr)

    df.drop(columns=drop_columns, inplace=True)

    return df


def impute_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values for `dtype(float64)` columns using the mean.
    """
    for column in df.columns:
        if not is_float_dtype(df[column].dtype):
            continue

        avg = df[column].mean()
        df[column].fillna(avg, inplace=True)

    return df


def drop_ignored_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the following columns based on preliminary data inspection

      1. `centers_of_excellence` - We don't really care which center processes the seizure
      2. `mode_of_transportation` - We don't care how the products are being transported
      3. `fy` - We have no control over the year so we will not be using it.
    """

    df.drop(
        columns=["centers_of_excellence", "mode_of_transportation", "fy"], inplace=True
    )

    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes string values of string columns
    """
    for column in df.columns:
        if not is_string_dtype(df[column].dtype):
            continue

        df[column] = df[column].apply(normalize_string_value)

    return df


def add_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One hot encode categorical columns (string)
    """

    string_cols = []
    for column in df.columns:
        if not is_string_dtype(df[column].dtype):
            continue

        string_cols.append(column)

    df = pd.get_dummies(df, columns=string_cols, dtype=int, drop_first=True)

    return df


def join_columns(
    df: pd.DataFrame, columns: Union[List, Tuple[str]], drop_joined=True
) -> pd.DataFrame:
    """ """
    if not all(map(lambda col: col in df.columns, columns)):
        return df

    new_name = "_".join(columns)
    print("join_columns:", new_name, file=sys.stderr)

    df[new_name] = df.apply(
        lambda obs: "-".join(map(lambda col: str(obs[col]), columns)), axis="columns"
    )

    if drop_joined:
        df.drop(columns=columns, inplace=True, errors="ignore")

    return df


def drop_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columsn that have the same cardinality as the number of observations
    """
    observations = len(df)
    columns_to_drop = set()
    for column in df.columns:
        if len(df[column].unique()) == observations:
            columns_to_drop.add(column)

    df.drop(columns=columns_to_drop, inplace=True)

    return df


def add_group_count_column(
    df: pd.DataFrame, groupby: Union[List, Tuple[str]], target_count: str, name: str
) -> pd.DataFrame:
    if not all(map(lambda col: col in df.columns, groupby)):
        return df

    group_counts = df.groupby(groupby, as_index=False)[target_count].count()
    group_counts.rename(columns={target_count: name}, inplace=True)

    result = pd.merge(df, group_counts, left_on=groupby, right_on=groupby)

    return result


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alphabetically reorders the column
    """
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the following transformations:

      1. Column name normalization
      2. Removing columns with more than 20% missing
      3. Imputing missing values with average
      4. Drop columns that have a cardinality equal to the number of observations
    """
    return (
        df.copy()
        .pipe(normalize_column_names)
        .pipe(remove_columns_missing)
        .pipe(impute_missing_numeric)
        .pipe(drop_ignored_columns)
        .pipe(normalize_string_columns)
        .pipe(add_dummy_columns)
        .pipe(
            add_group_count_column,
            groupby=["unique_seizure_id"],
            target_count="line",
            name="line_count",
        )
        .pipe(join_columns, columns=["unique_seizure_id", "line"])
        .pipe(drop_unique_columns)
    )
