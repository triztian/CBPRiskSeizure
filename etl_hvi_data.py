from typing import List, Iterable

import pandas as pd


def load_processed() -> pd.DataFrame:
    """
    Load the processed HVI data set.
    """
    return pd.read_csv("data/processed/hvi_data_processed.csv")


def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the following columns
    """
    df.drop(
        columns=[
            "category",
            "product",
            "individual_category",
            "description",
        ],
        inplace=True,
    )

    return df


def add_trading_partner_columns(
    df: pd.DataFrame, columns: List[str], trading_partner_dropped: str
) -> pd.DataFrame:
    """
    Adds the "dummy" columns for trading partners, these must be
    based on the same columns from the model training features
    """
    dfs = []
    for column in columns:
        dft = df.copy()

        dft[column] = 1
        for xcolumn in set(columns).difference([column]):
            dft[xcolumn] = 0

        dfs.append(dft)

    dummy = df.copy()
    for column in columns:
        dummy[column] = 0

    dfs.append(dummy)

    return pd.concat(dfs)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to match in features
    """
    df.rename(columns={"price_usd": "msrp"}, inplace=True)
    return df


def add_line_count_columns(
    df: pd.DataFrame, line_count_values: Iterable[int]
) -> pd.DataFrame:
    """ """
    dfs = []
    for line_count in line_count_values:
        dft = df.copy()
        dft["line_count"] = line_count
        dfs.append(dft)

    return pd.concat(dfs)


def transform(
    df: pd.DataFrame,
    trading_partner_columns: Iterable[str],
    trading_partner_dropped: str,
    line_count_values: Iterable[int],
    copy=True,
) -> pd.DataFrame:
    """
    Applies transformations specific to High-Value Imports product dataset
    """
    target = df
    if copy:
        target = df.copy()

    return (
        target.pipe(remove_columns)
        .pipe(
            add_trading_partner_columns,
            trading_partner_columns,
            trading_partner_dropped,
        )
        .pipe(rename_columns)
        .pipe(add_line_count_columns, line_count_values)
    )
