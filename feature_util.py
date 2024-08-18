from os import getcwd
from datetime import datetime
from typing import BinaryIO, TextIO, Union, Tuple, List, Set, Iterable, Callable

from pickle import load, dump

import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

FileOrPath = Union[str, BinaryIO, TextIO]


def create_rfe_df(
    df: pd.DataFrame, n_features: int, target_feature: str
) -> pd.DataFrame:
    """
    Performs Recursive Feature Extraction and provide a DataFrame with the feature's index and ranking
    """
    X = df[df.columns.difference([target_feature])]
    y = df[target_feature]

    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=n_features)
    rfe.fit(X, y)

    features = []
    for i in range(X.shape[1]):
        if not rfe.support_[i]:
            continue
        features.append(
            {"index": i, "feature_name": X.columns[i], "rank": rfe.ranking_[i]}
        )

    features_df = pd.DataFrame(features, columns=["index", "feature_name", "rank"])

    return features_df


def write_split(
    X,
    y,
    columns: Union[Tuple[str], List[str], Set[str]],
    f: Union[str, TextIO, BinaryIO],
):
    """
    Write an X and y split dataset
    """
    print(X, y)
    df = pd.DataFrame(X, columns=columns)
    df[columns[-1]] = y

    if type(f) == str:
        with open(f, "w") as f:
            df.to_csv(f, index=False)
    else:
        df.to_csv(f, index=False)

    return df


def load_split(
    file_or_path: FileOrPath, y_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read a data set that was serialized as a split for testing and training
    """
    df = pd.read_csv(file_or_path)
    return df[df.columns.difference([y_name])], df[y_name]


def load_model_pickle(file_or_path: FileOrPath):
    """
    Loads a serialized
    """
    with open(file_or_path, "rb") as f:
        return load(f)


def fit_and_save(
    model: Pipeline, X_train: Union[pd.DataFrame, np.array], name: str, increment=True
) -> Pipeline:
    """
    Fits (train) a model and saves it to disk (pickle5)
    """
    model.fit(X_train)

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    filename = f"{name}_{timestamp}.pkl"
    with open(filename, "wb") as f:
        dump(model, f, protocol=5)

    return model


def reaorder_columns(
    df: pd.DataFrame, orderby: Union[Iterable[str], Callable[[str], int]]
) -> pd.DataFrame:
    """
    Reorders columns so that they can be used with the model.
    """
    return df[orderby]
