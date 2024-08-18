import pandas as pd
from typing import Sequence, Union

IPR_DATA_FILES = [
    # We only load the last file becuase it's cummulative w
    # with previous years
    # "ipr-seizures-fy19-fy21.csv",
    # "ipr-seizures-fy19-fy22.csv",
    "data/raw/ipr-seizures-fy19-fy23_0.csv",
]

HVI_DATA_FILES = ["data/raw/hvi-products.csv"]


def load_ipr(files: Sequence[str] = None) -> Union[pd.DataFrame, None]:
    """
    Loads IPR data set files.
    """
    # print(files)
    paths = files if files is not None and len(files) > 0 else IPR_DATA_FILES
    # print(paths)
    return pd.concat([pd.read_csv(f) for f in paths])


def load_hvi(files: Sequence[str] = None) -> Union[pd.DataFrame, None]:
    """
    Loads the High-Value Imports, Inc product list.
    """
    paths = files if files is not None and len(files) > 0 else HVI_DATA_FILES
    return pd.concat([pd.read_csv(f) for f in paths])
