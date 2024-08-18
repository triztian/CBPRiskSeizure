from typing import Union


def to_percent(value: Union[float, int], total: Union[float, int]) -> float:
    """Converts a value into 2 decimals percent should mostlybe used for exploratory display as it reduces precision"""
    return round(float(value) / float(total) * 100.0, 2)
