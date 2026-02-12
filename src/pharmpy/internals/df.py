from __future__ import annotations

from hashlib import sha256
from typing import Iterator, Union, overload

from pharmpy.deps import pandas as pd


def _pd_hash_values(obj: Union[pd.Index, pd.Series, pd.DataFrame]) -> pd.Series:
    # NOTE: We explicit all arguments for future-proofing
    return pd.util.hash_pandas_object(  # pyright: ignore [reportAttributeAccessIssue]
        obj, index=False, encoding='utf8', hash_key='0123456789123456', categorize=True
    )


def _df_hash_values(df: pd.DataFrame) -> Iterator[pd.Series]:
    yield _pd_hash_values(df.columns)
    yield _pd_hash_values(df.index)
    yield _pd_hash_values(df.dtypes)
    yield _pd_hash_values(df)


def hash_df_runtime(df: pd.DataFrame) -> int:
    return hash(tuple(map(lambda series: tuple(series.values), _df_hash_values(df))))


def hash_df_fs(df: pd.DataFrame) -> str:
    h = sha256()
    for series in _df_hash_values(df):
        h.update(series.to_numpy())
    return h.hexdigest()


def create_series(x, name: str) -> pd.Series:
    """Create a pandas series with index starting from 1"""
    ser = pd.Series(x, name=name, index=range(1, len(x) + 1))
    return ser


@overload
def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    pass


@overload
def reset_index(df: pd.Series) -> pd.Series:
    pass


def reset_index(df):
    df = df.set_axis(range(1, len(df) + 1), axis=0)
    return df
