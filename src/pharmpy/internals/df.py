from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterator, Union, overload

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


def safe_convert_column_to_int32(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if all((_ids := df[col].astype('int32')) == df[col]):
        df = df.assign(**{col: _ids})
    return df


def pandas_to_dict(obj: pd.DataFrame | pd.Series) -> dict[str, Any]:
    # Serialize a pandas Series or DataFrame into a dict
    # We need this for json serialization because pandas
    # has multiple issues with its own json serialization
    # 1. Loss of precision of floating point numbers
    # 2. Loss of dtype of multiindex
    # 3. Duplication of column names bloating the file size

    is_series = isinstance(obj, pd.Series)
    df = obj.to_frame() if is_series else obj

    # Cannot use isinstance here since Index is both baseclass and specific Index
    if type(df.index) is pd.Index or type(df.index) is pd.MultiIndex:
        if isinstance(df.index, pd.MultiIndex):
            names = list(df.index.names)
        else:
            name = df.index.name
            # The default index name for anonymous index
            if name is None:
                name = 'index'
            names = [name]
        index = {'type': 'columns', 'columns': names}
        df = df.reset_index()
    elif isinstance(df.index, pd.RangeIndex):
        index = {
            'type': 'range',
            'start': df.index.start,
            'stop': df.index.stop,
            'step': df.index.step,
        }
    else:
        raise ValueError("Index type is not supported for serialization")

    dtypes = df.dtypes.astype(str).tolist()
    data = df.to_dict(orient="list")
    d = {'is_series': is_series, 'dtypes': dtypes, 'index': index, 'data': data}
    return d


def pandas_from_dict(d: dict[str, Any]) -> pd.Series | pd.DataFrame:
    df = pd.DataFrame(d['data'])
    dtype_dict = {col: dtype for col, dtype in zip(df.columns, d['dtypes'])}
    df = df.astype(dtype_dict)
    index = d['index']
    if index['type'] == 'columns':
        df = df.set_index(index['columns'])
        # Assume "index" to mean default nameless index
        if df.index.name == "index":
            df.index.name = None
    else:
        new_index = pd.RangeIndex(start=index['start'], stop=index['stop'], step=index['step'])
        df = df.set_index(new_index)
    if d['is_series']:
        df = df.iloc[:, 0]
    return df
