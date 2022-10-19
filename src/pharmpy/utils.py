from pathlib import Path
from typing import Union

from pharmpy.deps import pandas as pd


def normalize_user_given_path(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        path = Path(path)
    return path.expanduser()


def hash_df(df) -> int:
    values = pd.util.hash_pandas_object(df, index=True).values
    return hash(tuple(values))
