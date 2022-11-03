from pharmpy.deps import pandas as pd


def hash_df(df) -> int:
    values = pd.util.hash_pandas_object(df, index=True).values
    return hash(tuple(values))
