import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_pandas_display():
    """This must be in sync with docs/conf.py"""
    import pandas as pd

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
