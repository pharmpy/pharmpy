import pytest


# This hook is needed to remove all globals of the module for the
# function under test. Otherwise we cannot detect a missing import
# of any function or variable from that module.
# The implementation is based on a hack where we assume that
# everything in the globals dict for the test that comes
# after the __builtins__ is coming from our module.
def pytest_runtest_setup(item):
    # Check if the item is a doctest
    found = False
    kept = dict()
    for key, value in item.dtest.globs.items():
        if key == '__builtins__':
            found = True
        elif found:
            continue
        kept[key] = value
    item.dtest.globs = kept


@pytest.fixture(scope="session", autouse=True)
def configure_pandas_display():
    """This must be in sync with docs/conf.py"""
    import pandas as pd

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
