import pytest

@pytest.fixture(scope='module')
def records():
    from pysn import get_api
    return get_api('nonmem').records
