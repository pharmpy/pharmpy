import pytest

import pharmpy.config as config
from pharmpy.model import data


def test_config_item():
    class TestClass:
        item1 = config.ConfigItem(23, "A single number")

    obj = TestClass()
    assert obj.item1 == 23
    obj.item1 = 91
    assert obj.item1 == 91
    with pytest.raises(TypeError):
        obj.item1 = "A"


def test_data_config():
    assert data.conf.na_values == [-99]
