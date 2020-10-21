import pytest

import pharmpy.config as config
import pharmpy.data as data


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


def test_read_config():
    config.read_configuration()
