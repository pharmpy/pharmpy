import pytest

import pharmpy.plugins.utils as utils


def test_detect_model():
    with pytest.raises(utils.PluginError):
        utils.detect_model(None)


def test_load_plugins():
    plugins = utils.load_plugins()
    names = [x.__name__ for x in plugins]
    assert names == ['pharmpy.plugins.nonmem']
