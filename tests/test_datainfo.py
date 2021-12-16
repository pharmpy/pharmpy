import pytest

from pharmpy.datainfo import DataInfo


def test_id_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(KeyError):
        di.id_label = 'DUMMY'
    di.id_label = 'ID'
    assert di.id_label == 'ID'
