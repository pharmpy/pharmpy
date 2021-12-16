import pytest

from pharmpy.datainfo import DataInfo


def test_id_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(KeyError):
        di.id_label = 'DUMMY'
    di.id_label = 'ID'
    assert di.id_label == 'ID'


def test_dv_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(KeyError):
        di.dv_label = 'DUMMY'
    di.dv_label = 'DV'
    assert di.dv_label == 'DV'


def test_idv_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(KeyError):
        di.idv_label = 'DUMMY'
    di.idv_label = 'TIME'
    assert di.idv_label == 'TIME'


def test_get_set_column_type():
    di = DataInfo(['ID', 'TIME', 'DV'])
    di.set_column_type('ID', 'id')
    with pytest.raises(KeyError):
        di.set_column_type('DUMMY', 'id')
    with pytest.raises(KeyError):
        di.set_column_type('TIME', 'kzarqj')
    assert di.get_column_type('ID') == 'id'
