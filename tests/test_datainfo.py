from pathlib import Path

import pytest
import sympy
import sympy.physics.units

from pharmpy.datainfo import ColumnInfo, DataInfo


def test_columninfo_init():
    with pytest.raises(TypeError):
        ColumnInfo(1)


def test_columninfo_type():
    col = ColumnInfo("DUMMY")
    with pytest.raises(TypeError):
        col.type = "notaknowntype"
    col.type = 'id'
    assert col.type == 'id'

    col2 = ColumnInfo("DUMMY", type='dv')
    assert col2.type == 'dv'
    assert col2.continuous


def test_columninfo_descriptor():
    col = ColumnInfo("DUMMY", descriptor="body weight")
    assert col.descriptor == "body weight"
    col2 = ColumnInfo("DUMMY2")
    assert col2.descriptor is None
    with pytest.raises(TypeError):
        col2.descriptor = "notaknowndescriptor"


def test_columninfo_scale():
    col = ColumnInfo("DUMMY")
    with pytest.raises(TypeError):
        col.scale = 'notavalidscale'
    col.scale = 'nominal'
    assert col.scale == 'nominal'
    assert not col.continuous


def test_columninfo_unit():
    col = ColumnInfo("DUMMY")
    col.unit = "nospecialunit"
    assert col.unit == sympy.Symbol("nospecialunit")
    col.unit = "kg"
    assert col.unit == sympy.physics.units.kg


def test_columninfo_continuous():
    col = ColumnInfo("DUMMY", scale="nominal")
    with pytest.raises(ValueError):
        col.continuous = True


def test_columninfo_is_numerical():
    col = ColumnInfo("DUMMY", scale='nominal')
    assert not col.is_numerical()
    col = ColumnInfo("DUMMY", scale='ratio')
    assert col.is_numerical()


def test_columninfo_repr():
    col = ColumnInfo("DUMMY", scale='nominal')
    correct = """              DUMMY
type        unknown
scale       nominal
continuous    False
categories     None
unit              1
drop          False
datatype    float64
descriptor     None"""
    assert repr(col) == correct


def test_init():
    di = DataInfo()
    assert len(di) == 0


def test_eq():
    di1 = DataInfo()
    di2 = DataInfo(columns=["COL1", "COL2"])
    assert di1 != di2
    di3 = DataInfo(columns=["DUMMY1", "DUMMY2"])
    assert di2 != di3


def test_indexing():
    di = DataInfo(columns=["COL1", "COL2"])
    assert di[0].name == 'COL1'
    with pytest.raises(TypeError):
        di[1.0]
    col = ColumnInfo("COL3")
    di[0] = col
    assert di["COL3"].name == "COL3"
    del di[0]
    assert len(di) == 1
    di.insert(1, col)
    assert len(di) == 2


def test_id_column():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.id_column = 'DUMMY'
    di.id_column = 'ID'
    assert di.id_column.name == 'ID'


def test_dv_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.dv_column = 'DUMMY'
    di.dv_column = 'DV'
    assert di.dv_column.name == 'DV'


def test_idv_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.idv_column = 'DUMMY'
    di.idv_column = 'TIME'
    assert di.idv_column.name == 'TIME'


def test_get_set_column_type():
    di = DataInfo(['ID', 'TIME', 'DV'])
    di['ID'].type = 'id'
    with pytest.raises(IndexError):
        di['DUMMY'].type = 'id'
    with pytest.raises(TypeError):
        di['TIME'].type = 'kzarqj'
    assert di['ID'].type == 'id'


def test_get_column_label():
    di = DataInfo(['ID', 'TIME', 'DV', 'WGT', 'APGR'])
    di['ID'].type = 'id'
    di[['WGT', 'APGR']].types = 'covariate'
    assert di.typeix['id'].names == ['ID']
    assert di.typeix['covariate'].names == ['WGT', 'APGR']


def test_unit():
    di = DataInfo(['ID', 'TIME', 'DV', 'WGT', 'APGR'])
    assert di['ID'].unit == 1


def test_scale():
    col = ColumnInfo('WGT', scale='ratio')
    assert col
    with pytest.raises(TypeError):
        ColumnInfo('DUMMY', scale='dummy')


def test_json():
    col1 = ColumnInfo("ID", type='id', scale='nominal')
    col2 = ColumnInfo("TIME", type='idv', scale='ratio', unit="h")
    di = DataInfo([col1, col2])
    correct = '{"columns": [{"name": "ID", "type": "id", "scale": "nominal", "continuous": false, "categories": null, "unit": "1", "datatype": "float64"}, {"name": "TIME", "type": "idv", "scale": "ratio", "continuous": true, "categories": null, "unit": "hour", "datatype": "float64"}], "path": null}'  # noqa: E501
    assert di.to_json() == correct

    newdi = DataInfo.from_json(correct)
    assert newdi == di


def test_path():
    di = DataInfo(["C1", "C2"])
    di.path = "file.datainfo"
    assert di.path == Path("file.datainfo")


def test_types():
    di = DataInfo(['ID', 'TIME', 'DV'])
    di.id_column = 'ID'
    di.dv_column = 'DV'
    di.idv_column = 'TIME'
    assert di.types == ['id', 'idv', 'dv']
