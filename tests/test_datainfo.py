from pathlib import Path

import pytest
import sympy
import sympy.physics.units

from pharmpy.datainfo import ColumnInfo, DataInfo


def test_columninfo_init():
    with pytest.raises(TypeError):
        ColumnInfo(1)
    with pytest.raises(ValueError):
        ColumnInfo("WGT", datatype="notadtypeq")


def test_columninfo_type():
    with pytest.raises(TypeError):
        col = ColumnInfo("DUMMY", type="notaknowntype")
    col = ColumnInfo("DUMMY", type="id")
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
        ColumnInfo("DUMMY2", descriptor="notaknowndescriptor")


def test_columninfo_scale():
    with pytest.raises(TypeError):
        col = ColumnInfo("DUMMY", scale='notavalidscale')
    col = ColumnInfo("DUMMY", scale='nominal')
    assert col.scale == 'nominal'
    assert not col.continuous


def test_columninfo_unit():
    col = ColumnInfo("DUMMY", unit="nospecialunit")
    assert col.unit == sympy.Symbol("nospecialunit")
    col = ColumnInfo("DUMMY", unit="kg")
    assert col.unit == sympy.physics.units.kg


def test_columninfo_continuous():
    ColumnInfo("DUMMY", scale="nominal")
    with pytest.raises(ValueError):
        ColumnInfo("DUMMY", scale="nominal", continuous=True)


def test_columninfo_is_numerical():
    col = ColumnInfo("DUMMY", scale='nominal')
    assert not col.is_numerical()
    col = ColumnInfo("DUMMY", scale='ratio')
    assert col.is_numerical()


def test_columninfo_repr():
    col = ColumnInfo("DUMMY", scale='nominal')
    correct = """type          unknown
scale         nominal
continuous      False
categories       None
unit                1
drop            False
datatype      float64
descriptor       None
Name: DUMMY"""
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
    di = col + di[1:]
    assert di["COL3"].name == "COL3"
    di = di[1:]
    assert len(di) == 1
    di = di[0:1] + col
    assert len(di) == 2


def test_id_column():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di = di.set_id_column('DUMMY')
    di = di.set_id_column('ID')
    assert di.id_column.name == 'ID'


def test_dv_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.set_dv_column('DUMMY')
    di = di.set_dv_column('DV')
    assert di.dv_column.name == 'DV'


def test_idv_label():
    di = DataInfo(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.set_idv_column('DUMMY')
    di = di.set_idv_column('TIME')
    assert di.idv_column.name == 'TIME'


def test_get_set_column_type():
    di = DataInfo(['ID', 'TIME', 'DV'])
    di = di.set_id_column('ID')
    with pytest.raises(IndexError):
        di['DUMMY']
    with pytest.raises(TypeError):
        di['TIME'] = di['TIME'].derive(type='kzarqj')
    assert di['ID'].type == 'id'


def test_get_column_label():
    di = DataInfo(['ID', 'TIME', 'DV', 'WGT', 'APGR'])
    di = di.set_id_column('ID')
    di = di[0:3] + di[['WGT', 'APGR']].set_types('covariate')
    assert di.typeix['id'].names == ['ID']
    assert di.typeix['covariate'].names == ['WGT', 'APGR']
    with pytest.raises(IndexError):
        di.typeix['unknowntypeixq']


def test_unit():
    di = DataInfo(['ID', 'TIME', 'DV', 'WGT', 'APGR'])
    assert di['ID'].unit == 1


def test_scale():
    col = ColumnInfo('WGT', scale='ratio')
    assert col
    with pytest.raises(TypeError):
        ColumnInfo('DUMMY', scale='dummy')


def test_json(tmp_path):
    col1 = ColumnInfo("ID", type='id', scale='nominal')
    col2 = ColumnInfo("TIME", type='idv', scale='ratio', unit="h", descriptor='time after dose')
    di = DataInfo([col1, col2])
    correct = '{"columns": [{"name": "ID", "type": "id", "scale": "nominal", "continuous": false, "categories": null, "unit": "1", "datatype": "float64", "drop": false}, {"name": "TIME", "type": "idv", "scale": "ratio", "continuous": true, "categories": null, "unit": "hour", "datatype": "float64", "drop": false, "descriptor": "time after dose"}], "path": null, "separator": ","}'  # noqa: E501
    assert di.to_json() == correct

    newdi = DataInfo.from_json(correct)
    assert newdi == di

    di.to_json(tmp_path / 'my.datainfo')
    assert (tmp_path / 'my.datainfo').is_file()


def test_path():
    di = DataInfo(["C1", "C2"])
    di = di.derive(path="file.datainfo")
    assert di.path == Path("file.datainfo")


def test_types():
    di = DataInfo(['ID', 'TIME', 'DV'])
    di = di.set_id_column('ID').set_dv_column('DV').set_idv_column('TIME')
    assert di.types == ['id', 'idv', 'dv']


def test_descriptor_indexer():
    col1 = ColumnInfo("ID", type='id')
    col2 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    di = DataInfo([col1, col2])
    bwci = di.descriptorix['body weight']
    assert len(bwci) == 1
    with pytest.raises(IndexError):
        di.descriptorix['nonexistingdescriptorq']


def test_repr():
    col1 = ColumnInfo("ID", type='id')
    col2 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    di = DataInfo([col1, col2])
    assert type(repr(di)) == str


def test_read_json(testdata):
    di = DataInfo.read_json(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.datainfo')
    assert di['ID'].type == 'id'


def test_from_json():
    json = """
{
  "columns": [
    {
      "name": "ID",
      "type": "id",
      "unit": "1",
      "scale": "nominal",
      "datatype": "int32",
      "continuous": false
    },
    {
      "name": "WT",
      "type": "covariate",
      "unit": "1",
      "scale": "ratio",
      "datatype": "float64",
      "descriptor": "body weight"
    }
  ],
  "separator": ",",
  "path": "/home/mydir"
}
"""
    di = DataInfo.from_json(json)
    assert isinstance(di.path, Path)


def test_get_dtype_dict():
    col1 = ColumnInfo("ID", type='id', datatype='int32')
    col2 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    di = DataInfo([col1, col2])
    assert di.get_dtype_dict() == {'ID': 'int32', 'WGT': 'float64'}


def test_set_types():
    col1 = ColumnInfo("ID", type='id', datatype='int32')
    col2 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    di = DataInfo([col1, col2])
    di.set_types(['id', 'unknown'])
    with pytest.raises(ValueError):
        di.set_types(['id', 'unknown', 'unknown'])


def test_set_id_column():
    col1 = ColumnInfo("ID", type='id', datatype='int32')
    col2 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    di = DataInfo([col1, col2])
    di.set_id_column('ID')
    with pytest.raises(ValueError):
        di.set_id_column("WGT")


def test_set_column():
    col1 = ColumnInfo("ID", type='id', datatype='int32')
    col2 = ColumnInfo("WGT", type='unknown')
    di = DataInfo([col1, col2])
    assert di['WGT'].type == 'unknown'
    col3 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    di = di.set_column(col3)
    assert len(di) == 2
    assert di['WGT'].type == 'covariate'


def test_add():
    col1 = ColumnInfo("ID", type='id', datatype='int32')
    col2 = ColumnInfo("WGT", type='unknown')
    di = DataInfo([col1, col2])
    col3 = ColumnInfo("WGT", type='covariate', descriptor='body weight')
    newdi = (col3,) + di
    assert len(newdi) == 3

    newdi2 = di + (col3,)
    assert len(newdi2) == 3


def test_is_categorical():
    col1 = ColumnInfo("ID", scale='nominal', type='id', datatype='int32')
    assert col1.is_categorical()
    col2 = ColumnInfo("WGT", scale='ratio')
    assert not col2.is_categorical()


def test_convert_datatype_to_pd_dtype():
    dtype = ColumnInfo.convert_datatype_to_pd_dtype("nmtran-date")
    assert dtype == "str"
    dtype = ColumnInfo.convert_datatype_to_pd_dtype("float64")
    assert dtype == "float64"

    datatype = ColumnInfo.convert_pd_dtype_to_datatype("float64")
    assert datatype == 'float64'
