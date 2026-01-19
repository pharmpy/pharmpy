from pathlib import Path

import pytest

from pharmpy.basic import Expr, Unit
from pharmpy.model import ColumnInfo, DataInfo, DataVariable


def test_datavariable_create():
    with pytest.raises(TypeError):
        DataVariable.create(1)
    with pytest.raises(ValueError):
        DataVariable.create("WGT", type="notadtypeq")

    var = DataVariable.create("WGT")
    assert dict(var.properties) == {}
    assert var.count is False

    with pytest.raises(ValueError):
        DataVariable.create("WGT", properties={'myprop': 23})


def test_datavariable_descriptor():
    var = DataVariable.create("DUMMY", properties={'descriptor': "body weight"})
    assert var.properties['descriptor'] == "body weight"
    with pytest.raises(ValueError):
        DataVariable.create("DUMMY2", properties={'descriptor': "notaknowndescriptor"})


def test_datavariable_type():
    with pytest.raises(ValueError):
        DataVariable.create("DUMMY", type="notaknowntype")
    var = DataVariable.create("DUMMY", type="id")
    assert var.type == 'id'

    var2 = DataVariable.create("DUMMY", type='dv')
    assert var2.type == 'dv'
    assert var2.count is False


def test_datavariable_categories():
    var = DataVariable.create("DUMMY", properties={'categories': [1, 2, 3]})
    assert var.properties['categories'] == (1, 2, 3)
    var = DataVariable.create("DUMMY", properties={'categories': (1, 2, 3)})
    assert var.properties['categories'] == (1, 2, 3)
    with pytest.raises(TypeError):
        DataVariable.create("DUMMY", properties={'categories': 1})


def test_datavariable_scale():
    with pytest.raises(ValueError):
        DataVariable.create("DUMMY", scale='notavalidscale')
    var = DataVariable.create("DUMMY", scale='nominal')
    assert var.scale == 'nominal'
    assert not var.count


def test_datavariable_count():
    DataVariable.create("DUMMY", scale="nominal")
    with pytest.raises(ValueError):
        DataVariable.create("DUMMY", scale="nominal", count=True)


def test_datavariable_symbol():
    var = DataVariable.create("DUMMY")
    assert var.symbol == Expr.symbol("DUMMY")


def test_datavariable_unit():
    var = DataVariable.create("DUMMY", properties={'unit': "nospecialunit"})
    assert var.properties['unit'] == Unit("nospecialunit")
    var = DataVariable.create("DUMMY", properties={'unit': "kg"})
    assert var.properties['unit'].unicode() == "kg"


def test_datavariable_is_numerical():
    var = DataVariable.create("DUMMY", scale='nominal')
    assert not var.is_numerical()
    var = DataVariable.create("DUMMY", scale='ratio')
    assert var.is_numerical()


def test_datavariable_repr():
    var = DataVariable.create("DUMMY")
    assert isinstance(repr(var), str)


def test_datavariable_eq():
    var1 = DataVariable.create("DUMMY")
    assert var1.__eq__(23) is NotImplemented


def test_columninfo_create():
    with pytest.raises(TypeError):
        ColumnInfo.create(1)
    var = DataVariable.create("WGT")
    with pytest.raises(ValueError):
        ColumnInfo.create("WGT", var, datatype="notadtypeq")
    var1 = DataVariable.create("DV1")
    var2 = DataVariable.create("DV2")
    with pytest.raises(TypeError):
        ColumnInfo.create("DV", {1.0: var1, 2: var2}, variable_id="DVID")
    with pytest.raises(TypeError):
        ColumnInfo.create("DV", {1.0: var1, 2: var2}, variable_id=1)
    with pytest.raises(ValueError):
        ColumnInfo.create("DV", {1.0: var1, 2: var2}, variable_id=None)

    var1 = DataVariable.create("DV1", type="dv")
    var2 = DataVariable.create("DV2", type="idv")
    with pytest.raises(ValueError):
        ColumnInfo.create("DV", {1: var1, 2: var2}, variable_id="DVID")


def test_columninfo_eq():
    col = ColumnInfo.create("DUMMY")
    assert col == col
    col2 = ColumnInfo.create("DUMMY2")
    assert col != col2


def test_columninfo_is_integer():
    col = ColumnInfo.create("DUMMY")
    assert not col.is_integer()
    col2 = ColumnInfo.create("DUMMY2", datatype="int32")
    assert col2.is_integer()


def test_columninfo_indexing():
    var1 = DataVariable.create("DV1")
    var2 = DataVariable.create("DV2")
    col = ColumnInfo.create("DV", {1: var1, 2: var2}, variable_id="DVID")
    assert col["DV1"].name == "DV1"
    assert col[2] == var2
    with pytest.raises(KeyError):
        col["DV3"]
    with pytest.raises(ValueError):
        col.variable

    var1 = DataVariable.create("DVID")
    col = ColumnInfo.create("DVID", var1)
    with pytest.raises(KeyError):
        col["DVID"]


def test_columninfo_to_dict():
    var1 = DataVariable.create("DV1")
    var2 = DataVariable.create("DV2")
    col = ColumnInfo.create("DV", {1: var1, 2: var2}, variable_id="DVID")
    assert col.to_dict() == {
        'name': 'DV',
        'drop': False,
        'datatype': 'float64',
        'variable_id': 'DVID',
        'variable_mapping': {
            '1': {
                'name': 'DV1',
                'type': 'unknown',
                'scale': 'ratio',
                'count': False,
                'properties': {},
            },
            '2': {
                'name': 'DV2',
                'type': 'unknown',
                'scale': 'ratio',
                'count': False,
                'properties': {},
            },
        },
    }


def test_columninfo_repr():
    col = ColumnInfo.create("DUMMY")
    correct = """drop             False
datatype       float64
variable_id       None
variables        DUMMY
Name: DUMMY"""
    assert repr(col) == correct


def test_columninfo_hash():
    var = DataVariable("DUMMY")
    col1 = ColumnInfo.create("DUMMY", var, datatype='float32')
    col2 = ColumnInfo.create("DUMMY", var, datatype='float64')
    assert hash(col1) != hash(col2)


def test_create():
    di = DataInfo.create()
    assert len(di) == 0
    di = DataInfo.create(columns=["COL1", "COL2"])
    assert len(di) == 2
    ci1 = ColumnInfo.create("COL1")
    ci2 = ColumnInfo.create("COL2")
    di = DataInfo.create(columns=[ci1, ci2])
    assert len(di) == 2
    di = DataInfo.create(path='dummy_path')
    assert di.path is not None
    di = DataInfo.create(separator=',')
    assert di.separator == ','
    with pytest.raises(TypeError):
        DataInfo.create(columns=1)
    with pytest.raises(TypeError):
        DataInfo.create(columns=[1])

    var1 = DataVariable.create("DV1", type="dv")
    var2 = DataVariable.create("DV2", type="dv")
    col1 = ColumnInfo.create("DV", {1: var1, 2: var2}, variable_id="DVID")
    with pytest.raises(ValueError):
        DataInfo.create([col1])

    with pytest.raises(ValueError):
        DataInfo.create(["DV", "DV"])


def test_eq():
    di1 = DataInfo.create()
    di2 = DataInfo.create(columns=["COL1", "COL2"])
    assert di2 == di2
    assert di1 != di2
    di3 = DataInfo.create(columns=["DUMMY1", "DUMMY2"])
    assert di2 != di3
    assert di2 != 23


def test_indexing():
    di = DataInfo.create(columns=["COL1", "COL2"])
    assert di[0].name == 'COL1'
    with pytest.raises(TypeError):
        di[1.0]
    col = ColumnInfo.create("COL3")
    di = col + di[1:]
    assert di["COL3"].name == "COL3"
    di = di[1:]
    assert len(di) == 1
    di = di[0:1] + col
    assert len(di) == 2


def test_id_column():
    di = DataInfo.create(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di = di.set_id_column('DUMMY')
    di = di.set_id_column('ID')
    assert di.id_column.name == 'ID'


def test_dv_label():
    di = DataInfo.create(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.set_dv_column('DUMMY')
    di = di.set_dv_column('DV')
    assert di.dv_column.name == 'DV'


def test_idv_label():
    di = DataInfo.create(['ID', 'TIME', 'DV'])
    with pytest.raises(IndexError):
        di.set_idv_column('DUMMY')
    di = di.set_idv_column('TIME')
    assert di.idv_column.name == 'TIME'


def test_get_set_column_type():
    di = DataInfo.create(['ID', 'TIME', 'DV'])
    di = di.set_id_column('ID')
    with pytest.raises(IndexError):
        di['DUMMY']
    with pytest.raises(TypeError):
        di['TIME'] = di['TIME'].replace(type='kzarqj')
    assert di['ID'].type == 'id'


def test_get_column_label():
    di = DataInfo.create(['ID', 'TIME', 'DV', 'WGT', 'APGR'])
    di = di.set_id_column('ID')
    di = di[0:3] + di[['WGT', 'APGR']].set_types('covariate')
    assert di.typeix['id'].names == ['ID']
    assert di.typeix['covariate'].names == ['WGT', 'APGR']
    with pytest.raises(IndexError):
        di.typeix['unknowntypeixq']


def test_missing_data_token():
    di = DataInfo.create(['ID', 'TIME', 'DV', 'WGT', 'APGR'])
    assert di.missing_data_token == '-99'


def test_json(tmp_path):
    var1 = DataVariable.create("ID", type='id', scale='nominal')
    col1 = ColumnInfo.create("ID", var1)
    var2 = DataVariable.create(
        "TIME", type='idv', scale='ratio', properties={'unit': 'h', 'descriptor': 'time after dose'}
    )
    col2 = ColumnInfo.create("TIME", var2)
    di = DataInfo.create([col1, col2])
    correct = '{"columns": [{"name": "ID", "drop": false, "datatype": "float64", "variable_id": null, "variable_mapping": {"name": "ID", "type": "id", "scale": "nominal", "count": false, "properties": {}}}, {"name": "TIME", "drop": false, "datatype": "float64", "variable_id": null, "variable_mapping": {"name": "TIME", "type": "idv", "scale": "ratio", "count": false, "properties": {"unit": "hour", "descriptor": "time after dose"}}}], "path": null, "separator": ",", "missing_data_token": "-99", "__version__": 1}'  # noqa E501
    assert di.to_json() == correct

    newdi = DataInfo.from_json(correct)
    assert newdi == di

    di.to_json(tmp_path / 'my.datainfo')
    assert (tmp_path / 'my.datainfo').is_file()


def test_path():
    di = DataInfo.create(["C1", "C2"])
    expected = Path.cwd() / "file.datainfo"
    di = di.replace(path=str(expected))
    assert di.path == expected
    di = di.replace(path=None)
    assert di.path is None


def test_types():
    di = DataInfo.create(['ID', 'TIME', 'DV'])
    di = di.set_id_column('ID').set_dv_column('DV').set_idv_column('TIME')
    assert di.types == ['id', 'idv', 'dv']


def test_repr():
    col1 = ColumnInfo.create("ID")
    col2 = ColumnInfo.create("WGT")
    di = DataInfo.create([col1, col2])
    assert type(repr(di)) == str


def test_read_json(testdata):
    di = DataInfo.read_json(testdata / 'nonmem' / 'models' / 'mox_simulated_normal.datainfo')
    assert DataInfo.from_dict(di.to_dict()) == di
    assert di['ID'].type == 'id'


def test_from_json():
    json = """
{
  "__version__": 1,
  "columns": [
    {
      "name": "ID",
      "variable_mapping" : {
          "name": "ID",
          "type": "id",
          "unit": "1",
          "scale": "nominal",
          "count": false
      },
      "datatype": "int32"
    },
    {
      "name": "WT",
      "datatype": "float64",
      "variable_mapping" : {
          "name": "WT",
          "type": "covariate",
          "properties": {"descriptor": "body weight"}
      }
    },
    {
      "name": "DVID",
      "variable_mapping" : {
          "name": "DVID",
          "type": "dvid"
      }
    },
    {
      "name": "DV",
      "variable_id": "DVID",
      "variable_mapping" : {
        "1": {
          "name": "DV1",
          "type": "dv"
          },
        "2": {
          "name": "DV2",
          "type": "dv"
          }
      }
    }
  ],
  "separator": ",",
  "path": "/home/mydir"
}
"""
    di = DataInfo.from_json(json)
    assert isinstance(di.path, Path)


def test_get_dtype_dict():
    var1 = DataVariable.create("ID", type='id')
    col1 = ColumnInfo.create("ID", var1, datatype='int32')
    var2 = DataVariable.create("WGT", type='covariate', properties={'descriptor': 'body weight'})
    col2 = ColumnInfo.create("WGT", var2)
    di = DataInfo.create([col1, col2])
    assert di.get_dtype_dict() == {'ID': 'int32', 'WGT': 'float64'}


def test_set_types():
    var1 = DataVariable.create("ID", type='id')
    col1 = ColumnInfo.create("ID", var1, datatype='int32')
    var2 = DataVariable.create("WGT", type='covariate', properties={'descriptor': 'body weight'})
    col2 = ColumnInfo.create("WGT", var2)
    di = DataInfo.create([col1, col2])
    di.set_types(['id', 'unknown'])
    with pytest.raises(ValueError):
        di.set_types(['id', 'unknown', 'unknown'])

    var1 = DataVariable.create("DVID", type="dvid")
    col1 = ColumnInfo.create("DVID", var1)
    var2 = DataVariable.create("DV1", type="dv")
    var3 = DataVariable.create("DV2", type="dv")
    col2 = ColumnInfo.create("DV", {1: var2, 2: var3}, variable_id="DVID")
    di = DataInfo.create([col1, col2])
    assert di['DV'].type == 'dv'
    di2 = di.set_types(['dvid', 'idv'])
    assert di2['DV'].type == 'idv'

    di3 = di2.set_dv_column("DV")
    assert di3['DV'].type == 'dv'


def test_set_id_column():
    var1 = DataVariable.create("ID", type='id')
    col1 = ColumnInfo.create("ID", var1, datatype='int32')
    var2 = DataVariable.create("WGT", type='covariate', properties={'descriptor': 'body weight'})
    col2 = ColumnInfo.create("WGT", var2)
    di = DataInfo.create([col1, col2])
    di.set_id_column('ID')
    with pytest.raises(ValueError):
        di.set_id_column("WGT")


def test_set_column():
    var1 = DataVariable.create("ID", type='id')
    col1 = ColumnInfo.create("ID", var1, datatype='int32')
    var2 = DataVariable.create("WGT", type='unknown')
    col2 = ColumnInfo.create("WGT", var2)
    di = DataInfo.create([col1, col2])
    assert di['WGT'].type == 'unknown'
    var3 = DataVariable.create("WGT", type='covariate', properties={'descriptor': 'body weight'})
    col3 = ColumnInfo.create("WGT", var3)
    di = di.set_column(col3)
    assert len(di) == 2
    assert di['WGT'].type == 'covariate'


def test_add():
    col1 = ColumnInfo.create("ID", datatype='int32')
    col2 = ColumnInfo.create("WGT")
    di = DataInfo.create([col1, col2])
    col3 = ColumnInfo.create("WGT2")
    newdi = (col3,) + di
    assert len(newdi) == 3

    newdi2 = di + (col3,)
    assert len(newdi2) == 3

    assert 'WGT2' in newdi2
    assert 'OTHER' not in newdi2


def test_is_categorical():
    var1 = DataVariable.create("ID", scale='nominal', type='id')
    assert var1.is_categorical()
    var2 = DataVariable.create("WGT", scale='ratio')
    assert not var2.is_categorical()


def test_convert_datatype_to_pd_dtype():
    dtype = ColumnInfo.convert_datatype_to_pd_dtype("nmtran-date")
    assert dtype == "str"
    dtype = ColumnInfo.convert_datatype_to_pd_dtype("float64")
    assert dtype == "float64"

    datatype = ColumnInfo.convert_pd_dtype_to_datatype("float64")
    assert datatype == 'float64'


def test_hash():
    col1 = ColumnInfo.create("ID")
    col2 = ColumnInfo.create("WGT")
    di1 = DataInfo.create([col1, col2])
    di2 = DataInfo.create([col1])
    assert hash(di1) != hash(di2)


def test_dict():
    col1 = ColumnInfo.create("ID")
    d = col1.to_dict()
    assert d == {
        'name': 'ID',
        'drop': False,
        'datatype': 'float64',
        'variable_id': None,
        'variable_mapping': {
            'type': 'unknown',
            'count': False,
            'name': 'ID',
            'scale': 'ratio',
            'properties': {},
        },
    }
    col2 = ColumnInfo.from_dict(d)
    assert col1 == col2


def test_find_single_column_name():
    var1 = DataVariable.create("ID", type='id')
    col1 = ColumnInfo.create("ID", var1)
    var2 = DataVariable.create("WGT", type='covariate')
    col2 = ColumnInfo.create("WGT", var2)
    col3 = ColumnInfo.create("DVID")
    di1 = DataInfo.create([col1, col2, col3])
    assert di1.find_single_column_name('id') == 'ID'
    with pytest.raises(ValueError):
        di1.find_single_column_name('dvid')
    assert di1.find_single_column_name('dvid', 'DVID') == 'DVID'
    var4 = DataVariable.create("AGE", type='covariate')
    col4 = ColumnInfo.create("AGE", var4)
    di2 = di1 + [col4]
    with pytest.raises(ValueError):
        di2.find_single_column_name('covariate')


def test_names_and_symbols():
    col1 = ColumnInfo.create("ID", datatype='int32')
    col2 = ColumnInfo.create("DV", datatype='float64')
    col3 = ColumnInfo.create("WGT")
    di = DataInfo.create([col1, col2, col3])
    names = ["ID", "DV", "WGT"]
    assert di.names == names
    assert di.symbols == list(map(Expr.symbol, names))


def test_find_column_by_property():
    var1 = DataVariable.create("ID", type='id', properties={'unit': 'kg'})
    col1 = ColumnInfo.create("ID", var1)
    var2 = DataVariable.create("DV", type='dv', properties={'unit': 'kg'})
    col2 = ColumnInfo.create("DV", var2)
    di = DataInfo.create([col1, col2])
    assert di.find_column_by_property("unit", Unit("kg")) is None


def test_get_property():
    var1 = DataVariable.create("ID", type='id', properties={'unit': 'kg'})
    with pytest.raises(KeyError):
        var1.get_property("descriptor")
    with pytest.raises(ValueError):
        var1.get_property("unknownprop")


def test_mapped_variable():
    var1 = DataVariable.create("DVID")
    col1 = ColumnInfo.create("DVID", var1)
    dv1 = DataVariable.create("DV1", type="dv")
    dv2 = DataVariable.create("DV2", type="dv")
    col2 = ColumnInfo.create("DV", variable_id="DVID", variable_mapping={1: dv1, 2: dv2})
    assert col2.name == "DV"
    assert len(col1) == 1
    assert len(col2) == 2
    assert col2.variable_id == "DVID"
    assert len(col2.variables) == 2
    di = DataInfo.create([col1, col2])
    assert len(di.variables) == 3
