from pharmpy.basic import Unit
from pharmpy.model import Add, Assignment, Drop
from pharmpy.modeling import convert_unit, get_unit_of


def test_get_unit_of(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_real.mod")
    assert get_unit_of(model, "Y") == Unit("mg/L")
    assert get_unit_of(model, "V") == Unit("L")
    assert get_unit_of(model, "WGT") == Unit("kg")
    assert get_unit_of(model, "CL") == Unit("L/h")


def test_convert_unit(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    m2 = convert_unit(model, "WGT", "g")
    assert m2.statements[0] == Assignment.create("SCALED_WGT", "1000*WGT")
    assert len(m2.datainfo.provenance) == 1
    m2 = convert_unit(model, "WGT", "g", in_dataset=True)
    assert m2.statements[0].symbol.name == 'TVCL'
    assert m2.dataset['WGT'][1] == 1400.0
    assert Drop.create('WGT') in m2.datainfo.provenance
    assert Add.create('WGT') in m2.datainfo.provenance
    assert len(m2.datainfo.provenance) == 3

    m2 = convert_unit(model, "WGT", "kg", in_dataset=True)
    assert model == m2
    assert len(m2.datainfo.provenance) == 1
