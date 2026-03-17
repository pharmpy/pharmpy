from pharmpy.basic import Unit
from pharmpy.modeling import set_property, set_unit


def test_set_unit(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = set_unit(model, "WGT", "g")
    assert model.datainfo["WGT"].variable.get_property("unit") == Unit("g")


def test_set_property(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    cats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    model = set_property(model, "APGR", "categories", cats)
    assert model.datainfo['APGR'].variable.properties['categories'] == tuple(cats)
