from pharmpy.basic import Unit
from pharmpy.modeling import set_unit


def test_set_unit(load_example_model_for_test):
    model = load_example_model_for_test("pheno")
    model = set_unit(model, "WGT", "g")
    assert model.datainfo["WGT"].variable.get_property("unit") == Unit("g")
