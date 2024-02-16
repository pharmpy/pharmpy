from pharmpy.basic import Unit
from pharmpy.modeling import get_unit_of


def test_get_unit_of(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_real.mod")
    assert get_unit_of(model, "Y") == Unit("mg/l")
    assert get_unit_of(model, "V") == Unit("l")
    assert get_unit_of(model, "WGT") == Unit("kg")
