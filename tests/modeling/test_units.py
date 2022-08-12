from sympy.physics import units

from pharmpy.modeling import get_unit_of


def test_get_unit_of(load_model_for_test, testdata):
    model = load_model_for_test(testdata / "nonmem" / "pheno_real.mod")
    assert get_unit_of(model, "Y") == units.milligram / units.l
    assert get_unit_of(model, "V") == units.l
    assert get_unit_of(model, "WGT") == units.kilogram
