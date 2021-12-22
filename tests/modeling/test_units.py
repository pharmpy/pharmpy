from sympy.physics import units

from pharmpy.modeling import get_unit_of, read_model


def test_get_unit_of(testdata):
    model = read_model(testdata / "nonmem" / "pheno_real.mod")
    assert get_unit_of(model, "Y") == units.milligram / units.l
    assert get_unit_of(model, "V") == units.l
    assert get_unit_of(model, "WGT") == units.kilogram
